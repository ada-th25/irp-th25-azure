"""
Small hyperparameter sweep for HAUNet Stage 1, run against splits_small before
committing to the full dataset.
"""

import argparse
import copy
import csv
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

BASE_CONFIG_PATH = "tune_base_config.yaml"
TRAIN_SCRIPT = "src/train.py"
TRIAL_TIMEOUT_S = 2100   # 35 min hard cap per trial -- sized for 3 epochs at ~9min/epoch on a T4

LOSS_TYPE_WEIGHTS = {
    "focal_dice": 0.5,
    "focal_tversky": 0.25,
    "gaussian_focal": 0.25,
}


def sample_loss_type() -> str:
    types = list(LOSS_TYPE_WEIGHTS.keys())
    weights = list(LOSS_TYPE_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


def sample_loss_params(loss_type: str) -> dict:
    if loss_type == "focal_dice":
        return {
            "alpha": random.uniform(0.75, 0.9),
            "gamma": random.uniform(1.0, 2.0),
            "dice_weight": random.uniform(0.7, 1.3),
            "smooth": random.choice([0.001, 0.01, 0.1]),
        }
    elif loss_type == "focal_tversky":
        return {
            "alpha": random.uniform(0.6, 0.9),
            "gamma": random.uniform(1.0, 2.5),
            "tversky_alpha": random.uniform(0.2, 0.4),
            "tversky_beta": random.uniform(0.6, 0.8),
        }
    elif loss_type == "gaussian_focal":
        return {
            "gaussian_alpha": random.choice([1.5, 2.0, 2.5]),
            "gaussian_beta": random.choice([2.0, 4.0, 6.0]),
        }
    else:
        raise ValueError(f"Unhandled loss_type in sweep: {loss_type}")


def sample_trial_config(base_cfg: dict, trial_id: int) -> tuple[dict, dict]:
    cfg = copy.deepcopy(base_cfg)

    loss_type = sample_loss_type()
    loss_params = sample_loss_params(loss_type)
    learning_rate = 10 ** random.uniform(-5, -3.3)
    dilate_target = False   # pinned -- matches proven e2 config, not swept

    exp_name = f"tune_haunet_stage1_t{trial_id:02d}_{loss_type}"
    cfg["experiment"]["name"] = exp_name
    cfg["checkpointing"]["gcs_checkpoint_prefix"] = f"processed_dataset_gcp/checkpoints/{exp_name}"

    cfg["training"]["learning_rate"] = learning_rate
    cfg["data"]["dilate_target"] = dilate_target

    cfg["loss"] = {"type": loss_type, **loss_params}
    if loss_type in ("focal_dice", "focal_tversky"):
        cfg["loss"]["focal_weight"] = 1.0
    if loss_type == "focal_dice":
        cfg["loss"]["dice_weight"] = loss_params["dice_weight"]
    if loss_type == "focal_tversky":
        cfg["loss"]["tversky_weight"] = 1.0

    sampled = {
        "loss_type": loss_type,
        "learning_rate": learning_rate,
        "dilate_target": dilate_target,
        **loss_params,
    }
    return cfg, sampled


def _write_sidecar(cfg: dict, sampled: dict, log_dir: str, exp_name: str) -> None:
    """Permanent record of exactly what this trial was configured with --
    the CSV log only has epoch metrics, never the sampled hyperparameters,
    so without this sidecar there's no reliable way to know later (e.g.
    across multiple sweep runs) what config produced a given result."""
    sidecar_path = Path(log_dir) / f"{exp_name}.sampled.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump({"exp_name": exp_name, **sampled}, f, indent=2)


def _read_best_auprc(csv_path: Path) -> tuple[float, int]:
    if not csv_path.exists():
        return float("-inf"), 0
    best = float("-inf")
    n = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            val = row.get("val_auprc", "")
            if val:
                best = max(best, float(val))
                n += 1
    return best, n


def run_trial(cfg: dict, sampled: dict, epochs: int) -> tuple[float, str]:
    """Writes cfg to a temp YAML plus a permanent sidecar JSON, runs train.py
    with output streamed live and a hard timeout, returns (best_val_auprc,
    csv_path). On timeout or failure, still tries to recover whatever
    epochs' val_auprc were already logged before the process was killed."""
    cfg["training"]["epochs"] = epochs
    cfg["training"]["val_every_n_epochs"] = 1
    cfg["training"]["save_every_n_epochs"] = epochs

    exp_name = cfg["experiment"]["name"]
    log_dir = cfg["logging"]["log_dir"]
    csv_path = Path(log_dir) / f"{exp_name}.csv"

    _write_sidecar(cfg, sampled, log_dir, exp_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmp_config_path = f.name

    print(f"\n{'='*60}\nTrial: {exp_name}\n{'='*60}")
    print("Config:", tmp_config_path)
    with open(tmp_config_path) as _f:
        print(_f.read())

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT, "--config", tmp_config_path, "--no-wandb"],
            timeout=TRIAL_TIMEOUT_S,
        )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] Trial exceeded {TRIAL_TIMEOUT_S}s (ran {elapsed:.0f}s) -- killed. "
              f"Checking for partial results...")
        best_auprc, n_rows = _read_best_auprc(csv_path)
        if n_rows > 0:
            print(f"  Recovered {n_rows} epoch(s) of partial results. "
                  f"Best val_auprc before timeout: {best_auprc:.4f}")
            return best_auprc, str(csv_path)
        print("  No usable partial results found.")
        return float("-inf"), None

    elapsed = time.time() - t0
    print(f"  Trial wall time: {elapsed:.0f}s")

    if returncode != 0:
        print(f"  [FAILED] train.py exited non-zero (code {returncode}). "
              f"See output above for the traceback. If there was no traceback "
              f"printed, check `dmesg` for an OOM kill.")
        best_auprc, n_rows = _read_best_auprc(csv_path)
        if n_rows > 0:
            print(f"  Recovered {n_rows} epoch(s) of partial results despite failure. "
                  f"Best val_auprc: {best_auprc:.4f}")
            return best_auprc, str(csv_path)
        return float("-inf"), None

    if not csv_path.exists():
        print(f"  [WARN] Expected CSV log not found at {csv_path}")
        return float("-inf"), None

    best_auprc, n_rows = _read_best_auprc(csv_path)
    print(f"  Best val_auprc this trial: {best_auprc:.4f}")
    return best_auprc, str(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--base-config", default=BASE_CONFIG_PATH)
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    results = []
    for trial_id in range(1, args.n_trials + 1):
        cfg, sampled = sample_trial_config(base_cfg, trial_id)
        score, csv_path = run_trial(cfg, sampled, args.epochs)
        results.append({"trial": trial_id, "val_auprc": score, **sampled})

        if trial_id == 1:
            print(f"\n[INFO] First trial complete. If wall time above was long, "
                  f"consider lowering --epochs or --n-trials for the remaining "
                  f"{args.n_trials - 1} trials before they all run.\n")

    results.sort(key=lambda r: r["val_auprc"], reverse=True)

    print(f"\n{'='*60}\nSweep complete -- ranked by val_auprc\n{'='*60}")
    for r in results:
        extra = ", ".join(f"{k}={v}" for k, v in r.items()
                           if k not in ("trial", "val_auprc", "loss_type"))
        print(f"  trial={r['trial']:2d}  loss={r['loss_type']:15s}  "
              f"val_auprc={r['val_auprc']:.4f}  {extra}")

    if results:
        best = results[0]
        print(f"\nBest trial: {best['trial']} (loss={best['loss_type']}, val_auprc={best['val_auprc']:.4f})")
        print("Sampled hyperparameters for this trial:")
        for k, v in best.items():
            if k not in ("trial", "val_auprc"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
