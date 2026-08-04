"""
Evaluate a trained checkpoint on a held-out split (default: test).

Produces:
  - <log_dir>/<exp_name>_test_per_sample.csv   one row per test sample
  - <log_dir>/<exp_name>_test_summary.json     mean/std/median per metric

Usage:
    python src/evaluate.py \
        --config experiments/e2_haunet_cascaded_stage1_test.YAML \
        --checkpoint checkpoints/e2_haunet_cascaded_stage1_test/best.pt \
        --split test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import math
import yaml
from torch.utils.data import DataLoader
import importlib

sys.path.append(str(Path(__file__).parent))
from data.wildfire_dataset import WildfireDataset, load_split_ids
from metrics import (
    compute_iou, compute_soft_iou, compute_ipsnr, compute_boundary_f1,
    compute_precision_recall, compute_auprc, compute_centroid_distance,
)
from utils.postprocess import postprocess_prediction

MODEL_REGISTRY = {
    "haunet": "models.haunet",
    "autoencoder": "models.autoencoder",
}

GAUSSIAN_LOSS_TYPES = {"gaussian_focal", "gaussianfocalloss", "focal_dice_gaussian", "focal_dice_gaussian_spatial"}

# Column order for the per-sample CSV.
CSV_COLUMNS = [
    "sample_id", "iou", "iou_t03", "soft_iou", "ipsnr", "boundary_f1",
    "precision", "recall", "pixel_f1", "auprc", "centroid_dist_m",
    "pp_iou", "pp_boundary_f1", "pp_precision", "pp_recall",
    "pp_pixel_f1", "pp_centroid_dist_m",
]


def build_model(cfg, device):
    model_name = cfg["model"]["name"]
    make_model = importlib.import_module(MODEL_REGISTRY[model_name]).make_model
    model = make_model(
        goes_channels=cfg["data"]["goes_channels"],
        cov_channels=cfg["data"]["cov_channels"],
        stage1_channels=cfg["data"].get("stage1_channels", 0),
        prior_fire_fraction=cfg["model"].get("prior_fire_fraction", 0.0085),
        use_cov_pyramid=cfg["model"].get("use_cov_pyramid", True),
    ).to(device)
    return model


@torch.no_grad()
def evaluate_sample(model, batch, device, threshold, low_threshold,
                     compute_ipsnr_flag, pixel_size_m,
                     do_postprocess, postprocess_cfg):
    """Run inference + compute all metrics for a single sample (batch_size=1)."""
    inp = batch["input"].to(device)
    target = batch["target"].to(device)

    logits = model(inp)
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.adaptive_max_pool2d(logits, target.shape[-2:])

    row = {
        "iou": compute_iou(logits, target, threshold=threshold),
        "iou_t03": compute_iou(logits, target, threshold=low_threshold),
        "soft_iou": compute_soft_iou(logits, target),
        "boundary_f1": compute_boundary_f1(logits, target, threshold=threshold),
        "auprc": compute_auprc(logits, target),
        "centroid_dist_m": compute_centroid_distance(logits, target, pixel_size_m=pixel_size_m),
    }

    pr = compute_precision_recall(logits, target, threshold=threshold)
    row["precision"] = pr["precision"]
    row["recall"] = pr["recall"]
    row["pixel_f1"] = pr["f1"]

    if compute_ipsnr_flag:
        ipsnr = compute_ipsnr(logits, target, threshold=threshold)
        row["ipsnr"] = ipsnr if ipsnr != float("inf") else float("nan")
    else:
        row["ipsnr"] = float("nan")

    if do_postprocess:
        probs = torch.sigmoid(logits)
        method = postprocess_cfg.get("method", "percentile")
        percentile = postprocess_cfg.get("percentile", 99.5)
        fixed_threshold = postprocess_cfg.get("fixed_threshold", 0.5)
        min_component_size = postprocess_cfg.get("min_component_size", 4)

        mask = postprocess_prediction(
            probs, method=method, percentile=percentile,
            fixed_threshold=fixed_threshold, min_component_size=min_component_size,
        )
        pseudo_logits = (mask * 20.0) - 10.0  # sigmoid(±10) ≈ 0/1

        row["pp_iou"] = compute_iou(pseudo_logits, target, threshold=0.5)
        row["pp_boundary_f1"] = compute_boundary_f1(pseudo_logits, target, threshold=0.5)
        pp_pr = compute_precision_recall(pseudo_logits, target, threshold=0.5)
        row["pp_precision"] = pp_pr["precision"]
        row["pp_recall"] = pp_pr["recall"]
        row["pp_pixel_f1"] = pp_pr["f1"]
        row["pp_centroid_dist_m"] = compute_centroid_distance(pseudo_logits, target, pixel_size_m=pixel_size_m)

    return row

@torch.no_grad()
def evaluate_baseline_sample(batch, device, threshold, low_threshold,
                              compute_ipsnr_flag, pixel_size_m):
    """Same metrics as evaluate_sample, but for a trivial always-background
    prediction i.e. logits fixed at a large negative value everywhere,
    so sigmoid(logits) ≈ 0 for every pixel."""
    target = batch["target"].to(device)
    # Large negative constant logit -> sigmoid ≈ 0 across the whole image
    logits = torch.full_like(target, -20.0)

    row = {
        "iou": compute_iou(logits, target, threshold=threshold),
        "iou_t03": compute_iou(logits, target, threshold=low_threshold),
        "soft_iou": compute_soft_iou(logits, target),
        "boundary_f1": compute_boundary_f1(logits, target, threshold=threshold),
        "auprc": compute_auprc(logits, target),
        "centroid_dist_m": compute_centroid_distance(logits, target, pixel_size_m=pixel_size_m),
    }
    pr = compute_precision_recall(logits, target, threshold=threshold)
    row["precision"] = pr["precision"]
    row["recall"] = pr["recall"]
    row["pixel_f1"] = pr["f1"]
    row["ipsnr"] = float("nan")  # undefined/meaningless for a constant predictor
    return row

@torch.no_grad()
def evaluate_prevalence_baseline_sample(batch, device, threshold, low_threshold,
                                         pixel_size_m, prior_logit):
    """Constant-probability baseline: every pixel gets the same predicted
    probability, equal to the training-set positive-pixel prevalence. Unlike
    the all-zero baseline, it can produce nonzero precision/recall at
    low thresholds."""
    target = batch["target"].to(device)
    logits = torch.full_like(target, prior_logit)

    row = {
        "iou": compute_iou(logits, target, threshold=threshold),
        "iou_t03": compute_iou(logits, target, threshold=low_threshold),
        "soft_iou": compute_soft_iou(logits, target),
        "boundary_f1": compute_boundary_f1(logits, target, threshold=threshold),
        "auprc": compute_auprc(logits, target),
        "centroid_dist_m": compute_centroid_distance(logits, target, pixel_size_m=pixel_size_m),
    }
    pr = compute_precision_recall(logits, target, threshold=threshold)
    row["precision"] = pr["precision"]
    row["recall"] = pr["recall"]
    row["pixel_f1"] = pr["f1"]
    row["ipsnr"] = float("nan")
    return row


def summarize(rows: list) -> dict:
    """Mean / std / median / min / max per numeric column, ignoring NaNs."""
    summary = {}
    if not rows:
        return summary
    keys = [k for k in rows[0].keys() if k != "sample_id"]
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(len(vals)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained wildfire model on a test split.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file to evaluate.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test).")
    parser.add_argument("--save-preds", action="store_true",
                        help="Also save raw sigmoid prediction arrays (.npy) per sample.")
    parser.add_argument("--preds-dir", default=None,
                        help="Directory to save predictions to (default: <log_dir>/<exp_name>_preds).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    branch = cfg["experiment"]["branch"]
    bucket_name = cfg["data"]["bucket_name"]
    prefix = cfg["data"]["output_prefix"]
    loss_type = cfg["loss"]["type"].lower()

    print(f"\n{'='*60}")
    print(f"Evaluating : {exp_name}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"{'='*60}\n")

    default_heatmap = loss_type in GAUSSIAN_LOSS_TYPES
    heatmap_target = cfg["data"].get("heatmap_target", default_heatmap)
    dilate_target = False #cfg["data"].get("dilate_target", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    splits_dir = cfg["data"].get("splits_dir", "splits")
    test_ids = load_split_ids(bucket_name, branch, args.split, prefix, splits_dir)
    print(f"Loaded {len(test_ids)} '{args.split}' sample ids.")

    use_stage1_pred = cfg["data"].get("use_stage1_pred", False)
    stage1_pred_key = cfg["data"].get("stage1_pred_key", "stage1_pred")
    test_ds = WildfireDataset(
        bucket_name, prefix, branch, split_ids=test_ids,
        dilate_target=False, heatmap_target=False,
        goes_mean=cfg["data"].get("goes_mean"),
        goes_std=cfg["data"].get("goes_std"),
        dem_mean=cfg["data"].get("dem_mean"),
        dem_std=cfg["data"].get("dem_std"),
        heatmap_sigma=cfg["data"].get("heatmap_sigma", 0.6),
        use_stage1_pred=use_stage1_pred, stage1_pred_key=stage1_pred_key, augment=False,
    )
    # batch_size=1 so each forward pass corresponds to exactly one sample_id
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg["data"]["num_workers"])
    print(f"Test dataset: {len(test_ds)} samples.")
    if len(test_ds) == 0:
        raise RuntimeError(
            f"Test dataset is empty after filtering: check stage1_pred_key "
            f"('{stage1_pred_key}') matches the manifest paths, or that "
            f"'{args.split}' samples actually have that key populated."
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)

    model_cfg = ckpt.get("config", cfg)
    if "config" not in ckpt:
        print("  [WARN] Checkpoint has no saved 'config' -- building model "
              "from --config instead. Verify channel counts match training.")
    elif model_cfg.get("data", {}).get("goes_channels") != cfg["data"]["goes_channels"] or \
         model_cfg.get("data", {}).get("cov_channels") != cfg["data"]["cov_channels"] or \
         model_cfg.get("model", {}).get("use_cov_pyramid", True) != cfg.get("model", {}).get("use_cov_pyramid", True):
        print("  [WARN] --config differs from the config saved in the "
              "checkpoint (channels or use_cov_pyramid). Using the checkpoint's "
              "config for model architecture; eval settings still come from --config.")

    model = build_model(model_cfg, device)
    result = model.load_state_dict(ckpt["model"], strict=False)
    if result.missing_keys:
        print(f"  [WARN] Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  [WARN] Unexpected keys: {result.unexpected_keys}")
    if not result.missing_keys and not result.unexpected_keys:
        print("  Model weights loaded successfully — all keys matched.")
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_best_metric = ckpt.get("best_metric", "?")
    ckpt_best_scores = ckpt.get("best_scores", {})
    ckpt_best_score = ckpt_best_scores.get(ckpt_best_metric, ckpt.get("best_score", "?"))
    print(f"  Checkpoint from epoch {ckpt_epoch} "
          f"({ckpt_best_metric}={ckpt_best_score})")

    # ------------------------------------------------------------------
    # Eval config (mirrors train.py so numbers are directly comparable)
    # ------------------------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    eval_threshold = float(eval_cfg.get("threshold", 0.5))
    eval_low_threshold = float(eval_cfg.get("low_threshold", 0.3))
    eval_compute_ipsnr = bool(eval_cfg.get("compute_ipsnr", True))
    eval_pixel_size_m = float(eval_cfg.get("pixel_size_m", 384.0))

    do_postprocess = bool(eval_cfg.get("log_postprocessed", False))
    postprocess_cfg = cfg.get("postprocess", {})

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    ckpt_tag = Path(args.checkpoint).stem   # e.g. "best_val_auprc" from "best_val_auprc.pt"

    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"{exp_name}_{ckpt_tag}_{args.split}_per_sample.csv"
    summary_path = log_dir / f"{exp_name}_{ckpt_tag}_{args.split}_summary.json"

    preds_dir = None
    if args.save_preds:
        preds_dir = Path(args.preds_dir) if args.preds_dir else (log_dir / f"{exp_name}_preds")
        preds_dir.mkdir(parents=True, exist_ok=True)

    columns = [c for c in CSV_COLUMNS if do_postprocess or not c.startswith("pp_")]

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for i, batch in enumerate(test_loader):
            sample_id = batch["sample_id"][0]  # batch_size=1; dataset returns manifest["id"] per item

            row = evaluate_sample(
                model, batch, device,
                threshold=eval_threshold,
                low_threshold=eval_low_threshold,
                compute_ipsnr_flag=eval_compute_ipsnr,
                pixel_size_m=eval_pixel_size_m,
                do_postprocess=do_postprocess,
                postprocess_cfg=postprocess_cfg,
            )
            row["sample_id"] = sample_id
            rows.append(row)

            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items() if k in columns})

            if args.save_preds:
                inp = batch["input"].to(device)
                with torch.no_grad():
                    logits = model(inp)
                    target_shape = batch["target"].shape[-2:]
                    if logits.shape[-2:] != target_shape:
                        logits = F.adaptive_max_pool2d(logits, target_shape)
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                np.save(preds_dir / f"{sample_id}.npy", probs)

            if (i + 1) % 25 == 0 or (i + 1) == len(test_ds):
                print(f"  [{i+1:4d}/{len(test_ds)}] {sample_id}: "
                      f"iou={row['iou']:.4f} bf1={row['boundary_f1']:.4f}", flush=True)

    print(f"\nPer-sample metrics written to {csv_path}")

    # ------------------------------------------------------------------
    # All-zero baseline
    # ------------------------------------------------------------------
    baseline_rows = []
    for batch in test_loader:
        brow = evaluate_baseline_sample(
            batch, device,
            threshold=eval_threshold,
            low_threshold=eval_low_threshold,
            compute_ipsnr_flag=eval_compute_ipsnr,
            pixel_size_m=eval_pixel_size_m,
        )
        baseline_rows.append(brow)
    baseline_summary = summarize(baseline_rows)

    # ------------------------------------------------------------------
    # Prevalence-rate constant baseline
    # ------------------------------------------------------------------
    prior_p = cfg["model"].get("prior_fire_fraction", 0.0085)
    prior_logit = math.log(prior_p / (1 - prior_p))
    prevalence_rows = []
    for batch in test_loader:
        prow = evaluate_prevalence_baseline_sample(
            batch, device,
            threshold=eval_threshold,
            low_threshold=eval_low_threshold,
            pixel_size_m=eval_pixel_size_m,
            prior_logit=prior_logit,
        )
        prevalence_rows.append(prow)
    prevalence_summary = summarize(prevalence_rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = summarize(rows)
    summary_out = {
        "experiment": exp_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt_epoch,
        "split": args.split,
        "n_samples": len(rows),
        "metrics": summary,
        "baseline_all_zero": baseline_summary,
        "baseline_prevalence_rate": prevalence_summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Summary written to {summary_path}\n")

    print(f"{'Metric':<18}{'Model':>10}{'AllZero':>10}{'Prevalence':>12}")
    print("-" * 50)
    for k, v in summary.items():
        b0 = baseline_summary.get(k, {}).get("mean", float("nan"))
        bp = prevalence_summary.get(k, {}).get("mean", float("nan"))
        print(f"{k:<18}{v['mean']:>10.4f}{b0:>10.4f}{bp:>12.4f}")


if __name__ == "__main__":
    main()