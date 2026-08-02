"""
Plot GOES bands, covariates, true target, and predicted fire probability
for a handful of val/train/test samples, side by side.

Usage:
    # Basic prediction plot
    python -m src.utils.plot_predictions \
        --config experiments/e2_haunet_cascaded_stage1_test.yaml \
        --checkpoint checkpoints_test/e2_haunet_viirs_cascaded_stage1_test/epoch_010.pt \
        --n-samples 4 \
        --out predictions.png

    # With post-processing overlay and seed
    python -m src.utils.plot_predictions \
        --config experiments/e2_haunet_cascaded_stage1_test.yaml \
        --checkpoint checkpoints_test/e2_haunet_viirs_cascaded_stage1_test/epoch_010.pt \
        --show-postprocessed \
        --pp-method percentile \
        --pp-percentile 99.5 \
        --seed 42
"""
import argparse
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from src.data.wildfire_dataset import WildfireDataset, load_split_ids
from postprocess import postprocess_prediction

from src.models.autoencoder import make_model as make_autoencoder_model
from src.models.haunet import make_model as make_haunet_model


@contextmanager
def timed(label):
    t0 = time.time()
    yield
    print(f"[timing] {label}: {time.time() - t0:.2f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to sample from (train, val, or test).",
    )
    p.add_argument("--out", default="predictions.png")
    p.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. 0 is usually faster for a single batch.")
    p.add_argument("--seed", type=int, default=None,
                    help="Optional seed for reproducible sample selection.")
    p.add_argument("--show-postprocessed", action="store_true",
                    help="Add columns showing the post-processed binary mask alongside raw prediction.")
    p.add_argument("--pp-method", default="percentile", choices=["percentile", "fixed"])
    p.add_argument("--pp-percentile", type=float, default=99.5)
    p.add_argument("--pp-fixed-threshold", type=float, default=0.5)
    p.add_argument("--pp-min-component-size", type=int, default=4)
    return p.parse_args()


def build_model(cfg: dict, config_path: str) -> torch.nn.Module:
    model_type = cfg.get("model", {}).get("name", "").lower()
    exp_name = cfg.get("experiment", {}).get("name", "").lower()
    config_str = config_path.lower()

    # Route to Autoencoder if specified in model type, experiment name, or config filename
    if "autoencoder" in model_type or "autoencoder" in exp_name or "autoencoder" in config_str:
        return make_autoencoder_model(
            goes_channels=cfg["data"]["goes_channels"],
            cov_channels=cfg["data"]["cov_channels"],
            stage1_channels=cfg["data"].get("stage1_channels", 0),
        )
    
    return make_haunet_model(
        goes_channels=cfg["data"]["goes_channels"],
        cov_channels=cfg["data"]["cov_channels"],
        stage1_channels=cfg["data"].get("stage1_channels", 0),
    )


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    with timed("load config"):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    branch = cfg["experiment"]["branch"]
    bucket_name = cfg["data"]["bucket_name"]
    prefix = cfg["data"]["output_prefix"]
    splits_dir = cfg["data"].get("splits_dir", "splits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with timed("load split ids"):
        ids = load_split_ids(bucket_name, branch, args.split, prefix, splits_dir)
    print(f"  -> {len(ids)} ids in split '{args.split}'")

    with timed("construct WildfireDataset"):
        ds = WildfireDataset(
            bucket_name, prefix, branch, split_ids=ids,
            goes_mean=cfg["data"].get("goes_mean"),
            goes_std=cfg["data"].get("goes_std"),
            use_stage1_pred=cfg["data"].get("use_stage1_pred", False),
            stage1_pred_key=cfg["data"].get("stage1_pred_key", "stage1_pred"),
        )

    n = min(args.n_samples, len(ds))
    chosen_idx = random.sample(range(len(ds)), n)

    if args.num_workers == 0:
        with timed(f"fetch {n} samples directly (no DataLoader workers)"):
            samples = [ds[i] for i in chosen_idx]
        batch = torch.utils.data.dataloader.default_collate(samples)
    else:
        from torch.utils.data import Subset
        subset = Subset(ds, chosen_idx)
        loader = DataLoader(subset, batch_size=n, shuffle=False, num_workers=args.num_workers)
        with timed(f"fetch {n} samples via DataLoader (num_workers={args.num_workers})"):
            batch = next(iter(loader))

    inp = batch["input"].to(device)
    target = batch["target"].to(device)
    goes = batch["goes"]
    cov = batch["covariates"]

    use_stage1_pred = cfg["data"].get("use_stage1_pred", False)
    stage1_pred_np = None
    if use_stage1_pred:
        # input channel layout in WildfireDataset.__getitem__: [goes_up, stage1_pred(if enabled), cov]
        goes_ch = cfg["data"]["goes_channels"]
        stage1_pred_np = inp[:, goes_ch:goes_ch+1].cpu().numpy()  # (B, 1, H, W)

    with timed("build model"):
        model = build_model(cfg, args.config).to(device)

    if args.checkpoint:
        with timed("load checkpoint"):
            ckpt = torch.load(args.checkpoint, map_location=device)
            # Handle checkpoints saved under "model" or "state_dict" keys
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', 'N/A')})")
    else:
        print("No checkpoint given -- using freshly initialised weights.")

    model.eval()

    with timed("inference"):
        with torch.no_grad():
            logits = model(inp)
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.adaptive_avg_pool2d(logits, target.shape[-2:])
            probs = torch.sigmoid(logits)

            if args.show_postprocessed:
                pp_mask = postprocess_prediction(
                    probs, method=args.pp_method, percentile=args.pp_percentile,
                    fixed_threshold=args.pp_fixed_threshold,
                    min_component_size=args.pp_min_component_size,
                )

    base_titles = ["GOES C07", "GOES C14", "GOES TFI", "DEM (cov[0])", "Land cover (cov[1])"]
    if use_stage1_pred:
        base_titles.append("Stage1 P(fire) input")
    base_titles += ["True target", "Predicted P(fire)"]
    col_titles = list(base_titles)
    if args.show_postprocessed:
        col_titles.append("Post-processed mask")
    n_cols = len(col_titles)

    with timed("plot + save figure"):
        fig, axes = plt.subplots(n, n_cols, figsize=(3.2 * n_cols, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row in range(n):
            goes_np = goes[row].numpy()
            cov_np = cov[row].numpy()
            target_np = target[row, 0].cpu().numpy()
            prob_np = probs[row, 0].cpu().numpy()

            panels = [goes_np[0], goes_np[1], goes_np[2], cov_np[0], cov_np[1]]
            cmaps = ["inferno", "inferno", "inferno", "terrain", "tab20"]
            if use_stage1_pred:
                panels.append(stage1_pred_np[row, 0])
                cmaps.append("hot")
            panels += [target_np, prob_np]
            cmaps += ["hot", "hot"]
            if args.show_postprocessed:
                panels.append(pp_mask[row, 0].cpu().numpy())
                cmaps.append("gray")

            for col in range(n_cols):
                ax = axes[row, col]
                im = ax.imshow(panels[col], cmap=cmaps[col])
                
                # Add colorbar specifically for target and prediction to show value ranges
                if col_titles[col] in ["True target", "Predicted P(fire)", "Stage1 P(fire) input"]:
                    im.set_clim(0.0, 1.0)  # Pin range to [0, 1] for probability scale
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)

                if row == 0:
                    ax.set_title(col_titles[col], fontsize=10)
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()