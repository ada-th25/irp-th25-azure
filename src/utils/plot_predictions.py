"""
Plot GOES bands, covariates, true target, and predicted fire probability
for a handful of validation samples, side by side.

Usage:
    python -m src.utils.plot_predictions --config experiments/e2_haunet_cascaded_stage1_test.YAML \
        --checkpoint checkpoints_test/e2_haunet_viirs_cascaded_stage1_test/epoch_010.pt \
        --n-samples 4 --out predictions.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless -- no display needed on the GCP instance
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from src.data.wildfire_dataset import WildfireDataset, load_split_ids
from src.models.haunet import make_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--out", default="predictions.png")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    branch = cfg["experiment"]["branch"]
    bucket_name = cfg["data"]["bucket_name"]
    prefix = cfg["data"]["output_prefix"]
    splits_dir = cfg["data"].get("splits_dir", "splits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids = load_split_ids(bucket_name, branch, args.split, prefix, splits_dir)
    ds = WildfireDataset(bucket_name, prefix, branch, split_ids=ids)
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=True, num_workers=4)

    model = make_model(
        goes_channels=cfg["data"]["goes_channels"],
        cov_channels=cfg["data"]["cov_channels"],
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')})")
    else:
        print("No checkpoint given -- using freshly initialised weights.")

    model.eval()

    batch = next(iter(loader))
    inp = batch["input"].to(device)
    inp[:, :3] = 0.0 # zero the GOES channels (first 3), keep DEM/land cover
    target = batch["target"].to(device)
    goes = batch["goes"]          # (B, 3, goes_H, goes_W) -- raw, low-res
    cov = batch["covariates"]     # (B, 2, cov_H, cov_W) -- as fed to the model
    sample_ids = batch["sample_id"]

    with torch.no_grad():
        logits = model(inp)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.adaptive_avg_pool2d(logits, target.shape[-2:])
        probs = torch.sigmoid(logits)

    n = inp.shape[0]
    # Columns: GOES C07, GOES C14, GOES TFI, DEM, land cover, true target, predicted prob
    n_cols = 7
    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["GOES C07", "GOES C14", "GOES TFI", "DEM (cov[0])",
                  "Land cover (cov[1])", "True target", "Predicted P(fire)"]

    for row in range(n):
        goes_np = goes[row].numpy()      # (3, gH, gW)
        cov_np = cov[row].numpy()        # (2, cH, cW)
        target_np = target[row, 0].cpu().numpy()   # (H, W)
        prob_np = probs[row, 0].cpu().numpy()      # (H, W)

        panels = [
            goes_np[0], goes_np[1], goes_np[2],
            cov_np[0], cov_np[1],
            target_np, prob_np,
        ]
        cmaps = ["inferno", "inferno", "inferno", "terrain", "tab20", "hot", "hot"]
        vmins_vmaxs = [
            (None, None), (None, None), (None, None),
            (None, None), (None, None),
            (0, 1), (None, None),   # True target stays fixed, prediction auto-scales
        ]

        for col in range(n_cols):
            ax = axes[row, col]
            vmin, vmax = vmins_vmaxs[col]
            im = ax.imshow(panels[col], cmap=cmaps[col], vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(sample_ids[row], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nSaved figure to {args.out}")
    print(f"Sample IDs plotted: {list(sample_ids)}")
    print(f"Predicted prob range: {probs.min().item():.4f} - {probs.max().item():.4f}")
    print(f"Target range: {target.min().item():.4f} - {target.max().item():.4f}")


if __name__ == "__main__":
    main()