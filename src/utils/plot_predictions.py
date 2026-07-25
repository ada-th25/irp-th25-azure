"""
Plot GOES bands, covariates, true target, and predicted fire probability
for a handful of validation samples, side by side.

Usage:
    python -m src.utils.plot_predictions --config experiments/e2_haunet_cascaded_stage1_test.YAML \
        --checkpoint checkpoints_test/e2_haunet_viirs_cascaded_stage1_test/epoch_010.pt \
        --n-samples 4 --out predictions.png

For a quick look at *why* this is slow, timing is printed to stdout for each
stage (config/split loading, dataset construction, batch fetch, model +
checkpoint load, inference). Compare --num-workers 0 vs a higher value --
for a single batch, worker process spawn/import overhead often costs more
than it saves, especially if __getitem__ does its own remote (GCS) reads.
"""

import argparse
import random
import sys
import time
from contextlib import contextmanager
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
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--out", default="predictions.png")
    p.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. 0 is usually faster for a single batch, "
                         "since each worker pays process-spawn + import overhead before "
                         "it even starts fetching data.")
    p.add_argument("--seed", type=int, default=None,
                    help="Optional seed for reproducible sample selection.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    t_start = time.time()

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
        )

    # Instead of asking the DataLoader to shuffle the *entire* dataset and
    # spin up worker processes for a single batch, just pick n random
    # indices ourselves and fetch them directly. This sidesteps:
    #   (a) worker process spawn/import overhead for one batch
    #   (b) any up-front full-dataset shuffling/indexing cost
    n = min(args.n_samples, len(ds))
    chosen_idx = random.sample(range(len(ds)), n)

    if args.num_workers == 0:
        with timed(f"fetch {n} samples directly (no DataLoader workers)"):
            samples = [ds[i] for i in chosen_idx]
        batch = torch.utils.data.dataloader.default_collate(samples)
    else:
        from torch.utils.data import Subset
        subset = Subset(ds, chosen_idx)
        loader = DataLoader(subset, batch_size=n, shuffle=False,
                             num_workers=args.num_workers)
        with timed(f"fetch {n} samples via DataLoader (num_workers={args.num_workers})"):
            batch = next(iter(loader))

    inp = batch["input"].to(device)
    #inp[:, :3] = 0.0 # zero the GOES channels (first 3), keep DEM/land cover
    target = batch["target"].to(device)
    goes = batch["goes"]          # (B, 3, goes_H, goes_W) -- raw, low-res
    cov = batch["covariates"]     # (B, 2, cov_H, cov_W) -- as fed to the model
    sample_ids = batch["sample_id"]

    with timed("build model"):
        model = make_model(
            goes_channels=cfg["data"]["goes_channels"],
            cov_channels=cfg["data"]["cov_channels"],
        ).to(device)

    if args.checkpoint:
        with timed("load checkpoint"):
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')})")
    else:
        print("No checkpoint given -- using freshly initialised weights.")

    model.eval()

    with timed("inference"):
        with torch.no_grad():
            logits = model(inp)
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.adaptive_avg_pool2d(logits, target.shape[-2:])
            probs = torch.sigmoid(logits)

    # Columns: GOES C07, GOES C14, GOES TFI, DEM, land cover, true target, predicted prob
    n_cols = 7
    with timed("plot + save figure"):
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
    print(f"\n[timing] TOTAL: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()