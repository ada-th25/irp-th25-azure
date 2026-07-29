"""
Export model predictions from a trained checkpoint as either raw probability
maps or thresholded binary masks. Kept separate from train.py/validate() on
purpose: training metrics need to stay stable/comparable across experiments,
while "give me a clean binary map for review" is an opt-in, one-off action.

Usage:
    python src/export_predictions.py \
        --config experiments/e2_haunet_cascaded_stage1_test.YAML \
        --checkpoint checkpoints/e2_haunet_cascaded_stage1_test/best.pt \
        --split val \
        --out-dir exports/e2_stage1_val \
        --binarize --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from data.wildfire_dataset import WildfireDataset, load_split_ids
import importlib

MODEL_REGISTRY = {
    "haunet": "models.haunet",
    "autoencoder": "models.autoencoder",
}


@torch.no_grad()
def export_predictions(model, loader, device, out_dir: Path, threshold: float, binarize: bool):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for batch in loader:
        inp = batch["input"].to(device)
        logits = model(inp)
        probs = torch.sigmoid(logits)

        target_shape = batch["target"].shape[-2:]
        if probs.shape[-2:] != target_shape:
            probs = F.adaptive_max_pool2d(probs, target_shape)

        output = (probs >= threshold).float() if binarize else probs
        output_np = output.cpu().numpy()

        for i, sample_id in enumerate(batch["sample_id"]):
            np.save(out_dir / f"{sample_id}.npy", output_np[i, 0])
            n_written += 1

    print(f"Wrote {n_written} {'binary masks' if binarize else 'probability maps'} to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export predictions from a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True, help="Directory to write .npy outputs.")
    parser.add_argument("--binarize", action="store_true",
                        help="Threshold output to a binary {0,1} mask. Omit for raw probability maps.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    branch = cfg["experiment"]["branch"]
    bucket_name = cfg["data"]["bucket_name"]
    prefix = cfg["data"]["output_prefix"]
    splits_dir = cfg["data"].get("splits_dir", "splits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_ids = load_split_ids(bucket_name, branch, args.split, prefix, splits_dir)
    ds = WildfireDataset(
        bucket_name, prefix, branch, split_ids=split_ids,
        dilate_target=False, heatmap_target=False,  # target repr irrelevant for export/input pipeline
        goes_mean=cfg["data"].get("goes_mean"),
        goes_std=cfg["data"].get("goes_std"),
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"])

    model_name = cfg["model"]["name"]
    make_model = importlib.import_module(MODEL_REGISTRY[model_name]).make_model
    model = make_model(
        goes_channels=cfg["data"]["goes_channels"],
        cov_channels=cfg["data"]["cov_channels"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"({ckpt.get('best_metric', 'val_iou')}={ckpt.get('best_score', ckpt.get('best_iou')):.4f})")

    export_predictions(model, loader, device, Path(args.out_dir),
                       threshold=args.threshold, binarize=args.binarize)


if __name__ == "__main__":
    main()