import argparse
import csv
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from google.cloud import storage
from google.auth import default

# Local modules
sys.path.append(str(Path(__file__).parent))
from data.wildfire_dataset import WildfireDataset, load_split_ids, compute_pos_weight
from models.haunet.haunet import make_model
from metrics import compute_iou, compute_ipsnr, compute_boundary_f1


# ------------------------------------------------------------------
# GCS Checkpoint Helpers
# ------------------------------------------------------------------

def upload_checkpoint(bucket, local_path: str, gcs_prefix: str) -> None:
    fname = os.path.basename(local_path)
    gcs_path = f"{gcs_prefix}/{fname}"
    bucket.blob(gcs_path).upload_from_filename(local_path)
    print(f"  [GCS] Uploaded checkpoint → gs://{bucket.name}/{gcs_path}")


def prune_old_checkpoints(bucket, gcs_prefix: str, keep_last_n: int) -> None:
    """Delete older epoch checkpoints from GCS, keeping only the most recent N."""
    blobs = sorted(
        [b for b in bucket.list_blobs(prefix=gcs_prefix)
         if "epoch_" in b.name and b.name.endswith(".pt")],
        key=lambda b: b.name,
    )
    to_delete = blobs[:-keep_last_n] if len(blobs) > keep_last_n else []
    for b in to_delete:
        b.delete()
        print(f"  [GCS] Pruned old checkpoint: {b.name}")


# ------------------------------------------------------------------
# CSV Logger
# ------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self.columns = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, row: dict) -> None:
        write_header = not os.path.exists(self.path)
        if self.columns is None:
            self.columns = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            if write_header:
                writer.writeheader()
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in row.items()})


# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, batch in enumerate(loader):
        inp = batch["input"].to(device)     # (B, 5, 256, 256)
        target = batch["target"].to(device)    # (B, 1, 256, 256)

        optimiser.zero_grad()
        logits = model(inp)                    # (B, 1, 256, 256)
        loss = criterion(logits, target)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()
        total_loss += loss.item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(f"  Epoch {epoch:3d} | batch {batch_idx+1:4d}/{n_batches} "
                  f"| loss {loss.item():.4f}", flush=True)

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_ipsnr = 0.0
    total_bf1 = 0.0
    n_ipsnr = 0       # ipsnr is skipped when no fire pixels exist
    n_batches = len(loader)

    for batch in loader:
        inp = batch["input"].to(device)
        target = batch["target"].to(device)

        logits = model(inp)
        loss = criterion(logits, target)

        total_loss += loss.item()
        total_iou += compute_iou(logits, target)
        total_bf1 += compute_boundary_f1(logits, target)

        ipsnr = compute_ipsnr(logits, target)
        if ipsnr != float("inf"):
            total_ipsnr += ipsnr
            n_ipsnr += 1

    return {
        "val_loss": total_loss / n_batches,
        "val_iou": total_iou / n_batches,
        "val_ipsnr": total_ipsnr / max(n_ipsnr, 1),
        "val_boundary_f1": total_bf1 / n_batches,
    }