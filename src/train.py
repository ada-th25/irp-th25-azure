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
from data.wildfire_dataset import WildfireDataset, load_split_ids
from models.haunet.haunet import make_model
from metrics import compute_iou, compute_ipsnr, compute_boundary_f1
from losses import build_loss


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

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Train wildfire SR model.")
    parser.add_argument("--config",   required=True,
                        help="Path to experiment YAML config.")
    parser.add_argument("--resume",   default=None,
                        help="Path to checkpoint .pt file to resume from.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (use CSV only).")
    args = parser.parse_args()
 
    # ------------------------------------------------------------------
    # Load Config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
 
    exp_name = cfg["experiment"]["name"]
    branch = cfg["experiment"]["branch"]
    bucket_name = cfg["data"]["bucket_name"]
    prefix = cfg["data"]["output_prefix"]
 
    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Branch     : {branch}")
    print(f"Config     : {args.config}")
    print(f"{'='*60}\n")
 
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg["checkpointing"]["checkpoint_dir"]) / exp_name
    log_dir = Path(cfg["logging"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)
 
    # ------------------------------------------------------------------
    # Weights & Biases (W&B) logging
    # ------------------------------------------------------------------
    use_wandb = cfg["logging"]["use_wandb"] and not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project = cfg["logging"]["wandb_project"],
                entity = cfg["logging"].get("wandb_entity"),
                name = exp_name,
                config = cfg,
                resume = "allow",
            )
            print("W&B initialised.")
        except ImportError:
            print("[WARN] wandb not installed. Falling back to CSV logging.")
            print("       Install with: pip install wandb --break-system-packages")
            use_wandb = False
 
    csv_logger = CSVLogger(str(log_dir / f"{exp_name}.csv"))
 
    # ------------------------------------------------------------------
    # Data Splits
    # ------------------------------------------------------------------
    print("\nLoading data splits from GCS...")
    train_ids = load_split_ids(bucket_name, branch, "train", prefix)
    val_ids = load_split_ids(bucket_name, branch, "val",   prefix)
 
    # ------------------------------------------------------------------
    # Datasets and Loaders
    # ------------------------------------------------------------------
    print("\nBuilding datasets...")
    train_ds = WildfireDataset(bucket_name, prefix, branch, split_ids=train_ids)
    val_ds = WildfireDataset(bucket_name, prefix, branch, split_ids=val_ids)
 
    train_loader = DataLoader(
        train_ds,
        batch_size = cfg["training"]["batch_size"],
        shuffle = True,
        num_workers = cfg["data"]["num_workers"],
        pin_memory = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = cfg["training"]["batch_size"],
        shuffle = False,
        num_workers = cfg["data"]["num_workers"],
        pin_memory = (device.type == "cuda"),
    )
    print(f"  Train: {len(train_ds)} samples / {len(train_loader)} batches")
    print(f"  Val  : {len(val_ds)}   samples / {len(val_loader)}   batches")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nBuilding model...")
    model = make_model(
        goes_channels = cfg["data"]["goes_channels"],
        cov_channels = cfg["data"]["cov_channels"],
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  HAUNet parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss, Optimiser, Scheduler
    # ------------------------------------------------------------------
    criterion = build_loss(cfg["loss"]).to(device)
 
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )
 
    n_epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max = n_epochs,
        eta_min = cfg["training"]["min_lr"],
    )

    # ------------------------------------------------------------------
    # GCS client (for checkpoint upload)
    # ------------------------------------------------------------------
    credentials, _ = default()
    gcs_bucket = storage.Client(credentials=credentials).bucket(bucket_name)
    gcs_ckpt_prefix = cfg["checkpointing"]["gcs_checkpoint_prefix"]

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    best_iou    = 0.0
 
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_iou={best_iou:.4f}")

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    val_every = cfg["training"]["val_every_n_epochs"]
    save_every = cfg["training"]["save_every_n_epochs"]
    keep_last = cfg["checkpointing"]["keep_last_n"]
 
    print(f"\nStarting training for {n_epochs} epochs "
          f"(val every {val_every}, save every {save_every})...\n")
 
    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
 
        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        train_loss = train_one_epoch(
            model, train_loader, optimiser, criterion, device,
            grad_clip=cfg["training"]["grad_clip"],
            epoch=epoch,
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
 
        log_row = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "lr":         lr_now,
        }

        # ------------------------------------------------------------------
        # Validate
        # ------------------------------------------------------------------
        if epoch % val_every == 0 or epoch == n_epochs:
            print(f"\n  [Epoch {epoch}] Running validation...", flush=True)
            val_metrics = validate(model, val_loader, criterion, device)
            log_row.update(val_metrics)
 
            is_best = val_metrics["val_iou"] > best_iou
            if is_best:
                best_iou = val_metrics["val_iou"]
                best_path = str(ckpt_dir / "best.pt")
                torch.save({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_iou":  best_iou,
                    "config":    cfg,
                }, best_path)
                upload_checkpoint(gcs_bucket, best_path, gcs_ckpt_prefix)
                print(f"  ★ New best val_iou={best_iou:.4f} — saved best.pt")
 
            print(
                f"  [Epoch {epoch:3d}] "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_iou={val_metrics['val_iou']:.4f} | "
                f"val_ipsnr={val_metrics['val_ipsnr']:.2f} dB | "
                f"val_bf1={val_metrics['val_boundary_f1']:.4f} | "
                f"lr={lr_now:.2e} | "
                f"time={time.time()-t0:.0f}s",
                flush=True,
            )
        else:
            print(
                f"  [Epoch {epoch:3d}] "
                f"train_loss={train_loss:.4f} | "
                f"lr={lr_now:.2e} | "
                f"time={time.time()-t0:.0f}s",
                flush=True,
            )
        
        # ── Log ───────────────────────────────────────────────────────────
        csv_logger.log(log_row)
 
        if use_wandb:
            import wandb
            wandb.log(log_row, step=epoch)
 
        # ── Checkpoint ────────────────────────────────────────────────────
        if epoch % save_every == 0 or epoch == n_epochs:
            ckpt_path = str(ckpt_dir / f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_iou":  best_iou,
                "config":    cfg,
            }, ckpt_path)
            upload_checkpoint(gcs_bucket, ckpt_path, gcs_ckpt_prefix)
            prune_old_checkpoints(gcs_bucket, gcs_ckpt_prefix, keep_last)
 
    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\nTraining complete. Best val_iou: {best_iou:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()
 
 
if __name__ == "__main__":
    main()