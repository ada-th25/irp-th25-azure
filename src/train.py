"""
Usage example: python src/train.py --config experiments/e2_haunet_cascaded_stage1_test.YAML
"""

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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from google.cloud import storage
from google.auth import default
from concurrent.futures import ThreadPoolExecutor

# Local modules
sys.path.append(str(Path(__file__).parent))
from data.wildfire_dataset import WildfireDataset, load_split_ids
from metrics import compute_iou, compute_soft_iou, compute_ipsnr, compute_boundary_f1, compute_precision_recall, compute_auprc, compute_centroid_distance
from losses import build_loss
from utils.postprocess import postprocess_prediction
import importlib

MODEL_REGISTRY = {
    "haunet": "models.haunet",
    "autoencoder": "models.autoencoder",
}

# Loss types that expect a continuous [0,1] Gaussian heatmap target rather
# than a binary {0,1} mask. Used to auto-derive heatmap_target from loss.type
# and to sanity-check explicit overrides.
GAUSSIAN_LOSS_TYPES = {"gaussian_focal", "gaussianfocalloss", "focal_dice_gaussian", "focal_dice_gaussian_spatial"}

# Metrics where higher is better vs lower is better, used to generalise the
# "is_best" checkpoint comparison beyond a hardcoded val_iou.
HIGHER_IS_BETTER = {
    "val_iou": True, "val_soft_iou": True, "val_iou_t03": True, "val_ipsnr": True,
    "val_boundary_f1": True, "val_precision": True, "val_recall": True,
    "val_pixel_f1": True, "val_loss": False,
    "val_auprc": True, "val_centroid_dist_m": False,
}

# ------------------------------------------------------------------
# Active Pixel Counter (for WeightedRandomSampler)
# ------------------------------------------------------------------
def load_positive_pixel_counts(bucket, branch: str, split_ids: list[str], output_prefix: str) -> dict:
    """
    Read n_active_pixels (or n_detections) directly from manifest JSONs,
    keyed by sample id. Avoids loading full raster data just to get a
    pixel count.
    """
    prefix = f"{output_prefix}/{branch}_pairs/manifest/"
    split_id_set = set(split_ids)
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".json")]

    def _read(blob):
        m = json.loads(blob.download_as_text())
        sid = m.get("id")
        if sid in split_id_set:
            return sid, (m.get("n_active_pixels") or m.get("n_detections") or 1)
        return None

    counts = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        for result in ex.map(_read, blobs):
            if result:
                counts[result[0]] = result[1]
    return counts

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
    DEFAULT_COLUMNS = [
        "epoch", "train_loss", "lr",
        "val_loss", "val_iou","val_iou_t03", "val_soft_iou", "val_ipsnr",
        "val_boundary_f1", "val_precision", "val_recall", "val_pixel_f1",
        "val_auprc", "val_centroid_dist_m",
        "pp_iou", "pp_boundary_f1", "pp_precision", "pp_recall",
        "pp_pixel_f1", "pp_centroid_dist_m",
    ]
 
    def __init__(self, path: str, columns: list = None, resume: bool = False):
        self.path = path
        self.columns = columns or list(self.DEFAULT_COLUMNS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if not resume and os.path.exists(self.path):
            backup_path = self.path + ".bak"
            print(f"  [CSVLogger] Existing log found at {self.path} but this is a fresh "
                  f"(non-resumed) run -- backing it up to {backup_path} and starting a new file.")
            os.replace(self.path, backup_path)
 
    def log(self, row: dict) -> None:
        write_header = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, restval="", extrasaction="ignore")
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
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    component_totals = {"fd": 0.0, "gf": 0.0, "spatial": 0.0}
    has_components = hasattr(criterion, "forward_with_components")
 
    optimiser.zero_grad()   # zero once, before the accumulation loop starts
 
    for batch_idx, batch in enumerate(loader):
        inp = batch["input"].to(device)     # (B, 5, 256, 256)
        target = batch["target"].to(device)    # (B, 1, 256, 256)
 
        logits = model(inp)                    # (B, 1, H_in, W_in)
        
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.adaptive_max_pool2d(logits, target.shape[-2:])
            
        if hasattr(criterion, "forward_with_components"):
            loss, components = criterion.forward_with_components(logits, target)
            for k, v in components.items():
                component_totals[k] += v
        else:
            loss = criterion(logits, target)
            components = None

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"    [WARNING] logits contain NaN/Inf at batch {batch_idx}")

        if batch_idx == 0:
            print(f"    [Epoch {epoch}] logit range: {logits.min().item():.4f} - {logits.max().item():.4f}, "
                  f"logit std: {logits.std().item():.6f}")
            print(f"    input stats: min={inp.min().item():.4f} max={inp.max().item():.4f} "
                  f"mean={inp.mean().item():.4f} has_nan={torch.isnan(inp).any().item()}")
            print(f"    target stats: min={target.min().item():.4f} max={target.max().item():.4f} "
                  f"sum={target.sum().item():.1f} has_nan={torch.isnan(target).any().item()}")
            
            probs = torch.sigmoid(logits)
            print(f"    [Epoch {epoch}] sample pred range: {probs.min().item():.4f} - {probs.max().item():.4f}", flush=True)
 
        # Scale the loss down by accum_steps so that summing gradients
        # over accum_steps batches
        (loss / accum_steps).backward()
 
        # Only step the optimiser (and clip/zero) every accum_steps
        # batches, or on the final batch of the epoch, so a trailing
        # partial group of batches still gets applied instead
        is_accum_boundary = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == n_batches)
        if is_accum_boundary:
            clip_val = grad_clip if grad_clip > 0 else float("inf")
    
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                
            optimiser.step()
            optimiser.zero_grad()
 
        total_loss += loss.item()   # log the TRUE (unscaled) loss value
 
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(f"  Epoch {epoch:3d} | batch {batch_idx+1:4d}/{n_batches} "
                  f"| loss {loss.item():.4f}", flush=True)
            
    if has_components:
        avg_components = {k: v / n_batches for k, v in component_totals.items()}
        print(f"  Epoch {epoch:3d} | component losses: "
              f"fd={avg_components['fd']:.4f} gf={avg_components['gf']:.4f} "
              f"spatial={avg_components['spatial']:.4f}", flush=True)
 
    return total_loss / n_batches


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    threshold: float = 0.5,
    low_threshold: float = 0.3,
    compute_ipsnr_flag: bool = True,
    pixel_size_m: float = 384.0,
) -> dict:
    """
    Args:
        threshold:           probability threshold used for val_iou, val_boundary_f1,
                              val_precision/recall/pixel_f1.
        low_threshold:        secondary, looser threshold used for val_iou_t03 —
                              useful for Gaussian-heatmap targets where the core
                              peak region undercounts a 0.5 threshold.
        compute_ipsnr_flag:   if False, skips ipsnr entirely (returns 0.0). ipsnr is
                              the one metric that scores raw probabilities rather
                              than a thresholded mask, so it's most meaningful for
                              Stage-1 (heatmap) runs and can be skipped elsewhere
                              to save time.
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_iou_lowthresh = 0.0
    total_soft_iou = 0.0
    total_ipsnr = 0.0
    total_bf1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_pixel_f1 = 0.0
    total_auprc = 0.0
    total_centroid = 0.0
    n_centroid = 0
    n_ipsnr = 0       # ipsnr is skipped when no fire pixels exist
    n_batches = len(loader)
 
    for batch in loader:
        inp = batch["input"].to(device)
        target = batch["target"].to(device)
 
        logits = model(inp)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.adaptive_max_pool2d(logits, target.shape[-2:])
        
        loss = criterion(logits, target)
        total_loss += loss.item()
        
        total_iou += compute_iou(logits, target, threshold=threshold)
        total_iou_lowthresh += compute_iou(logits, target, threshold=low_threshold)
        total_soft_iou += compute_soft_iou(logits, target)
        total_bf1 += compute_boundary_f1(logits, target, threshold=threshold)
 
        pr = compute_precision_recall(logits, target, threshold=threshold)
        total_precision += pr["precision"]
        total_recall += pr["recall"]
        total_pixel_f1 += pr["f1"]

        total_auprc += compute_auprc(logits, target)

        cdist = compute_centroid_distance(logits, target, pixel_size_m=pixel_size_m)
        if not np.isnan(cdist):
            total_centroid += cdist
            n_centroid += 1
 
        if compute_ipsnr_flag:
            ipsnr = compute_ipsnr(logits, target, threshold=threshold)
            if ipsnr != float("inf"):
                total_ipsnr += ipsnr
                n_ipsnr += 1
 
    return {
        "val_loss": total_loss / n_batches,
        "val_iou": total_iou / n_batches,
        "val_iou_t03": total_iou_lowthresh / n_batches,
        "val_soft_iou": total_soft_iou / n_batches,
        "val_ipsnr": (total_ipsnr / max(n_ipsnr, 1)) if compute_ipsnr_flag else 0.0,
        "val_boundary_f1": total_bf1 / n_batches,
        "val_precision": total_precision / n_batches,
        "val_recall": total_recall / n_batches,
        "val_pixel_f1": total_pixel_f1 / n_batches,
        "val_auprc": total_auprc / n_batches,
        "val_centroid_dist_m": (total_centroid / max(n_centroid, 1)),
    }

@torch.no_grad()
def compute_postprocessed_metrics(
    model, loader, device, pixel_size_m, postprocess_cfg,
) -> dict:
    """
    Same metrics as validate(), computed on the post-processed binary
    mask instead of raw sigmoid output. Kept as a separate pass -- never
    used for checkpoint selection (see main(): best_metric is always
    computed from raw validate() output).
    """
    model.eval()
    total_iou, total_bf1, total_precision, total_recall, total_pixel_f1 = 0, 0, 0, 0, 0
    total_centroid, n_centroid = 0.0, 0
    n_batches = len(loader)

    method = postprocess_cfg.get("method", "percentile")
    percentile = postprocess_cfg.get("percentile", 99.5)
    fixed_threshold = postprocess_cfg.get("fixed_threshold", 0.5)
    min_component_size = postprocess_cfg.get("min_component_size", 4)

    for batch in loader:
        inp = batch["input"].to(device)
        target = batch["target"].to(device)
        logits = model(inp)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.adaptive_max_pool2d(logits, target.shape[-2:])
        probs = torch.sigmoid(logits)

        mask = postprocess_prediction(
            probs, method=method, percentile=percentile,
            fixed_threshold=fixed_threshold, min_component_size=min_component_size,
        )
        # feed mask as pseudo-logits (already 0/1) at threshold=0.5
        pseudo_logits = (mask * 20.0) - 10.0  # sigmoid(±10) ≈ 0/1, avoids logit(0)/(1) inf

        total_iou += compute_iou(pseudo_logits, target, threshold=0.5)
        total_bf1 += compute_boundary_f1(pseudo_logits, target, threshold=0.5)
        pr = compute_precision_recall(pseudo_logits, target, threshold=0.5)
        total_precision += pr["precision"]
        total_recall += pr["recall"]
        total_pixel_f1 += pr["f1"]
        cdist = compute_centroid_distance(pseudo_logits, target, pixel_size_m=pixel_size_m)
        if not np.isnan(cdist):
            total_centroid += cdist
            n_centroid += 1

    return {
        "pp_iou": total_iou / n_batches,
        "pp_boundary_f1": total_bf1 / n_batches,
        "pp_precision": total_precision / n_batches,
        "pp_recall": total_recall / n_batches,
        "pp_pixel_f1": total_pixel_f1 / n_batches,
        "pp_centroid_dist_m": (total_centroid / max(n_centroid, 1)),
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
    loss_type = cfg["loss"]["type"].lower()
 
    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Branch     : {branch}")
    print(f"Config     : {args.config}")
    print(f"{'='*60}\n")
 
    default_heatmap = loss_type in GAUSSIAN_LOSS_TYPES
    heatmap_target = cfg["data"].get("heatmap_target", default_heatmap)
    dilate_target = cfg["data"].get("dilate_target", False)

    if heatmap_target and loss_type not in GAUSSIAN_LOSS_TYPES:
        print(f"  [WARN] heatmap_target=True but loss.type='{loss_type}' expects a "
              f"binary target, not a continuous Gaussian heatmap. Check your config "
              f"unless this is intentional.")
    if (not heatmap_target) and (loss_type in GAUSSIAN_LOSS_TYPES):
        print(f"  [WARN] loss.type='{loss_type}' expects a continuous Gaussian heatmap "
              f"target but heatmap_target=False -- it will be trained against a "
              f"binary{'/dilated' if dilate_target else ''} target instead.")
    print(f"  Target representation: "
          f"{'heatmap' if heatmap_target else ('dilated' if dilate_target else 'binary')}")
 
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
    # GCS client (for checkpoint upload)
    # ------------------------------------------------------------------
    credentials, _ = default()
    gcs_bucket = storage.Client(credentials=credentials).bucket(bucket_name)
    gcs_ckpt_prefix = cfg["checkpointing"]["gcs_checkpoint_prefix"]
 
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
    splits_dir = cfg["data"].get("splits_dir", "splits")
    print(f"\nLoading data splits from GCS (splits_dir='{splits_dir}')...")
    train_ids = load_split_ids(bucket_name, branch, "train", prefix, splits_dir)
    val_ids = load_split_ids(bucket_name, branch, "val",   prefix, splits_dir)

    # ------------------------------------------------------------------
    # Weighted sampler (optional, config-driven)
    # ------------------------------------------------------------------
    use_weighted_sampler = cfg["training"].get("use_weighted_sampler", False)
    train_sampler = None
    if use_weighted_sampler:
        print("\nBuilding weighted sampler from manifest positive-pixel counts...")
        pos_counts = load_positive_pixel_counts(gcs_bucket, branch, train_ids, prefix)
        weights = [max(pos_counts.get(sid, 1), 1) for sid in train_ids]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train_ids), replacement=True
        )
        print(f"  Weighted sampler built ({len(weights)} samples).")
 
    # ------------------------------------------------------------------
    # Datasets and Loaders
    # ------------------------------------------------------------------
    print("\nBuilding datasets...")
    use_stage1_pred = cfg["data"].get("use_stage1_pred", False)

    train_ds = WildfireDataset(
        bucket_name, prefix, branch, split_ids=train_ids,
        dilate_target=dilate_target, heatmap_target=heatmap_target,
        goes_mean=cfg["data"].get("goes_mean"),
        goes_std=cfg["data"].get("goes_std"),
        dem_mean=cfg["data"].get("dem_mean"),
        dem_std=cfg["data"].get("dem_std"),
        heatmap_sigma=cfg["data"].get("heatmap_sigma", 0.6),
        use_stage1_pred=use_stage1_pred, augment=True
    )
    val_ds = WildfireDataset(
        bucket_name, prefix, branch, split_ids=val_ids,
        dilate_target=False, heatmap_target=False,
        goes_mean=cfg["data"].get("goes_mean"),
        goes_std=cfg["data"].get("goes_std"),
        dem_mean=cfg["data"].get("dem_mean"),
        dem_std=cfg["data"].get("dem_std"),
        heatmap_sigma=cfg["data"].get("heatmap_sigma", 0.6),
        use_stage1_pred=use_stage1_pred, augment=False
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size = cfg["training"]["batch_size"],
        shuffle = (train_sampler is None),   # sampler and shuffle are mutually exclusive
        sampler = train_sampler,
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
    model_name = cfg["model"]["name"]
    make_model = importlib.import_module(MODEL_REGISTRY[model_name]).make_model
    
    print(f"\nBuilding model ({model_name})...")
    model = make_model(
        goes_channels = cfg["data"]["goes_channels"],
        cov_channels = cfg["data"]["cov_channels"],
        stage1_channels = cfg["data"].get("stage1_channels", 0),
        prior_fire_fraction = cfg["model"].get("prior_fire_fraction", 0.0085),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_name} parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss, Optimiser, Scheduler
    # ------------------------------------------------------------------
    criterion = build_loss(cfg["loss"]).to(device)
 
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr = float(cfg["training"]["learning_rate"]),
        weight_decay = float(cfg["training"]["weight_decay"]),
    )
 
    n_epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max = n_epochs,
        eta_min = float(cfg["training"]["min_lr"]),
    )

    # ------------------------------------------------------------------
    # Evaluation config: thresholds + which metrics to bother computing.
    # ------------------------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    eval_threshold = float(eval_cfg.get("threshold", 0.5))
    eval_low_threshold = float(eval_cfg.get("low_threshold", 0.3))
    eval_compute_ipsnr = bool(eval_cfg.get("compute_ipsnr", True))
    eval_pixel_size_m = float(eval_cfg.get("pixel_size_m", 384.0))

    log_postprocessed = bool(eval_cfg.get("log_postprocessed", False))
    postprocess_cfg = cfg.get("postprocess", {})

    # Which logged metric(s) decide "best_<metric>.pt", and whether higher or lower is better.
    best_metric_cfg = cfg.get("checkpointing", {}).get("best_metric", "val_iou")
    best_metrics = [best_metric_cfg] if isinstance(best_metric_cfg, str) else list(best_metric_cfg)
    for m in best_metrics:
        if m not in HIGHER_IS_BETTER:
            raise ValueError(
                f"Unknown checkpointing.best_metric '{m}'. "
                f"Choose from: {sorted(HIGHER_IS_BETTER)}."
            )
    print(f"  Best-checkpoint metric(s): " +
          ", ".join(f"{m} ({'higher' if HIGHER_IS_BETTER[m] else 'lower'} is better)" for m in best_metrics))


    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    best_scores = {m: (float("-inf") if HIGHER_IS_BETTER[m] else float("inf")) for m in best_metrics}
 
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
    
        result = model.load_state_dict(ckpt["model"], strict=False)
        if result.missing_keys:
            print(f"  [WARN] Missing keys (in model, not in checkpoint): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  [WARN] Unexpected keys (in checkpoint, not in model): {result.unexpected_keys}")
        if not result.missing_keys and not result.unexpected_keys:
            print("  Model weights loaded successfully — all keys matched.")
    
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        # Backward-compatible with older checkpoints (single best_score/best_metric, or even older best_iou).
        ckpt_best_scores = ckpt.get("best_scores")
        if ckpt_best_scores:
            for m in best_metrics:
                if m in ckpt_best_scores:
                    best_scores[m] = ckpt_best_scores[m]
        elif "best_score" in ckpt or "best_iou" in ckpt:
            legacy_metric = ckpt.get("best_metric", "val_iou")
            legacy_score = ckpt.get("best_score", ckpt.get("best_iou"))
            if legacy_metric in best_scores:
                best_scores[legacy_metric] = legacy_score
        print(f"  Resumed at epoch {start_epoch}, best_scores={best_scores}")

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    val_every = cfg["training"]["val_every_n_epochs"]
    save_every = cfg["training"]["save_every_n_epochs"]
    keep_last = cfg["checkpointing"]["keep_last_n"]
    accum_steps = cfg["training"].get("grad_accum_steps", 1)   # 1 = no accumulation (default/backward compatible)
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps "
              f"(effective batch size = {cfg['training']['batch_size'] * accum_steps})")
 
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
            accum_steps=accum_steps,
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
            val_metrics = validate(
                model, val_loader, criterion, device,
                threshold=eval_threshold,
                low_threshold=eval_low_threshold,
                compute_ipsnr_flag=eval_compute_ipsnr,
                pixel_size_m=eval_pixel_size_m,
            )
            log_row.update(val_metrics)

            if log_postprocessed:
                pp_metrics = compute_postprocessed_metrics(
                    model, val_loader, device, eval_pixel_size_m, postprocess_cfg,
                )
                log_row.update(pp_metrics)
                print(f"  [Epoch {epoch:3d}] post-processed: "
                      f"pp_iou={pp_metrics['pp_iou']:.4f} pp_bf1={pp_metrics['pp_boundary_f1']:.4f} "
                      f"pp_prec={pp_metrics['pp_precision']:.4f} pp_rec={pp_metrics['pp_recall']:.4f}",
                      flush=True)
 
            for m in best_metrics:
                current_score = val_metrics[m]
                m_is_higher = HIGHER_IS_BETTER[m]
                is_best = (current_score > best_scores[m]) if m_is_higher else (current_score < best_scores[m])
                if is_best:
                    best_scores[m] = current_score
                    best_path = str(ckpt_dir / f"best_{m}.pt")
                    torch.save({
                        "epoch":       epoch,
                        "model":       model.state_dict(),
                        "optimiser":   optimiser.state_dict(),
                        "scheduler":   scheduler.state_dict(),
                        "best_scores": best_scores,
                        "best_metric": m,
                        "config":      cfg,
                    }, best_path)
                    upload_checkpoint(gcs_bucket, best_path, gcs_ckpt_prefix)
                    print(f"  ★ New best {m}={best_scores[m]:.4f} — saved best_{m}.pt")
 
            print(
                f"  [Epoch {epoch:3d}] "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_iou={val_metrics['val_iou']:.4f} | "
                f"val_iou@{eval_low_threshold}={val_metrics['val_iou_t03']:.4f} | "
                f"val_prec={val_metrics['val_precision']:.4f} | "
                f"val_rec={val_metrics['val_recall']:.4f} | "
                f"val_ipsnr={val_metrics['val_ipsnr']:.2f} dB | "
                f"val_bf1={val_metrics['val_boundary_f1']:.4f} | "
                f"val_auprc={val_metrics['val_auprc']:.4f} | "
                f"val_centroid_dist_m={val_metrics['val_centroid_dist_m']:.4f} | "
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
        
        # ------------------------------------------------------------------
        # Log
        # ------------------------------------------------------------------
        csv_logger.log(log_row)
 
        if use_wandb:
            import wandb
            wandb.log(log_row, step=epoch)
 
        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if epoch % save_every == 0 or epoch == n_epochs:
            ckpt_path = str(ckpt_dir / f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimiser":   optimiser.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_scores": best_scores,
                "config":      cfg,
            }, ckpt_path)
            upload_checkpoint(gcs_bucket, ckpt_path, gcs_ckpt_prefix)
            prune_old_checkpoints(gcs_bucket, gcs_ckpt_prefix, keep_last)
 
    print(f"\nTraining complete. Best scores: " +
          ", ".join(f"{m}={best_scores[m]:.4f}" for m in best_metrics))
    if use_wandb:
        import wandb
        wandb.finish()
 
 
if __name__ == "__main__":
    main()