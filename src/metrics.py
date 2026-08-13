"""
Evaluation Metrics.

Provides pixel-level, spatial, boundary-aware, and threshold-free evaluation metrics
to assess predicted fire masks against ground-truth targets.

Functions:
    - compute_iou: Thresholded Binary Intersection over Union (IoU).
    - compute_soft_iou: Probabilistic / threshold-free soft IoU.
    - compute_ipsnr: Intersection Peak Signal-to-Noise Ratio (dB).
    - compute_boundary_f1: Perimeter/Boundary-focused F1 score.
    - compute_precision_recall: Pixel-level Precision, Recall, and F1.
    - compute_auprc: Area Under Precision-Recall Curve.
    - compute_centroid_distance: Spatial distance between target and predicted peak intensity.
    - compute_all_metrics: Batch aggregator wrapper over all supported metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score


def compute_iou(preds: torch.Tensor, targets: torch.Tensor,
                threshold: float = 0.5) -> float:
    """
    Computes binary Intersection over Union (IoU) over a single batch.

    Args:
        preds: Raw model predictions/logits of shape (B, 1, H, W) or (B, H, W).
        targets: Ground truth binary targets {0, 1} matching preds shape.
        threshold: Probability cutoff to binarize predictions after sigmoid.

    Returns:
        float: Mean IoU score over the batch. Returns 1.0 if both prediction and target
        are entirely empty (true negative alignment).
    """
    pred_mask = (torch.sigmoid(preds) >= threshold).float()
    target_mask = (targets >= 0.5).float()
    intersection = (pred_mask * target_mask).sum()
    union = (pred_mask + target_mask).clamp(0, 1).sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_soft_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Computes soft (threshold-free) Intersection over Union using predicted probabilities.

    Unlike standard IoU, soft IoU evaluates continuously over probabilities. This prevents zero-value
    collapses caused by strict fixed thresholds when evaluating sparse target regions.

    Args:
        preds: Raw model logits of shape (B, 1, H, W) or (B, H, W).
        targets: Target ground truth values in range [0, 1] matching preds shape.
        eps: Small numerical stability constant to prevent division by zero.

    Returns:
        float: Soft IoU score averaged across samples in the batch.
    """
    probs = torch.sigmoid(preds)
    target = targets

    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = (probs + target - probs * target).sum(dim=(1, 2, 3))

    iou = torch.where(
        union < eps,
        torch.ones_like(union),
        intersection / (union + eps),
    )
    return iou.mean().item()


def compute_ipsnr(preds: torch.Tensor, targets: torch.Tensor,
                  threshold: float = 0.5, eps: float = 1e-8) -> float:
    """
    Calculates Intersection Peak Signal-to-Noise Ratio (IPSNR) over non-background regions.

    Restricts PSNR error calculation to pixels in the spatial union of true and predicted fire regions,
    isolating error magnitude from background dominance.

    Args:
        preds: Raw model predictions/logits of shape (B, 1, H, W).
        targets: Ground truth target maps of shape (B, 1, H, W).
        threshold: Decision cutoff threshold for building the prediction mask union.
        eps: Lower numerical limit bound for Mean Squared Error (MSE).

    Returns:
        float: Signal-to-noise ratio in decibels (dB), capped at 100.0 dB for ideal predictions.
        Returns float("inf") if the union region contains zero total fire pixels.
    """
    pred_prob = torch.sigmoid(preds)
    mask = ((pred_prob >= threshold) | (targets >= 0.5)).float()
    n = mask.sum().item()
    if n == 0:
        return float("inf")
    mse = ((pred_prob - targets) ** 2 * mask).sum().item() / n
    if mse < eps:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_boundary_f1(preds: torch.Tensor, targets: torch.Tensor,
                         threshold: float = 0.5) -> float:
    """
    Calculates the F1 score restricted strictly to the perimeter edges of fire boundaries.

    Boundaries are extracted using morphological erosion via min-pooling over a 3x3 kernel.

    Args:
        preds: Pre-activation logits tensor of shape (B, 1, H, W).
        targets: Ground-truth target tensor of shape (B, 1, H, W).
        threshold: Probability cutoff for output binarization.

    Returns:
        float: Calculated spatial Boundary F1 score bounded in range [0, 1].
    """
    pred_mask = (torch.sigmoid(preds) >= threshold).float()
    target_mask = (targets >= 0.5).float()

    def get_boundary(mask: torch.Tensor) -> torch.Tensor:
        eroded = -nn.functional.max_pool2d(
            -mask, kernel_size=3, stride=1, padding=1
        )
        return (mask - eroded).clamp(0, 1)

    pred_boundary   = get_boundary(pred_mask)
    target_boundary = get_boundary(target_mask)

    tp = (pred_boundary * target_boundary).sum().item()
    fp = (pred_boundary * (1 - target_boundary)).sum().item()
    fn = ((1 - pred_boundary) * target_boundary).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_precision_recall(preds: torch.Tensor, targets: torch.Tensor,
                             threshold: float = 0.5) -> dict:
    """
    Calculates pixel-level classification Precision, Recall, and F1-score metrics.

    Args:
        preds: Raw model prediction logits tensor.
        targets: Ground truth target mask.
        threshold: Decision threshold applied after sigmoid.

    Returns:
        Dict[str, float]: Output metric scores containing keys 'precision', 'recall', and 'f1'.
    """
    pred_mask = (torch.sigmoid(preds) >= threshold).float()
    target_mask = (targets >= 0.5).float()

    tp = (pred_mask * target_mask).sum().item()
    fp = (pred_mask * (1 - target_mask)).sum().item()
    fn = ((1 - pred_mask) * target_mask).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return {"precision": prec, "recall": rec, "f1": f1}


def compute_auprc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates Area Under Precision-Recall Curve (AUPRC / Average Precision).

    Evaluates predictions directly using raw sigmoid probabilities, avoiding threshold bias.

    Args:
        preds: Raw model output prediction logits.
        targets: Target raster maps. Evaluates positive regions where values > 0.

    Returns:
        float: Mean AUPRC score across batch samples containing positive target pixels.
        Returns float("nan") if no positive pixels exist in target batch samples.
    """
    probs = torch.sigmoid(preds).detach().cpu().numpy()
    tgts = targets.detach().cpu().numpy()

    scores = []
    for b in range(probs.shape[0]):
        t = tgts[b].reshape(-1)
        p = probs[b].reshape(-1)
        if t.max() <= 0:
            continue  # no positive signal in target, AP undefined
        # binarize target for AP's definition of "positive" if it's a
        # continuous heatmap; use >0 rather than >=0.5 so faint Gaussian
        # tails still count as the positive region
        scores.append(average_precision_score((t > 0).astype(int), p))
    return float(np.mean(scores)) if scores else float("nan")


def compute_centroid_distance(preds: torch.Tensor, targets: torch.Tensor,
                               pixel_size_m: float = None) -> float:
    """
    Calculates spatial Euclidean distance between target and predicted peak intensity coordinates.

    Finds spatial discrepancy by locating the argmax coordinate in predicted and target rasters.

    Args:
        preds: Predicted model logits of shape (B, 1, H, W).
        targets: Target spatial maps of shape (B, 1, H, W).
        pixel_size_m: Spatial resolution per pixel in meters. If None, returns distance in pixel units.

    Returns:
        float: Mean distance error for samples containing valid non-zero target pixels.
        Returns float("nan") if all targets in the batch are non-positive.
    """
    probs = torch.sigmoid(preds)  # (B, 1, H, W)
    B = probs.shape[0]
    dists = []
    for b in range(B):
        t = targets[b, 0]
        if t.max() <= 0:
            continue
        p = probs[b, 0]
        t_idx = torch.argmax(t)
        p_idx = torch.argmax(p)
        H, W = t.shape
        ty, tx = t_idx // W, t_idx % W
        py, px = p_idx // W, p_idx % W
        d = torch.sqrt(((ty - py).float()) ** 2 + ((tx - px).float()) ** 2).item()
        if pixel_size_m is not None:
            d *= pixel_size_m
        dists.append(d)
    return float(np.mean(dists)) if dists else float("nan")


def compute_all_metrics(preds: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5, pixel_size_m: float = None) -> dict:
    """
    Convenience wrapper function computing all metrics simultaneously for a input batch.

    Args:
        preds: Model execution output logits tensor.
        targets: Ground truth spatial targets tensor.
        threshold: Decision threshold applied for binary metrics.
        pixel_size_m: Ground spatial resolution footprint per pixel side in meters.

    Returns:
        Dict[str, float]: Mapping of metric names to calculated quantitative scores.
    """
    pr = compute_precision_recall(preds, targets, threshold)
    return {
        "iou":                  compute_iou(preds, targets, threshold),
        "soft_iou":             compute_soft_iou(preds, targets),
        "ipsnr":                compute_ipsnr(preds, targets, threshold),
        "boundary_f1":          compute_boundary_f1(preds, targets, threshold),
        "precision":            pr["precision"],
        "recall":               pr["recall"],
        "pixel_f1":             pr["f1"],
        "auprc":                compute_auprc(preds, targets),
        "centroid_dist":        compute_centroid_distance(preds, targets, pixel_size_m=pixel_size_m),
    }