import torch
import torch.nn as nn
import numpy as np


def compute_iou(preds: torch.Tensor, targets: torch.Tensor,
                threshold: float = 0.5) -> float:
    """
    Binary IoU (Intersection over Union) over a batch.

    Args:
        preds:     (B, 1, H, W) raw logits (pre-sigmoid)
        targets:   (B, 1, H, W) binary ground truth {0, 1}
        threshold: probability threshold for converting logits to a
                   binary prediction mask

    Returns:
        Scalar IoU averaged over the batch. Returns 1.0 if both
        prediction and target are empty (perfect agreement on "no fire").
    """
    pred_mask = (torch.sigmoid(preds) >= threshold).float()
    intersection = (pred_mask * targets).sum()
    union = (pred_mask + targets).clamp(0, 1).sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_ipsnr(preds: torch.Tensor, targets: torch.Tensor,
                  threshold: float = 0.5, eps: float = 1e-8) -> float:
    """
    Intersection PSNR: PSNR computed only over the union of predicted
    and true fire pixels (Badhan et al. 2020). Standard whole-image PSNR is dominated by the
    ~99.9% background pixels and is uninformative for this task; IPSNR
    isolates the signal that actually matters (placement accuracy of
    fire probability around the true and predicted fire regions).

    Returns:
        PSNR in dB, capped at 100.0 for near-perfect matches.
        Returns float('inf') if the union mask is empty (no fire pixels
        in prediction OR target), caller should exclude these from
        any running average rather than treating them as a real score.
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
    F1 score restricted to fire-perimeter pixels.

    Boundary pixels are found by morphological erosion (via negated
    max-pool), a pixel is on the boundary if it is fire but at least
    one of its 8 neighbours is not (or vice versa for the prediction).

    Returns:
        F1 in [0, 1]. Returns 0.0 if there are no boundary pixels in
        either prediction or target (degenerate case).
    """
    pred_mask = (torch.sigmoid(preds) >= threshold).float()

    def get_boundary(mask: torch.Tensor) -> torch.Tensor:
        eroded = -nn.functional.max_pool2d(
            -mask, kernel_size=3, stride=1, padding=1
        )
        return (mask - eroded).clamp(0, 1)

    pred_boundary   = get_boundary(pred_mask)
    target_boundary = get_boundary(targets.float())

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
    Pixel-level precision and recall (not boundary-restricted).

    Useful as a supplementary metric to IoU: tells whether the
    model is biased toward over-predicting (low precision, high recall)
    or under-predicting (high precision, low recall) fire, which a
    single IoU number can mask.

    Returns:
        dict with keys 'precision', 'recall', 'f1'
    """
    pred_mask = (torch.sigmoid(preds) >= threshold).float()

    tp = (pred_mask * targets).sum().item()
    fp = (pred_mask * (1 - targets)).sum().item()
    fn = ((1 - pred_mask) * targets).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return {"precision": prec, "recall": rec, "f1": f1}


def compute_all_metrics(preds: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5) -> dict:
    """
    Convenience wrapper computing every metric at once for a single batch.
    Used by evaluate.py to produce the final per-sample / aggregate
    test-set report.
    """
    pr = compute_precision_recall(preds, targets, threshold)
    return {
        "iou":          compute_iou(preds, targets, threshold),
        "ipsnr":        compute_ipsnr(preds, targets, threshold),
        "boundary_f1":  compute_boundary_f1(preds, targets, threshold),
        "precision":    pr["precision"],
        "recall":       pr["recall"],
        "pixel_f1":     pr["f1"],
    }