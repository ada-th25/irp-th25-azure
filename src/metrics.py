import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score


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
    target_mask = (targets >= 0.5).float()
    intersection = (pred_mask * target_mask).sum()
    union = (pred_mask + target_mask).clamp(0, 1).sum()
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
    Pixel-level precision and recall (not boundary-restricted).

    Useful as a supplementary metric to IoU: tells whether the
    model is biased toward over-predicting (low precision, high recall)
    or under-predicting (high precision, low recall) fire, which a
    single IoU number can mask.

    Returns:
        dict with keys 'precision', 'recall', 'f1'
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
    Average precision computed directly on continuous probabilities vs.
    continuous/binary targets, no thresholding. This is the fair way to
    compare losses whose output calibration (max prob, sharpness) differs,
    since compute_iou/precision/recall depend heavily on a fixed threshold
    that different losses clear very differently.

    Returns:
        Scalar AUPRC averaged over the batch. Skips samples with an
        all-zero target (undefined precision-recall curve).
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
    Euclidean distance (in pixels, or metres if pixel_size_m given) between
    the argmax location of the predicted heatmap and the argmax location
    of the target heatmap, per sample.

    This measures "how far off is the fire location",
    which IoU at a fixed threshold does not cleanly capture.

    Returns:
        Mean distance over samples where the target has a positive peak.
        Samples with an all-zero target are skipped (no true location to
        compare against).
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
    Convenience wrapper computing every metric at once for a single batch.

    Args:
        pixel_size_m: native pixel resolution of the target being evaluated
                      (e.g. 384.0 for VIIRS Stage-1, 30.0 for Landsat Stage-2).
                      Passed through to compute_centroid_distance so reported
                      distances are in metres; leave as None to get distance
                      in pixels instead (not comparable across branches).
    """
    pr = compute_precision_recall(preds, targets, threshold)
    return {
        "iou":                  compute_iou(preds, targets, threshold),
        "ipsnr":                compute_ipsnr(preds, targets, threshold),
        "boundary_f1":          compute_boundary_f1(preds, targets, threshold),
        "precision":            pr["precision"],
        "recall":               pr["recall"],
        "pixel_f1":             pr["f1"],
        "auprc":                compute_auprc(preds, targets),
        "centroid_dist":        compute_centroid_distance(preds, targets, pixel_size_m=pixel_size_m),
    }