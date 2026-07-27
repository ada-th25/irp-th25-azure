"""
Post-processing for final Landsat fire-mask output: adaptive thresholding
+ connected-component cleanup. Applied downstream of a trained model,
never during training-time checkpoint selection.
"""
import numpy as np
import torch
from scipy import ndimage
from skimage.filters import threshold_otsu


def postprocess_prediction(
    probs: torch.Tensor,
    method: str = "percentile",
    percentile: float = 99.5,
    fixed_threshold: float = 0.5,
    min_component_size: int = 4,
) -> torch.Tensor:
    """
    Args:
        probs: (B, 1, H, W) sigmoid probabilities (not logits).
        method: "percentile" (per-image adaptive threshold), "otsu" or "fixed".
        percentile: used when method="percentile" threshold at this
            percentile of each image's own probability distribution.
        fixed_threshold: used when method="fixed".
        min_component_size: connected components smaller than this many
            pixels are dropped as noise after thresholding.

    Returns:
        (B, 1, H, W) binary mask (float 0.0/1.0), same device as input.
    """
    device = probs.device
    probs_np = probs.detach().cpu().numpy()
    B = probs_np.shape[0]
    out = np.zeros_like(probs_np)

    for b in range(B):
        p = probs_np[b, 0]
        if method == "percentile":
            thresh = np.percentile(p, percentile)
        elif method == "otsu":
            # Otsu finds the threshold that best separates the image's
            # pixel-value histogram into two classes
            thresh = threshold_otsu(p)
        else:
            thresh = fixed_threshold
        mask = (p >= thresh).astype(np.uint8)

        if min_component_size > 0 and mask.sum() > 0:
            labeled, n_components = ndimage.label(mask)
            sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
            for comp_id, size in enumerate(sizes, start=1):
                if size < min_component_size:
                    mask[labeled == comp_id] = 0

        out[b, 0] = mask

    return torch.from_numpy(out).float().to(device)


def fit_threshold(
    model, loader, device, method="percentile", percentile_grid=None,
    fixed_grid=None,
) -> dict:
    """
    Fit the best threshold on a validation set by maximising pixel-F1,
    using compute_precision_recall. Run once, offline, after training --
    not during training-time checkpoint selection.

    Returns dict with the chosen threshold config and its val F1.
    """
    from metrics import compute_precision_recall
    import torch.nn.functional as F

    if method == "percentile":
        grid = percentile_grid or [95, 97, 99, 99.5, 99.9]
    else:
        grid = fixed_grid or [0.3, 0.4, 0.5, 0.6, 0.7]

    best = {"value": None, "f1": -1.0}
    model.eval()
    with torch.no_grad():
        for val in grid:
            total_f1, n = 0.0, 0
            for batch in loader:
                inp = batch["input"].to(device)
                target = batch["target"].to(device)
                logits = model(inp)
                if logits.shape[-2:] != target.shape[-2:]:
                    logits = F.adaptive_max_pool2d(logits, target.shape[-2:])
                probs = torch.sigmoid(logits)
                mask = postprocess_prediction(
                    probs, method=method,
                    percentile=val if method == "percentile" else 99.5,
                    fixed_threshold=val if method == "fixed" else 0.5,
                )
                # compute_precision_recall expects logits; feed mask as
                # pre-thresholded logits via a large scale trick
                pr = compute_precision_recall(
                    torch.logit(mask.clamp(1e-6, 1 - 1e-6)), target, threshold=0.5
                )
                total_f1 += pr["f1"]
                n += 1
            avg_f1 = total_f1 / max(n, 1)
            if avg_f1 > best["f1"]:
                best = {"value": val, "f1": avg_f1}
    return best