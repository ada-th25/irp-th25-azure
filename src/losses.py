import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) for positives, (1 - sigmoid(logit)) for
    negatives, and alpha_t = alpha for positives, (1-alpha) for negatives.

    Args:
        alpha  (float): Balancing factor for the positive class.
                        Typical range 0.25-0.75; increase toward 1.0 for
                        very sparse fire patches.
        gamma  (float): Focusing exponent. gamma=0 recovers weighted BCE.
                        gamma=2 is the standard recommendation; increase to
                        3-4 if the model still ignores fire pixels.
        reduction (str): 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output (before sigmoid)
            targets : (B, 1, H, W) float tensor with values in {0.0, 1.0}

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-pixel tensor.
        """
        # Numerically stable focal loss via BCE with logits
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )                                              # (B, 1, H, W)

        p_t = torch.exp(-bce)                         # probability of the true class

        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice Loss (Milletari et al., 2016).

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    Operates on soft probabilities, so it is fully differentiable.  The
    smoothing constant prevents division-by-zero on empty patches.

    Args:
        smooth    (float): Laplace smoothing added to numerator and denominator.
                           Default 1.0 is standard; reduce to 1e-6 for stricter
                           gradient signal on fire-present patches.
        reduction (str)  : 'mean' averages over batch; 'none' returns per-sample.
    """

    def __init__(self, smooth: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output (before sigmoid)
            targets : (B, 1, H, W) float tensor with values in {0.0, 1.0}

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)                  # (B, 1, H, W)

        # Flatten spatial dims; keep batch dimension separate
        probs_flat   = probs.view(probs.size(0), -1)   # (B, H*W)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)   # (B,)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        loss = 1.0 - dice                               # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Tversky Loss
# ---------------------------------------------------------------------------
 
class TverskyLoss(nn.Module):
    """
    Tversky Loss (Salehi et al., 2017): a generalisation of Dice that
    weights false positives and false negatives independently.
 
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    Loss = 1 - Tversky
 
    Setting beta > alpha penalises missed positives (false negatives
    i.e. missed fire pixels) more heavily than false alarms (false
    positives). For severe positive-class sparsity:
    with alpha == beta == 0.5, Tversky reduces to standard Dice, which
    is why plain Dice struggles when fire pixels are ~1:100+ vs background.
 
    Args:
        alpha  (float): Weight on false positives. Lower = fewer false-alarm
                        penalties.
        beta   (float): Weight on false negatives. Higher = missed fire
                        pixels penalised more. For ~1:100+ imbalance, try
                        beta in the 0.7-0.8 range with alpha = 1 - beta.
        smooth (float): Laplace smoothing constant.
        reduction (str): 'mean' | 'sum' | 'none'
    """
 
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output (before sigmoid)
            targets : (B, 1, H, W) float tensor with values in {0.0, 1.0}
 
        Returns:
            Scalar Tversky loss.
        """
        probs = torch.sigmoid(logits)
 
        probs_flat   = probs.view(probs.size(0), -1)     # (B, H*W)
        targets_flat = targets.view(targets.size(0), -1)
 
        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)
 
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = 1.0 - tversky
 
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Combined Focal + Dice Loss
# ---------------------------------------------------------------------------

class FocalDiceLoss(nn.Module):
    """
    Weighted sum of Focal Loss and Dice Loss.

        L = focal_weight * FocalLoss + dice_weight * DiceLoss

    alpha=0.75 (rather than the standard 0.25) because fire is the rare
    positive class we care about, so we up-weight it relative to background.

    Args:
        alpha        (float): Focal alpha; weight on the positive (fire) class.
        gamma        (float): Focal focusing exponent.
        focal_weight (float): Scale factor for the Focal term.
        dice_weight  (float): Scale factor for the Dice term.
        smooth       (float): Dice smoothing constant.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 0.1,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.dice  = DiceLoss(smooth=smooth, reduction="mean")
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output
            targets : (B, 1, H, W) float {0, 1}

        Returns:
            Combined scalar loss.
        """
        focal_loss = self.focal(logits, targets)
        dice_loss  = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

    def extra_repr(self) -> str:
        return (
            f"alpha={self.focal.alpha}, gamma={self.focal.gamma}, "
            f"focal_weight={self.focal_weight}, dice_weight={self.dice_weight}"
        )


# ---------------------------------------------------------------------------
# Combined Focal + Tversky Loss
# ---------------------------------------------------------------------------
 
class FocalTverskyLoss(nn.Module):
    """
    Weighted sum of Focal Loss and Tversky Loss.
 
        L = focal_weight * FocalLoss + tversky_weight * TverskyLoss
 
    Recommended for severe positive-class sparsity (measured ~1:117 fire
    vs background on this dataset). Tversky's beta > alpha explicitly
    penalises missed fire pixels harder than false alarms, giving a much
    stronger gradient signal than plain Dice when targets.sum() is tiny
    relative to H*W.
 
    Args:
        alpha          (float): Focal alpha; weight on the positive (fire) class.
        gamma          (float): Focal focusing exponent.
        tversky_alpha  (float): Tversky false-positive weight.
        tversky_beta   (float): Tversky false-negative weight (> tversky_alpha
                                to prioritise recall on fire pixels).
        focal_weight   (float): Scale factor for the Focal term.
        tversky_weight (float): Scale factor for the Tversky term.
        smooth         (float): Tversky smoothing constant.
    """
 
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.tversky = TverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta,
            smooth=smooth, reduction="mean",
        )
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output
            targets : (B, 1, H, W) float {0, 1}
 
        Returns:
            Combined scalar loss.
        """
        focal_loss = self.focal(logits, targets)
        tversky_loss = self.tversky(logits, targets)
        return self.focal_weight * focal_loss + self.tversky_weight * tversky_loss
 
    def extra_repr(self) -> str:
        return (
            f"alpha={self.focal.alpha}, gamma={self.focal.gamma}, "
            f"tversky_alpha={self.tversky.alpha}, tversky_beta={self.tversky.beta}, "
            f"focal_weight={self.focal_weight}, tversky_weight={self.tversky_weight}"
        )


# ---------------------------------------------------------------------------
# CenterNet Gaussian Focal Loss
# ---------------------------------------------------------------------------

class GaussianFocalLoss(nn.Module):
    """
    Penalty-reduced Focal Loss for Gaussian heatmaps (CenterNet / CornerNet).
    Operates directly on logits (internally applies sigmoid).

    Args:
        alpha     (float): Focal loss power term for modulating predictions. Default: 2.0.
        beta      (float): Power term for reducing background penalty near peak. Default: 4.0.
        reduction (str)  : 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model predictions.
            targets : (B, 1, H, W) continuous ground truth Gaussian heatmap [0.0, 1.0].
        """
        preds = torch.sigmoid(logits)

        eps = 1e-6
        preds = torch.clamp(preds, min=eps, max=1.0 - eps)

        pos_mask = (targets == 1.0)
        neg_mask = (targets < 1.0)

        pos_loss = -torch.log(preds) * torch.pow(1.0 - preds, self.alpha) * pos_mask
        neg_loss = -torch.log(1.0 - preds) * torch.pow(preds, self.alpha) * torch.pow(1.0 - targets, self.beta) * neg_mask

        # DEBUG: pinpoint which term produces NaN/Inf first
        if torch.isnan(pos_loss).any() or torch.isinf(pos_loss).any():
            print(f"  [LOSS DEBUG] pos_loss has nan/inf. preds range: {preds.min().item()}-{preds.max().item()}")
        if torch.isnan(neg_loss).any() or torch.isinf(neg_loss).any():
            print(f"  [LOSS DEBUG] neg_loss has nan/inf. preds range: {preds.min().item()}-{preds.max().item()}, "
                  f"targets range: {targets.min().item()}-{targets.max().item()}")

        if self.reduction == "mean":
            n_pos = pos_mask.float().sum().clamp(min=1.0)
            n_neg = neg_mask.float().sum().clamp(min=1.0)
            result = pos_loss.sum() / n_pos + neg_loss.sum() / n_neg
            if torch.isnan(result) or torch.isinf(result):
                print(f"  [LOSS DEBUG] final result nan/inf. pos_loss.sum()={pos_loss.sum().item()}, "
                      f"neg_loss.sum()={neg_loss.sum().item()}, n_pos={n_pos.item()}, n_neg={n_neg.item()}")
            return result
        elif self.reduction == "sum":
            return (pos_loss + neg_loss).sum()

        return pos_loss + neg_loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}, reduction={self.reduction}"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_loss(cfg: dict) -> nn.Module:
    """
    Instantiate a loss from a config dict (e.g. loaded from experiment YAML).
 
    Expected YAML structure:
        loss:
          type: focal_tversky       # 'focal', 'dice', 'focal_dice', 'tversky', 'focal_tversky'
          alpha: 0.75
          gamma: 2.0
          tversky_alpha: 0.3
          tversky_beta: 0.7
          focal_weight: 1.0
          tversky_weight: 1.0
          smooth: 1.0
 
    Returns:
        nn.Module ready to call as criterion(logits, targets).
    """
    loss_type = cfg.get("type", "focal_dice").lower()
 
    if loss_type == "focal":
        return FocalLoss(
            alpha=cfg.get("alpha", 0.25),
            gamma=cfg.get("gamma", 2.0),
        )
    elif loss_type == "dice":
        return DiceLoss(smooth=cfg.get("smooth", 0.1))
    elif loss_type == "tversky":
        return TverskyLoss(
            alpha=cfg.get("tversky_alpha", 0.3),
            beta=cfg.get("tversky_beta", 0.7),
            smooth=cfg.get("smooth", 1.0),
        )
    elif loss_type == "focal_dice":
        return FocalDiceLoss(
            alpha=cfg.get("alpha", 0.75),
            gamma=cfg.get("gamma", 2.0),
            focal_weight=cfg.get("focal_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
            smooth=cfg.get("smooth", 0.1),
        )
    elif loss_type == "focal_tversky":
        return FocalTverskyLoss(
            alpha=cfg.get("alpha", 0.75),
            gamma=cfg.get("gamma", 2.0),
            tversky_alpha=cfg.get("tversky_alpha", 0.3),
            tversky_beta=cfg.get("tversky_beta", 0.7),
            focal_weight=cfg.get("focal_weight", 1.0),
            tversky_weight=cfg.get("tversky_weight", 1.0),
            smooth=cfg.get("smooth", 1.0),
        )
    elif loss_type in ("gaussian_focal", "gaussianfocalloss"):
        return GaussianFocalLoss(
            alpha=cfg.get("gaussian_alpha", 2.0),
            beta=cfg.get("gaussian_beta", 4.0),
            reduction=cfg.get("reduction", "mean"),
        )
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. Choose from: "
            f"focal, dice, tversky, focal_dice, focal_tversky."
        )