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
# Convenience factory
# ---------------------------------------------------------------------------

def build_loss(cfg: dict) -> nn.Module:
    """
    Instantiate a loss from a config dict (e.g. loaded from experiment YAML).

    Expected YAML structure:
        loss:
          type: focal_dice          # 'focal', 'dice', or 'focal_dice'
          alpha: 0.75
          gamma: 2.0
          focal_weight: 1.0
          dice_weight: 1.0

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
    elif loss_type == "focal_dice":
        return FocalDiceLoss(
            alpha=cfg.get("alpha", 0.75),
            gamma=cfg.get("gamma", 2.0),
            focal_weight=cfg.get("focal_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
            smooth=cfg.get("smooth", 0.1),
        )
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. Choose from: focal, dice, focal_dice."
        )