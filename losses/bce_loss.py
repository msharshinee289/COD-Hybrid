# =============================================================
# losses/bce_loss.py
# Weighted Binary Cross-Entropy Loss for COD segmentation.
#
# Standard BCE treats every pixel equally. For COD this is
# a problem because:
#   - Most pixels are background (class imbalance)
#   - Hard pixels near ambiguous boundaries matter most
#
# This implementation adds two improvements:
#   1. Weighted BCE — foreground pixels get higher weight
#      when the dataset is imbalanced (more bg than fg)
#   2. The weight is computed PER IMAGE from its own mask
#      ratio, not as a global constant — so each image's
#      loss is appropriately balanced regardless of how
#      large or small the camouflaged object is
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Per-image weighted Binary Cross-Entropy Loss.

    For each image in the batch:
        num_pos = number of foreground pixels (mask == 1)
        num_neg = number of background pixels (mask == 0)
        weight_pos = num_neg / (num_pos + num_neg)
        weight_neg = num_pos / (num_pos + num_neg)

    This balances the contribution of positive and negative
    pixels regardless of object size — critical for COD where
    camouflaged objects can be very small.

    Args:
        reduction : 'mean' (default) or 'sum'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred : [B, 1, H, W]  sigmoid probability map ∈ [0, 1]
            mask : [B, 1, H, W]  binary ground truth ∈ {0, 1}

        Returns:
            loss : scalar tensor
        """
        assert pred.shape == mask.shape, \
            f'pred shape {pred.shape} != mask shape {mask.shape}'

        B = pred.shape[0]
        loss_total = 0.0

        for b in range(B):
            p = pred[b]   # [1, H, W]
            m = mask[b]   # [1, H, W]

            # Count positive and negative pixels
            num_pos = m.sum()
            num_neg = m.numel() - num_pos

            # Per-image adaptive weights
            total    = num_pos + num_neg
            w_pos    = (num_neg / total).clamp(min=0.1, max=0.9)
            w_neg    = (num_pos / total).clamp(min=0.1, max=0.9)

            # Build per-pixel weight map
            weight = torch.where(m >= 0.5, w_pos, w_neg)  # [1, H, W]

            # Binary cross entropy with per-pixel weights
            bce = F.binary_cross_entropy(p, m,
                                         weight=weight,
                                         reduction='mean')
            loss_total = loss_total + bce

        loss = loss_total / B

        return loss
