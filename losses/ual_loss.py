# =============================================================
# losses/ual_loss.py
# Uncertainty Awareness Loss (UAL) — from ZoomNeXt.
#
# Problem it solves:
#   BCE alone trains the model to be confident everywhere.
#   But for camouflaged objects, pixels near blended boundaries
#   are genuinely ambiguous — forcing high confidence there
#   leads to wrong predictions.
#
#   UAL adds a penalty specifically for uncertain predictions,
#   i.e. pixels where pred ≈ 0.5 (maximally uncertain).
#   This encourages the model to either be confidently right
#   or explicitly acknowledge uncertainty rather than guess.
#
# Formula:
#   i_UAL(p) = 1 - |2p - 1|^α
#
#   When p = 0.0 or 1.0 → |2p-1| = 1 → i_UAL = 0  (no penalty)
#   When p = 0.5        → |2p-1| = 0 → i_UAL = 1  (max penalty)
#
# Phase 1: fixed λ weight (UAL_LAMBDA_INIT from config)
# Phase 2: cosine-increasing λ schedule
#   λ starts near 0 so the model first learns basic segmentation
#   via BCE, then UAL gradually increases its influence
# =============================================================

import math
import torch
import torch.nn as nn

import configs.cod_hybrid as cfg


class UALLoss(nn.Module):
    """
    Uncertainty Awareness Loss.

    Args:
        alpha  : power factor in the UAL formula (default 1.0)
                 Higher α makes the penalty curve sharper —
                 penalises predictions closer to 0.5 more harshly
        lam    : weight of UAL term in total loss
                 Phase 1: fixed at UAL_LAMBDA_INIT
                 Phase 2: updated each epoch via cosine schedule

    Forward:
        pred   : [B, 1, H, W]  sigmoid probability map ∈ [0, 1]
        mask   : [B, 1, H, W]  binary ground truth (not used in
                 UAL formula but kept for interface consistency)

    Returns:
        loss   : scalar UAL loss term
    """

    def __init__(self,
                 alpha: float = cfg.UAL_ALPHA,
                 lam:   float = cfg.UAL_LAMBDA_INIT):
        super().__init__()
        self.alpha = alpha
        self.lam   = lam      # updated externally during Phase 2 training

    def forward(self,
                pred: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred : [B, 1, H, W]  values in [0, 1]
            mask : [B, 1, H, W]  not used in formula, kept for consistency

        Returns:
            loss : scalar
        """
        # i_UAL = 1 - |2p - 1|^α
        # High when p ≈ 0.5, zero when p ≈ 0 or p ≈ 1
        uncertainty = 1.0 - torch.abs(2.0 * pred - 1.0).pow(self.alpha)

        # Mean over all pixels and batch
        loss = uncertainty.mean()

        return self.lam * loss

    def set_lambda(self, lam: float):
        """
        Update the UAL weight λ externally.
        Called by the trainer each epoch in Phase 2.

        Args:
            lam : new λ value
        """
        self.lam = lam


# ── COSINE λ SCHEDULE (Phase 2) ──────────────────────────────
def cosine_lambda(epoch:      int,
                  total_epochs: int,
                  lam_init:   float = cfg.UAL_LAMBDA_INIT,
                  lam_max:    float = cfg.UAL_LAMBDA_MAX) -> float:
    """
    Computes the cosine-increasing λ for the current epoch.

    Starts at lam_init, smoothly increases to lam_max.
    The cosine curve ensures a slow start and gradual ramp-up
    rather than a sudden jump that could destabilise training.

    Formula:
        progress = epoch / total_epochs  ∈ [0, 1]
        λ = lam_init + (lam_max - lam_init) * 0.5 * (1 - cos(π * progress))

    Args:
        epoch        : current epoch (0-indexed)
        total_epochs : total number of training epochs
        lam_init     : starting λ value
        lam_max      : maximum λ value

    Returns:
        lam : float  λ value for this epoch
    """
    progress = epoch / max(total_epochs - 1, 1)
    cosine   = 0.5 * (1.0 - math.cos(math.pi * progress))
    lam      = lam_init + (lam_max - lam_init) * cosine
    return lam
