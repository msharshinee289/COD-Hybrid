# =============================================================
# models/fpq.py
# Fusion Previous Query (FPQ) — from MilDetr.
#
# Phase 1: Placeholder — passes features through unchanged.
#          The HD decoder works without FPQ in Phase 1.
#
# Phase 2: Full implementation with:
#   - Geometric Sequence Sum Fusion (GSSF)
#       FPQ(Q_l, GF) = Σ_{i=0}^{l-1} [GF^i(GF-1)/(GF^{l-1}-1)] * Q_i + Q_l
#   - Fusion Gradient Truncation (FGT)
#       Stop gradients beyond GRL layers back to prevent
#       redundant backpropagation through distant connections
#
# Role in architecture:
#   Applied at each level of the HD decoder.
#   Fuses the current decoder level's features with all
#   previous decoder level outputs — so each level refines
#   rather than re-learns from scratch.
#
# Why it matters for COD:
#   Camouflaged object boundaries are ambiguous. Each decoder
#   level sees them at a different resolution. FPQ ensures the
#   high-resolution levels KNOW what the low-resolution levels
#   already decided, avoiding contradictory predictions.
# =============================================================

import torch
import torch.nn as nn

import configs.cod_hybrid as cfg


class FPQ(nn.Module):
    """
    Phase 1: Identity placeholder.
    Accepts the current query and history but returns the
    current query unchanged. This lets the HD decoder run
    correctly in Phase 1 without FPQ logic.

    Args:
        channels : feature channels (unused in Phase 1)

    Forward:
        query   : [B, C, H, W]  current decoder level features
        history : list of previous decoder level features
                  (ignored in Phase 1)

    Returns:
        query unchanged : [B, C, H, W]
    """

    def __init__(self, channels: int = cfg.COMPRESSED_CHANNELS):
        super().__init__()
        self.channels = channels

        # Phase 1: no learnable parameters
        # Phase 2: will add GSSF projection layers here

    def forward(self,
                query:   torch.Tensor,
                history: list = None) -> torch.Tensor:
        """
        Phase 1 — identity pass-through.

        Args:
            query   : [B, C, H, W]
            history : list of previous queries (ignored in Phase 1)

        Returns:
            query   : [B, C, H, W]  unchanged
        """
        # ── Phase 2 implementation goes here ────────────────
        # Step 1: If history is empty, return query unchanged
        # Step 2: Compute geometric fusion weights using GF factor
        # Step 3: Apply FGT — detach gradients beyond GRL layers
        # Step 4: Weighted sum of history + current query
        # Step 5: Return fused query
        # ────────────────────────────────────────────────────

        return query   # Phase 1: pass-through
