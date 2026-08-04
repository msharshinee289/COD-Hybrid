# =============================================================
# models/r3fn.py
# Reverse Features Feed Forward Network (R3FN) — from MilDetr.
#
# Problem it solves:
#   After MHSIU's attention-based fusion, global patterns are
#   well captured but LOCAL spatial information gets suppressed.
#   For camouflaged objects, local texture cues are often the
#   ONLY discriminative signal — so we must re-inject them.
#
# How it works (3 steps):
#   1. Reverse  — reshape 1D token sequence → 2D spatial map
#   2. Convolve — SConv (standard) OR EfConv (efficient) to
#                 aggregate dense local neighbourhood info
#   3. Residual — add back to the input (skip connection)
#
# Adapted from MilDetr's encoder insertion point:
#   In MilDetr: inserted after MSDA in each encoder layer
#   Here:       inserted after MHSIU, once per pyramid level
# =============================================================

import torch
import torch.nn as nn

import configs.cod_hybrid as cfg


# ── SCONV BLOCK ──────────────────────────────────────────────
class SConv(nn.Module):
    """
    Stand Conv block — aggregates local information within a
    single-scale feature map.

    Architecture:
        3×3 Conv → GroupNorm → GELU → 3×3 Conv

    Computational cost: O(HWC²)
    Used for large-scale feature maps (C2, C3).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── EFCONV BLOCK ─────────────────────────────────────────────
class EfConv(nn.Module):
    """
    Efficient Conv block — replaces conventional conv with a
    split-channel depthwise separable design to reduce compute.

    Architecture:
        Split channels in half:
            Path A (C/2): standard 3×3 Conv
            Path B (C/2): DwConv (depthwise separable, kernel C/2)
        Concatenate A and B → C channels

    Computational cost: O(HWC²/2 + HWC/2)
        vs SConv's      O(HWC²)
    Used for small-scale feature maps (C4, C5) where C is larger
    relative to spatial size, making compute savings more valuable.
    """

    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2

        # Path A — standard conv on first half of channels
        self.conv_a = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=half),
            nn.GELU(),
        )

        # Path B — depthwise separable conv on second half
        # groups=half makes it depthwise (one filter per channel)
        self.conv_b = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=3, padding=1,
                      groups=half, bias=False),          # depthwise
            nn.Conv2d(half, half, kernel_size=1, bias=False),  # pointwise
            nn.GroupNorm(num_groups=4, num_channels=half),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along channel dimension
        half    = x.shape[1] // 2
        x_a     = x[:, :half, :, :]      # first half
        x_b     = x[:, half:, :, :]      # second half

        out_a   = self.conv_a(x_a)
        out_b   = self.conv_b(x_b)

        return torch.cat([out_a, out_b], dim=1)   # [B, C, H, W]


# ── SINGLE-LEVEL R3FN ────────────────────────────────────────
class R3FNLevel(nn.Module):
    """
    R3FN applied to one pyramid level.

    Steps:
        1. Input is already a 2D spatial feature map [B, C, H, W]
           (no reshape needed here since MHSIU outputs 2D maps)
        2. Apply SConv (large levels) or EfConv (small levels)
        3. Add residual connection back to input

    Args:
        channels   : feature channels (64)
        use_efconv : if True use EfConv, else SConv
    """

    def __init__(self, channels: int, use_efconv: bool = False):
        super().__init__()
        if use_efconv:
            self.conv = EfConv(channels)
        else:
            self.conv = SConv(channels)

        # Layer norm applied before conv (pre-norm style)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W]  fused feature from MHSIU

        Returns:
            out : [B, C, H, W]  locally enhanced feature
        """
        # Pre-norm → conv → residual add
        out = self.conv(self.norm(x))
        return x + out     # residual connection


# ── FULL R3FN (ALL 4 PYRAMID LEVELS) ─────────────────────────
class R3FN(nn.Module):
    """
    Applies R3FNLevel to each of the 4 pyramid levels.

    Level assignment:
        C2, C3 (larger spatial maps) → SConv
        C4, C5 (smaller spatial maps) → EfConv

    Input:
        fused_features : list of 4 tensors from MHSIU
            [f2, f3, f4, f5]  each [B, 64, H_i, W_i]

    Output:
        enhanced_features : list of 4 tensors
            [f2, f3, f4, f5]  same shapes, local info re-injected
    """

    def __init__(self):
        super().__init__()
        C = cfg.R3FN_CHANNELS   # 64

        self.levels = nn.ModuleList([
            R3FNLevel(C, use_efconv=False),   # C2 — SConv
            R3FNLevel(C, use_efconv=False),   # C3 — SConv
            R3FNLevel(C, use_efconv=True),    # C4 — EfConv
            R3FNLevel(C, use_efconv=True),    # C5 — EfConv
        ])

    def forward(self, fused_features: list) -> list:
        """
        Args:
            fused_features : [f2, f3, f4, f5] from MHSIU

        Returns:
            enhanced : [f2, f3, f4, f5] with local info restored
        """
        return [
            level(feat)
            for level, feat in zip(self.levels, fused_features)
        ]
