# =============================================================
# models/rgpu.py
# Rich Granularity Perception Unit (RGPU) — from ZoomNeXt.
#
# Phase 1: Basic but functional version.
# Phase 2: Full iterative channel-group mixing with feature
#          modulation vector ω as described in ZoomNeXt Sec 3.4
#
# Problem it solves:
#   After R3FN, features are locally enhanced but still treat
#   all channels equally. Camouflaged objects have subtle cues
#   spread across different channel groups — some channels
#   respond to texture, others to edges, others to colour.
#   RGPU iteratively mixes these channel groups so later groups
#   can build on refined representations from earlier ones.
#
# Phase 1 design:
#   G groups of channels processed sequentially.
#   Each group goes through a CBR block (Conv-BN-ReLU).
#   Groups are concatenated and projected back to C channels.
#   A lightweight channel attention (SE-style) reweights the
#   final features — this is the simplified modulation vector.
# =============================================================

import torch
import torch.nn as nn

import configs.cod_hybrid as cfg


# ── CBR BLOCK ────────────────────────────────────────────────
class CBR(nn.Module):
    """Conv → BatchNorm → ReLU  (standard building block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── CHANNEL ATTENTION (SE-STYLE) ─────────────────────────────
class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Produces a per-channel weight vector ω ∈ (0,1)^C
    that reweights the concatenated group features.

    This is the Phase 1 approximation of RGPU's feature
    modulation vector described in ZoomNeXt.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)      # global avg pool
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.pool(x)          # [B, C, 1, 1]
        w = self.fc(w)            # [B, C]
        w = w.view(B, C, 1, 1)   # [B, C, 1, 1]
        return x * w              # channel-wise reweighting


# ── SINGLE-LEVEL RGPU ────────────────────────────────────────
class RGPULevel(nn.Module):
    """
    RGPU applied to one pyramid level.

    Phase 1 design (G groups, sequential processing):

        Input f : [B, C, H, W]

        Split f into G groups along channel dim → each [B, C/G, H, W]

        Group 1:  g1 = CBR(group_1_features)
        Group j>1: g_j = CBR( concat(group_j_features, g_{j-1}) )
            ↑ each group sees the previous group's output,
              allowing progressive refinement

        Concatenate all g_j → [B, C, H, W]
        Apply channel attention (modulation vector ω)
        Residual add with input

    Args:
        channels : feature channels C (64)
        groups   : number of channel groups G (default 4)
    """

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        assert channels % groups == 0, \
            f'channels ({channels}) must be divisible by groups ({groups})'

        self.channels    = channels
        self.groups      = groups
        self.group_ch    = channels // groups   # channels per group

        # One CBR per group
        # Group 1: input is group_ch channels
        # Groups 2..G: input is group_ch (current) + group_ch (previous) = 2*group_ch
        self.group_convs = nn.ModuleList()
        for g in range(groups):
            in_ch = self.group_ch if g == 0 else self.group_ch * 2
            self.group_convs.append(CBR(in_ch, self.group_ch))

        # Channel attention — the modulation vector ω
        self.channel_attn = ChannelAttention(channels)

        # Final projection after concatenation
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W]  R3FN-enhanced features

        Returns:
            out : [B, C, H, W]  channel-refined features
        """
        # Split into G groups along channel dim
        groups = torch.chunk(x, self.groups, dim=1)   # G × [B, C/G, H, W]

        # Sequential group processing
        group_outputs = []
        prev = None
        for g, (conv, group_feat) in enumerate(
                zip(self.group_convs, groups)):
            if prev is None:
                # First group — process alone
                out_g = conv(group_feat)
            else:
                # Subsequent groups — concatenate with previous output
                out_g = conv(torch.cat([group_feat, prev], dim=1))
            group_outputs.append(out_g)
            prev = out_g

        # Concatenate all group outputs → [B, C, H, W]
        concat = torch.cat(group_outputs, dim=1)

        # Project
        concat = self.proj(concat)

        # Channel attention (modulation vector ω)
        concat = self.channel_attn(concat)

        # Residual connection
        return x + concat


# ── FULL RGPU (ALL 4 PYRAMID LEVELS) ─────────────────────────
class RGPU(nn.Module):
    """
    Applies RGPULevel independently to each of the 4 pyramid levels.

    Input:
        enhanced_features : list of 4 tensors from R3FN
            [f2, f3, f4, f5]  each [B, 64, H_i, W_i]

    Output:
        refined_features : list of 4 tensors
            [f2, f3, f4, f5]  same shapes, channel-refined
    """

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleList([
            RGPULevel(
                channels = cfg.COMPRESSED_CHANNELS,   # 64
                groups   = cfg.RGPU_GROUPS,           # 4
            )
            for _ in range(4)
        ])

    def forward(self, enhanced_features: list) -> list:
        """
        Args:
            enhanced_features : [f2, f3, f4, f5] from R3FN

        Returns:
            refined : [f2, f3, f4, f5]
        """
        return [
            level(feat)
            for level, feat in zip(self.levels, enhanced_features)
        ]
