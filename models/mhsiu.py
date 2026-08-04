# =============================================================
# models/mhsiu.py
# Multi-Head Scale Integration Unit (MHSIU) — from ZoomNeXt.
#
# Takes the three sets of scale features from the backbone
# (×0.5, ×1.0, ×1.5) and fuses them into one enriched feature
# map per pyramid level using attention-based scale weighting.
#
# This is the "zoom in and out" fusion step — the model learns
# which scale is most informative for each spatial location.
#
# Operates independently on each of the 4 pyramid levels.
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import configs.cod_hybrid as cfg


# ── SINGLE-LEVEL MHSIU ───────────────────────────────────────
class MHSIULevel(nn.Module):
    """
    MHSIU applied to one pyramid level.

    For one level, we have three feature maps:
        f_small  : from ×0.5 scale  [B, C, H, W]
        f_main   : from ×1.0 scale  [B, C, H, W]   (reference)
        f_large  : from ×1.5 scale  [B, C, H, W]

    All three are already at the same spatial size (same level)
    but may differ slightly due to zoom → the small/large are
    resized to match the main scale's spatial dimensions.

    Steps:
        1. Scale alignment  — resize f_small and f_large to match f_main
        2. Group-wise transformation φ and γ — two separate linear
           transformations applied to the aligned features per group
        3. Softmax attention — compute scale weights A^k_m per group m
        4. Weighted fusion — sum across scales and groups

    Args:
        channels  : number of feature channels (64)
        num_heads : number of attention groups M (default 4)
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels  = channels
        self.num_heads = num_heads

        # Group-wise transformation φ (for attention weight generation)
        # Applied independently per scale — shared across scales
        self.phi = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, groups=num_heads, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Group-wise transformation γ (for value features)
        self.gamma = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, groups=num_heads, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Attention score generator: maps 3 transformed features → 3 weights
        # Input: concatenation of φ outputs from all 3 scales → 3C channels
        # Output: 3 × num_heads attention maps
        self.attn_conv = nn.Conv2d(
            channels * 3, num_heads * 3,
            kernel_size=1, bias=True,
        )

        # Final fusion projection
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,
                f_small: torch.Tensor,
                f_main:  torch.Tensor,
                f_large: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_small : [B, C, H, W]  features from ×0.5 zoom
            f_main  : [B, C, H, W]  features from ×1.0 zoom (reference size)
            f_large : [B, C, H, W]  features from ×1.5 zoom

        Returns:
            fused   : [B, C, H, W]  fused multi-scale feature
        """
        B, C, H, W = f_main.shape

        # ── 1. Scale alignment ──────────────────────────────
        # Resize small (from ×0.5) and large (from ×1.5) to match main
        f_small = F.interpolate(f_small, size=(H, W),
                                mode='bilinear', align_corners=False)
        f_large = F.interpolate(f_large, size=(H, W),
                                mode='bilinear', align_corners=False)

        scales = [f_small, f_main, f_large]   # 3 aligned feature maps

        # ── 2. Group-wise transformations ───────────────────
        # φ: for attention score computation
        # γ: for value (content) features
        phi_feats   = [self.phi(f)   for f in scales]   # 3 × [B, C, H, W]
        gamma_feats = [self.gamma(f) for f in scales]   # 3 × [B, C, H, W]

        # ── 3. Softmax attention across scales ──────────────
        # Concatenate φ outputs → [B, 3C, H, W]
        phi_cat = torch.cat(phi_feats, dim=1)            # [B, 3C, H, W]

        # Generate raw attention scores → [B, 3*num_heads, H, W]
        attn_raw = self.attn_conv(phi_cat)

        # Reshape to [B, 3, num_heads, H, W] then softmax over scale dim (dim=1)
        attn_raw = attn_raw.view(B, 3, self.num_heads, H, W)
        attn     = torch.softmax(attn_raw, dim=1)        # [B, 3, M, H, W]

        # ── 4. Weighted sum across scales ───────────────────
        # Expand γ features to match attention shape
        # Each γ_feat: [B, C, H, W] → [B, C/M, M, H, W] per head group
        # For simplicity we treat all channels together (C = num_heads × C//num_heads)
        fused = torch.zeros_like(f_main)

        for k, gamma_f in enumerate(gamma_feats):
            # attn[:, k, :, :, :] → [B, M, H, W]
            # Expand to [B, C, H, W] by repeating M groups across channels
            weight = attn[:, k, :, :, :]                 # [B, M, H, W]
            # Repeat each head weight across C//M channels
            weight = weight.repeat_interleave(C // self.num_heads, dim=1)
            fused  = fused + weight * gamma_f

        # ── 5. Final projection ─────────────────────────────
        fused = self.fuse(fused)

        return fused                                      # [B, C, H, W]


# ── FULL MHSIU (ALL 4 PYRAMID LEVELS) ────────────────────────
class MHSIU(nn.Module):
    """
    Applies MHSIULevel independently to each of the 4 pyramid levels.

    Input:
        scale_features : dict from TripletBackbone
            {
              0.5 : [f2, f3, f4, f5],
              1.0 : [f2, f3, f4, f5],
              1.5 : [f2, f3, f4, f5],
            }
            each fi : [B, 64, H_i, W_i]

    Output:
        fused_features : list of 4 fused tensors [f2, f3, f4, f5]
            each : [B, 64, H_i, W_i]
    """

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleList([
            MHSIULevel(
                channels  = cfg.COMPRESSED_CHANNELS,
                num_heads = cfg.MHSIU_NUM_HEADS,
            )
            for _ in range(4)    # one per pyramid level C2..C5
        ])

    def forward(self, scale_features: dict) -> list:
        """
        Args:
            scale_features : dict {0.5: [...], 1.0: [...], 1.5: [...]}

        Returns:
            fused : list of 4 tensors, one per pyramid level
        """
        feats_small = scale_features[0.5]
        feats_main  = scale_features[1.0]
        feats_large = scale_features[1.5]

        fused = []
        for i, level in enumerate(self.levels):
            fused_i = level(
                f_small = feats_small[i],
                f_main  = feats_main[i],
                f_large = feats_large[i],
            )
            fused.append(fused_i)

        return fused    # [f2_fused, f3_fused, f4_fused, f5_fused]
