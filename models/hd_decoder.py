# =============================================================
# models/hd_decoder.py
# Hierarchical Difference Propagation Decoder (HD Decoder).
#
# Takes the 4 pyramid features from RGPU and decodes them
# top-down into a single high-resolution feature map that
# the prediction head turns into a probability map.
#
# Key idea — difference propagation:
#   At each level, instead of simply adding top-down features
#   to skip-connected encoder features, we compute the
#   DIFFERENCE between them. This difference highlights what
#   the high-level semantic representation missed at each
#   local spatial scale — exactly the subtle boundary and
#   texture cues that camouflage exploits.
#
# Top-down flow (C5 → C4 → C3 → C2):
#   Level 4 (deepest, most semantic):
#       query_4 = FPQ(f5, history=[])
#       out_4   = decode(query_4)
#
#   Level 3:
#       upsample out_4 to match f4 spatial size
#       diff_3  = upsample(out_4) - f4        ← difference signal
#       query_3 = FPQ(f4 + diff_3, history=[out_4])
#       out_3   = decode(query_3)
#
#   Level 2, Level 1: same pattern
#
#   Final output: out_1  [B, 64, H/4, W/4]
#   (prediction head upsamples this to full resolution)
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import configs.cod_hybrid as cfg
from models.fpq import FPQ


# ── DECODE BLOCK ─────────────────────────────────────────────
class DecodeBlock(nn.Module):
    """
    Single decode step applied at each pyramid level.

    Architecture:
        Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU

    Takes fused features (encoder + top-down + difference)
    and produces refined decoder features at that level.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── DIFFERENCE FUSION BLOCK ──────────────────────────────────
class DifferenceFusion(nn.Module):
    """
    Computes and fuses the difference signal between
    top-down features and encoder skip features.

    Steps:
        1. Upsample top-down features to match encoder spatial size
        2. Compute difference: diff = upsampled - encoder_feat
        3. Concatenate encoder_feat with diff signal
        4. Project fused features through a conv layer

    Args:
        channels : feature channels (64)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,
                top_down:     torch.Tensor,
                encoder_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            top_down     : [B, C, H_low, W_low]  from previous decoder level
            encoder_feat : [B, C, H_high, W_high] skip connection from RGPU

        Returns:
            fused : [B, C, H_high, W_high]
        """
        H, W = encoder_feat.shape[-2], encoder_feat.shape[-1]

        # 1. Upsample top-down to match encoder spatial size
        top_up = F.interpolate(top_down, size=(H, W),
                               mode='bilinear', align_corners=False)

        # 2. Compute difference — highlights what top-down missed
        diff = top_up - encoder_feat

        # 3. Concatenate encoder features with difference signal
        fused = torch.cat([encoder_feat, diff], dim=1)  # [B, 2C, H, W]

        # 4. Project back to C channels
        return self.proj(fused)                          # [B, C, H, W]


# ── HD DECODER ───────────────────────────────────────────────
class HDDecoder(nn.Module):
    """
    Hierarchical Difference Propagation Decoder.

    Processes 4 pyramid levels top-down (C5 → C4 → C3 → C2).
    At each level:
        - DifferenceFusion merges top-down + skip features
        - FPQ refines the query using history of previous levels
        - DecodeBlock produces the output for this level

    Input:
        refined_features : list of 4 tensors from RGPU
            [f2, f3, f4, f5]
            f2: [B, 64, H/4,  W/4 ]   ← finest, most spatial
            f3: [B, 64, H/8,  W/8 ]
            f4: [B, 64, H/16, W/16]
            f5: [B, 64, H/32, W/32]   ← coarsest, most semantic

    Output:
        out      : [B, 64, H/4, W/4]  final decoded feature map
        side_out : list of 3 intermediate outputs for deep supervision
    """

    def __init__(self):
        super().__init__()
        C = cfg.DECODER_CHANNELS   # 64

        # FPQ module (Phase 1: pass-through; Phase 2: full fusion)
        self.fpq = FPQ(channels=C)

        # Difference fusion blocks (3 transitions between 4 levels)
        self.diff_fusion = nn.ModuleList([
            DifferenceFusion(C),   # C5 → C4
            DifferenceFusion(C),   # C4 → C3
            DifferenceFusion(C),   # C3 → C2
        ])

        # Decode blocks at all 4 levels
        self.decode_blocks = nn.ModuleList([
            DecodeBlock(C),        # level 4 (C5)
            DecodeBlock(C),        # level 3 (C4)
            DecodeBlock(C),        # level 2 (C3)
            DecodeBlock(C),        # level 1 (C2)
        ])

    def forward(self, refined_features: list) -> tuple:
        """
        Args:
            refined_features : [f2, f3, f4, f5] from RGPU

        Returns:
            final_out    : [B, 64, H/4, W/4]
            side_outputs : list of 3 intermediate decoder outputs
                           [out_4, out_3, out_2]
        """
        f2, f3, f4, f5 = refined_features
        history      = []    # stores previous decoder outputs for FPQ
        side_outputs = []    # intermediate outputs for deep supervision

        # ── Level 4 — start from deepest features (C5) ──────
        q4    = self.fpq(f5, history)
        out_4 = self.decode_blocks[0](q4)
        history.append(out_4)
        side_outputs.append(out_4)

        # ── Level 3 — C5 → C4 ───────────────────────────────
        fused_3 = self.diff_fusion[0](out_4, f4)
        q3      = self.fpq(fused_3, history)
        out_3   = self.decode_blocks[1](q3)
        history.append(out_3)
        side_outputs.append(out_3)

        # ── Level 2 — C4 → C3 ───────────────────────────────
        fused_2 = self.diff_fusion[1](out_3, f3)
        q2      = self.fpq(fused_2, history)
        out_2   = self.decode_blocks[2](q2)
        history.append(out_2)
        side_outputs.append(out_2)

        # ── Level 1 — C3 → C2 ───────────────────────────────
        fused_1 = self.diff_fusion[2](out_2, f2)
        q1      = self.fpq(fused_1, history)
        out_1   = self.decode_blocks[3](q1)

        return out_1, side_outputs
        # out_1        → prediction head
        # side_outputs → deep supervision in Phase 2
