# =============================================================
# models/prediction_head.py
# Prediction Head — converts decoder features to a probability map.
#
# Takes the final output of the HD decoder [B, 64, H/4, W/4]
# and produces a single-channel probability map [B, 1, H, W]
# where each pixel value ∈ [0, 1] represents the probability
# of that pixel belonging to a camouflaged object.
#
# Architecture:
#   3×3 Conv → BN → ReLU    (feature refinement)
#   3×3 Conv → BN → ReLU    (feature refinement)
#   1×1 Conv                 (channel squeeze to 1)
#   Bilinear upsample ×4     (H/4, W/4 → H, W)
#   Sigmoid                  (scores → probabilities)
#
# The head is also applied to side outputs from the decoder
# for deep supervision in Phase 2 — so it is defined once
# and reused across all decoder levels.
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import configs.cod_hybrid as cfg


class PredictionHead(nn.Module):
    """
    Converts decoder feature maps to binary segmentation
    probability maps.

    Args:
        in_channels : channels from decoder output (64)
        out_size    : (H, W) of the final output map.
                      If None, upsamples by scale_factor instead.
        scale_factor: upsample factor when out_size is None (default 4,
                      since decoder output is H/4 × W/4)

    Forward:
        x   : [B, C, H', W']  decoder feature map
        size: optional (H, W) tuple to upsample to — used during
              eval when input images may not be 352×352

    Returns:
        prob_map : [B, 1, H, W]  sigmoid probability map
    """

    def __init__(self,
                 in_channels:  int = cfg.DECODER_CHANNELS,
                 scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_block = nn.Sequential(
            # Refine decoder features
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),

            # Squeeze to single channel
            nn.Conv2d(in_channels // 2, 1,
                      kernel_size=1, bias=True),
        )

    def forward(self,
                x:    torch.Tensor,
                size: tuple = None) -> torch.Tensor:
        """
        Args:
            x    : [B, C, H', W']
            size : (H, W) to upsample to — if None uses scale_factor

        Returns:
            prob_map : [B, 1, H, W]  values in [0, 1]
        """
        # ── Conv block ──────────────────────────────────────
        out = self.conv_block(x)          # [B, 1, H', W']

        # ── Upsample to input resolution ────────────────────
        if size is not None:
            out = F.interpolate(out, size=size,
                                mode='bilinear', align_corners=False)
        else:
            out = F.interpolate(out, scale_factor=self.scale_factor,
                                mode='bilinear', align_corners=False)

        # ── Sigmoid → probability map ────────────────────────
        out = torch.sigmoid(out)          # [B, 1, H, W]

        return out
