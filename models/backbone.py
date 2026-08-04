# =============================================================
# models/backbone.py
# Shared backbone — PVTv2-B2 triplet feature encoder.
#
# Takes three zoomed versions of the input image (×0.5, ×1.0, ×1.5)
# and extracts multi-scale pyramid features from each using the
# SAME backbone weights (shared / tied weights).
#
# Output: three sets of compressed feature maps, one per zoom scale,
# each at four pyramid levels [C2, C3, C4, C5] → 64 channels each.
# =============================================================

import torch
import torch.nn as nn
import timm

import configs.cod_hybrid as cfg
from data.transforms import ZoomTransform


# ── CHANNEL COMPRESSION BLOCK ────────────────────────────────
class ChannelCompress(nn.Module):
    """
    1×1 Conv → BN → ReLU
    Reduces each backbone output level to COMPRESSED_CHANNELS (64).
    One instance per pyramid level.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── BACKBONE ─────────────────────────────────────────────────
class TripletBackbone(nn.Module):
    """
    Shared PVTv2-B2 backbone that processes three zoom scales.

    For each scale s ∈ {0.5, 1.0, 1.5}:
        1. The zoomed image is passed through the shared PVTv2-B2
        2. Four feature maps are extracted [C2, C3, C4, C5]
        3. Each level is channel-compressed to 64-d via a 1×1 conv

    The shared weights mean all three scales see the same learned
    feature extractor — this is the "triplet encoder" from ZoomNeXt.

    Args:
        pretrained : load ImageNet pretrained weights for PVTv2

    Forward input:
        image : torch.Tensor  [B, 3, H, W]  normalised image batch

    Forward output:
        dict with keys 0.5, 1.0, 1.5
        each value is a list of 4 tensors:
            [feat_C2, feat_C3, feat_C4, feat_C5]
            shapes: [B, 64, H/4, W/4]
                    [B, 64, H/8, W/8]
                    [B, 64, H/16, W/16]
                    [B, 64, H/32, W/32]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Shared PVTv2-B2 ─────────────────────────────────
        # features_only=True returns intermediate feature maps
        # out_indices=(0,1,2,3) → C2, C3, C4, C5
        self.backbone = timm.create_model(
            cfg.BACKBONE,
            pretrained   = pretrained,
            features_only= True,
            out_indices  = (0, 1, 2, 3),
        )

        # ── Channel compression (one per pyramid level) ──────
        # cfg.FEATURE_CHANNELS = [64, 128, 320, 512] for PVTv2-B2
        self.compress = nn.ModuleList([
            ChannelCompress(in_ch, cfg.COMPRESSED_CHANNELS)
            for in_ch in cfg.FEATURE_CHANNELS
        ])

        # ── Zoom transform ───────────────────────────────────
        # Produces the three scaled images from one input image
        self.zoom = ZoomTransform(
            scales     = cfg.ZOOM_SCALES,
            image_size = cfg.IMAGE_SIZE,
        )

    def _extract(self, x: torch.Tensor) -> list:
        """
        Run a single image tensor through the backbone and compress.

        Returns list of 4 compressed feature tensors [C2, C3, C4, C5].
        """
        raw_features = self.backbone(x)          # list of 4 tensors
        compressed   = [
            self.compress[i](raw_features[i])
            for i in range(len(raw_features))
        ]
        return compressed                        # [f2, f3, f4, f5]

    def forward(self, image: torch.Tensor) -> dict:
        """
        Args:
            image : [B, 3, H, W]

        Returns:
            scale_features : dict
                {
                  0.5 : [f2, f3, f4, f5],   each fi: [B, 64, ...]
                  1.0 : [f2, f3, f4, f5],
                  1.5 : [f2, f3, f4, f5],
                }
        """
        # Build the three zoom versions
        # zoomed: dict {scale: [B, 3, H, W]}
        zoomed = self.zoom(image)

        scale_features = {}
        for scale in cfg.ZOOM_SCALES:
            scale_features[scale] = self._extract(zoomed[scale])

        return scale_features
