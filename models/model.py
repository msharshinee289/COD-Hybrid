# =============================================================
# models/model.py
# Top-level CODHybrid model — wires all modules together.
#
# Forward pass:
#   image [B,3,H,W]
#     → TripletBackbone   → scale_features (dict of 3 × 4 tensors)
#     → MHSIU             → fused_features (list of 4 tensors)
#     → R3FN              → enhanced_features (list of 4 tensors)
#     → RGPU              → refined_features (list of 4 tensors)
#     → HDDecoder         → (final_feat, side_outputs)
#     → PredictionHead    → prob_map [B,1,H,W]
#
# Returns a dict:
#   {
#     'pred'  : [B,1,H,W]          ← main prediction (always present)
#     'sides' : list of [B,1,H',W'] ← side predictions (Phase 2 deep sup.)
#   }
# =============================================================

import torch
import torch.nn as nn

import configs.cod_hybrid as cfg
from models.backbone        import TripletBackbone
from models.mhsiu           import MHSIU
from models.r3fn            import R3FN
from models.rgpu            import RGPU
from models.hd_decoder      import HDDecoder
from models.prediction_head import PredictionHead


class CODHybrid(nn.Module):
    """
    Unified Hybrid Architecture for Camouflaged Object Detection.

    Combines:
      - ZoomNeXt's triplet zoom strategy, MHSIU, RGPU, HD Decoder
      - MilDetr's R3FN (local feature re-injection) and FPQ
        (hierarchical query fusion, active in Phase 2)
      - Uncertainty-Aware Loss support via multi-output dict

    Args:
        pretrained : whether to load ImageNet weights for backbone
    """

    def __init__(self, pretrained: bool = cfg.BACKBONE_PRETRAINED):
        super().__init__()

        # ── Encoder pipeline ─────────────────────────────────
        self.backbone = TripletBackbone(pretrained=pretrained)
        self.mhsiu    = MHSIU()
        self.r3fn     = R3FN()
        self.rgpu     = RGPU()

        # ── Decoder pipeline ─────────────────────────────────
        self.decoder  = HDDecoder()

        # ── Prediction head ───────────────────────────────────
        # Shared head — used for main output and side outputs
        self.head     = PredictionHead(
            in_channels  = cfg.DECODER_CHANNELS,
            scale_factor = 4,
        )

    def forward(self, image: torch.Tensor) -> dict:
        """
        Args:
            image : [B, 3, H, W]  normalised input image batch

        Returns:
            dict {
                'pred'  : [B, 1, H, W]   main probability map
                'sides' : list of tensors  side predictions at
                          lower resolutions (empty in Phase 1,
                          used for deep supervision in Phase 2)
            }
        """
        H, W = image.shape[-2], image.shape[-1]

        # ── Encoder ──────────────────────────────────────────
        scale_feats    = self.backbone(image)       # {0.5,1.0,1.5} → [f2..f5]
        fused_feats    = self.mhsiu(scale_feats)    # [f2..f5] fused across scales
        enhanced_feats = self.r3fn(fused_feats)     # [f2..f5] local info restored
        refined_feats  = self.rgpu(enhanced_feats)  # [f2..f5] channel refined

        # ── Decoder ──────────────────────────────────────────
        final_feat, side_feats = self.decoder(refined_feats)
        # final_feat : [B, 64, H/4, W/4]
        # side_feats : [out_4, out_3, out_2] at lower resolutions

        # ── Main prediction ───────────────────────────────────
        pred = self.head(final_feat, size=(H, W))   # [B, 1, H, W]

        # ── Side predictions (Phase 2 deep supervision) ───────
        # In Phase 1 the trainer ignores these.
        # In Phase 2 the trainer computes loss on each of them.
        sides = [
            self.head(sf, size=(H, W))
            for sf in side_feats
        ]

        return {
            'pred'  : pred,    # [B, 1, H, W]
            'sides' : sides,   # list of [B, 1, H, W]
        }


# ── BUILD FUNCTION ────────────────────────────────────────────
def build_model(pretrained: bool = cfg.BACKBONE_PRETRAINED) -> CODHybrid:
    """
    Convenience factory used by train.py and eval.py.

    Usage:
        from models.model import build_model
        model = build_model()
    """
    model = CODHybrid(pretrained=pretrained)
    return model
