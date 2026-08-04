# =============================================================
# configs/cod_hybrid.py
# Central configuration for the COD Hybrid architecture.
# Every other file imports from here — never hard-code values.
# =============================================================

import os

# ── PATHS ────────────────────────────────────────────────────
# ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_ROOT  = os.path.join(ROOT_DIR, 'data', 'datasets')   # CAMO / COD10K here
# CKPT_DIR   = os.path.join(ROOT_DIR, 'checkpoints') 
# LOG_DIR    = os.path.join(ROOT_DIR, 'logs')


# ── PATHS - Google Colab ────────────────────────────────────────────────────
ROOT_DIR  = '/content/drive/MyDrive/cod_hybrid_project/cod_hybrid'
DATA_ROOT = '/content/drive/MyDrive/cod_hybrid_project/cod_hybrid/data/datasets'
CKPT_DIR  = '/content/drive/MyDrive/cod_hybrid_project/cod_hybrid/checkpoints'
LOG_DIR   = '/content/drive/MyDrive/cod_hybrid_project/cod_hybrid/logs'

# Dataset paths (update these to match your local folder layout)
CAMO_ROOT    = os.path.join(DATA_ROOT, 'CAMO')
COD10K_ROOT  = os.path.join(DATA_ROOT, 'COD10K')

# ── INPUT ────────────────────────────────────────────────────
IMAGE_SIZE   = 352          # training crop resolution (352 × 352)
ZOOM_SCALES  = [0.5, 1.0, 1.5]   # zoom-out, main, zoom-in

# ── BACKBONE ─────────────────────────────────────────────────
BACKBONE          = 'pvt_v2_b2'   # timm model name
BACKBONE_PRETRAINED = True
FEATURE_CHANNELS  = [64, 128, 320, 512]   # PVTv2-B2 output channels at C2-C5
COMPRESSED_CHANNELS = 64                  # after 1×1 channel compression

# ── MHSIU ────────────────────────────────────────────────────
MHSIU_NUM_HEADS = 4      # number of attention groups M

# ── R3FN ─────────────────────────────────────────────────────
R3FN_CHANNELS = 64       # operates on compressed feature channels

# ── RGPU ─────────────────────────────────────────────────────
RGPU_GROUPS = 4          # number of channel groups G (>= 2)

# ── FPQ (Phase 2) ────────────────────────────────────────────
FPQ_GEOMETRIC_FACTOR = 2    # GF — controls fusion weight decay
FPQ_GRADIENT_REFLOW  = 4    # GRL — how many layers back to allow gradients

# ── HD DECODER ───────────────────────────────────────────────
DECODER_CHANNELS = 64    # uniform channel width through decoder pyramid

# ── LOSS ─────────────────────────────────────────────────────
# BCE loss
BCE_REDUCTION = 'mean'

# UAL loss  (Phase 1: fixed lambda; Phase 2: cosine schedule)
UAL_ALPHA          = 1.0     # power factor α in  1 - |2p - 1|^α
UAL_LAMBDA_INIT    = 0.1     # starting weight of UAL term
UAL_LAMBDA_MAX     = 1.0     # maximum weight (reached at end of training)
# Phase 1 uses UAL_LAMBDA_INIT as a fixed value throughout

# ── TRAINING ─────────────────────────────────────────────────
BATCH_SIZE     = 8
#NUM_WORKERS    = 4
NUM_WORKERS = 2     # reduced from 4 — better for Drive I/O
#NUM_EPOCHS     = 100         # Phase 1: run fewer (e.g. 30) to verify pipeline
NUM_EPOCHS     = 30         # Phase 1: run fewer (e.g. 30) to verify pipeline
LR             = 1e-4        # AdamW learning rate
LR_BACKBONE    = 1e-5        # lower LR for pretrained backbone
WEIGHT_DECAY   = 1e-4
# Cosine annealing scheduler
LR_MIN         = 1e-6        # minimum LR at end of schedule

# ── EVALUATION ───────────────────────────────────────────────
EVAL_DATASETS  = ['CAMO', 'COD10K']   # Phase 2 adds NC4K, CHAMELEON
EVAL_SIZE      = 352

# ── CHECKPOINTING ────────────────────────────────────────────
#SAVE_EVERY     = 10          # save checkpoint every N epochs
SAVE_EVERY  = 5     # reduced from 10 — saves checkpoint every 5 epochs
RESUME         = None        # path to checkpoint to resume from, or None

# ── MISC ─────────────────────────────────────────────────────
SEED           = 42
DEVICE         = 'cuda'      # 'cuda' or 'cpu'
