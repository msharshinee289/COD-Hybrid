# =============================================================
# train.py
# Entry point for training the COD Hybrid model.
#
# Usage:
#   # Train from scratch
#   python train.py
#
#   # Resume from checkpoint
#   python train.py --resume checkpoints/ckpt_epoch_010.pth
#
#   # Train on CPU (for debugging)
#   python train.py --device cpu
# =============================================================

import argparse
import torch

import configs.cod_hybrid as cfg
from utils.misc      import set_seed, get_device
from models          import build_model
from data.build      import build_train_loader
from engine.trainer  import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train COD Hybrid model'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=cfg.RESUME,
        help='Path to checkpoint to resume from (default: None)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=cfg.DEVICE,
        help='Device to train on: "cuda" or "cpu"'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=cfg.NUM_EPOCHS,
        help=f'Number of training epochs (default: {cfg.NUM_EPOCHS})'
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device(args.device)

    # ── Reproducibility ──────────────────────────────────────
    set_seed(cfg.SEED)

    # ── Dataloader ───────────────────────────────────────────
    print('\n[train] Building dataloader...')
    train_loader = build_train_loader()

    # ── Model ────────────────────────────────────────────────
    print('[train] Building model...')
    model = build_model(pretrained=cfg.BACKBONE_PRETRAINED)

    # # ── SANITY CHECK — remove after verification ──────────────
    # print('\n[sanity check] Running single batch forward pass...')
    # model.eval()
    # with torch.no_grad():
    #     for batch in train_loader:
    #         image = batch['image']
    #         mask  = batch['mask']
    #         out   = model(image)
    #         print(f'✅ Forward pass successful!')
    #         print(f'   Input shape  : {image.shape}')
    #         print(f'   Mask shape   : {mask.shape}')
    #         print(f'   Output shape : {out["pred"].shape}')
    #         print(f'   Output range : [{out["pred"].min():.3f}, {out["pred"].max():.3f}]')
    #         print(f'   Side outputs : {len(out["sides"])} tensors')
    #         break
    # print('[sanity check] Done. Pipeline is working correctly.\n')
    # # ── END SANITY CHECK ──────────────────────────────────────
    
    # ── Trainer ──────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        device       = device,
    )

    # ── Resume if checkpoint provided ────────────────────────
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # ── Train ────────────────────────────────────────────────
    trainer.train()


if __name__ == '__main__':
    main()
