# =============================================================
# engine/trainer.py
# Training loop for the COD Hybrid model.
#
# Phase 1: Basic but complete training loop.
#   - AdamW optimizer with separate LR for backbone vs new modules
#   - CosineAnnealingLR scheduler
#   - BCE + UAL loss (fixed λ in Phase 1)
#   - Checkpoint saving every SAVE_EVERY epochs
#   - Only uses main prediction (ignores side outputs)
#
# Phase 2 upgrades:
#   - Cosine-increasing λ schedule for UAL
#   - Deep supervision on side outputs
#   - Full evaluation metrics after each epoch
# =============================================================

import os
import torch
import torch.nn as nn
from torch.optim        import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import configs.cod_hybrid as cfg
from losses   import BCELoss, UALLoss
from utils.misc import (AverageMeter, log_epoch,
                         save_checkpoint, ensure_dirs,
                         count_parameters)


class Trainer:
    """
    Manages the full training loop.

    Args:
        model      : CODHybrid model instance
        train_loader : DataLoader from build_train_loader()
        device     : torch.device

    Usage:
        trainer = Trainer(model, train_loader, device)
        trainer.train()
    """

    def __init__(self, model, train_loader, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.device       = device

        # ── Losses ───────────────────────────────────────────
        self.bce_loss = BCELoss(reduction='mean')
        self.ual_loss = UALLoss(alpha=cfg.UAL_ALPHA,
                                lam=cfg.UAL_LAMBDA_INIT)

        # ── Optimizer ────────────────────────────────────────
        # Separate parameter groups:
        #   backbone → lower LR (pretrained, should update slowly)
        #   new modules → higher LR (trained from scratch)
        backbone_params = list(model.backbone.parameters())
        backbone_ids    = set(id(p) for p in backbone_params)
        new_params      = [p for p in model.parameters()
                           if id(p) not in backbone_ids]

        self.optimizer = AdamW([
            {'params': backbone_params, 'lr': cfg.LR_BACKBONE},
            {'params': new_params,      'lr': cfg.LR},
        ], weight_decay=cfg.WEIGHT_DECAY)

        # ── Scheduler ────────────────────────────────────────
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max  = cfg.NUM_EPOCHS,
            eta_min= cfg.LR_MIN,
        )

        # ── Dirs ─────────────────────────────────────────────
        ensure_dirs(cfg.CKPT_DIR, cfg.LOG_DIR)

        # ── State ────────────────────────────────────────────
        self.start_epoch = 0

        # Print model summary
        count_parameters(self.model)

    # ── RESUME FROM CHECKPOINT ───────────────────────────────
    def load_checkpoint(self, path: str):
        """Resume training from a saved checkpoint."""
        from utils.misc import load_checkpoint
        self.start_epoch = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler
        )
        print(f'[trainer] Resuming from epoch {self.start_epoch}')

    # ── SINGLE EPOCH ─────────────────────────────────────────
    def _train_one_epoch(self, epoch: int) -> dict:
        """
        Runs one full pass over the training data.

        Returns:
            dict of AverageMeter objects with loss values
        """
        self.model.train()

        meters = {
            'loss' : AverageMeter('loss'),
            'bce'  : AverageMeter('bce'),
            'ual'  : AverageMeter('ual'),
        }

        for batch_idx, batch in enumerate(self.train_loader):
            image = batch['image'].to(self.device)   # [B, 3, H, W]
            mask  = batch['mask'].to(self.device)    # [B, 1, H, W]
            B     = image.shape[0]

            # ── Forward pass ─────────────────────────────────
            outputs = self.model(image)
            pred    = outputs['pred']    # [B, 1, H, W]
            # sides = outputs['sides'] — ignored in Phase 1

            # ── Compute losses ───────────────────────────────
            loss_bce = self.bce_loss(pred, mask)
            loss_ual = self.ual_loss(pred, mask)
            loss     = loss_bce + loss_ual

            # ── Backward pass ────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=1.0)

            self.optimizer.step()

            # ── Update meters ────────────────────────────────
            meters['loss'].update(loss.item(),     B)
            meters['bce'].update(loss_bce.item(),  B)
            meters['ual'].update(loss_ual.item(),  B)

            # ── Batch log (every 50 batches) ─────────────────
            if (batch_idx + 1) % 50 == 0:
                print(f'  [Epoch {epoch:03d}]'
                      f'  batch {batch_idx+1}/{len(self.train_loader)}'
                      f'  loss: {meters["loss"].avg:.4f}'
                      f'  bce: {meters["bce"].avg:.4f}'
                      f'  ual: {meters["ual"].avg:.4f}')

        return meters

    # ── FULL TRAINING LOOP ────────────────────────────────────
    def train(self):
        """Run the full training loop for NUM_EPOCHS epochs."""
        print(f'\n[trainer] Starting training for {cfg.NUM_EPOCHS} epochs')
        print(f'[trainer] Device: {self.device}')
        print(f'[trainer] Batch size: {cfg.BATCH_SIZE}')
        print(f'[trainer] LR (new modules): {cfg.LR}')
        print(f'[trainer] LR (backbone): {cfg.LR_BACKBONE}\n')

        for epoch in range(self.start_epoch + 1,
                           cfg.NUM_EPOCHS + 1):

            # ── Train one epoch ───────────────────────────────
            meters = self._train_one_epoch(epoch)

            # ── Step scheduler ────────────────────────────────
            self.scheduler.step()

            # ── Phase 2: update UAL λ (cosine schedule) ───────
            # Uncomment in Phase 2:
            # from losses import cosine_lambda
            # lam = cosine_lambda(epoch, cfg.NUM_EPOCHS)
            # self.ual_loss.set_lambda(lam)

            # ── Epoch log ────────────────────────────────────
            log_epoch(epoch, cfg.NUM_EPOCHS, meters)

            # ── Save checkpoint ───────────────────────────────
            if epoch % cfg.SAVE_EVERY == 0 or epoch == cfg.NUM_EPOCHS:
                save_checkpoint(
                    state={
                        'epoch'    : epoch,
                        'model'    : self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    },
                    save_dir = cfg.CKPT_DIR,
                    epoch    = epoch,
                )

        print('\n[trainer] Training complete.')
