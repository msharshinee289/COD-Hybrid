# =============================================================
# utils/misc.py
# Shared utility functions used across the entire project.
# =============================================================

import os
import random
import numpy as np
import torch


# ── REPRODUCIBILITY ──────────────────────────────────────────
def set_seed(seed: int = 42):
    """Fix all random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes convolutions deterministic (slight speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── DEVICE ───────────────────────────────────────────────────
def get_device(device_str: str = 'cuda') -> torch.device:
    """Return a torch.device, falling back to CPU if CUDA unavailable."""
    if device_str == 'cuda' and not torch.cuda.is_available():
        print('[misc] CUDA not available — falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device_str)


# ── CHECKPOINTING ────────────────────────────────────────────
def save_checkpoint(state: dict, save_dir: str, epoch: int):
    """
    Save a training checkpoint.

    state should contain at minimum:
        {
            'epoch':       int,
            'model':       model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'scheduler':   scheduler.state_dict(),
        }
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'ckpt_epoch_{epoch:03d}.pth')
    torch.save(state, path)
    print(f'[misc] Checkpoint saved → {path}')


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint back into model (and optionally optimizer/scheduler).
    Returns the epoch number stored in the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'[misc] Checkpoint not found: {path}')

    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f'[misc] Model weights loaded from {path}')

    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        print('[misc] Optimizer state restored.')

    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
        print('[misc] Scheduler state restored.')

    return ckpt.get('epoch', 0)


# ── METRIC TRACKING ──────────────────────────────────────────
class AverageMeter:
    """
    Tracks a running average of any scalar (loss, metric, etc.).

    Usage:
        meter = AverageMeter('loss')
        meter.update(loss.item(), batch_size)
        print(meter.avg)
    """
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f'{self.name}: {self.avg:.4f}'


# ── LOGGING ──────────────────────────────────────────────────
def log_epoch(epoch: int, total_epochs: int, meters: dict):
    """
    Print a single-line training summary for the current epoch.

    meters: dict of {name: AverageMeter}

    Example output:
        [Epoch 003/100]  loss: 0.4231  bce: 0.3891  ual: 0.0340
    """
    parts = [f'[Epoch {epoch:03d}/{total_epochs}]']
    for name, meter in meters.items():
        parts.append(f'{name}: {meter.avg:.4f}')
    print('  '.join(parts))


# ── DIRECTORY HELPERS ────────────────────────────────────────
def ensure_dirs(*dirs):
    """Create directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ── PARAMETER COUNTING ───────────────────────────────────────
def count_parameters(model) -> int:
    """Return total number of trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[misc] Trainable parameters: {total / 1e6:.2f}M')
    return total
