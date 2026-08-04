# =============================================================
# data/build.py
# Factory functions that build ready-to-use DataLoaders.
#
# train.py and engine/trainer.py import from here — they never
# instantiate datasets or transforms directly.
# =============================================================

import os
from torch.utils.data import DataLoader

from data.dataset    import CODDataset, CombinedCODDataset
from data.transforms import TrainTransform, EvalTransform
import configs.cod_hybrid as cfg


# ── TRAINING DATALOADER ──────────────────────────────────────
def build_train_loader() -> DataLoader:
    """
    Builds the training DataLoader.

    - Combines CAMO (Train) + COD10K (Train) into one dataset
    - Applies TrainTransform (augmentation + normalisation)
    - Shuffles every epoch
    - Drops the last incomplete batch so batch size is always consistent

    Returns:
        DataLoader yielding dicts with keys: 'image', 'mask', 'img_name'
    """
    transform = TrainTransform(image_size=cfg.IMAGE_SIZE)

    camo_train = CODDataset(
        root         = cfg.CAMO_ROOT,
        split        = 'Train',
        transform    = transform,
        image_subdir = os.path.join('Images', 'Train'),
        mask_subdir  = 'GT',
    )
    # cod10k_train = CODDataset( - # Use CAMO only for Phase 1 (1000 images instead of 7000)
    #     root         = cfg.COD10K_ROOT,
    #     split        = 'Train',
    #     transform    = transform,
    #     image_subdir = os.path.join('Train', 'Image'),
    #     mask_subdir  = os.path.join('Train', 'GT_Object'),
    # )

    # combined = CombinedCODDataset([camo_train, cod10k_train]) - # Use CAMO only for Phase 1 (1000 images instead of 7000), Reduces epoch time from ~27 minutes to ~4 minutes
    combined = camo_train

    loader = DataLoader(
        combined,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = True,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = True,     # faster CPU→GPU transfer
        drop_last   = True,     # avoid inconsistent batch sizes
    )

    print(f'[build] Train loader — '
          f'{len(combined)} samples, '
          f'{len(loader)} batches per epoch '
          f'(batch size {cfg.BATCH_SIZE})')

    return loader


# ── EVALUATION DATALOADERS ───────────────────────────────────
def build_eval_loader(dataset_name: str) -> DataLoader:
    """
    Builds an evaluation DataLoader for a single benchmark.

    Args:
        dataset_name : one of 'CAMO' or 'COD10K'

    Returns:
        DataLoader yielding dicts with keys: 'image', 'mask', 'img_name'

    Note:
        - batch_size is fixed at 1 for evaluation so predictions can be
          saved individually with their correct filenames
        - No shuffling — consistent ordering across runs
    """
    transform = EvalTransform(image_size=cfg.EVAL_SIZE)

    # Map dataset name to its root path
    roots = {
        'CAMO'  : (cfg.CAMO_ROOT,
                os.path.join('Images', 'Test'),
                'GT'),
        'COD10K': (cfg.COD10K_ROOT,
                os.path.join('Test', 'Image'),
                os.path.join('Test', 'GT_Object')),
    }

    if dataset_name not in roots:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(roots.keys())}"
        )

    dataset = CODDataset(
        root         = roots[dataset_name][0],
        split        = 'Test',
        transform    = transform,
        image_subdir = roots[dataset_name][1],
        mask_subdir  = roots[dataset_name][2],
    )

    loader = DataLoader(
        dataset,
        batch_size  = 1,         # one image at a time during evaluation
        shuffle     = False,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = True,
    )

    print(f'[build] Eval loader ({dataset_name}) — '
          f'{len(dataset)} test samples')

    return loader


# ── CONVENIENCE: ALL EVAL LOADERS ────────────────────────────
def build_all_eval_loaders() -> dict:
    """
    Builds eval DataLoaders for every dataset listed in cfg.EVAL_DATASETS.

    Returns:
        dict mapping dataset name → DataLoader
        e.g. {'CAMO': <DataLoader>, 'COD10K': <DataLoader>}
    """
    return {
        name: build_eval_loader(name)
        for name in cfg.EVAL_DATASETS
    }
