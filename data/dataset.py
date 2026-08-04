# =============================================================
# data/dataset.py
# PyTorch Dataset classes for CAMO and COD10K.
#
# Expected folder layout for each dataset:
#
#   CAMO/
#     ├── Image/
#     │     ├── Train/       ← .jpg files
#     │     └── Test/       
#     └── GT/                ← .png binary masks (0 or 255)
#
#   COD10K/
#     ├── Train/
#     │     ├── Image/
#     │     └── GT/
#     └── Test/
#           ├── Image/
#           └── GT/
#
# Each mask is a grayscale PNG where:
#   255 = camouflaged object pixel
#     0 = background pixel
# =============================================================

import os
from PIL import Image
from torch.utils.data import Dataset


# ── SINGLE DATASET ───────────────────────────────────────────
class CODDataset(Dataset):
    """
    Loads (image, mask) pairs from a single COD benchmark folder
    (CAMO or COD10K).

    Args:
        root      : path to the dataset root (e.g. /data/datasets/CAMO)
        split     : 'Train' or 'Test'
        transform : a TrainTransform or EvalTransform instance
    """

    def __init__(self, root: str, split: str = 'Train',
             transform=None,
             image_subdir: str = None,
             mask_subdir:  str = None):
        super().__init__()
        assert split in ('Train', 'Test'), \
            f"split must be 'Train' or 'Test', got '{split}'"

        self.transform = transform
        self.image_dir = os.path.join(root, image_subdir)
        self.mask_dir  = os.path.join(root, mask_subdir)
        # Collect all image filenames
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.images) == 0:
            raise RuntimeError(
                f'No images found in {self.image_dir}. '
                f'Check your dataset path and folder structure.'
            )

        print(f'[dataset] {os.path.basename(root)} / {split} — '
              f'{len(self.images)} samples found.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name  = self.images[idx]
        # Mask has same name but .png extension
        mask_name = os.path.splitext(img_name)[0] + '.png'

        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir,  mask_name)

        # ── Load ────────────────────────────────────────────
        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')    # grayscale

        # ── Transform ───────────────────────────────────────
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            'image'    : image,      # [3, H, W]  float32, normalised
            'mask'     : mask,       # [1, H, W]  float32, values in {0,1}
            'img_name' : img_name,   # kept for saving predictions during eval
        }


# ── COMBINED DATASET ─────────────────────────────────────────
class CombinedCODDataset(Dataset):
    """
    Concatenates multiple CODDataset instances into one.
    Used to train on CAMO + COD10K simultaneously.

    Args:
        datasets : list of CODDataset instances
    """

    def __init__(self, datasets: list):
        super().__init__()
        self.datasets = datasets
        self.lengths  = [len(d) for d in datasets]
        self.total    = sum(self.lengths)

        print(f'[dataset] Combined dataset — '
              f'{self.total} total samples '
              f'({" + ".join(str(l) for l in self.lengths)})')

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        # Map global index to the correct sub-dataset and local index
        for dataset, length in zip(self.datasets, self.lengths):
            if idx < length:
                return dataset[idx]
            idx -= length
        raise IndexError('Index out of range in CombinedCODDataset.')
