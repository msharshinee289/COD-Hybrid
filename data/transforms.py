# =============================================================
# data/transforms.py
# Image + mask augmentations for training and evaluation.
#
# IMPORTANT: Every spatial transform (flip, crop, resize) is
# applied identically to both the image AND its binary mask
# so they stay perfectly aligned.
# =============================================================

import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


# ── TRAINING TRANSFORMS ──────────────────────────────────────
class TrainTransform:
    """
    Augmentation pipeline used during training.

    Steps (applied to image + mask together):
        1. Resize to slightly larger than target (scale jitter)
        2. Random horizontal flip
        3. Random crop to IMAGE_SIZE × IMAGE_SIZE
        4. Color jitter on image only (not mask)
        5. Convert to tensor and normalise image
    """

    def __init__(self, image_size: int = 352):
        self.image_size = image_size
        # Resize to 1.25× before cropping gives room to crop randomly
        self.resize_to  = int(image_size * 1.25)     # 440 for size=352

    def __call__(self, image: Image.Image, mask: Image.Image):
        # ── 1. Resize ──────────────────────────────────────
        image = TF.resize(image, (self.resize_to, self.resize_to),
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (self.resize_to, self.resize_to),
                          interpolation=TF.InterpolationMode.NEAREST)
        # NEAREST for mask — we never want interpolated label values

        # ── 2. Random horizontal flip ───────────────────────
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # ── 3. Random crop ──────────────────────────────────
        i, j, h, w = _random_crop_params(image, self.image_size)
        image = TF.crop(image, i, j, h, w)
        mask  = TF.crop(mask,  i, j, h, w)

        # ── 4. Color jitter (image only) ────────────────────
        image = _color_jitter(image,
                              brightness=0.3,
                              contrast=0.3,
                              saturation=0.3,
                              hue=0.05)

        # ── 5. To tensor + normalise ────────────────────────
        image = TF.to_tensor(image)            # [3, H, W]  float32 in [0,1]
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])

        # Mask → single-channel float tensor in {0, 1}
        mask  = torch.from_numpy(
                    np.array(mask, dtype=np.float32) / 255.0
                ).unsqueeze(0)                 # [1, H, W]

        return image, mask


# ── EVALUATION TRANSFORMS ────────────────────────────────────
class EvalTransform:
    """
    Deterministic pipeline used during validation and testing.

    Steps:
        1. Resize image and mask to IMAGE_SIZE × IMAGE_SIZE
        2. Convert to tensor and normalise image
    No random ops — evaluation must be reproducible.
    """

    def __init__(self, image_size: int = 352):
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        # ── 1. Resize ──────────────────────────────────────
        image = TF.resize(image, (self.image_size, self.image_size),
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (self.image_size, self.image_size),
                          interpolation=TF.InterpolationMode.NEAREST)

        # ── 2. To tensor + normalise ────────────────────────
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])

        mask  = torch.from_numpy(
                    np.array(mask, dtype=np.float32) / 255.0
                ).unsqueeze(0)                 # [1, H, W]

        return image, mask


# ── ZOOM SCALE TRANSFORM ─────────────────────────────────────
class ZoomTransform:
    """
    Produces the three zoomed versions of a single image tensor
    that the triplet backbone expects.

    Input : image tensor  [3, H, W]   (already normalised)
    Output: dict with keys 0.5, 1.0, 1.5 → each a [3, H, W] tensor
            All three are resized back to the same H×W so they can
            be stacked and passed through the shared backbone together.
    """

    def __init__(self, scales: list = None, image_size: int = 352):
        self.scales     = scales or [0.5, 1.0, 1.5]
        self.image_size = image_size

    def __call__(self, image: torch.Tensor) -> dict:
        H, W = image.shape[-2], image.shape[-1]
        zoomed = {}
        for scale in self.scales:
            # Compute the scaled resolution
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))

            # Resize to scaled size, then back to original size
            # so all three share the same spatial dimensions
            scaled = TF.resize(image, (new_h, new_w),
                               interpolation=TF.InterpolationMode.BILINEAR)
            scaled = TF.resize(scaled, (H, W),
                               interpolation=TF.InterpolationMode.BILINEAR)
            zoomed[scale] = scaled
        return zoomed


# ── PRIVATE HELPERS ──────────────────────────────────────────
def _random_crop_params(image: Image.Image, crop_size: int):
    """Return (top, left, height, width) for a random crop."""
    w, h = image.size
    assert h >= crop_size and w >= crop_size, \
        f'Image ({h}×{w}) is smaller than crop size ({crop_size}).'
    top  = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return top, left, crop_size, crop_size


def _color_jitter(image: Image.Image,
                  brightness: float = 0.3,
                  contrast:   float = 0.3,
                  saturation: float = 0.3,
                  hue:        float = 0.05) -> Image.Image:
    """Apply random color jitter to a PIL image."""
    # Randomly decide whether to apply each jitter in a random order
    transforms = []

    if brightness > 0:
        b = random.uniform(max(0, 1 - brightness), 1 + brightness)
        transforms.append(lambda img, b=b: TF.adjust_brightness(img, b))

    if contrast > 0:
        c = random.uniform(max(0, 1 - contrast), 1 + contrast)
        transforms.append(lambda img, c=c: TF.adjust_contrast(img, c))

    if saturation > 0:
        s = random.uniform(max(0, 1 - saturation), 1 + saturation)
        transforms.append(lambda img, s=s: TF.adjust_saturation(img, s))

    if hue > 0:
        h = random.uniform(-hue, hue)
        transforms.append(lambda img, h=h: TF.adjust_hue(img, h))

    # Shuffle order for diversity
    random.shuffle(transforms)
    for t in transforms:
        image = t(image)

    return image
