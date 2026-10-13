# Vendored (trimmed) from mlx-examples/clip/image_processor.py —
# Copyright © 2023-2024 Apple Inc., MIT License.

import numpy as np
from PIL import Image

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def resize(image: Image.Image, short_size: int) -> Image.Image:
    width, height = image.size
    short = min(width, height)
    long = max(width, height)
    if short == short_size:
        return image
    new_long = int(short_size * long / short)
    new_size = (short_size, new_long) if width <= height else (new_long, short_size)
    return image.resize(new_size, Image.Resampling.BICUBIC)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    original_width, original_height = image.size
    top = (original_height - size) // 2
    left = (original_width - size) // 2
    return image.crop((left, top, left + size, top + size))


def process_image(image: Image.Image, size: int = 224) -> np.ndarray:
    """PIL image -> (size, size, 3) float32 array, CLIP-normalized."""
    image = center_crop(resize(image, size), size)
    arr = np.asarray(image, dtype=np.float32) * (1 / 255.0)
    arr = (arr - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(
        CLIP_STD, dtype=np.float32
    )
    return arr
