# src/utils.py
from PIL import Image
import numpy as np
from typing import Tuple

def load_image(path: str) -> Image.Image:
    """
    Load image as PIL RGB Image (no resizing).
    Returns the PIL Image.
    """
    img = Image.open(path).convert("RGB")
    return img


def preprocess_for_model(image: Image.Image, target_size: Tuple[int,int]=(256,256)) -> np.ndarray:
    """
    Preprocess a PIL Image for model inference.
    Returns a numpy array shaped (1, H, W, 3), dtype float32, values in [0,1].
    """
    img_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr
