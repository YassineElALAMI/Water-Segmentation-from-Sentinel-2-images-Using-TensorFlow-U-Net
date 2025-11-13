# src/visualization.py
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from typing import Tuple

def create_mask_visualization(binary_mask: np.ndarray, water_color: Tuple[int,int,int]=(0,120,255)) -> np.ndarray:
    """
    Convert binary mask (H,W) -> RGB uint8 visualization (H,W,3).
    water_color is RGB tuple.
    """
    h, w = binary_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask_bool = binary_mask == 1
    vis[mask_bool] = water_color
    return vis


def create_overlay(original_image: Image.Image, binary_mask: np.ndarray, alpha: float=0.4,
                   water_color: Tuple[int,int,int]=(0,120,255)) -> np.ndarray:
    """
    Blend mask on top of the original PIL image.
    - Resizes binary_mask to original image size using NEAREST neighbor before blending.
    Returns overlay as uint8 RGB ndarray.
    """
    orig_arr = np.array(original_image).astype("float32")
    # Resize mask to original
    mask_img = Image.fromarray((binary_mask * 255).astype('uint8'))
    mask_resized = mask_img.resize((orig_arr.shape[1], orig_arr.shape[0]), Image.NEAREST)
    mask_arr = np.array(mask_resized)
    # convert to binary 0/1
    mask_bin = (mask_arr > 127).astype(np.uint8)

    # build color layer
    color_layer = np.zeros_like(orig_arr, dtype=np.float32)
    color_layer[..., 0] = water_color[0]
    color_layer[..., 1] = water_color[1]
    color_layer[..., 2] = water_color[2]

    mask_3c = np.stack([mask_bin]*3, axis=-1).astype(np.float32)

    blended = ((1 - mask_3c * alpha) * orig_arr + (mask_3c * alpha) * color_layer)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def create_probability_heatmap(prob_map: np.ndarray, colormap: str="jet") -> np.ndarray:
    """
    Convert probability map (H,W) with values 0..1 into RGB heatmap uint8.
    """
    prob = np.clip(prob_map, 0.0, 1.0)
    cmap = cm.get_cmap(colormap)
    colored = cmap(prob)[:, :, :3]  # float in 0..1
    colored = (colored * 255).astype(np.uint8)
    return colored


def blend_heatmap_with_image(original_image: Image.Image, heatmap_rgb: np.ndarray, alpha: float=0.5) -> np.ndarray:
    """
    Resize heatmap to original image and blend with original (alpha).
    Returns uint8 RGB ndarray.
    """
    orig_arr = np.array(original_image).astype(np.float32)
    heat_pil = Image.fromarray(heatmap_rgb)
    heat_resized = heat_pil.resize((orig_arr.shape[1], orig_arr.shape[0]), Image.BILINEAR)
    heat_arr = np.array(heat_resized).astype(np.float32)
    blended = (alpha * heat_arr + (1 - alpha) * orig_arr).astype(np.uint8)
    return blended


def save_mask(mask_vis: np.ndarray, path: str):
    Image.fromarray(mask_vis.astype(np.uint8)).save(path)


def save_overlay(overlay_arr: np.ndarray, path: str):
    Image.fromarray(overlay_arr.astype(np.uint8)).save(path)


def save_heatmap(heatmap_arr: np.ndarray, path: str):
    Image.fromarray(heatmap_arr.astype(np.uint8)).save(path)
