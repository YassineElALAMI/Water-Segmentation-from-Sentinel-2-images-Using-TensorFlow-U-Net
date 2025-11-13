# src/inference.py
import numpy as np
from typing import Tuple
from tensorflow.keras.models import load_model as keras_load_model

def load_model(model_path):
    """
    Load a saved Keras model (.keras).
    """
    model = keras_load_model(str(model_path), compile=False)
    return model


def predict_mask(preprocessed_img: np.ndarray, threshold: float, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on preprocessed image and return:
      - binary_mask: uint8 array shape (H,W) values {0,1}
      - prob_map: float32 array shape (H,W) values [0,1]
    preprocessed_img must be shape (1,H,W,3).
    """
    preds = model.predict(preprocessed_img)  # (1,H,W,1) or (1,H,W)
    pred = preds[0]
    # ensure 2D
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[:,:,0]
    prob_map = np.clip(pred.astype("float32"), 0.0, 1.0)
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    return binary_mask, prob_map


def calculate_water_coverage(binary_mask: np.ndarray) -> dict:
    """
    Calculate water / land percentage from binary mask (0/1).
    """
    total = binary_mask.size
    water = int(binary_mask.sum())
    water_pct = (water / total) * 100.0
    return {"water_percentage": water_pct, "land_percentage": 100.0 - water_pct}
