# 🌊 GeoAI Water Body Detection

A clean, production-ready semantic segmentation toolkit for detecting water bodies in Sentinel‑2 satellite imagery using a U‑Net model. Includes a simple desktop GUI to test images, plus modular Python utilities for preprocessing, inference, and visualization.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9+-orange.svg)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-ff69b4.svg)
![Pillow](https://img.shields.io/badge/Pillow-10.0%2B-informational.svg)

## 🎯 Overview

This project provides a complete evaluation setup for a U‑Net model trained to detect water surfaces in satellite imagery:

- Modular Python codebase: preprocessing (`src/utils.py`), inference (`src/inference.py`), visualization (`src/visualization.py`)
- Desktop GUI (Tkinter) for quick, interactive testing
- Polished visual outputs: binary masks and color overlays
- Clear project structure and straightforward setup

## 📁 Project Structure

```
Water-Segmentation-from-Sentinel-2-Using-TensorFlow-U-Net/
│
├── data/
│   ├── sample_inputs/          # Put test images here (jpg/png)
│   └── outputs/                # Generated outputs (masks, overlays)
│
├── models/
│   └── unet_water_best.keras   # Trained model (place here; ignored by git)
│
├── src/
│   ├── utils.py                # Image I/O + preprocessing
│   ├── inference.py            # Model loading + prediction + stats
│   └── visualization.py        # Mask/overlay visualization + saving
│
├── gui/
│   └── app_tkinter.py          # Desktop GUI entry point
│
├── notebooks/
│   └── tenser_code.ipynb       # Exploration/experiments (optional)
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Place the model file

Copy your trained Keras model to the `models/` folder (default filename expected by the app is `unet_water_best.keras`):

```
models/
└── unet_water_best.keras
```

> Note: `*.keras` files are ignored by git (see `.gitignore`).

### 3) Run the GUI

```powershell
python gui/app_tkinter.py
```

The interface lets you upload an image, adjust detection threshold and overlay transparency, and save the results.

## 💻 Usage

### GUI (Recommended)

1. Run `python gui/app_tkinter.py`
2. Click “Upload Image” (supports JPG/PNG)
3. Adjust the detection threshold (0.0–1.0) and overlay transparency
4. View tabs for Original, Mask, and Overlay
5. Save the generated mask/overlay

### Python API

Use the building blocks directly in your own scripts:

```python
from pathlib import Path
from PIL import Image
from src.utils import load_image, preprocess_for_model
from src.inference import load_model, predict_mask, calculate_water_coverage
from src.visualization import create_overlay, create_mask_visualization

# Load model
model_path = Path("models") / "unet_water_best.keras"
model = load_model(model_path)

# Load + preprocess
img = load_image("data/sample_inputs/example.jpg")
inp = preprocess_for_model(img, target_size=(256, 256))

# Predict
mask = predict_mask(inp, threshold=0.5, model=model)
stats = calculate_water_coverage(mask)
print(stats)

# Visualize
overlay_rgb = create_overlay(img, mask, alpha=0.4)
mask_rgb = create_mask_visualization(mask)
```

## 🧩 Modules

- `src/utils.py`
	- `load_image(path)` – Load JPG/PNG as PIL (RGB)
	- `preprocess_for_model(pil_image, target_size)` – Resize + normalize to `(1,H,W,3)`

- `src/inference.py`
	- `load_model(model_path)` – Load the Keras model
	- `predict_mask(image_array, threshold, model)` – Predict binary mask
	- `calculate_water_coverage(mask)` – Compute water/land percentages

- `src/visualization.py`
	- `create_mask_visualization(mask)` – Colorized RGB mask
	- `create_overlay(original_image, mask, alpha)` – Blend mask on original
	- `save_mask(mask_rgb, path)` / `save_overlay(img_rgb, path)` – Save results

## 🎨 Features

- ✅ U‑Net inference on 256×256 RGB images
- ✅ Adjustable detection threshold and overlay alpha
- ✅ Ready‑to‑use GUI for quick testing
- ✅ Clean visual outputs (mask + overlay)
- ✅ Lightweight, modular codebase

## 📊 Model Info

- Architecture: U‑Net (TensorFlow/Keras)
- Task: Semantic Segmentation (water/non‑water)
- Input: 256×256 RGB
- Output: Binary mask (`0/1`)

## 🔧 Troubleshooting

- Missing model file: ensure `models/unet_water_best.keras` exists
- Import errors: run `pip install -r requirements.txt`
- TensorFlow CPU notices: informational logs about available CPU instructions are expected

## 🗂️ Git Tips (large files)

Keras models (`*.keras`) are ignored by default. If you accidentally added one before the rule, untrack it and recommit:

```powershell
git rm --cached models/unet_water_best.keras
git commit -m "Stop tracking large model file"
git push
```

## 🤝 Contributing

Issues and PRs are welcome. Please keep changes focused and documented.

## 📄 License

If you plan to publish, consider adding a LICENSE file (e.g., MIT) and badge.


