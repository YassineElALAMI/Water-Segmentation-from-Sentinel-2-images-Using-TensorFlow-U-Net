import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import io

# Import from project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_image, preprocess_for_model
from src.inference import load_model, predict_mask, calculate_water_coverage
from src.visualization import (
    create_mask_visualization,
    create_overlay,
    create_probability_heatmap,
    blend_heatmap_with_image,
)

# -------------------------------------------------------------
# Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="GeoAI Water Segmentation",
    layout="wide",
    page_icon="🌊",
)

st.markdown("<h1 style='text-align:center;'>🌊 GeoAI Water Body Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>U-Net Water Segmentation for Sentinel-2 (RGB)</p>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Load model
# -------------------------------------------------------------
@st.cache_resource
def load_unet():
    model_path = Path(__file__).parent.parent / "models" / "unet_water_best.keras"
    return load_model(model_path)

model = load_unet()
st.success("Model Loaded Successfully!")

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.4, 0.05)

uploaded_file = st.sidebar.file_uploader("Upload Image (RGB)", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------------
# Main Content
# -------------------------------------------------------------
if uploaded_file:

    # Load + show original image
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # Preprocess for model
    preprocessed = preprocess_for_model(image)

    # Predict
    binary_mask, prob_map = predict_mask(preprocessed, threshold, model)

    # Stats
    stats = calculate_water_coverage(binary_mask)
    st.sidebar.markdown(f"### Water: **{stats['water_percentage']:.2f}%**")
    st.sidebar.markdown(f"### Land: **{stats['land_percentage']:.2f}%**")

    # Visualizations
    mask_vis = create_mask_visualization(binary_mask)
    overlay = create_overlay(image, binary_mask, alpha=alpha)
    heatmap = create_probability_heatmap(prob_map)
    heat_blend = blend_heatmap_with_image(image, heatmap, alpha=alpha)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎭 Mask", "🌈 Overlay", "🔥 Heatmap", "🔥+📷 Heatmap Blend"]
    )

    # Mask Tab
    with tab1:
        st.image(mask_vis, caption="Binary Mask")

        # Download mask
        mask_bytes = io.BytesIO()
        Image.fromarray(mask_vis).save(mask_bytes, format="PNG")
        st.download_button("Download Mask", mask_bytes.getvalue(), "mask.png")

    # Overlay Tab
    with tab2:
        st.image(overlay, caption="Overlay")

        overlay_bytes = io.BytesIO()
        Image.fromarray(overlay).save(overlay_bytes, format="PNG")
        st.download_button("Download Overlay", overlay_bytes.getvalue(), "overlay.png")

    # Heatmap
    with tab3:
        st.image(heatmap, caption="Probability Heatmap")

        heat_bytes = io.BytesIO()
        Image.fromarray(heatmap).save(heat_bytes, format="PNG")
        st.download_button("Download Heatmap", heat_bytes.getvalue(), "heatmap.png")

    # Heatmap Blend
    with tab4:
        st.image(heat_blend, caption="Heatmap + Original Blend")

        heatblend_bytes = io.BytesIO()
        Image.fromarray(heat_blend).save(heatblend_bytes, format="PNG")
        st.download_button("Download Blended Heatmap", heatblend_bytes.getvalue(), "heatmap_blend.png")

else:
    st.info("⬅️ Upload an RGB image to begin.")

