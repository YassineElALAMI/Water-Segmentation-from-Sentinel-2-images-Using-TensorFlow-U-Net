import folium
import numpy as np
from streamlit_folium import st_folium
from PIL import Image
import tempfile
from pathlib import Path

def display_geospatial_image(image_path):
    """
    Shows an image on an interactive folium map.
    Works for RGB images and later for GeoTIFF.
    """

    # Load image
    img = Image.open(image_path)
    img = img.convert("RGB")

    width, height = img.size

    # Create a temp file for the overlay
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")

    # Create base map
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        control_scale=True
    )

    # If later image has geospatial bounds, replace these
    bounds = [[-50, -50], [50, 50]]

    # Add image overlay
    folium.raster_layers.ImageOverlay(
        name="Input Image",
        image=tmp.name,
        bounds=bounds,
        opacity=1,
        interactive=True,
        cross_origin=False,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    return m
