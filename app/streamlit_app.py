from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from bc_predictor.config import ARTIFACT_DIR
from bc_predictor.image_model import load_checkpoint, predict_image


st.set_page_config(page_title="Breast Cancer Image Predictor", layout="centered")
st.title("Breast Cancer Image Predictor")

checkpoint_path = ARTIFACT_DIR / "image_model.pt"

uploaded = st.file_uploader("Upload a histopathology image patch", type=["png", "jpg", "jpeg", "tif", "tiff"])

if not checkpoint_path.exists():
    st.warning("Train the image model first to create artifacts/image_model.pt.")
elif uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = Path(tmp.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint_path, device)
    result = predict_image(model, tmp_path, device)

    st.metric("Prediction", result["predicted_class"].title())
    st.metric("IDC positive probability", f"{result['positive_probability']:.2%}")
    st.caption("For academic demonstration only. Not for medical diagnosis.")

