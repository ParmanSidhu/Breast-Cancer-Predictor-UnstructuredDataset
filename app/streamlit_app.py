from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from sklearn.datasets import load_breast_cancer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bc_predictor.config import ARTIFACT_DIR
from bc_predictor.image_model import load_checkpoint, predict_image
from bc_predictor.tabular_model import train_wisconsin_baseline


st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("Breast Cancer Predictor")

checkpoint_path = ARTIFACT_DIR / "image_model.pt"
tabular_model_path = ARTIFACT_DIR / "wisconsin_baseline.joblib"

structured_tab, image_tab = st.tabs(["30-Feature Wisconsin Baseline", "Image Prediction"])


@st.cache_resource
def get_tabular_model():
    if tabular_model_path.exists():
        return joblib.load(tabular_model_path)
    model, _ = train_wisconsin_baseline()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, tabular_model_path)
    return model


with structured_tab:
    st.subheader("Wisconsin 30-Feature Prediction")
    dataset = load_breast_cancer()
    defaults = pd.DataFrame(dataset.data, columns=dataset.feature_names).median()

    selected_profile = st.selectbox(
        "Load sample values",
        ["Median patient profile", "Example benign case", "Example malignant case"],
    )
    if selected_profile == "Example benign case":
        default_values = pd.Series(dataset.data[dataset.target == 1][0], index=dataset.feature_names)
    elif selected_profile == "Example malignant case":
        default_values = pd.Series(dataset.data[dataset.target == 0][0], index=dataset.feature_names)
    else:
        default_values = defaults

    values = {}
    for index, feature_name in enumerate(dataset.feature_names):
        column = dataset.data[:, index]
        values[feature_name] = st.slider(
            feature_name,
            min_value=float(column.min()),
            max_value=float(column.max()),
            value=float(default_values[feature_name]),
            step=float((column.max() - column.min()) / 1000),
        )

    tabular_model = get_tabular_model()
    input_frame = pd.DataFrame([values], columns=dataset.feature_names)
    benign_probability = tabular_model.predict_proba(input_frame)[0, 1]
    prediction = "Benign" if benign_probability >= 0.5 else "Malignant"

    st.metric("Prediction", prediction)
    st.metric("Benign probability", f"{benign_probability:.2%}")
    st.caption(
        "This tab uses the structured Wisconsin dataset with 30 numeric features. "
        "It is a baseline, not the main unstructured image model."
    )

with image_tab:
    st.subheader("Histopathology Image Prediction")
    uploaded = st.file_uploader(
        "Upload a histopathology image patch",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

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
