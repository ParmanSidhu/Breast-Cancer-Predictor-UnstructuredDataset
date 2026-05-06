# Breast Cancer Prediction with Structured and Unstructured Data

This project compares a classic structured-data breast cancer baseline with an
unstructured image-based deep learning pipeline.

The Wisconsin Breast Cancer dataset is included only as a baseline because it is
small, clean, and already engineered into numeric features. The main model uses
histopathology image patches from the Breast Histopathology Images / IDC dataset
and evaluates with patient-wise splits to reduce data leakage.

## Project Goals

- Build a transparent Wisconsin tabular baseline.
- Train an image classifier on unstructured histopathology images.
- Avoid inflated accuracy by splitting image patches by patient ID.
- Report accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and class balance.
- Provide a small Streamlit app for image prediction.

## Dataset Choice

Main unstructured dataset:

- Breast Histopathology Images / IDC dataset
- Image patches: 50 x 50 histopathology crops
- Labels: IDC negative and IDC positive
- Why this dataset: it uses raw image patches instead of engineered tabular features.

Baseline structured dataset:

- Wisconsin Breast Cancer dataset from scikit-learn
- Used only for comparison, not as the main contribution.

## Repository Layout

```text
.
├── app/
│   └── streamlit_app.py
├── docs/
│   ├── dataset_strategy.md
│   └── report_alignment.md
├── scripts/
│   ├── prepare_idc_dataset.py
│   ├── train_image_model.py
│   ├── train_wisconsin_baseline.py
│   └── predict_image.py
├── src/
│   └── bc_predictor/
│       ├── config.py
│       ├── data.py
│       ├── image_model.py
│       ├── metrics.py
│       └── tabular_model.py
└── tests/
    └── test_data.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

Download the IDC dataset from Kaggle:

```text
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
```

Extract it under:

```text
data/raw/breast-histopathology-images
```

Then create a patient-wise train/validation/test split:

```bash
python scripts/prepare_idc_dataset.py \
  --raw-dir data/raw/breast-histopathology-images \
  --output-dir data/processed/idc
```

The processed dataset will be organized as:

```text
data/processed/idc/
├── train/
│   ├── negative/
│   └── positive/
├── val/
│   ├── negative/
│   └── positive/
├── test/
│   ├── negative/
│   └── positive/
└── manifest.csv
```

## Run the Wisconsin Baseline

```bash
python scripts/train_wisconsin_baseline.py
```

## Train the Image Model

```bash
python scripts/train_image_model.py \
  --data-dir data/processed/idc \
  --epochs 10 \
  --batch-size 64
```

Model outputs are saved to:

```text
artifacts/
```

## Predict One Image

```bash
python scripts/predict_image.py \
  --checkpoint artifacts/image_model.pt \
  --image path/to/patch.png
```

## Launch the App

```bash
streamlit run app/streamlit_app.py
```

## Important Note

This project is for academic and research demonstration only. It is not a
medical device and must not be used for clinical diagnosis.
