# Dataset Strategy

The original report uses the Wisconsin Breast Cancer dataset. That dataset is
useful for teaching machine learning, but it is structured and engineered. It
contains numeric cell features rather than raw medical content.

This project keeps Wisconsin as a baseline and adds an unstructured image
dataset as the main experiment.

## Main Dataset

The selected unstructured dataset is the Breast Histopathology Images / IDC
dataset. It contains histopathology image patches labeled as IDC negative or IDC
positive.

## Why Not Only Wisconsin?

High accuracy on Wisconsin can happen because:

- The dataset is small and clean.
- Features are already engineered.
- It is heavily used in examples and tutorials.
- The task is not equivalent to raw image diagnosis.

## Leakage Control

The IDC dataset contains many image patches per patient. A random image-level
split can place patches from the same patient in both training and testing. That
can inflate performance because the model sees highly related tissue patterns
during training.

This project uses patient-wise splitting:

- All patches from one patient stay in one split.
- Train, validation, and test patients do not overlap.
- Metrics are more realistic than a naive random patch split.

## Recommended Reporting

Report both results, but explain that they answer different questions:

- Wisconsin baseline: performance on structured engineered features.
- IDC image model: performance on raw unstructured histopathology images.

The image model is the stronger project contribution because it demonstrates
deep learning on unstructured data.

