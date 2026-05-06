# Report Alignment

This repository can be described as an improved version of a breast cancer
prediction project.

## Existing Synopsis Direction

The synopsis mentions breast cancer prediction using a structured dataset such
as Wisconsin. That approach is valid for a baseline model, but it can produce
very high accuracy because the features are already clean and numeric.

## Improved Direction

This project adds an image-based pipeline using histopathology patches. The
model learns from pixels instead of precomputed clinical measurements.

## Suggested Methodology Section

1. Train a baseline classifier on Wisconsin Breast Cancer data.
2. Prepare IDC histopathology image data.
3. Split image data by patient ID to prevent leakage.
4. Train a transfer-learning CNN using MobileNetV2.
5. Evaluate on a held-out patient-wise test set.
6. Compare structured and unstructured performance.

## Suggested Conclusion

The Wisconsin dataset is useful for demonstrating basic classification, but the
image-based model better represents real unstructured medical AI workflows. The
patient-wise split makes evaluation more trustworthy than a random patch-level
split.

