from pathlib import Path

import pandas as pd

from bc_predictor.data import IDCSample, build_patient_wise_split, parse_idc_filename


def test_parse_idc_filename():
    path = Path("8863_idx5_x451_y1451_class0.png")
    sample = parse_idc_filename(path)
    assert sample is not None
    assert sample.patient_id == "8863"
    assert sample.label == 0
    assert sample.class_name == "negative"


def test_patient_wise_split_has_no_patient_overlap():
    samples = [
        IDCSample(Path(f"{patient}_idx5_x1_y1_class{label}.png"), str(patient), label)
        for patient in range(1000, 1040)
        for label in [0, 1]
    ]
    manifest = build_patient_wise_split(samples)
    split_patients = {
        split: set(manifest.loc[manifest["split"] == split, "patient_id"])
        for split in ["train", "val", "test"]
    }
    assert not split_patients["train"] & split_patients["val"]
    assert not split_patients["train"] & split_patients["test"]
    assert not split_patients["val"] & split_patients["test"]
    assert isinstance(manifest, pd.DataFrame)

