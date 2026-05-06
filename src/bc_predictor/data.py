from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


IDC_PATTERN = re.compile(r"(?P<patient>\d+)_idx\d+_x\d+_y\d+_class(?P<label>[01])")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class IDCSample:
    source_path: Path
    patient_id: str
    label: int

    @property
    def class_name(self) -> str:
        return "positive" if self.label == 1 else "negative"


def parse_idc_filename(path: Path) -> IDCSample | None:
    """Parse patient ID and class label from an IDC patch filename."""
    match = IDC_PATTERN.search(path.name)
    if not match:
        return None
    return IDCSample(
        source_path=path,
        patient_id=match.group("patient"),
        label=int(match.group("label")),
    )


def discover_idc_samples(raw_dir: Path) -> list[IDCSample]:
    samples: list[IDCSample] = []
    for path in raw_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        sample = parse_idc_filename(path)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise ValueError(f"No IDC image patches found under {raw_dir}")
    return samples


def build_patient_wise_split(
    samples: list[IDCSample],
    random_state: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> pd.DataFrame:
    """Create train/val/test splits without sharing patients across splits."""
    if train_size <= 0 or val_size <= 0 or train_size + val_size >= 1:
        raise ValueError("Expected train_size > 0, val_size > 0, and train+val < 1")

    frame = pd.DataFrame(
        {
            "source_path": [str(sample.source_path) for sample in samples],
            "patient_id": [sample.patient_id for sample in samples],
            "label": [sample.label for sample in samples],
            "class_name": [sample.class_name for sample in samples],
        }
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(
        splitter.split(frame, frame["label"], groups=frame["patient_id"])
    )

    frame["split"] = "temp"
    frame.loc[train_idx, "split"] = "train"

    temp = frame.iloc[temp_idx].copy()
    relative_val_size = val_size / (1 - train_size)
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_size,
        random_state=random_state,
    )
    val_local_idx, test_local_idx = next(
        splitter.split(temp, temp["label"], groups=temp["patient_id"])
    )

    frame.loc[temp.iloc[val_local_idx].index, "split"] = "val"
    frame.loc[temp.iloc[test_local_idx].index, "split"] = "test"
    return frame


def materialize_image_folder(manifest: pd.DataFrame, output_dir: Path) -> None:
    """Copy files into ImageFolder-compatible split/class directories."""
    for split in ["train", "val", "test"]:
        for class_name in ["negative", "positive"]:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for row in manifest.itertuples(index=False):
        source = Path(row.source_path)
        destination = output_dir / row.split / row.class_name / source.name
        if not destination.exists():
            shutil.copy2(source, destination)

    manifest.to_csv(output_dir / "manifest.csv", index=False)


def validate_patient_isolation(manifest: pd.DataFrame) -> None:
    split_patients = {
        split: set(manifest.loc[manifest["split"] == split, "patient_id"])
        for split in ["train", "val", "test"]
    }
    overlaps = {
        ("train", "val"): split_patients["train"] & split_patients["val"],
        ("train", "test"): split_patients["train"] & split_patients["test"],
        ("val", "test"): split_patients["val"] & split_patients["test"],
    }
    leaking = {pair: ids for pair, ids in overlaps.items() if ids}
    if leaking:
        raise ValueError(f"Patient leakage detected across splits: {leaking}")

