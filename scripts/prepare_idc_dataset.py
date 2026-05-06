from __future__ import annotations

import argparse
from pathlib import Path

from bc_predictor.data import (
    build_patient_wise_split,
    discover_idc_samples,
    materialize_image_folder,
    validate_patient_isolation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IDC image dataset.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = discover_idc_samples(args.raw_dir)
    manifest = build_patient_wise_split(samples, random_state=args.random_state)
    validate_patient_isolation(manifest)
    materialize_image_folder(manifest, args.output_dir)

    print(f"Prepared {len(manifest)} images at {args.output_dir}")
    print(manifest.groupby(["split", "class_name"]).size())


if __name__ == "__main__":
    main()

