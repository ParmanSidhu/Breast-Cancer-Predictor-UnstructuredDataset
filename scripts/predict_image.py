from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bc_predictor.image_model import load_checkpoint, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict IDC status from an image patch.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    print(predict_image(model, args.image, device))


if __name__ == "__main__":
    main()

