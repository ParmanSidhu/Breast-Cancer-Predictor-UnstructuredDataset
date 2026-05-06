from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from bc_predictor.config import ARTIFACT_DIR
from bc_predictor.image_model import (
    build_model,
    create_dataloaders,
    evaluate_model,
    get_default_device,
    save_checkpoint,
    train_one_epoch,
)
from bc_predictor.metrics import save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-based IDC classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Use a class-balanced subset for quick demo training.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train without downloading ImageNet weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    print(f"Using device: {device}")
    loaders = create_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.max_samples_per_split,
    )
    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val_f1 = -1.0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_metrics = evaluate_model(model, loaders["val"], criterion, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_metrics = {"val": val_metrics, "epoch": epoch}
            save_checkpoint(model, ARTIFACT_DIR / "image_model.pt", best_metrics)

    checkpoint_model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(ARTIFACT_DIR / "image_model.pt", map_location=device)
    checkpoint_model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = evaluate_model(
        checkpoint_model,
        loaders["test"],
        criterion,
        device,
    )
    best_metrics["test"] = test_metrics
    best_metrics["test_loss"] = test_loss
    save_metrics(best_metrics, ARTIFACT_DIR / "image_metrics.json")
    save_checkpoint(checkpoint_model, ARTIFACT_DIR / "image_model.pt", best_metrics)
    print(best_metrics)


if __name__ == "__main__":
    main()
