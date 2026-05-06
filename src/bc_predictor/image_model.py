from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from bc_predictor.config import CLASS_NAMES, IMAGE_SIZE
from bc_predictor.metrics import binary_classification_metrics


def build_transforms(train: bool) -> transforms.Compose:
    augmentations = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if train:
        augmentations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
            ]
        )
    augmentations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(augmentations)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 2,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = datasets.ImageFolder(
            data_dir / split,
            transform=build_transforms(train=split == "train"),
        )
        if dataset.classes != CLASS_NAMES:
            raise ValueError(
                f"Expected classes {CLASS_NAMES}, found {dataset.classes} in {data_dir / split}"
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )
    return loaders


def build_model(num_classes: int = 2) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_score = []

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        y_score.extend(probabilities.cpu().numpy().tolist())

    metrics = binary_classification_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_score),
    )
    return running_loss / len(loader.dataset), metrics


def save_checkpoint(model: nn.Module, path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "metrics": metrics,
            "image_size": IMAGE_SIZE,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(num_classes=len(checkpoint.get("class_names", CLASS_NAMES)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: nn.Module, image_path: Path, device: torch.device) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor = build_transforms(train=False)(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "positive_probability": float(probabilities[1]),
        "negative_probability": float(probabilities[0]),
    }

