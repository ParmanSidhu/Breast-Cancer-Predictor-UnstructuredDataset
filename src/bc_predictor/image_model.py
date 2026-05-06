from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
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
    max_samples_per_split: int | None = None,
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
        if max_samples_per_split is not None and len(dataset) > max_samples_per_split:
            dataset = balanced_subset(dataset, max_samples_per_split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )
    return loaders


def balanced_subset(dataset: datasets.ImageFolder, max_samples: int) -> Subset:
    """Keep a small class-balanced subset for quick local demo training."""
    targets = np.asarray(dataset.targets)
    per_class = max(1, max_samples // len(dataset.classes))
    indices = []
    rng = np.random.default_rng(42)
    for class_index in range(len(dataset.classes)):
        class_indices = np.where(targets == class_index)[0]
        selected = rng.choice(
            class_indices,
            size=min(per_class, len(class_indices)),
            replace=False,
        )
        indices.extend(selected.tolist())
    rng.shuffle(indices)
    return Subset(dataset, indices)


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    loaded_pretrained = pretrained
    try:
        model = models.mobilenet_v2(weights=weights)
    except Exception as exc:
        print(f"Could not load pretrained weights, using random initialization: {exc}")
        model = models.mobilenet_v2(weights=None)
        loaded_pretrained = False
    if loaded_pretrained:
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
    model = build_model(
        num_classes=len(checkpoint.get("class_names", CLASS_NAMES)),
        pretrained=False,
    )
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
