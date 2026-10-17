"""Training script for FarmWatch AI Custom PyTorch CNN from scratch.

Features:
- Pure random initialization (no pretrained weights).
- Image-level grouped split loading (zero leakage).
- PyTorch Dataset with data augmentation on train split only.
- Validation checkpointing (saves best model by validation accuracy).
- Training curves plotting (loss and accuracy vs epoch).
- Automatic device fallback (Intel Arc XPU or CPU).
- Exports:
  - models/cnn_from_scratch.pth
  - models/class_to_idx.json
  - models/preprocess_config.json
  - reports/training_curves.png
"""

import os
import time
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.models.custom_cnn import CustomCropCNN
from src.utils.device import get_torch_device, get_device_info
from src.utils.dataset_paths import resolve_image_path, get_safe_path_str


class PlantDocDataset(Dataset):
    """Dataset loader reading from pre-split JSON manifests with robust path resolution."""

    def __init__(self, manifest_path: Path, class_to_idx: Dict[str, int], transform=None):
        self.manifest_path = Path(manifest_path)
        self.class_to_idx = class_to_idx
        self.transform = transform

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        resolved_path = resolve_image_path(item)
        safe_path = get_safe_path_str(resolved_path)

        with Image.open(safe_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[item["class_name"]]
        return image, label_idx, item["image_id"]


def get_transforms(input_size: Tuple[int, int] = (128, 128)) -> Tuple[transforms.Compose, transforms.Compose]:
    """Define data transformations with augmentation for training only."""
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, eval_transform


def train_model(
    project_root: Path,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    input_size: Tuple[int, int] = (128, 128),
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete CNN training loop from scratch with validation and checkpointing."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_torch_device()
    device_info = get_device_info()
    print(f"Starting training on device: {device} ({device_info.get('device_name', '')})")

    manifests_dir = project_root / "data" / "manifests"
    train_manifest = manifests_dir / "train_manifest.json"
    val_manifest = manifests_dir / "val_manifest.json"
    class_to_idx_path = project_root / "models" / "class_to_idx.json"

    if not class_to_idx_path.exists():
        # Fallback to data/splits if not in models
        class_to_idx_path = project_root / "data" / "splits" / "class_to_idx.json"

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_tf, eval_tf = get_transforms(input_size)
    train_dataset = PlantDocDataset(train_manifest, class_to_idx, transform=train_tf)
    val_dataset = PlantDocDataset(val_manifest, class_to_idx, transform=eval_tf)

    # Use num_workers=0 for safe Windows dataloading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Initialize model from scratch
    model = CustomCropCNN(num_classes=num_classes, in_channels=3, dropout_rate=0.3)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CustomCropCNN initialized from random weights. Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "cnn_from_scratch.pth"

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["epoch_time"].append(epoch_duration)

        scheduler.step(epoch_val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
            f"Duration: {epoch_duration:.1f}s"
        )

        # Save best model checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "input_size": list(input_size),
                "architecture": "CustomCropCNN",
                "device": str(device),
                "device_name": device_info.get("device_name", ""),
                "pytorch_version": torch.__version__,
                "random_seed": seed,
            }
            torch.save(checkpoint_data, checkpoint_path)
            # Also save to artifacts/models/ for full system compatibility
            (project_root / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, project_root / "artifacts" / "models" / "best_custom_cnn.pt")
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Acc: {best_val_acc * 100:.2f}%) to {checkpoint_path}")

    total_training_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_training_time:.1f}s. Best Epoch: {best_epoch} with Val Acc: {best_val_acc * 100:.2f}%")

    # Save preprocessing configuration
    preprocess_config = {
        "input_resolution": list(input_size),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "channel_order": "RGB",
        "interpolation": "BILINEAR",
        "num_classes": num_classes,
        "classes": [idx_to_class[i] for i in range(num_classes)],
    }
    with open(models_dir / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, indent=2)
    with open(project_root / "artifacts" / "models" / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, indent=2)

    # Save class mapping
    with open(models_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(project_root / "artifacts" / "models" / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker="o", color="#2d6a4f")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker="s", color="#d90429")
    plt.title("Custom CNN: Loss vs Epoch (Scratch Training)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Accuracy", marker="o", color="#2d6a4f")
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Accuracy", marker="s", color="#1b4332")
    plt.title("Custom CNN: Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curves_path = reports_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {curves_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_time_seconds": total_training_time,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    train_model(root, epochs=8, batch_size=16, learning_rate=1e-3)
