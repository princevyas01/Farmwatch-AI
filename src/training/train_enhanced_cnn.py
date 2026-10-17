"""Enhanced training pipeline for FarmWatch AI on the focused well-represented dataset.

Improvements:
- Enhanced 4-block deep CNN (EnhancedCropCNN).
- Class-weighted CrossEntropyLoss with 0.05 label smoothing.
- Cosine Annealing learning rate schedule.
- Moderate, realistic augmentation (random flips, rotation, color jitter).
- Checkpoints best model by validation accuracy to models/enhanced_cnn.pth.
- Generates training curves in reports/focused_training_curves.png.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from src.models.custom_cnn import EnhancedCropCNN
from src.training.train_cnn import PlantDocDataset
from src.utils.device import get_torch_device, get_device_info


def get_enhanced_transforms(input_size: Tuple[int, int] = (128, 128)):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, eval_transform


def compute_class_weights(dataset: PlantDocDataset, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = [0] * num_classes
    for item in dataset.items:
        idx = dataset.class_to_idx[item["class_name"]]
        counts[idx] += 1
    total = len(dataset)
    weights = [total / (num_classes * max(1, c)) for c in counts]
    # Normalize weights
    norm_weights = [w / sum(weights) * num_classes for w in weights]
    return torch.tensor(norm_weights, dtype=torch.float32, device=device)


def train_enhanced_model(
    project_root: Path,
    epochs: int = 12,
    batch_size: int = 16,
    initial_lr: float = 1.5e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = get_torch_device()
    device_info = get_device_info()
    print(f"Starting Enhanced Training on: {device} ({device_info.get('device_name', '')})")

    manifests_dir = project_root / "data" / "manifests"
    train_manifest = manifests_dir / "focused_train_manifest.json"
    val_manifest = manifests_dir / "focused_val_manifest.json"
    class_to_idx_path = project_root / "models" / "focused_class_to_idx.json"

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_tf, eval_tf = get_enhanced_transforms((128, 128))
    train_dataset = PlantDocDataset(train_manifest, class_to_idx, transform=train_tf)
    val_dataset = PlantDocDataset(val_manifest, class_to_idx, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples across {num_classes} classes.")

    model = EnhancedCropCNN(num_classes=num_classes, in_channels=3, dropout_rate=0.3)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EnhancedCropCNN initialized from random weights. Trainable parameters: {total_params:,}")

    class_weights = compute_class_weights(train_dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = project_root / "models" / "enhanced_cnn.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
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
        ep_dur = time.time() - ep_start

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["epoch_time"].append(ep_dur)

        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | Dur: {ep_dur:.1f}s"
        )

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
                "input_size": [128, 128],
                "architecture": "EnhancedCropCNN",
                "device": str(device),
                "device_name": device_info.get("device_name", ""),
                "pytorch_version": torch.__version__,
                "random_seed": seed,
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  --> Saved new best enhanced checkpoint at epoch {epoch} (Val Acc: {best_val_acc * 100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nEnhanced training finished in {total_time:.1f}s. Best Epoch: {best_epoch} with Val Acc: {best_val_acc * 100:.2f}%")

    # Save preprocessing config
    preprocess_config = {
        "model_architecture": "EnhancedCropCNN",
        "input_resolution": [128, 128],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "channel_order": "RGB",
        "interpolation": "BILINEAR",
        "num_classes": num_classes,
        "classes": [idx_to_class[i] for i in range(num_classes)],
    }
    with open(project_root / "models" / "focused_preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, indent=2)

    # Plot curves
    reports_dir = project_root / "reports"
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker="o", color="#2d6a4f")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker="s", color="#d90429")
    plt.title("Enhanced CNN: Loss vs Epoch (11 Well-Represented Classes)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy + Smoothing)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Accuracy", marker="o", color="#2d6a4f")
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Accuracy", marker="s", color="#1b4332")
    plt.title("Enhanced CNN: Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curves_path = reports_dir / "focused_training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved enhanced training curves to {curves_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_time_seconds": total_time,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }


if __name__ == "__main__":
    train_enhanced_model(PROJECT_ROOT, epochs=12, batch_size=16, initial_lr=1.5e-3)
