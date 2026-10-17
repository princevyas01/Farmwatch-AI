"""MobileNetV3-Small Transfer-Learning Experiment for FarmWatch AI.

Controlled Transfer Learning on the PlantDoc 11-class subset.
- Architecture: torchvision.models.mobilenet_v3_small(weights=DEFAULT)
- Pretrained weights: ImageNet-1K (MobileNet_V3_Small_Weights.IMAGENET1K_V1)
- Replacement head: Linear(1024, 11)
- Two-stage training:
    Stage 1: Frozen backbone (epochs 1 to warm_up_epochs), train classifier head only.
    Stage 2: Unfreeze top feature blocks (features[9:]), fine-tune with differential learning rates.
- Selection criteria: Best validation Macro F1 (with val loss as tiebreaker).
- Zero data leakage: Identical splits to scratch-trained baseline (focused manifests).
- Held-out test set is NEVER touched during training or checkpoint selection.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.training.train_cnn import PlantDocDataset
from src.utils.device import get_torch_device, get_device_info


def get_mobilenetv3_transforms(input_size: Tuple[int, int] = (224, 224)):
    """Standard ImageNet preprocessing transforms for MobileNetV3."""
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
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
    norm_weights = [w / sum(weights) * num_classes for w in weights]
    return torch.tensor(norm_weights, dtype=torch.float32, device=device)


def build_mobilenetv3_model(num_classes: int = 11, pretrained: bool = False) -> nn.Module:
    """Build MobileNetV3-Small with optional ImageNet weights and custom classifier head.
    
    For inference where a checkpoint will supply all weights, pass pretrained=False
    to prevent downloading or verifying ImageNet weights unnecessarily.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    # Replace final classification layer (index 3 in classifier Sequential)
    in_features = model.classifier[3].in_features  # 1024
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_mobilenetv3_model(
    project_root: Path = PROJECT_ROOT,
    epochs: int = 14,
    warmup_epochs: int = 4,
    batch_size: int = 16,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_torch_device()
    device_info = get_device_info()
    print(f"==================================================")
    print(f"FarmWatch AI - MobileNetV3-Small Transfer Learning")
    print(f"Device: {device} ({device_info.get('device_name', '')})")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Random Seed: {seed}")
    print(f"Smoke Test Mode: {smoke_test}")
    print(f"==================================================")

    manifests_dir = project_root / "data" / "manifests"
    train_manifest = manifests_dir / "focused_train_manifest.json"
    val_manifest = manifests_dir / "focused_val_manifest.json"
    class_to_idx_path = project_root / "models" / "focused_class_to_idx.json"

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Save dedicated mobilenetv3_class_to_idx.json
    with open(project_root / "models" / "mobilenetv3_class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    train_tf, eval_tf = get_mobilenetv3_transforms((224, 224))
    train_dataset = PlantDocDataset(train_manifest, class_to_idx, transform=train_tf)
    val_dataset = PlantDocDataset(val_manifest, class_to_idx, transform=eval_tf)

    if smoke_test:
        epochs = 1
        warmup_epochs = 1
        # Truncate dataset items for quick smoke test
        train_dataset.items = train_dataset.items[:32]
        val_dataset.items = val_dataset.items[:16]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Classes={num_classes}")

    # Build model with pretrained weights for transfer learning
    model = build_mobilenetv3_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV3-Small loaded with ImageNet weights. Total parameters: {total_params:,}")

    # Save preprocessing config
    preprocess_config = {
        "model_architecture": "MobileNetV3-Small",
        "is_pretrained": True,
        "pretrained_weights": "ImageNet-1K (MobileNet_V3_Small_Weights.IMAGENET1K_V1)",
        "input_resolution": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "channel_order": "RGB",
        "interpolation": "BILINEAR",
        "num_classes": num_classes,
        "classes": class_names,
    }
    with open(project_root / "models" / "mobilenetv3_preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, indent=2)

    class_weights = compute_class_weights(train_dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Stage 1: Freeze all backbone feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Classifier parameters are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_p_stage1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 (Head Warm-up, {warmup_epochs} epochs): Trainable parameters: {trainable_p_stage1:,}")

    optimizer = torch.optim.AdamW(
        [{"params": model.classifier.parameters(), "lr": head_lr}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=warmup_epochs, eta_min=1e-5
    )

    history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rate": [],
        "epoch_time_s": [],
        "selected_as_best": [],
    }

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path = project_root / "models" / "mobilenetv3_small_transfer.pth"

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        # Check if transitioning to Stage 2 (Unfreeze top blocks: features[9:])
        if epoch == warmup_epochs + 1 and not smoke_test:
            print("\n>>> Transitioning to Stage 2: Fine-Tuning Top Feature Blocks (features[9:])...")
            for idx in range(9, len(model.features)):
                for param in model.features[idx].parameters():
                    param.requires_grad = True

            trainable_p_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Stage 2 Trainable parameters: {trainable_p_stage2:,}")

            # Reconfigure optimizer with differential learning rates
            remaining_epochs = epochs - warmup_epochs
            fine_tune_params = [
                {"params": model.features[9:].parameters(), "lr": backbone_lr},
                {"params": model.classifier.parameters(), "lr": head_lr * 0.5},
            ]
            optimizer = torch.optim.AdamW(fine_tune_params, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=1e-6
            )

        current_stage = 1 if epoch <= warmup_epochs else 2

        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
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

            running_train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                val_preds_all.extend(preds.cpu().tolist())
                val_targets_all.extend(labels.cpu().tolist())

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = accuracy_score(val_targets_all, val_preds_all)
        _, _, epoch_val_macro_f1, _ = precision_recall_fscore_support(
            val_targets_all, val_preds_all, average="macro", zero_division=0
        )

        ep_duration = time.time() - ep_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Selection rule: Highest Validation Macro F1 (with val loss as tiebreaker)
        is_best = False
        if (epoch_val_macro_f1 > best_val_macro_f1) or (
            abs(epoch_val_macro_f1 - best_val_macro_f1) < 1e-5 and epoch_val_loss < best_val_loss
        ):
            best_val_macro_f1 = epoch_val_macro_f1
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            is_best = True

            # Save best checkpoint
            checkpoint_data = {
                "epoch": epoch,
                "stage": current_stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "val_macro_f1": epoch_val_macro_f1,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "input_size": [224, 224],
                "architecture": "MobileNetV3-Small",
                "is_pretrained": True,
                "pretrained_source": "ImageNet-1K (MobileNet_V3_Small_Weights.DEFAULT)",
                "device": str(device),
                "device_name": device_info.get("device_name", ""),
                "pytorch_version": torch.__version__,
                "random_seed": seed,
            }
            torch.save(checkpoint_data, checkpoint_path)

        scheduler.step()

        # Log history
        history["epoch"].append(epoch)
        history["stage"].append(current_stage)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["val_macro_f1"].append(round(epoch_val_macro_f1, 4))
        history["learning_rate"].append(round(current_lr, 7))
        history["epoch_time_s"].append(round(ep_duration, 2))
        history["selected_as_best"].append(is_best)

        best_mark = " [BEST CANDIDATE SAVED]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] (S{current_stage}) "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
            f"Val Macro F1: {epoch_val_macro_f1 * 100:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {ep_duration:.1f}s{best_mark}"
        )

    total_duration = time.time() - total_start_time
    print(f"\n==================================================")
    print(f"Training completed in {total_duration:.1f}s ({total_duration/60:.2f} mins)")
    print(f"Best Candidate Checkpoint: Epoch {best_epoch} with Val Macro F1: {best_val_macro_f1 * 100:.2f}% (Saved to {checkpoint_path})")
    print(f"==================================================")

    # Save CSV history
    csv_path = project_root / "reports" / "mobilenetv3_training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(history.keys()))
        for i in range(len(history["epoch"])):
            writer.writerow([history[k][i] for k in history.keys()])
    print(f"Saved CSV training history to: {csv_path}")

    # Save validation metrics JSON
    val_metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "total_epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "total_training_time_seconds": round(total_duration, 2),
        "device": str(device),
        "device_name": device_info.get("device_name", ""),
        "checkpoint_path": str(checkpoint_path.relative_to(project_root)),
        "history": history,
    }
    with open(project_root / "reports" / "mobilenetv3_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Saved validation metrics JSON to: reports/mobilenetv3_validation_metrics.json")

    # Plot training and validation curves
    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", color="#1d3557")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s", color="#e63946")
    plt.axvline(x=warmup_epochs + 0.5, color="gray", linestyle=":", label="Stage 2 Unfreeze")
    plt.title("Loss vs. Epoch (CrossEntropy + Smoothing)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o", color="#2a9d8f")
    plt.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Val Acc", marker="s", color="#e76f51")
    plt.axvline(x=warmup_epochs + 0.5, color="gray", linestyle=":")
    plt.title("Accuracy vs. Epoch (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Macro F1 plot
    plt.subplot(1, 3, 3)
    plt.plot(history["epoch"], [f * 100 for f in history["val_macro_f1"]], label="Val Macro F1", marker="^", color="#457b9d")
    plt.scatter([best_epoch], [best_val_macro_f1 * 100], color="#d62828", s=100, zorder=5, label=f"Best Ep {best_epoch}")
    plt.axvline(x=warmup_epochs + 0.5, color="gray", linestyle=":")
    plt.title("Validation Macro F1 vs. Epoch (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1 (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curves_path = project_root / "reports" / "mobilenetv3_training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved MobileNetV3 training curves to: {curves_path}")

    return val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small Transfer Learning Model")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test with mini batch")
    parser.add_argument("--epochs", type=int, default=14, help="Total epochs")
    parser.add_argument("--warmup", type=int, default=4, help="Stage 1 warm-up epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_mobilenetv3_model(
        epochs=args.epochs,
        warmup_epochs=args.warmup,
        batch_size=args.batch_size,
        smoke_test=args.smoke,
    )
