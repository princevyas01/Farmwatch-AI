"""Custom PyTorch Convolutional Neural Network for Leaf Condition Classification.

Trained from random initialization (scratch training baseline).
Architecture:
- 3 Convolutional Blocks (Conv2d, BatchNorm2d, ReLU, MaxPool2d)
- Adaptive Global Average Pooling
- Dropout regularizer
- Fully connected linear classification head
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class CustomCropCNN(nn.Module):
    """Custom compact CNN architecture for field leaf crop condition classification."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: Input (B, 3, H, W) -> (B, 32, H/2, W/2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        # Note: block3[0] is the primary target for Grad-CAM activation visualization
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling: (B, 128, H/8, W/8) -> (B, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Explicitly initialize weights using Kaiming Normal for ReLU stability
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial convolutional feature maps up to the last conv layer."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the final convolutional layer targeted for Grad-CAM."""
        return self.block3[0]

    def get_model_metadata(self) -> Dict[str, Any]:
        """Return architectural specification dictionary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_architecture": "CustomCropCNN",
            "num_blocks": 3,
            "channels": [32, 64, 128],
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "initialized_from": "random_initialization (kaiming_normal)",
        }


class EnhancedCropCNN(nn.Module):
    """Enhanced deep convolutional architecture for fine-grained leaf pathology classification.

    Features:
    - 4 progressive convolutional blocks with double-conv feature extraction.
    - Batch normalization & LeakyReLU / ReLU activations for stable gradient propagation.
    - Dropout regularization (spatial & dense) to combat overfitting.
    - Multi-scale receptive field suitable for localized fungal spots and broad blights.
    - Kaiming Normal weight initialization from scratch.
    """

    def __init__(self, num_classes: int = 11, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 128x128 -> 64x64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: 64x64 -> 32x32
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: 32x32 -> 16x16
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 4: 16x16 -> 8x8 (Target layer for Grad-CAM is block4[3])
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the final convolutional layer targeted for Grad-CAM."""
        return self.block4[3]

    def get_model_metadata(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_architecture": "EnhancedCropCNN",
            "num_blocks": 4,
            "channels": [32, 64, 128, 256],
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "initialized_from": "random_initialization (kaiming_normal)",
        }
