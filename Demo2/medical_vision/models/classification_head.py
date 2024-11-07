import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        return self.classifier(x)