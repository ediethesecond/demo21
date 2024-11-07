import torch
import torch.nn as nn
import timm
from typing import Dict, List, Optional
from .classification_head import ClassificationHead
from .segmentation_head import SegmentationHead

class IntegratedVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.model.num_classes
        
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=config.model.pretrained,
            features_only=True,
            in_chans=1
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, config.data.img_size, config.data.img_size)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        self.classification_head = ClassificationHead(
            in_channels=self.feature_dims[-1],
            num_classes=self.num_classes
        )
        
        self.segmentation_head = SegmentationHead(feature_dims=self.feature_dims)
        
    def forward(self, x: torch.Tensor, mode: str = 'classification') -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        
        if mode == 'classification':
            return {'classification': self.classification_head(features[-1])}
        elif mode == 'segmentation':
            return {'segmentation': self.segmentation_head(features)}
        elif mode == 'all':
            return {
                'classification': self.classification_head(features[-1]),
                'segmentation': self.segmentation_head(features)
            }
        else:
            raise ValueError(f"Invalid mode: {mode}")