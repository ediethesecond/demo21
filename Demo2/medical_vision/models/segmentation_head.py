import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .decoder import DecoderBlock

class SegmentationHead(nn.Module):
    def __init__(self, feature_dims: List[int]):
        super().__init__()
        
        self.decoder_blocks = nn.ModuleList()
        rev_features = list(reversed(feature_dims))
        
        for i in range(len(rev_features) - 1):
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=rev_features[i],
                    skip_channels=rev_features[i + 1] if i < len(rev_features) - 1 else 0,
                    out_channels=rev_features[i + 1]
                )
            )
        
        self.final_upscale = nn.Sequential(
            nn.ConvTranspose2d(rev_features[-1], 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]
        x = features[0]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[i + 1] if i < len(features) - 1 else None
            x = decoder_block(x, skip)
        
        x = self.final_upscale(x)
        
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        
        return x