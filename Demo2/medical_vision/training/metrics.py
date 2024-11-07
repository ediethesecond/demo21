import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict

class ClassificationMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.predictions.extend(torch.sigmoid(preds).detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        try:
            auc = roc_auc_score(
                np.array(self.targets),
                np.array(self.predictions),
                average='macro'
            )
        except ValueError:
            auc = 0.0
        return {'auc': auc}

class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.total_iou = 0.0
        self.num_samples = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = (torch.sigmoid(preds) > self.threshold).float()
        intersection = (preds * targets).sum((1, 2, 3))
        union = preds.sum((1, 2, 3)) + targets.sum((1, 2, 3)) - intersection
        
        iou = (intersection / union.clamp(min=1e-6)).mean()
        
        self.total_iou += iou.item() * preds.size(0)
        self.num_samples += preds.size(0)
    
    def compute(self) -> Dict[str, float]:
        return {'iou': self.total_iou / max(self.num_samples, 1)}