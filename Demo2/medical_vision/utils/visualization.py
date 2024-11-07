import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
from typing import List, Union, Optional

def plot_roc_curves(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Union[str, Path]] = None
):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # For each class
    for i in range(y_true.shape[1]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f'Class {i} (AUC = {roc_auc:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None
):
    """Plot confusion matrix for each class"""
    n_classes = y_true.shape[1]
    fig, axes = plt.subplots(
        (n_classes + 3) // 4, 4,
        figsize=(20, 5 * ((n_classes + 3) // 4))
    )
    axes = axes.ravel()
    
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    for i in range(n_classes):
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            ax=axes[i],
            cmap='Blues',
            cbar=False
        )
        axes[i].set_title(f'Class {i}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    # Remove empty subplots
    for i in range(n_classes, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_segmentation_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_dir: Union[str, Path],
    threshold: float = 0.5
):
    """
    Plot segmentation predictions alongside ground truth
    
    Args:
        predictions: Model predictions (N, H, W)
        targets: Ground truth masks (N, H, W)
        save_dir: Directory to save plots
        threshold: Threshold for binary prediction
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(predictions)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot ground truth
        axes[0].imshow(targets[i], cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Plot prediction
        axes[1].imshow(predictions[i], cmap='gray')
        axes[1].set_title('Raw Prediction')
        axes[1].axis('off')
        
        # Plot thresholded prediction
        axes[2].imshow((predictions[i] >= threshold).astype(float), cmap='gray')
        axes[2].set_title(f'Thresholded Prediction (t={threshold})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{i}.png')
        plt.close()