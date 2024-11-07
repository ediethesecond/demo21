# scripts/evaluate.py
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
from medical_vision.models import IntegratedVisionModel
from medical_vision.data import create_dataloaders
from medical_vision.training.metrics import ClassificationMetrics, SegmentationMetrics
from medical_vision.utils.logger import setup_logger
from medical_vision.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_segmentation_results
)

def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

class ModelEvaluator:
    def __init__(self, model, dataloaders, config, device):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.config = config
        self.device = device
        self.cls_metrics = ClassificationMetrics()
        self.seg_metrics = SegmentationMetrics()
        self.logger = setup_logger(__name__)

    @torch.no_grad()
    def evaluate(self, split: str = 'test', save_predictions: bool = True):
        self.model.eval()
        results = {
            'classification': {'predictions': [], 'targets': [], 'image_ids': []},
            'segmentation': {'predictions': [], 'targets': [], 'image_ids': [], 'ious': []}
        }

        # Evaluate both tasks
        for task in ['classification', 'segmentation']:
            self.logger.info(f"Evaluating {task} on {split} set...")
            
            for batch in tqdm(self.dataloaders[task][split]):
                batch_results = self.evaluate_batch(batch, task)
                
                # Store results
                for key in results[task].keys():
                    if key in batch_results:
                        results[task][key].extend(batch_results[key])

        # Calculate and save metrics
        metrics = self.calculate_metrics(results)
        
        if save_predictions:
            self.save_results(results, metrics, split)
        
        return metrics, results

    def evaluate_batch(self, batch, task):
        images = batch['image'].to(self.device)
        targets = batch['target'].to(self.device)
        image_ids = batch['image_id'] if 'image_id' in batch else None

        outputs = self.model(images, mode=task)
        predictions = outputs[task]

        batch_results = {
            'predictions': predictions.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'image_ids': image_ids
        }

        if task == 'segmentation':
            # Calculate per-image IoU
            ious = self.calculate_batch_ious(predictions, targets)
            batch_results['ious'] = ious.cpu().numpy()

        return batch_results

    def calculate_metrics(self, results):
        metrics = {}
        
        # Classification metrics
        cls_preds = np.array(results['classification']['predictions'])
        cls_targets = np.array(results['classification']['targets'])
        metrics['classification'] = {
            'auc': self.cls_metrics.compute_auc(cls_preds, cls_targets),
            'per_class_auc': self.cls_metrics.compute_per_class_auc(cls_preds, cls_targets)
        }
        
        # Segmentation metrics
        seg_preds = np.array(results['segmentation']['predictions'])
        seg_targets = np.array(results['segmentation']['targets'])
        metrics['segmentation'] = {
            'mean_iou': np.mean(results['segmentation']['ious']),
            'per_image_iou': results['segmentation']['ious']
        }
        
        return metrics

    def save_results(self, results, metrics, split):
        save_dir = Path(self.config.logging.save_dir) / f"evaluation_{split}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save classification results
        cls_df = pd.DataFrame({
            'image_id': results['classification']['image_ids'],
            'predictions': results['classification']['predictions'].tolist(),
            'targets': results['classification']['targets'].tolist()
        })
        cls_df.to_csv(save_dir / 'classification_results.csv', index=False)

        # Save segmentation results
        seg_df = pd.DataFrame({
            'image_id': results['segmentation']['image_ids'],
            'iou': results['segmentation']['ious']
        })
        seg_df.to_csv(save_dir / 'segmentation_results.csv', index=False)

        # Save metrics summary
        with open(save_dir / 'metrics_summary.yaml', 'w') as f:
            yaml.dump(metrics, f)

        # Generate and save visualizations
        self.save_visualizations(results, save_dir)

    def save_visualizations(self, results, save_dir):
        # Plot ROC curves
        plot_roc_curves(
            results['classification']['predictions'],
            results['classification']['targets'],
            save_path=save_dir / 'roc_curves.png'
        )

        # Plot confusion matrix
        plot_confusion_matrix(
            results['classification']['predictions'],
            results['classification']['targets'],
            save_path=save_dir / 'confusion_matrix.png'
        )

        # Plot sample segmentation results
        plot_segmentation_results(
            results['segmentation']['predictions'][:5],
            results['segmentation']['targets'][:5],
            save_dir=save_dir / 'segmentation_samples'
        )

def main():
    parser = argparse.ArgumentParser(description='Evaluate medical vision model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on')
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(__name__)

    # Load model
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = IntegratedVisionModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model, dataloaders, config, device)
    metrics, results = evaluator.evaluate(split=args.split)

    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"Classification AUC: {metrics['classification']['auc']:.4f}")
    logger.info("Per-class AUC scores:")
    for i, auc in enumerate(metrics['classification']['per_class_auc']):
        logger.info(f"  Class {i}: {auc:.4f}")
    logger.info(f"\nSegmentation Mean IoU: {metrics['segmentation']['mean_iou']:.4f}")

if __name__ == '__main__':
    main()