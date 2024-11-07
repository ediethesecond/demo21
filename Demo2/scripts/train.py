import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_vision.models.integrated_vision import IntegratedVisionModel
import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace
from medical_vision.training.trainer import ModelTrainer
from medical_vision.data.dataset import create_dataloaders
from medical_vision.utils.logger import setup_logger
import torch
import gc

def dict_to_namespace(d):
    """Recursively converts a dictionary to a SimpleNamespace"""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)

def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return dict_to_namespace(config)

def main():
    parser = argparse.ArgumentParser(description='Train medical vision model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    logger = setup_logger(__name__)
    
    # Setup CUDA and clear memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cudnn.benchmark = True
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    # Create model
    model = IntegratedVisionModel(config)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        config=config  # Now passing the full config object
    )
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.training.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        
        # Train both tasks
        train_metrics_cls = trainer.train_epoch('classification')
        train_metrics_seg = trainer.train_epoch('segmentation')
        
        # Evaluate both tasks
        val_metrics_cls = trainer.evaluate('classification')
        val_metrics_seg = trainer.evaluate('segmentation')
        
        # Log metrics
        logger.info("\nClassification Metrics:")
        logger.info(f"Train - Loss: {train_metrics_cls['loss']:.4f}, AUC: {train_metrics_cls['auc']:.4f}")
        logger.info(f"Val - Loss: {val_metrics_cls['loss']:.4f}, AUC: {val_metrics_cls['auc']:.4f}")
        
        logger.info("\nSegmentation Metrics:")
        logger.info(f"Train - Loss: {train_metrics_seg['loss']:.4f}, IoU: {train_metrics_seg['iou']:.4f}")
        logger.info(f"Val - Loss: {val_metrics_seg['loss']:.4f}, IoU: {val_metrics_seg['iou']:.4f}")
        
        # Update learning rate based on validation loss
        trainer.scheduler.step(val_metrics_cls['loss'] + val_metrics_seg['loss'])
        
        # Log best metrics
        logger.info("\nBest Metrics:")
        logger.info(f"Classification AUC: {trainer.best_metrics['classification']['auc']:.4f}")
        logger.info(f"Segmentation IoU: {trainer.best_metrics['segmentation']['iou']:.4f}")

if __name__ == '__main__':
    main()