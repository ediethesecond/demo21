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

def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

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
        batch_size=8,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = IntegratedVisionModel(
        backbone_name=config.model.backbone_name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        learning_rate=config.training.learning_rate,
        mixed_precision=config.training.mixed_precision,
        batch_size=config.training.batch_size
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