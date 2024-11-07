# medical_vision/training/__init__.py
from .trainer import ModelTrainer
from .metrics import ClassificationMetrics, SegmentationMetrics

class TrainingConfig:
    """Training configuration with default values"""
    def __init__(self, **kwargs):
        # Basic training parameters
        self.batch_size = kwargs.get('batch_size', 8)
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        
        # Device and memory settings
        self.mixed_precision = kwargs.get('mixed_precision', True)
        self.num_workers = kwargs.get('num_workers', 4)
        self.pin_memory = kwargs.get('pin_memory', True)
        self.gradient_clip_val = kwargs.get('gradient_clip_val', 1.0)
        
        # Learning rate scheduling
        self.scheduler_patience = kwargs.get('scheduler_patience', 5)
        self.scheduler_factor = kwargs.get('scheduler_factor', 0.5)
        self.min_lr = kwargs.get('min_lr', 1e-6)
        
        # Early stopping
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 1e-4)
        
        # Validation settings
        self.val_check_interval = kwargs.get('val_check_interval', 1.0)
        self.val_check_min_steps = kwargs.get('val_check_min_steps', 500)
        
        # Task weights for multi-task learning
        self.task_weights = kwargs.get('task_weights', {
            'classification': 1.0,
            'segmentation': 1.0
        })
        
        # Checkpointing
        self.save_top_k = kwargs.get('save_top_k', 3)
        self.checkpoint_metric = kwargs.get('checkpoint_metric', 'val_loss')
        self.save_last = kwargs.get('save_last', True)

class EarlyStoppingCallback:
    """Early stopping implementation"""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class ModelCheckpoint:
    """Model checkpoint handler"""
    def __init__(self, save_dir: str, save_top_k: int = 3, mode: str = 'min'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_metrics = []
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        metrics: Dict[str, float]
    ):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save current checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Update best metrics
        self.best_metrics.append((metric, checkpoint_path))
        self.best_metrics.sort(key=lambda x: x[0], reverse=self.mode == 'max')
        
        # Remove older checkpoints if exceeding save_top_k
        while len(self.best_metrics) > self.save_top_k:
            _, path = self.best_metrics.pop()
            if path.exists():
                path.unlink()

__all__ = [
    'ModelTrainer',
    'ClassificationMetrics',
    'SegmentationMetrics',
    'TrainingConfig',
    'EarlyStoppingCallback',
    'ModelCheckpoint'
]