# configs/__init__.py
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    name: str
    backbone: Dict[str, Any] = field(default_factory=dict)
    classification_head: Dict[str, Any] = field(default_factory=dict)
    segmentation_head: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**config_dict)

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    optimizer: Dict[str, Any] = field(default_factory=dict)
    scheduler: Dict[str, Any] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 8
    num_epochs: int = 50
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**config_dict)

@dataclass
class DataConfig:
    """Data configuration dataclass"""
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"
    img_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        return cls(**config_dict)

@dataclass
class LoggingConfig:
    """Logging configuration dataclass"""
    wandb_project: str = "medical_vision"
    wandb_entity: Optional[str] = None
    log_interval: int = 10
    save_dir: str = "checkpoints"
    experiment_name: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoggingConfig':
        return cls(**config_dict)

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(
            model=ModelConfig.from_dict(config_dict['model']),
            training=TrainingConfig.from_dict(config_dict['training']),
            data=DataConfig.from_dict(config_dict.get('data', {})),
            logging=LoggingConfig.from_dict(config_dict.get('logging', {}))
        )

class ConfigManager:
    """Configuration management class"""
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.model_configs_dir = self.config_dir / "model_configs"
        self.default_config_path = self.config_dir / "default_config.yaml"
        
    def load_config(self, 
                   model_name: str, 
                   override_config: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        """
        Load and merge configurations
        
        Args:
            model_name: Name of the model configuration to load
            override_config: Optional dictionary to override configuration values
            
        Returns:
            Complete experiment configuration
        """
        # Load default configuration
        default_config = self._load_yaml(self.default_config_path)
        
        # Load model-specific configuration
        model_config_path = self.model_configs_dir / f"{model_name}.yaml"
        if not model_config_path.exists():
            raise ValueError(f"Model configuration not found: {model_name}")
        
        model_config = self._load_yaml(model_config_path)
        
        # Merge configurations
        merged_config = self._merge_configs(default_config, model_config)
        
        # Apply overrides if provided
        if override_config:
            merged_config = self._merge_configs(merged_config, override_config)
        
        return ExperimentConfig.from_dict(merged_config)
    
    def list_available_models(self) -> list:
        """List all available model configurations"""
        return [f.stem for f in self.model_configs_dir.glob("*.yaml")]
    
    def save_config(self, config: ExperimentConfig, save_path: Union[str, Path]):
        """Save configuration to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'model': vars(config.model),
            'training': vars(config.training),
            'data': vars(config.data),
            'logging': vars(config.logging)
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @staticmethod
    def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(path) as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _merge_configs(base_config: Dict[str, Any], 
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        merged = deepcopy(base_config)
        
        for key, value in override_config.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merged[key] = ConfigManager._merge_configs(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        
        return merged

def load_experiment_config(
    model_name: str,
    config_dir: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """
    Helper function to load experiment configuration
    
    Args:
        model_name: Name of the model configuration to load
        config_dir: Optional directory containing configurations
        override_config: Optional dictionary to override configuration values
        
    Returns:
        Complete experiment configuration
    """
    config_manager = ConfigManager(config_dir)
    return config_manager.load_config(model_name, override_config)

# Example usage
"""
# Load configuration
config = load_experiment_config(
    model_name='swin_base',
    override_config={
        'training': {
            'batch_size': 16,
            'num_epochs': 100
        }
    }
)

# Access configuration
print(config.model.backbone['name'])
print(config.training.batch_size)
"""

__all__ = [
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'LoggingConfig',
    'ExperimentConfig',
    'ConfigManager',
    'load_experiment_config'
]