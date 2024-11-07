"""
medical_vision: A package for medical image analysis combining classification and segmentation.
"""
from pathlib import Path
import logging

# Version of the medical_vision package
__version__ = "0.1.0"  # Double underscores, not asterisks

# Establish the package base directory as a global variable
PACKAGE_ROOT = Path(__file__).parent.absolute()

# Import main components
try:
    from medical_vision.models import IntegratedVisionModel
    from medical_vision.training import ModelTrainer
    from medical_vision.data import create_dataloaders
except ImportError as e:
    logging.warning(f"Some imports failed: {e}")

# Configure logging with a null handler by default
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Create a convenient function to get the version
def get_version():
    return __version__  # Double underscores, not asterisks

# Export commonly used classes and functions
__all__ = [  # Double underscores, not asterisks
    'IntegratedVisionModel',
    'ModelTrainer',
    'create_dataloaders',
    'get_version',
    'PACKAGE_ROOT'
]

# Optional: Add any package-level configurations or initialization here
def setup_package(log_level=logging.INFO):
    """
    Configure the package with custom settings.
    
    Args:
        log_level: The logging level to use for the package
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Optional: Add any package metadata
__author__ = "Your Name"  # Double underscores, not asterisks
__email__ = "your.email@example.com"  # Double underscores, not asterisks
__description__ = "A deep learning package for medical image analysis"  # Double underscores, not asterisks