# medical_vision/data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Union, Tuple
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ChestXrayDataset(Dataset):
    """
    Dataset class for chest X-ray images supporting both classification and segmentation tasks.
    
    Args:
        task (str): Either 'classification' or 'segmentation'
        split (str): One of 'train', 'val', or 'test'
        transform (Optional[transforms.Compose]): Custom transforms for data augmentation
        data_dir (Optional[str]): Root directory containing both datasets
    """
    
    CONDITIONS = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    def __init__(
        self,
        task: str = 'classification',
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        data_dir: Optional[str] = None
    ):
        self.task = task
        self.split = split
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / 'data'
        self.transform = transform if transform else self._get_default_transforms()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if task not in ['classification', 'segmentation']:
            raise ValueError(f"Invalid task: {task}. Must be 'classification' or 'segmentation'")
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
        self.data_info = self._setup_dataset()
        logger.info(f"Initialized {task} dataset for {split} split with {len(self.data_info)} samples")
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transformations for images"""
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        
        if self.task != 'segmentation':
            base_transforms.append(transforms.Normalize(mean=[0.485], std=[0.229]))
            
        return transforms.Compose(base_transforms)
    
    def _setup_dataset(self) -> List[Tuple]:
        """Set up dataset based on task type"""
        if self.task == 'classification':
            return self._setup_chestx14_classification()
        return self._setup_montgomery_segmentation()

    def _process_chestx14_data(self, df: pd.DataFrame, base_path: Path) -> List[Tuple[str, np.ndarray]]:
        """Process ChestX-ray14 dataset entries"""
        processed_data = []
        images_dir = base_path / 'images'
        
        for _, row in df.iterrows():
            img_path = images_dir / row['Image Index']
            if img_path.exists():
                processed_data.append((str(img_path), row['Labels']))
                
        return processed_data
    
    def _process_montgomery_data(
        self,
        data_pairs: List[Tuple[str, str]],
        base_path: Path
    ) -> List[Tuple[str, str]]:
        """Process Montgomery dataset entries"""
        processed_data = []
        images_dir = base_path / 'CXR_png'
        masks_dir = base_path / 'masks'
        
        for img_file, mask_file in data_pairs:
            img_path = images_dir / img_file
            mask_path = masks_dir / mask_file
            if img_path.exists() and mask_path.exists():
                processed_data.append((str(img_path), str(mask_path)))
                
        return processed_data
    
    def _setup_chestx14_classification(self) -> List[Tuple[str, np.ndarray]]:
        """Set up ChestX-ray14 classification dataset"""
        base_path = self.data_dir / 'ChestX-ray14'
        data_entry = pd.read_csv(base_path / 'Data_Entry_2017.csv')
        
        data_entry['Labels'] = data_entry['Finding Labels'].apply(self._convert_to_multilabel)
        
        train_val_df, test_df = train_test_split(data_entry, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
        
        split_map = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        return self._process_chestx14_data(split_map[self.split], base_path)
    
    def _setup_montgomery_segmentation(self) -> List[Tuple[str, str]]:
        """Set up Montgomery segmentation dataset"""
        base_path = self.data_dir / 'Montgomery'
        images = sorted((base_path / 'CXR_png').glob('*.png'))
        masks = sorted((base_path / 'masks').glob('*.png'))
        
        data_pairs = list(zip(
            [img.name for img in images],
            [mask.name for mask in masks]
        ))
        
        train_val_pairs, test_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
        train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.2, random_state=42)
        
        split_map = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        return self._process_montgomery_data(split_map[self.split], base_path)

    def _convert_to_multilabel(self, finding_labels: str) -> np.ndarray:
        """Convert text labels to multi-hot encoded array"""
        labels = np.zeros(len(self.CONDITIONS))
        for finding in finding_labels.split('|'):
            if finding in self.CONDITIONS:
                labels[self.CONDITIONS.index(finding)] = 1
        return labels
    
    def _get_classification_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a classification data item"""
        img_path, labels = self.data_info[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'target': torch.FloatTensor(labels)
        }

    def _get_segmentation_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a segmentation data item"""
        img_path, mask_path = self.data_info[idx]
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return {
            'image': image,
            'target': mask
        }
    
    def __len__(self) -> int:
        return len(self.data_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data item"""
        if self.task == 'classification':
            return self._get_classification_item(idx)
        return self._get_segmentation_item(idx)

def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Create data loaders for both classification and segmentation tasks.
    
    Args:
        data_dir: Root directory containing both datasets
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory in GPU
        
    Returns:
        Dictionary containing dataloaders for each task and split
    """
    tasks = ['classification', 'segmentation']
    splits = ['train', 'val', 'test']
    
    dataloaders = {}
    
    for task in tasks:
        dataloaders[task] = {}
        for split in splits:
            dataset = ChestXrayDataset(
                task=task,
                split=split,
                data_dir=data_dir
            )
            
            dataloaders[task][split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            logger.info(f"Created {task} dataloader for {split} split with "
                       f"{len(dataset)} samples")
    
    return dataloaders

