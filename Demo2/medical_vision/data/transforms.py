import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mode: str, img_size: int = 224, augment: bool = True):
    """
    Get image transforms for different dataset modes
    
    Args:
        mode: 'train', 'val', or 'test'
        img_size: Target image size
        augment: Whether to apply data augmentation
    """
    if mode == 'train' and augment:
        return A.Compose([
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.2),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])

def create_dataloaders(config):
    """Create data loaders for training and validation"""
    datasets = {
        'train': {},
        'val': {}
    }
    
    for task in ['classification', 'segmentation']:
        for mode in ['train', 'val']:
            datasets[mode][task] = MedicalImageDataset(
                data_dir=config.data[f'{mode}_dir'],
                mode=mode,
                task=task,
                img_size=config.data.img_size,
                augment=mode == 'train'
            )
    
    dataloaders = {
        'train': {},
        'val': {}
    }
    
    for mode in ['train', 'val']:
        for task in ['classification', 'segmentation']:
            dataloaders[mode][task] = DataLoader(
                datasets[mode][task],
                batch_size=config.training.batch_size,
                shuffle=mode == 'train',
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                drop_last=mode == 'train'
            )
    
    return dataloaders