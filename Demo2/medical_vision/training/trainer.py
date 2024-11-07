import torch
import torch.nn as nn
from typing import Dict, Any
from tqdm import tqdm

class ModelTrainer:
    def __init__(
        self,
        model,  # Changed to match the format
        dataloaders: Dict[str, Any],
        config  # Changed to accept config instead of device and learning_rate
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.dataloaders = dataloaders

        # Loss functions
        self.classification_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.localization_criterion = nn.SmoothL1Loss().to(self.device)
        self.segmentation_criterion = nn.BCEWithLogitsLoss().to(self.device)

        self.setup_training_components()

    def setup_training_components(self):
        """Setup training components with config values"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.0001
        )

    def train_epoch(self, task: str):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.dataloaders[task]['train']):
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images, mode=task)

            # Get appropriate loss
            if task == 'classification':
                loss = self.classification_criterion(outputs[task], targets)
            elif task == 'localization':
                loss = self.localization_criterion(outputs[task], targets)
            else:  # segmentation
                loss = self.segmentation_criterion(outputs[task], targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return {'loss': running_loss / len(self.dataloaders[task]['train'])}

    def evaluate(self, task: str, split: str = 'val'):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.dataloaders[task][split], desc=f'Evaluating {task}'):
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(images, mode=task)

                if task == 'classification':
                    loss = self.classification_criterion(outputs[task], targets)
                elif task == 'localization':
                    loss = self.localization_criterion(outputs[task], targets)
                else:  # segmentation
                    loss = self.segmentation_criterion(outputs[task], targets)

                running_loss += loss.item()

        return {'loss': running_loss / len(self.dataloaders[task][split])}

    def save_checkpoint(self, metrics: Dict[str, float], epoch: int):
        """Save model checkpoint"""
        save_path = Path(self.config.logging.save_dir) / f"checkpoint_epoch_{epoch}.pt"
        save_path.parent.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)