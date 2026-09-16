import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseDetector(nn.Module, ABC):


    def __init__(self, config=None):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, images: torch.Tensor, category_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        raise NotImplementedError

    @abstractmethod
    def get_losses(self, pred_dict: Dict[str, torch.Tensor], labels: torch.Tensor,
                   expert_domain_labels: torch.Tensor, criterion: nn.Module) -> Dict[str, torch.Tensor]:

        raise NotImplementedError

    def set_training_mode(self, mode: str, **kwargs):

        pass

    def load_pretrained(self, checkpoint_path: str):

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            self.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
