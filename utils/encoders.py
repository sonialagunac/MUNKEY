import torch.nn as nn
from models.models import VIT


class ViTEncoder(nn.Module):
    def __init__(self, models_path, config):
        super().__init__()
        self.model = VIT(config, models_path=models_path, pretrained=True)
        self.hidden_size = self.model.hidden_size

    def forward(self, x):
        return self.model(x)
