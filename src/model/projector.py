import torch
import torch.nn as nn
from model.mobilenet_DINO import MobileNet_DINO
from model.resnet_DINO import ResNet18_DINO, ResNet34_DINO
from utils.utils import load_checkpoint

class Projector(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        out_dim: int = 10,
        name: str = "",          
    ):
        super().__init__()
        
        if name == "mobilenet_dino":
            model = MobileNet_DINO(c_in=3)
            in_dim = 1024
        elif name == "resnet18_dino":
            model = ResNet18_DINO()
            in_dim = 512
        elif name == "resnet34_dino":
            model = ResNet34_DINO()
            in_dim = 512
        
        hidden_dim = 2 * in_dim
        
        model = load_checkpoint(model, pretrained_path)
        self.backbone = model.teacher_backbone  # (B, 3, H, W) -> (B, in_dim)
        
        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        """
        x: tensor of shape (B, 3, H, W)
        Returns: projected features of shape (B, out_dim)
        """
        with torch.no_grad():
            features = self.backbone(x) # (B, in_dim)
        
        out = self.projector(features)  # (B, out_dim)
        return out