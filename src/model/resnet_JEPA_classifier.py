import torch
import torch.nn as nn 
import torch.nn.functional as F
from pathlib import Path
from timm.models import create_model
from types import MethodType

class ResNetJEPAClassifier(nn.Module):
    def __init__(self, num_classes=10, backbone_name='resnet18', freeze_backbone=False):
        super(ResNetJEPAClassifier, self).__init__()
        
        # Create backbone WITHOUT classification head (matching JEPA pretraining)
        self.backbone = create_model(
            backbone_name, 
            pretrained=False, 
            num_classes=0,  # No classification head - returns features
            global_pool=''  # No pooling - we'll do it ourselves
        )
        
        # Add custom methods to backbone for hierarchical forward (from resnet_JEPA.py)
        self._add_hierarchical_forward()
        
        # Get feature dimension
        self.num_features = self.backbone.num_features
        
        # Create our own classification head
        self.fc = nn.Linear(self.num_features, num_classes)
        
        self.freeze_backbone = freeze_backbone
    
    def _add_hierarchical_forward(self):
        """Add hierarchical forward method to backbone (from resnet_JEPA.py)"""
        
        def hierarchical_forward(self, x):
            """Extract hierarchical features from ResNet"""
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.maxpool(x)

            ls = []
            x = self.layer1(x); ls.append(x)
            x = self.layer2(x); ls.append(x)
            x = self.layer3(x); ls.append(x)
            x = self.layer4(x); ls.append(x)
            return ls
        
        # Bind the hierarchical_forward method to the backbone instance
        self.backbone.hierarchical_forward = MethodType(hierarchical_forward, self.backbone)
        
    def load_jepa_weights(self, checkpoint_path, strict=False):
        """Load pretrained I-JEPA weights into the backbone
        
        Args:
            checkpoint_path: Path to the JEPA checkpoint (.pth file)
            strict: If True, requires exact key matching
        """
        print(f"Loading JEPA weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract backbone weights from the checkpoint
        # The checkpoint contains: backbone, backbone_momentum, projection_head, predictor, mask_token
        backbone_state_dict = {}
        for key, value in checkpoint.items():
            # Only load backbone weights (not momentum, projection, or predictor)
            if key.startswith('backbone_momentum.') and not key.startswith('backbone.'):
                # Remove 'backbone.' prefix to match our backbone keys
                new_key = key.replace('backbone_momentum.', '')
                backbone_state_dict[new_key] = value
        
        # Load the weights into backbone only
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_state_dict, 
            strict=strict
        )
        
        print(f"✓ Loaded {len(backbone_state_dict)} backbone parameters")
        if missing_keys:
            print(f"⚠ Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
        if unexpected_keys:
            print(f"⚠ Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
        
        # Optionally freeze backbone for linear evaluation
        if self.freeze_backbone:
            self._freeze_backbone()
            print("✓ Backbone frozen - only training classifier head")
        else:
            print("✓ Full model unfrozen - training all parameters")
        
        return self
    
    def _freeze_backbone(self):
        """Freeze all parameters except the final classification layer"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep fc trainable
        for param in self.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("Backbone unfrozen - full model training enabled")

    def forward(self, x):
        """Forward pass with proper feature extraction and pooling
        
        Following the online_classification_benchmark approach:
        1. Extract features from backbone (returns feature maps)
        2. Pool feature maps to get global representation
        3. Apply classification head
        """
        # Get hierarchical features - returns list of feature maps
        features = self.backbone.hierarchical_forward(x)
        
        # Take the last layer's feature maps
        features = features[-1]  # Shape: (B, C, H, W)
        
        # Pool to get global features (like online_classification_benchmark)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1)  # (B, C, 1, 1)
            features = torch.flatten(features, start_dim=1)  # (B, C)
        
        # Apply classifier
        logits = self.fc(features)
        
        return logits

if __name__ == "__main__":
    # Simple test
    model = ResNetJEPAClassifier(num_classes=10, backbone_name='resnet18', freeze_backbone=True)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(logits.shape)  # Should be [4, 10]