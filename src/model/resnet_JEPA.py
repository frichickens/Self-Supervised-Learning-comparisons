import torch.nn as nn
import torch
import sys
from pathlib import Path
from typing import List
from timm.models.resnet import ResNet
from timm.models import create_model as timm_create_model

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pretrain.ijepa_mask import MultiBlockMask

def get_downsample_ratio(self: ResNet):
    return 32

def get_feature_map_channels(self: ResNet) -> List[int]:
    # `self.feature_info` is maintained by `timm`
    return [info['num_chs'] for info in self.feature_info[1:]]


# hack: override the forward function of `timm.models.resnet.ResNet`
def forward(self, x, hierarchical=False):
    """ this forward function is a modified version of `timm.models.resnet.ResNet.forward`
    >>> ResNet.forward
    """
    if hierarchical:
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
    else:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

ResNet.get_downsample_ratio = get_downsample_ratio
ResNet.get_feature_map_channels = get_feature_map_channels
ResNet.forward = forward

@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('resnet50')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()

    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])

    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio

    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch

class ResnetJEPA(nn.Module):
    """I-JEPA with ResNet backbone for self-supervised pretraining"""
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # Create backbone (encoder)
        self.backbone = timm_create_model(
            opt['backbone']['name'],
            pretrained=False,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimensions
        self.downsample_ratio = self.backbone.get_downsample_ratio()
        self.feature_channels = self.backbone.get_feature_map_channels()[-1]
        
        # Momentum encoder (target network)
        self.backbone_momentum = timm_create_model(
            opt['backbone']['name'],
            pretrained=False,
            num_classes=0
        )
        self.backbone_momentum.load_state_dict(self.backbone.state_dict())
        
        # Freeze momentum encoder
        for param in self.backbone_momentum.parameters():
            param.requires_grad = False
        
        # Projection heads (optional)
        if opt.get('use_projection_head', True):
            proj_dim = opt.get('projection_dim', 256)
            self.projection_head = nn.Sequential(
                nn.Conv2d(self.feature_channels, proj_dim, 1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True)
            )
            self.projection_head_momentum = nn.Sequential(
                nn.Conv2d(self.feature_channels, proj_dim, 1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True)
            )
            self.projection_head_momentum.load_state_dict(self.projection_head.state_dict())
            for param in self.projection_head_momentum.parameters():
                param.requires_grad = False
            feature_dim = proj_dim
        else:
            self.projection_head = None
            self.projection_head_momentum = None
            feature_dim = self.feature_channels
        
        # Predictor (decodes masked tokens)
        n_layers = opt.get('predictor_layers', 2)
        predictor_layers = []
        for i in range(n_layers):
            predictor_layers.extend([
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ])
        self.predictor = nn.Sequential(*predictor_layers)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def forward_encoder(self, x, mask=None):
        """Forward through encoder with optional masking"""
        # Get features
        feats = self.backbone(x, hierarchical=True)
        features = feats[-1]  # Take last layer features
        
        if self.projection_head is not None:
            features = self.projection_head(features)
        
        # Apply mask if provided
        if mask is not None:
            # mask: (B, 1, H, W) where 1=visible, 0=masked
            features = features * mask
        
        return features
    
    @torch.no_grad()
    def forward_momentum(self, x):
        """Forward through momentum encoder"""
        feats = self.backbone_momentum(x, hierarchical=True)
        features = feats[-1]
        
        if self.projection_head_momentum is not None:
            features = self.projection_head_momentum(features)
        
        return features
    
    def forward(self, x):
        """Full forward pass with masking"""
        B, C, H, W = x.shape
        
        # Generate masks
        fmap_h = H // self.downsample_ratio
        fmap_w = W // self.downsample_ratio
        
        context_mask, target_mask = self.get_mask(B, fmap_h, fmap_w, x.device)
        
        # Encode with context mask
        features = self.forward_encoder(x, context_mask)
        
        # Fill masked positions with mask tokens
        mask_tokens = self.mask_token.expand_as(features)
        features_with_mask = torch.where(
            context_mask.expand_as(features), 
            features, 
            mask_tokens
        )
        
        # Predict
        predictions = self.predictor(features_with_mask)
        
        # Get target features (no masking, no gradients)
        with torch.no_grad():
            targets = self.forward_momentum(x)
        
        return predictions, targets, context_mask, target_mask
    
    def get_mask(self, B, H, W, device):
        """Generate context and target masks"""
        if not hasattr(self, 'mask_generator'):
            raise RuntimeError("Mask generator not initialized. Call setup_mask_generator first.")
        
        context_mask, target_mask = self.mask_generator(B)
        context_mask = context_mask.unsqueeze(1).to(device, dtype=torch.bool)
        target_mask = target_mask.unsqueeze(1).to(device, dtype=torch.bool)
        
        return context_mask, target_mask
    
    def setup_mask_generator(self, input_size):
        """Setup the mask generator"""
        self.mask_generator = MultiBlockMask(
            input_size=input_size,
            patch_size=self.downsample_ratio,
            **self.opt['mask']
        )
    
    @torch.no_grad()
    def update_momentum_encoder(self, momentum):
        """Update momentum encoder with EMA"""
        for param, param_m in zip(self.backbone.parameters(), 
                                   self.backbone_momentum.parameters()):
            param_m.data = param_m.data * momentum + param.data * (1.0 - momentum)
        
        if self.projection_head is not None:
            for param, param_m in zip(self.projection_head.parameters(),
                                       self.projection_head_momentum.parameters()):
                param_m.data = param_m.data * momentum + param.data * (1.0 - momentum)

if __name__ == '__main__':
    convnet_test()