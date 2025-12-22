import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """Loss function for I-JEPA (Joint-Embedding Predictive Architecture)
    
    Computes smooth L1 loss between predictions and targets only on masked regions.
    """
    def __init__(self, loss_type='smooth_l1', normalize=True):
        """
        Args:
            loss_type: Type of loss ('smooth_l1', 'mse', 'cosine')
            normalize: Whether to normalize features before computing loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
    
    def forward(self, predictions, targets, target_mask):
        """
        Args:
            predictions: Predicted features (B, C, H, W)
            targets: Target features from momentum encoder (B, C, H, W)
            target_mask: Binary mask indicating target regions (B, 1, H, W)
        
        Returns:
            loss: Scalar loss value
        """
        # Normalize features if requested
        if self.normalize:
            predictions = F.normalize(predictions, dim=1)
            targets = F.normalize(targets, dim=1)
        
        # Compute loss based on type
        if self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'cosine':
            # Cosine similarity loss (1 - cosine_similarity)
            loss = 1 - F.cosine_similarity(predictions, targets, dim=1, keepdim=True)
            loss = loss.unsqueeze(1)  # Add channel dim back
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Sum across channels
        loss = loss.sum(dim=1, keepdim=True)
        
        # Apply mask and compute mean over masked regions
        loss = loss.mul_(target_mask).sum() / (target_mask.sum() + 1e-8)
        
        return loss


class JEPAProjectionHead(nn.Module):
    """Projection head for I-JEPA
    
    Projects feature maps to a lower-dimensional space.
    """
    def __init__(self, in_channels, out_channels, use_bn=True):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.projection(x)


class JEPAPredictor(nn.Module):
    """Predictor network for I-JEPA
    
    Predicts target representations from context representations with mask tokens.
    """
    def __init__(self, feature_dim, n_layers=2, hidden_dim=None):
        """
        Args:
            feature_dim: Dimension of input/output features
            n_layers: Number of predictor layers
            hidden_dim: Hidden dimension (defaults to feature_dim)
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim
        
        layers = []
        for i in range(n_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            out_dim = feature_dim if i == n_layers - 1 else hidden_dim
            
            layers.extend([
                nn.Conv2d(in_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ])
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.predictor(x)


def load_jepa_weights(model, checkpoint_path, strict=True):
    """Load I-JEPA pretrained weights into model
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce that keys match
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'backbone' in checkpoint:
        state_dict = checkpoint['backbone']
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print(f"Loaded I-JEPA weights from {checkpoint_path}")
    return model


def load_jepa_backbone_only(model, checkpoint_path):
    """Load only the backbone weights from I-JEPA checkpoint
    
    Useful for transfer learning where you only want the encoder.
    
    Args:
        model: Model to load backbone weights into
        checkpoint_path: Path to checkpoint file
    
    Returns:
        model: Model with loaded backbone weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Filter only backbone weights
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            # Remove 'backbone.' prefix
            new_key = k.replace('backbone.', '')
            backbone_state_dict[new_key] = v
    
    # Load into model backbone
    if hasattr(model, 'backbone'):
        model.backbone.load_state_dict(backbone_state_dict, strict=False)
        print(f"Loaded I-JEPA backbone weights from {checkpoint_path}")
    else:
        print("Warning: Model does not have 'backbone' attribute")
    
    return model


@torch.no_grad()
def momentum_update(model, model_momentum, momentum):
    """Update momentum encoder using exponential moving average (EMA)
    
    Args:
        model: Online/student model
        model_momentum: Momentum/teacher model
        momentum: Momentum coefficient (typically 0.996-0.999)
    """
    for param, param_m in zip(model.parameters(), model_momentum.parameters()):
        param_m.data = param_m.data * momentum + param.data * (1.0 - momentum)


def get_momentum_schedule(base_momentum, final_momentum, epochs):
    """Generate momentum schedule for training
    
    Args:
        base_momentum: Starting momentum value
        final_momentum: Final momentum value
        epochs: Total number of epochs
    
    Returns:
        momentum_schedule: List of momentum values per epoch
    """
    momentum_schedule = []
    for epoch in range(epochs):
        m = base_momentum + (final_momentum - base_momentum) * (epoch / epochs)
        momentum_schedule.append(m)
    return momentum_schedule
