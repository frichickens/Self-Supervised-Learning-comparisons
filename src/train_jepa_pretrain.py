import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import wandb
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

import options.options as option
from utils.utils import *
from pretrain.ijepa_mask import MultiBlockMask
from model.resnet_JEPA import ResNet
from timm.models import create_model as timm_create_model


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)
opt = option.dict_to_nonedict(opt)


class IJEPAResNet(nn.Module):
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


def train():
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = IJEPAResNet(opt)
    model.setup_mask_generator(opt['input_size'])
    model = model.to(device)
    
    # CIFAR-10 normalization constants
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    
    # STL-10 normalization constants
    STL10_MEAN = (0.44671088457107544, 0.43981093168258667, 0.40664660930633545)
    STL10_STD = (0.26034072041511536, 0.256574809551239, 0.2712670564651489)
    
    # Get dataset root from config
    dataset_root = opt['datasets']['train'].get('dataroot', './datasets')
    input_size = opt['input_size']
    
    # Training transforms with resize
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=input_size//7),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])
    
    # Create datasets using torchvision.datasets.CIFAR10
    # train_set = datasets.CIFAR10(
    #     root=dataset_root,
    #     train=True,
    #     download=False,
    #     transform=train_transform
    # )
    
    # valid_set = datasets.CIFAR10(
    #     root=dataset_root,
    #     train=False,
    #     download=False,
    #     transform=val_transform
    # )
    # Create datasets using torchvision.datasets.STL10
    
    train_set = datasets.STL10(
        root=dataset_root,
        split='unlabeled',
        download=True,
        transform=train_transform
    )
    
    valid_set = datasets.STL10(
        root=dataset_root,
        split='test',
        download=True,
        transform=val_transform
    )
    
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt['datasets']['train']['batch_size'],
        shuffle=True,
        num_workers=opt['datasets']['train'].get('n_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=opt['datasets']['val']['batch_size'],
        shuffle=False,
        num_workers=opt['datasets']['val'].get('n_workers', 4),
        pin_memory=True,
        drop_last=False
    )
    
    # Optimizer and scheduler
    train_hypers = opt['hyperparameters']
    optimizer = create_optimizer(model.parameters(), train_hypers)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        train_hypers['epochs'], 
        train_hypers['eta_min']
    )
    
    # Training loop
    best_loss = float('inf')
    global_step = 0
    momentum = opt.get('momentum_start', 0.996)
    momentum_end = opt.get('momentum_end', 1.0)
    
    get_model_parameters_number(model)
    
    for epoch in range(train_hypers['epochs']):
        model.train()
        total_train_loss = 0
        
        # Update momentum
        m = momentum + (momentum_end - momentum) * (epoch / train_hypers['epochs'])
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            imgs, _ = batch  # We don't need labels for pretraining
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, targets, context_mask, target_mask = model(imgs)
            
            # Normalize features
            predictions = F.normalize(predictions, dim=1)
            targets = F.normalize(targets, dim=1)
            
            # Compute loss only on target (masked) regions
            loss = F.smooth_l1_loss(predictions, targets, reduction='none').sum(dim=1, keepdim=True)
            loss = loss.mul_(target_mask).sum() / (target_mask.sum() + 1e-8)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update momentum encoder
            model.update_momentum_encoder(m)
            
            total_train_loss += loss.item() * batch_size
            global_step += 1
            
            # Log to wandb
            if global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/momentum': m,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'step': global_step
                })
        
        # Epoch metrics
        avg_train_loss = total_train_loss / len(train_set)
        print(f"[Train] Epoch {epoch} | Loss: {avg_train_loss:.4f}")
        wandb.log({'train/epoch_loss': avg_train_loss, 'epoch': epoch})
        
        # Validation
        if epoch % train_hypers.get('validate_epochs_freq', 5) == 0:
            val_loss = validate(model, valid_loader, device)
            print(f"[Val] Epoch {epoch} | Loss: {val_loss:.4f}")
            wandb.log({'val/loss': val_loss, 'epoch': epoch})
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = opt['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                ckpt_name = f"{opt['name']}_best.pth"
                ckpt_path = os.path.join(checkpoint_path, ckpt_name)
                save_checkpoint(model, ckpt_path)
                print(f"Saved best model with loss: {best_loss:.4f}")
        
        lr_scheduler.step()
    
    # Save final model
    checkpoint_path = opt['checkpoint_path']
    ckpt_name = f"{opt['name']}_final.pth"
    ckpt_path = os.path.join(checkpoint_path, ckpt_name)
    save_checkpoint(model, ckpt_path)
    print("Training completed!")


@torch.no_grad()
def validate(model, dataloader, device):
    """Validation function"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        imgs, _ = batch
        imgs = imgs.to(device)
        batch_size = imgs.shape[0]
        
        # Forward pass
        predictions, targets, context_mask, target_mask = model(imgs)
        
        # Normalize features
        predictions = F.normalize(predictions, dim=1)
        targets = F.normalize(targets, dim=1)
        
        # Compute loss
        loss = F.smooth_l1_loss(predictions, targets, reduction='none').sum(dim=1, keepdim=True)
        loss = loss.mul_(target_mask).sum() / (target_mask.sum() + 1e-8)
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    model.train()
    return total_loss / total_samples


if __name__ == '__main__':
    load_dotenv()
    wandb_key = os.getenv('WANDB_KEY')
    timestamp = get_timestamp()
    
    wandb.init(
        project="SSL_Comparison",
        name=f"{opt['name']}-{timestamp}"
    )
    
    wandb.config.update({
        'learning_rate': opt['hyperparameters']['lr'],
        'epochs': opt['hyperparameters']['epochs'],
        'batch_size': opt['datasets']['train']['batch_size'],
        'backbone': opt['backbone']['name'],
        'input_size': opt['input_size'],
        'mask_strategy': opt['mask']
    })
    
    train()
    wandb.finish()
