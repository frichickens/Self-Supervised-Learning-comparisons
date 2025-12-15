import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model.mobilenet_JEPA import MobileNet_JEPA

def create_random_masks(batch_size, num_patches, context_ratio=0.7, device='cuda'):
    """
    Create random masks for context and target patches
    
    Args:
        batch_size: number of images in batch
        num_patches: total number of patches (256 for 16x16 grid)
        context_ratio: ratio of patches to keep as context (0.7 = 70%)
        device: torch device
    
    Returns:
        context_mask: indices of context patches [B, N_context]
        target_mask: indices of target patches [B, N_target]
    """
    num_context = int(num_patches * context_ratio)
    num_target = int(num_patches * (1 - context_ratio))
    
    # Create different random masks for each image in batch
    context_mask = torch.stack([
        torch.randperm(num_patches)[:num_context] 
        for _ in range(batch_size)
    ]).to(device)
    
    target_mask = torch.stack([
        torch.randperm(num_patches)[:num_target] 
        for _ in range(batch_size)
    ]).to(device)
    
    return context_mask, target_mask


def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args):
    """Train for one epoch"""
    model.train()
    # Set target encoder to eval mode (it should not be trained)
    model.target_encoder.eval()
    
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        
        # Create random masks for this batch
        context_mask, target_mask = create_random_masks(
            batch_size, 
            args.num_patches, 
            args.context_ratio,
            device
        )
        
        # Forward pass through target encoder (no grad)
        with torch.no_grad():
            target_features = model.target_encoder(images, masks=[target_mask])
            # Normalize target features over feature dimension (as in original I-JEPA)
            target_features = F.layer_norm(target_features, (target_features.size(-1),))
        
        # Forward pass through context encoder and predictor
        context_features = model.context_encoder(images, masks=[context_mask])
        predictions = model.predictor(context_features, masks_x=[context_mask], masks=[target_mask])
        
        # Compute loss (smooth L1 loss as in I-JEPA paper)
        loss = F.smooth_l1_loss(predictions, target_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        
        # Step scheduler (per iteration)
        if scheduler is not None:
            scheduler.step()
        
        # Update target encoder with EMA (as in original I-JEPA)
        with torch.no_grad():
            for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(args.ema_momentum).add_(param_q.data, alpha=1.0 - args.ema_momentum)
        
        # Track loss
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar with VRAM info
        mem_mb = torch.cuda.max_memory_allocated() / 1024.**2 if torch.cuda.is_available() else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'mem_MB': f'{mem_mb:.0f}'
        })
    
    return total_loss / num_batches


def validate(model, val_loader, device, args):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            
            # Create random masks
            context_mask, target_mask = create_random_masks(
                batch_size, 
                args.num_patches, 
                args.context_ratio,
                device
            )
            
            # Forward pass through target encoder
            target_features = model.target_encoder(images, masks=[target_mask])
            target_features = F.layer_norm(target_features, (target_features.size(-1),))
            
            # Forward pass through context encoder and predictor
            context_features = model.context_encoder(images, masks=[context_mask])
            predictions = model.predictor(context_features, masks_x=[context_mask], masks=[target_mask])
            
            # Compute loss
            loss = F.smooth_l1_loss(predictions, target_features)
            total_loss += loss.item()
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, args, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': args
    }
    
    save_path = os.path.join(args.save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved to {save_path}')


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data augmentation for I-JEPA (minimal, since we're doing masking)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Batch size: {args.batch_size}')
    print(f'Train batches: {len(train_loader)}')
    
    # Create model
    model = MobileNet_JEPA(
        c_in=3,
        embed_dim=args.embed_dim,
        predictor_embed_dim=args.predictor_embed_dim,
        num_patches=args.num_patches,
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        momentum=args.ema_momentum
    ).to(device)
    
    # Freeze target encoder parameters (as in original I-JEPA)
    for p in model.target_encoder.parameters():
        p.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel created:')
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')
    print(f'Target encoder parameters (frozen): {sum(p.numel() for p in model.target_encoder.parameters()) / 1e6:.2f}M')
    
    # Create optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup (per iteration, not per epoch)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup * len(train_loader)
    
    if args.use_scheduler:
        # Warmup + Cosine annealing scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('loss', float('inf'))
            print(f'Resumed from epoch {start_epoch}')
        else:
            print(f'No checkpoint found at {args.resume}')
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args)
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            val_loss = validate(model, val_loader, device, args)
            print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, args, 'best_model.pth')
                print(f'New best model saved! Val Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}')
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, args, f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs-1, train_loss, args, 'final_model.pth')
    print('\nTraining completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train MobileNet-JEPA on CIFAR-10')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data', 
                        help='Path to CIFAR-10 data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension for encoder')
    parser.add_argument('--predictor_embed_dim', type=int, default=256,
                        help='Embedding dimension for predictor')
    parser.add_argument('--predictor_depth', type=int, default=6,
                        help='Number of transformer blocks in predictor')
    parser.add_argument('--predictor_num_heads', type=int, default=8,
                        help='Number of attention heads in predictor')
    parser.add_argument('--num_patches', type=int, default=256,
                        help='Number of patches (16x16 for 32x32 images)')
    
    # Masking arguments
    parser.add_argument('--context_ratio', type=float, default=0.7,
                        help='Ratio of patches to use as context (0.7 = 70%)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--ema_momentum', type=float, default=0.996,
                        help='EMA momentum for target encoder')
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help='Use cosine annealing scheduler with warmup')
    
    # Checkpointing arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints_jepa_pretrain',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Print configuration
    print('='*60)
    print('I-JEPA Pre-training Configuration')
    print('='*60)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('='*60 + '\n')
    
    main(args)
