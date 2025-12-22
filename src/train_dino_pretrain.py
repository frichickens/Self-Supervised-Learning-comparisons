import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np
from dotenv import load_dotenv

import options.options as option
from model.mobilenet_DINO import MobileNet_DINO
from utils.dino_utils import DINOLoss
from model.resnet_DINO import ResNet18_DINO, ResNet34_DINO
from data.dino_data import get_dino_dataloader
from utils.utils import save_checkpoint, load_checkpoint, get_timestamp

load_dotenv()

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, 
                     warmup_epochs=0, start_warmup_value=0):
    """Cosine learning rate schedule with linear warmup"""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train_one_epoch(model, criterion, dataloader, optimizer, scaler, 
                   epoch, device, opt, lr_schedule, wd_schedule, momentum_schedule):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    # Track metrics for collapse detection
    student_stds = []
    teacher_stds = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
    
    for it, crops in enumerate(progress_bar):
        # Update learning rate and weight decay
        step = epoch * len(dataloader) + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[step]
            if i == 0:
                param_group['weight_decay'] = wd_schedule[step]
        
        # Move data to device
        crops = [crop.to(device, non_blocking=True) for crop in crops]
        
        # IMPORTANT: Don't concatenate different sized crops!
        # Process global and local crops separately
        
        # Only 2 global crops (same size) go through teacher
        teacher_output = model.teacher_backbone(torch.cat([crops[0], crops[1]]))
        teacher_output = model.teacher_head(teacher_output)
        
        # Process each crop through student (handles different sizes)
        student_outputs = []
        for crop in crops:
            out = model.student_backbone(crop)
            out = model.student_head(out)
            student_outputs.append(out)
        
        # Concatenate student outputs (same feature dimension)
        student_output = torch.cat(student_outputs)
        
        # Forward pass with mixed precision
        if opt['training']['use_amp']:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                loss = criterion(student_output, teacher_output, epoch)
        else:
            loss = criterion(student_output, teacher_output, epoch)
        
        # Track output statistics for collapse detection
        with torch.no_grad():
            student_stds.append(student_output.std().item())
            teacher_stds.append(teacher_output.std().item())
        
        # Backward pass
        optimizer.zero_grad()
        
        if opt['training']['use_amp']:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.student_backbone.parameters(), 
                                          opt['training']['gradient_clip'])
            torch.nn.utils.clip_grad_norm_(model.student_head.parameters(), 
                                          opt['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.student_backbone.parameters(), 
                                          opt['training']['gradient_clip'])
            torch.nn.utils.clip_grad_norm_(model.student_head.parameters(), 
                                          opt['training']['gradient_clip'])
            optimizer.step()
        
        # EMA update of teacher
        with torch.no_grad():
            m = momentum_schedule[step]
            model.update_teacher(momentum=m)
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Log to wandb
        if opt['logging']['use_wandb'] and it % opt['logging']['log_freq'] == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/wd': optimizer.param_groups[0]['weight_decay'],
                'train/momentum': m,
                'train/student_std': student_output.std().item(),
                'train/teacher_std': teacher_output.std().item(),
                'step': step
            })
    
    # Check for collapse
    avg_student_std = np.mean(student_stds)
    avg_teacher_std = np.mean(teacher_stds)
    
    if avg_student_std < 0.01 or avg_teacher_std < 0.01:
        print(f"\nPotential collapse detected!")
        print(f"Student std: {avg_student_std:.4f}, Teacher std: {avg_teacher_std:.4f}")
    
    return total_loss / len(dataloader), avg_student_std, avg_teacher_std


def main(opt):
    # Create directories
    os.makedirs(opt['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(opt['paths']['log_dir'], exist_ok=True)
    
    # Initialize wandb
    if opt['logging']['use_wandb']:
        # Login with API key if provided
        load_dotenv()
        wandb_key = os.getenv('WANDB_KEY')
        if wandb_key:
            wandb.login(key=wandb_key)
        
        timestamp = get_timestamp()
        wandb.init(
            project=opt['logging']['project_name'],
            config=opt,
            name=f"{opt['model']['name']}_{opt['data']['dataset']}_{timestamp}"
        )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)           
    dataloader = get_dino_dataloader(opt)
    print(f"Dataset: {opt['data']['dataset']}, Batches: {len(dataloader)}")
    
    # Model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    arch = opt['model']['architecture']
    if arch == 'mobilenet':
        model = MobileNet_DINO(
            c_in=opt['model']['backbone']['c_in'],
            out_dim=opt['model']['head']['out_dim'],
            use_bn_in_head=opt['model']['head']['use_bn_in_head'],
            norm_last_layer=opt['model']['head']['norm_last_layer'],
            hidden_dim=opt['model']['head']['hidden_dim'],
            bottleneck_dim=opt['model']['head']['bottleneck_dim'],
            nlayers=opt['model']['head']['nlayers'],
            momentum_teacher=opt['model']['momentum_teacher']
        )
    elif arch == 'resnet18':
        model = ResNet18_DINO(
            out_dim=opt['model']['head']['out_dim'],
            use_bn_in_head=opt['model']['head']['use_bn_in_head'],
            norm_last_layer=opt['model']['head']['norm_last_layer'],
            hidden_dim=opt['model']['head']['hidden_dim'],
            bottleneck_dim=opt['model']['head']['bottleneck_dim'],
            nlayers=opt['model']['head']['nlayers'],
            momentum_teacher=opt['model']['momentum_teacher']
        )
    elif arch == 'resnet34':
        model = ResNet34_DINO(
            out_dim=opt['model']['head']['out_dim'],
            use_bn_in_head=opt['model']['head']['use_bn_in_head'],
            norm_last_layer=opt['model']['head']['norm_last_layer'],
            hidden_dim=opt['model']['head']['hidden_dim'],
            bottleneck_dim=opt['model']['head']['bottleneck_dim'],
            nlayers=opt['model']['head']['nlayers'],
            momentum_teacher=opt['model']['momentum_teacher']
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    model = model.to(device)
    print(f"Model: {opt['model']['name']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss
    ncrops = 2 + opt['training']['n_local_crops']
    criterion = DINOLoss(
        out_dim=opt['model']['head']['out_dim'],
        ncrops=ncrops,
        warmup_teacher_temp=opt['training']['warmup_teacher_temp'],
        teacher_temp=opt['training']['teacher_temp'],
        warmup_teacher_temp_epochs=opt['training']['warmup_teacher_temp_epochs'],
        nepochs=opt['training']['epochs'],
        student_temp=opt['training']['student_temp'],
        center_momentum=opt['training']['center_momentum']
    ).to(device)
    
    # Optimizer
    params_groups = [
        {'params': model.student_backbone.parameters()},
        {'params': model.student_head.parameters()},
    ]
    
    if opt['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(params_groups, lr=0, weight_decay=0)
    elif opt['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params_groups, lr=0, weight_decay=0, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt['training']['optimizer']}")
    
    # Schedules
    print("\n" + "="*60)
    print("Creating learning rate schedules...")
    print("="*60)
    
    lr_schedule = cosine_scheduler(
        base_value=opt['training']['base_lr'] * opt['training']['batch_size'] / 256,
        final_value=1e-6,
        epochs=opt['training']['epochs'],
        niter_per_ep=len(dataloader),
        warmup_epochs=opt['training']['warmup_epochs'],
    )
    
    wd_schedule = cosine_scheduler(
        base_value=opt['training']['weight_decay'],
        final_value=opt['training']['weight_decay'],
        epochs=opt['training']['epochs'],
        niter_per_ep=len(dataloader),
    )
    
    momentum_schedule = cosine_scheduler(
        base_value=opt['training']['momentum_teacher_start'],
        final_value=opt['training']['momentum_teacher_end'],
        epochs=opt['training']['epochs'],
        niter_per_ep=len(dataloader),
    )
    
    print(f"Base LR: {opt['training']['base_lr'] * opt['training']['batch_size'] / 256:.6f}")
    print(f"Final LR: 1e-6")
    print(f"Warmup epochs: {opt['training']['warmup_epochs']}")
    print(f"Weight decay: {opt['training']['weight_decay']}")
    
    # Mixed precision scaler
    scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu') if opt['training']['use_amp'] else None
    
    # Training loop
    print("\n" + "="*60)
    print("Starting DINO pretraining...")
    print("="*60 + "\n")
    
    for epoch in range(opt['training']['epochs']):
        avg_loss, student_std, teacher_std = train_one_epoch(
            model, criterion, dataloader, optimizer, scaler,
            epoch, device, opt, lr_schedule, wd_schedule, momentum_schedule
        )
        
        print(f"\nEpoch {epoch + 1}/{opt['training']['epochs']}:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Student Std: {student_std:.4f}")
        print(f"  Teacher Std: {teacher_std:.4f}")
        
        if opt['logging']['use_wandb']:
            wandb.log({
                'epoch': epoch,
                'epoch/avg_loss': avg_loss,
                'epoch/student_std': student_std,
                'epoch/teacher_std': teacher_std
            })
        
        # Save checkpoint
        if (epoch + 1) % opt['logging']['save_freq'] == 0:
            checkpoint_path = os.path.join(
                opt['paths']['checkpoint_dir'], 
                f"dino_{arch}_epoch{epoch + 1}.pth"
            )
            save_checkpoint(model, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(opt['paths']['checkpoint_dir'], f"dino_{arch}_final.pth")
    save_checkpoint(model, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    if opt['logging']['use_wandb']:
        wandb.finish()
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('-root', type=str, default=None, choices=['.'])
    args = parser.parse_args()
    
    opt = option.parse(args.opt, root=args.root)
    opt = option.dict_to_nonedict(opt)
    
    main(opt)