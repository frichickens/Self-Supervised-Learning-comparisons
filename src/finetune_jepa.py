"""
Fine-tune I-JEPA pretrained model on CIFAR-10 classification

Usage:
    # Linear evaluation (freeze backbone, train only classifier)
    python src/finetune_jepa.py -opt src/options/finetune_jepa.yml --linear_eval
    
    # Full fine-tuning (train entire model)
    python src/finetune_jepa.py -opt src/options/finetune_jepa.yml
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import wandb
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

import options.options as option
from utils.utils import *
from model.resnet_JEPA_classifier import ResNetJEPAClassifier


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
parser.add_argument('--linear_eval', action='store_true', help='Freeze backbone for linear evaluation')
args = parser.parse_args()

opt = option.parse(args.opt, root=args.root)
opt = option.dict_to_nonedict(opt)


def train():
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ResNetJEPAClassifier(
        num_classes=opt['out_nc'],
        backbone_name=opt['backbone']['name'],
        freeze_backbone=args.linear_eval or opt.get('freeze_backbone', False)
    )
    
    # Load pretrained JEPA weights if provided
    if opt.get('pretrain_path') and opt['pretrain_path'] != "":
        model.load_jepa_weights(opt['pretrain_path'])
    else:
        print("No pretrained weights provided - training from scratch")
    
    model = model.to(device)
    
    # CIFAR-10 normalization constants
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    
    # Get dataset configuration
    dataset_root = opt['datasets']['train'].get('dataroot', './datasets')
    input_size = opt.get('input_size', 224)
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=input_size//7),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # Create datasets
    train_set = datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=False,
        transform=train_transform
    )
    
    valid_set = datasets.CIFAR10(
        root=dataset_root,
        train=False,
        download=False,
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
    
    # Setup training
    train_hypers = opt['hyperparameters']
    loss_func = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model.parameters(), train_hypers)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        train_hypers['epochs'], 
        train_hypers['eta_min']
    )
    
    # Training loop
    best_acc = float('-inf')
    global_step = 0
    
    get_model_parameters_number(model)
    
    for epoch in range(train_hypers['epochs']):
        model.train()
        total_train_loss = 0
        total_preds = []
        total_gts = []
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = imgs.shape[0]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(imgs)
            loss = loss_func(logits, labels)
            
            total_preds.append(logits.detach().cpu())
            total_gts.append(labels.detach().cpu())
            
            total_train_loss += loss.item() * batch_size
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # Log to wandb
            if global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'step': global_step
                })
        
        # Compute training metrics
        total_preds = torch.cat(total_preds, dim=0)
        total_gts = torch.cat(total_gts, dim=0)
        
        train_loss = total_train_loss / len(train_set)
        train_metrics = calculate_metrics(total_preds, total_gts, num_classes=opt['out_nc'])
        
        print(f"[Train] Epoch {epoch} | Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        wandb.log({f"train_epoch/{k}": v for k, v in train_metrics.items()})
        wandb.log({'train_epoch/loss': train_loss, 'epoch': epoch})
        
        # Validation
        if epoch % train_hypers.get('validate_epochs_freq', 1) == 0:
            val_loss, val_metrics = validate(model, valid_loader, loss_func, device, opt['out_nc'])
            print(f"[Val] Epoch {epoch} | Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log({'val/loss': val_loss, 'epoch': epoch})
            
            # Save best model
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                checkpoint_path = opt['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                ckpt_name = f"{opt['name']}_best.pth"
                ckpt_path = os.path.join(checkpoint_path, ckpt_name)
                save_checkpoint(model, ckpt_path)
                print(f"Saved best model with accuracy: {best_acc:.4f}")
        
        lr_scheduler.step()
    
    # Save final model
    checkpoint_path = opt['checkpoint_path']
    ckpt_name = f"{opt['name']}_final.pth"
    ckpt_path = os.path.join(checkpoint_path, ckpt_name)
    save_checkpoint(model, ckpt_path)
    print(f"Training completed! Best accuracy: {best_acc:.4f}")


@torch.no_grad()
def validate(model, dataloader, loss_func, device, num_classes):
    """Validation function"""
    model.eval()
    total_loss = 0
    total_preds = []
    total_gts = []
    
    for batch in tqdm(dataloader, desc="Validating"):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.shape[0]
        
        # Forward pass
        logits = model(imgs)
        loss = loss_func(logits, labels)
        
        total_preds.append(logits.detach().cpu())
        total_gts.append(labels.detach().cpu())
        
        total_loss += loss.item() * batch_size
    
    total_preds = torch.cat(total_preds, dim=0)
    total_gts = torch.cat(total_gts, dim=0)
    
    val_loss = total_loss / len(dataloader.dataset)
    val_metrics = calculate_metrics(total_preds, total_gts, num_classes=num_classes)
    
    model.train()
    return val_loss, val_metrics


if __name__ == '__main__':
    load_dotenv()
    wandb_key = os.getenv('WANDB_KEY')
    timestamp = get_timestamp()
    
    wandb.login(key=wandb_key)
    
    mode = "linear_eval" if args.linear_eval else "finetune"
    wandb.init(
        project="SSL_Comparison",
        name=f"{opt['name']}-{mode}-{timestamp}"
    )
    
    wandb.config.update({
        'learning_rate': opt['hyperparameters']['lr'],
        'epochs': opt['hyperparameters']['epochs'],
        'batch_size': opt['datasets']['train']['batch_size'],
        'backbone': opt['backbone']['name'],
        'input_size': opt.get('input_size', 224),
        'mode': mode,
        'freeze_backbone': args.linear_eval or opt.get('freeze_backbone', False)
    })
    
    train()
    wandb.finish()
