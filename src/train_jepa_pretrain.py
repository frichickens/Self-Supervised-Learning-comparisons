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
from utils.jepa_utils import JEPALoss
from model.resnet_JEPA import ResnetJEPA

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)
opt = option.dict_to_nonedict(opt)


def train():
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ResnetJEPA(opt)
    model.setup_mask_generator(opt['input_size'])
    model = model.to(device)
    
    # Create loss function
    criterion = JEPALoss(loss_type='smooth_l1', normalize=True)
    
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
            
            # Compute loss only on target (masked) regions
            loss = criterion(predictions, targets, target_mask)
            
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
            val_loss = validate(model, valid_loader, device, criterion)
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
def validate(model, dataloader, device, criterion):
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
        
        # Compute loss
        loss = criterion(predictions, targets, target_mask)
        
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
