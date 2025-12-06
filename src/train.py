import os
import torch.nn as nn
import torch
from torchvision import transforms, datasets
from model import create_model
import wandb
import datetime
import options.options as option
import argparse

from dotenv import load_dotenv
# from huggingface_hub import login
from utils.utils import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)
opt = option.dict_to_nonedict(opt)
model = create_model(opt)
train_set, train_loader, valid_set, valid_loader, test_set, test_loader = create_ds(opt)


train_hypers = opt['hyperparameters']
checkpoint_path = opt['checkpoint_path']

loss_func = nn.CrossEntropyLoss()
optimizer = create_optimizer(model.parameters(), train_hypers)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_hypers['epochs'], train_hypers['eta_min'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_checkpoint(model, opt['pretrain_path'])
model.to(device)


def train():
    best_acc = float('-inf')
    global_step = 0
    
    for epoch in range(train_hypers['epochs']):
        model.train()
        total_train_loss = 0
        total_preds = []
        total_gts = []
        #Training process
        for batch in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            imgs, gts = batch
            batch_size = imgs.shape[0]
            
            imgs = imgs.to(device)
            gts = gts.to(device)
            
            pred = model(imgs)
            loss = loss_func(pred, gts)
            
            total_preds.append(pred.detach().cpu())
            total_gts.append(gts.detach().cpu())
            
            total_train_loss += loss.item() * batch_size
            
            #Backward and optimize
            loss.backward()
            optimizer.step()
            
            global_step += 1
    
    
                    
            if train_hypers['validate_mode'] == 'step':
                if global_step % (train_hypers['validate_step_freq']) == 0:
                    eval_loss, eval_metrics = validate()
                    print(f"[EVAL] Step {global_step}|{eval_loss}")
                    wandb.log({f"eval_step_{k}": v for k, v in eval_metrics.items()})
                    
                    model.train()
                    
                    if eval_metrics['accuracy'] > best_acc:
                        best_acc = eval_metrics['accuracy']
                        ckpt_name = f"{opt['model']}_{timestamp}_best.pth"
                        ckpt_path = os.path.join(checkpoint_path, ckpt_name)
                        save_checkpoint(model, ckpt_path)
                        
        if train_hypers['validate_mode'] == 'epoch':
            if epoch % int(train_hypers['validate_epochs_freq']) == 0:
                eval_loss, eval_metrics = validate()
                print(f"[EVAL] Epoch {epoch}|{eval_loss}")
                wandb.log({f"eval_epoch_{k}": v for k, v in eval_metrics.items()})
                
                model.train()
                
                if eval_metrics['accuracy'] > best_acc:
                    best_acc = eval_metrics['accuracy']
                    ckpt_name = f"{opt['model']}_{timestamp}_best.pth"
                    ckpt_path = os.path.join(checkpoint_path, ckpt_name)
                    save_checkpoint(model, ckpt_path)
                    
                    
        total_preds = torch.cat(total_preds, dim=0)
        total_gts = torch.cat(total_gts, dim=0)
        
        train_loss = total_train_loss / len(train_set)
        print(f"[Train] Epoch {epoch}|{train_loss}")
        train_metrics = calculate_metrics(total_preds, total_gts, num_classes=10)
        wandb.log({f"train_epoch_{k}": v for k, v in train_metrics.items()})
        
        lr_scheduler.step()
    
def validate():
    model.eval()
    with torch.no_grad():
        total_valid_loss = 0
        total_preds = []
        total_gts = []
        for batch in tqdm(valid_loader, total=len(valid_loader)):  
            imgs, gts = batch
            batch_size = imgs.shape[0]
            
            imgs = imgs.to(device)
            gts = gts.to(device)
            
            pred = model(imgs)
            loss = loss_func(pred, gts)
            
            total_preds.append(pred.detach().cpu())
            total_gts.append(gts.detach().cpu())
            
            total_valid_loss += loss.item() * batch_size
        
        total_preds = torch.cat(total_preds, dim=0)
        total_gts = torch.cat(total_gts, dim=0)
        
        eval_loss = total_valid_loss / len(valid_set)
        eval_metrics = calculate_metrics(total_preds, total_gts, num_classes=10)
    return eval_loss, eval_metrics


def test():
    model.eval()
    
    with torch.no_grad():
        total_test_loss = 0
        total_preds = []
        total_gts = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            imgs, gts = batch
            batch_size = imgs.shape[0]
            
            imgs = imgs.to(device)
            gts = gts.to(device)
            
            pred = model(imgs)
            loss = loss_func(pred, gts)
            total_preds.append(pred.detach().cpu())
            total_gts.append(gts.detach().cpu())
            total_test_loss += loss.item() * batch_size
        
        total_preds = torch.cat(total_preds, dim=0)
        total_gts = torch.cat(total_gts, dim=0)
        test_loss = total_test_loss / len(test_set)
        test_metrics = calculate_metrics(total_preds, total_gts, num_classes=10)
    for k, v in test_metrics.items():
        print(f"[TEST] {k}: {v}")
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == '__main__':

    load_dotenv()
    wandb_key=os.getenv('WANDB_KEY')
    timestamp = get_timestamp()
    wandb.login(key=wandb_key)
    wandb.init(
        project="SSL_Comparison",
        name=str(opt['model'] + "- time stamp -" + timestamp))

    wandb.config.learning_rate = opt['hyperparameters']['lr']
    wandb.config.epochs = opt['hyperparameters']['epochs']
    wandb.config.beta1 = opt['hyperparameters']['beta1']
    wandb.config.beta2 = opt['hyperparameters']['beta2']
    wandb.config.eta_min = opt['hyperparameters']['eta_min']
    wandb.config.weight_decay = opt['hyperparameters']['weight_decay']
    wandb.config.seed = opt['hyperparameters']['seed']

        
    if opt['is_train']:
        train()
    elif opt['is_test']:
        test()