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
from huggingface_hub import login
from utils.utils import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_valid_loss = 0
        
        #Training process
        for i, data in enumerate(train_loader):  
            #Move tensors to the configured device
            images, labels = data
            optimizer.zero_grad()
                
            #Forward pass
            outputs = model(images.to(device))
            loss = loss_func(outputs, labels.to(device))
            
            total_train_loss += loss.item() * images.size(0)
            #Backward and optimize
            loss.backward()
            optimizer.step()
            
            #Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.to(device)).sum().item()
            train_acc = float(correct)/float(images.to(device).shape[0])
        
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_train_loss/10500))
    
    
    
def validate():
    model=model.eval()
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        total_valid_loss += loss_func(outputs, labels).item() * labels.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs
        
    # calculate_metrics()    
    total_train_loss = total_train_loss/10500
    total_valid_loss = total_valid_loss/3000



def test():
    model=model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    #calculate_metrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('-root', type=str, default=None, choices=['.'])
    args = parser.parse_args()
    opt = option.parse(args.opt, root=args.root)
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    train_set, train_loader, valid_set, valid_loader, test_set, test_loader = create_ds(opt)
    model = load_checkpoint(model, opt['pretrain_path'])
    
    train_hypers = opt['hyperparameters']
    num_epochs = 70
    learning_rate = 0.006

    #Loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model.parameters(), train_hypers)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_hypers['epochs'], train_hypers['eta_min'])
    
    if opt['use_wandb']:
        load_dotenv()
        wandb_key=os.getenv('WANDB_KEY')
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        wandb.login(key=wandb_key)
        wandb.init(
            project="SSL_Comparison",
            name=str(opt['name'] + "- time stamp -" + timestamp))

        wandb.config.learning_rate = opt['hyperparameters']['lr']
        wandb.config.epochs = opt['hyperparameters']['epochs']
        wandb.config.beta1 = opt['hyperparameters']['beta1']
        wandb.config.beta2 = opt['hyperparameters']['beta2']
        wandb.config.eta_min = opt['hyperparameters']['eta_min']
        wandb.config.seed = opt['hyperparameters']['seed']
        
    
    if opt['is_train']:
        train()
    elif opt['is_test']:
        test()