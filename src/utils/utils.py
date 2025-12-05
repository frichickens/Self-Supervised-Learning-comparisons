from data import *
import torch


def create_ds(opt):
    for phase, dataset_opt in opt['datasets'].items():
        if phase=='train': 
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, opt, None)
        elif phase=='valid': 
            valid_set = create_dataset(dataset_opt)
            valid_loader = create_dataloader(valid_set, dataset_opt, opt, None)
        elif phase=='test':
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt, opt, None)
    return train_set, train_loader, valid_set, valid_loader, test_set, test_loader

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model

def create_optimizer(params, opt):
    return torch.optim.Adam(
        params,
        lr = opt['lr_G'], betas=(opt['beta1'], opt['beta2']),
        weight_decay = opt['weight_decay'],
        
    )
    
    
def calculate_metrics():
    pass