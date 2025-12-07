from data import *
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import numpy as np
from datetime import datetime
from sklearn.preprocessing import label_binarize
from collections import OrderedDict
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



def create_ds(opt):
    for phase, dataset_opt in opt['datasets'].items():
        if phase=='train': 
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, opt, None)
        elif phase=='val': 
            valid_set = create_dataset(dataset_opt)
            valid_loader = create_dataloader(valid_set, dataset_opt, opt, None)
        elif phase=='test':
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt, opt, None)
    return train_set, train_loader, valid_set, valid_loader, test_set, test_loader



def load_checkpoint(model, checkpoint_path):
    if checkpoint_path == "":
        print("No checkpoint path provided, training from scratch.")
        return model

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model



def create_optimizer(params, opt):
    return torch.optim.Adam(
        params,
        lr = opt['lr'], betas=(opt['beta1'], opt['beta2']),
        weight_decay = float(opt['weight_decay']),
        
    )


def save_checkpoint(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def calculate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10):
    """
    Compute classification metrics for multi-class 

    Args:
        logits: (B, C) raw scores from the model
        labels: (B,) ground-truth class indices
        num_classes: 10 for CIFAR-10

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc, pr_auc (macro-averaged)
    """
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    gts   = labels.cpu().numpy()

    metrics = {
        "accuracy" : accuracy_score(gts, preds),
        "precision": precision_score(gts, preds, average='macro', zero_division=0),
        "recall"   : recall_score(gts, preds, average='macro', zero_division=0),
        "f1"       : f1_score(gts, preds, average='macro', zero_division=0),
    }

    gts_bin = label_binarize(gts, classes=range(num_classes))
    if gts_bin.shape[1] == 1:                     # edge case (only 1 class in batch)
        gts_bin = np.hstack((1 - gts_bin, gts_bin))

    try:
        metrics["roc_auc"] = roc_auc_score(gts_bin, probs, average='macro', multi_class='ovr')
    except ValueError:
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(gts_bin, probs, average='macro')
    except ValueError:
        metrics["pr_auc"] = 0.0

    return metrics



def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def get_model_parameters_number(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params}")
    return total_params