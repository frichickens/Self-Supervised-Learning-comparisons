from data import *
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    label_binarize,
)
import numpy as np




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