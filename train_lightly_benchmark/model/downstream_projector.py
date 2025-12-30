import torch
import torch. nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from pytorch_lightning. loggers import WandbLogger
import wandb
from models import *
from lightly.transforms.utils import IMAGENET_NORMALIZE


def calculate_metrics(
    logits:  torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10):
    """
    Compute classification metrics for multi-class. 
    """
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = torch. argmax(logits, dim=1).cpu().numpy()
    gts = labels.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(gts, preds),
        "precision": precision_score(gts, preds, average='macro', zero_division=0),
        "recall": recall_score(gts, preds, average='macro', zero_division=0),
        "f1": f1_score(gts, preds, average='macro', zero_division=0),
    }

    gts_bin = label_binarize(gts, classes=range(num_classes))
    if gts_bin.shape[1] == 1:
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




# UNCOMMENT FOR CIFAR

# normalize = transforms.Normalize(
#     mean=[0.4914, 0.4822, 0.4465],
#     std=[0.2470, 0.2435, 0.2616]
# )


# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize,
# ])

# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     normalize,
# ])

# train_dataset = torchvision.datasets.CIFAR10(
#     root="datasets/cifar10", train=True, download=False, transform=train_transform
# )
# test_dataset = torchvision.datasets.CIFAR10(
#     root="datasets/cifar10", train=False, download=False, transform=test_transform
# )



input_size = 128
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_NORMALIZE["mean"],
        std=IMAGENET_NORMALIZE["std"]
    ),
])

test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_NORMALIZE["mean"],
        std=IMAGENET_NORMALIZE["std"]
    ),
])


train_dataset = torchvision.datasets.Imagenette(
    root="/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/imagenette2-160", size="160px", split="train", download=False, transform=train_transform
)
test_dataset = torchvision.datasets.Imagenette(
    root="/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/imagenette2-160", size="160px", split="val", download=False, transform=test_transform
)


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

# CHANGE THESE (BYOL)
ckpt_path = "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/benchmark_logs/imagenette/version_0/BYOL/checkpoints/epoch=199-step=14600.ckpt"
# pretrained_byol = BYOL.load_from_checkpoint(ckpt_path, strict=False)

ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']
backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])  # Your backbone definition
backbone.load_state_dict(backbone_state_dict, strict=True)  # Or False if minor mismatches

# backbone = pretrained_byol.backbone

for param in backbone.parameters():
    param.requires_grad = False


# ─────────────────────────────────────────────────────────────────────────────
# Linear Classifier Setup
# ─────────────────────────────────────────────────────────────────────────────
num_classes = 10
classifier = nn.Linear(512, num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone.to(device)
classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.3, momentum=0.9, weight_decay=5e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
num_epochs = 100
best_accuracy = 0.0

# Learning rate scheduler (recommended for linear eval)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ─────────────────────────────────────────────────────────────────────────────
# Wandb Setup
# ─────────────────────────────────────────────────────────────────────────────
wandb.init(
    project="SSL_Comp",  # Change to your project name
    name="linear-byol-resnet18-run",
    config={
        "learning_rate": 0.3,
        "epochs": num_epochs,
        "batch_size": 256,
        "optimizer": "SGD",
        "scheduler": "CosineAnnealingLR",
    }
)



print("Starting linear evaluation...")

for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    all_train_logits = []
    all_train_labels = []

    for images, labels in train_loader: 
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            features = backbone(images).flatten(start_dim=1)

        logits = classifier(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_train_logits.append(logits.detach())
        all_train_labels.append(labels.detach())

    # Step the scheduler
    scheduler.step()

    # Calculate training metrics
    train_logits = torch.cat(all_train_logits, dim=0)
    train_labels = torch.cat(all_train_labels, dim=0)
    train_metrics = calculate_metrics(train_logits, train_labels, num_classes)
    avg_train_loss = running_loss / len(train_loader)

    # ─────────────────────────────────────────────────────────────────────────
    # Validation after each epoch
    # ─────────────────────────────────────────────────────────────────────────
    classifier.eval()
    all_val_logits = []
    all_val_labels = []
    val_loss = 0.0

    with torch. no_grad():
        for images, labels in test_loader: 
            images, labels = images.to(device), labels.to(device)
            features = backbone(images).flatten(start_dim=1)
            logits = classifier(features)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            all_val_logits.append(logits)
            all_val_labels.append(labels)

    val_logits = torch.cat(all_val_logits, dim=0)
    val_labels = torch. cat(all_val_labels, dim=0)
    val_metrics = calculate_metrics(val_logits, val_labels, num_classes)
    avg_val_loss = val_loss / len(test_loader)

    # Track best accuracy
    if val_metrics["accuracy"] > best_accuracy: 
        best_accuracy = val_metrics["accuracy"]
        # Optionally save best classifier
        torch.save(classifier.state_dict(), "checkpoints/linear_classifier_best.pth")

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────
    current_lr = scheduler.get_last_lr()[0]

    wandb.log({
        "epoch": epoch + 1,
        "train_loss":  avg_train_loss,
        "val_loss": avg_val_loss,
        "learning_rate": current_lr,
        # Train metrics
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_roc_auc": train_metrics["roc_auc"],
        "train_pr_auc": train_metrics["pr_auc"],
        # Validation metrics
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall":  val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        # Best tracking
        "best_val_accuracy": best_accuracy,
    })

    # Print epoch summary
    print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
          f"LR: {current_lr:.6f} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_metrics['accuracy']*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"{'='*60}")
print(f"Best Validation Accuracy:  {best_accuracy * 100:.2f}%")
print(f"Final Validation Metrics:")
for k, v in val_metrics.items():
    print(f"  {k}: {v:. 4f}")

wandb.finish()