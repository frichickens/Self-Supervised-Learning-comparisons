import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract L2-normalized features from a DataLoader."""
    features = []
    labels = []
    model.eval()
    for imgs, targets in tqdm(loader, desc="Extracting features", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        feats = model(imgs)
        feats = F.normalize(feats, dim=1)  # Important for SSL methods like BYOL/SimCLR
        features.append(feats.cpu())
        labels.append(targets)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


@torch.no_grad()
def knn_evaluate(
    model,
    train_root,
    test_root,
    device,
    k=200,
    batch_size=256,
    num_workers=16,
    dataset="imagenette"  # "imagenette" or "cifar10"
):
    """
    KNN evaluation for Imagenette or CIFAR-10 using ImageFolder structure.
    
    Assumptions:
    - Model is the backbone only (e.g., ResNet with fc replaced by Identity).
    - Outputs raw features → will be L2-normalized.
    - Simple majority voting (unweighted KNN).
    
    Args:
    - train_root: path to training folder (ImageFolder format: train/class_name/images...)
    - test_root:  path to test/val folder (same structure)
    - dataset:   "imagenette" → 224x224 ImageNet-style preprocessing
                 "cifar10"   → 32x32 with CIFAR-10 mean/std
    - k:         number of neighbors (200 is standard for 10-class datasets like these)
    
    Returns:
        top1_accuracy (float)
    """
    # Preprocessing based on dataset
    if dataset.lower() == "imagenette":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        expected_classes = 10
    elif dataset.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
        ])
        expected_classes = 10
    else:
        raise ValueError("dataset must be 'imagenette' or 'cifar10'")

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform)

    if len(train_dataset.classes) != expected_classes:
        logger.warning(f"Found {len(train_dataset.classes)} classes, expected {expected_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    logger.info(f"Dataset: {dataset.upper()}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Extract features
    logger.info("Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device)

    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)

    # Move to device for fast similarity computation
    train_features = train_features.to(device)
    test_features = test_features.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    num_classes = len(train_dataset.classes)
    total_correct = 0
    total_samples = test_features.size(0)

    # KNN in batches to save memory
    for start_idx in tqdm(range(0, total_samples, batch_size), desc="KNN prediction"):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_features = test_features[start_idx:end_idx]

        # Cosine similarity
        sim = torch.mm(batch_features, train_features.t())

        # Get top-k nearest neighbors
        _, topk_indices = sim.topk(k=k, dim=1, largest=True)

        # Gather neighbor labels
        neighbor_labels = train_labels[topk_indices]  # [batch, k]

        # Majority voting: count votes per class
        votes = torch.zeros(end_idx - start_idx, num_classes, device=device)
        votes.scatter_add_(1, neighbor_labels, torch.ones_like(neighbor_labels, dtype=votes.dtype))

        # Predict class with most votes
        _, preds = votes.max(dim=1)
        batch_labels = test_labels[start_idx:end_idx]

        total_correct += (preds == batch_labels).sum().item()

    accuracy = 100.0 * total_correct / total_samples
    print(f"KNN (k={k}) Top-1 Accuracy: {accuracy:.2f}%")
    logger.info(f"KNN (k={k}) Top-1 Accuracy: {accuracy:.2f}%")
    return accuracy



############################################################################################################

from model import *
# Change these!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(c_out=10)  # or ResNet34(10), depending on your checkpoint

# Replace the final linear layer with Identity to get raw features
model.linear = nn.Identity()  # This removes the classification head

# Load the saved checkpoint
checkpoint_path = "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/src/checkpoints/vanilla/resnet18_251208-155838_best.pth"  # Change to your actual file
state_dict = torch.load(checkpoint_path, map_location=device)

# If your checkpoint has BYOL prefixes like "online_encoder.", strip them
if list(state_dict.keys())[0].startswith("online_encoder."):
    state_dict = {k.replace("online_encoder.", ""): v for k, v in state_dict.items()}
elif list(state_dict.keys())[0].startswith("encoder."):
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

# Load weights (strict=True now possible since linear is Identity and matches feature dim)
model.load_state_dict(state_dict, strict=False)  # Should work cleanly

model = model.to(device)
model.eval()

# # For Imagenette
# acc_imagenette = knn_evaluate(
#     model=model,
#     train_root="path/to/imagenette2/train",
#     test_root="path/to/imagenette2/val",
#     device=device,
#     k=200,
#     dataset="imagenette"
# )

# For CIFAR-10 (from ImageFolder)
acc_cifar10 = knn_evaluate(
    model=model,
    train_root="/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/src/datasets/cifar10_images/train",
    test_root="/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/src/datasets/cifar10_images/test",
    device=device,
    k=200,
    dataset="cifar10"
)