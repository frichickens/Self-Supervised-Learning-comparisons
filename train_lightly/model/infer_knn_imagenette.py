import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from lightly.models.modules.heads import (
    BYOLProjectionHead,
    BYOLPredictionHead,
    DINOProjectionHead,
    SimCLRProjectionHead,
)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import BenchmarkModule  # For inheritance trick
import copy
import numpy as np

path_to_train = "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/imagenette2-160/train"   
path_to_test  = "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/imagenette2-160/val"     
checkpoint_path = "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/benchmark_logs/imagenette/version_0/BYOL/checkpoints/epoch=199-step=14600.ckpt"  # UPDATE THIS
model_type = "BYOL"  # Change to "DINO" or "SimCLR"

input_size = 128
batch_size = 128
num_workers = 12
knn_k = 200
num_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
])

dataset_train_kNN = ImageFolder(root=path_to_train, transform=test_transforms)
dataset_test = ImageFolder(root=path_to_test, transform=test_transforms)

dataloader_train_knn = torch.utils.data.DataLoader(
    dataset_train_knn,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

dataloader_test_knn = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

class SSLModelForEval(BenchmarkModule):
    
    def __init__(self, model_type: str, num_classes: int = 10):
        super().__init__(dataloader_kNN=None, num_classes=num_classes)
        
        resnet = torchvision.models.resnet18(pretrained=False)
        feature_dim = 512
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if model_type == "BYOL":
            self.projection_head = BYOLProjectionHead(feature_dim, 4096, 256)
            self.prediction_head = BYOLPredictionHead(256, 4096, 256)
            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)
        elif model_type == "DINO":
            self.head = DINOProjectionHead(feature_dim, 2048, 256, 2048, batch_norm=True)
            self.teacher_backbone = copy.deepcopy(self.backbone)
            self.teacher_head = DINOProjectionHead(feature_dim, 2048, 256, 2048, batch_norm=True)
        elif model_type == "SimCLR":
            self.projection_head = SimCLRProjectionHead(feature_dim, feature_dim, 128)
        else:
            raise ValueError("model_type must be BYOL, DINO, or SimCLR")
        
        self.model_type = model_type

    def forward(self, x):
        return self.backbone(x).flatten(start_dim=1)

print(f"Loading checkpoint: {checkpoint_path}")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

model = SSLModelForEval.load_from_checkpoint(
    checkpoint_path,
    model_type=model_type,
    num_classes=num_classes
).to(device)

model.eval()
backbone = model.backbone  

@torch.no_grad()
def extract_features(dataloader):
    features = []
    labels = []
    for batch in dataloader:
        images, target = batch[:2]  
        images = images.to(device)
        feats = backbone(images)
        feats = nn.functional.normalize(feats, dim=1) 
        features.append(feats.cpu())
        labels.append(target)
    return torch.cat(features), torch.cat(labels)

@torch.no_grad()
def knn_predict(train_feats, train_labels, test_feats, test_labels, k=200):

    total_correct = 0
    total = 0
    batch_size = 128

    for i in range(0, len(test_feats), batch_size):
        batch_feats = test_feats[i:i+batch_size].to(device)
        sim = torch.mm(batch_feats, train_feats.T.to(device))  
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)

        pred = torch.zeros(batch_feats.size(0), num_classes, device=device)
        for j in range(k):
            pred.scatter_add_(1, train_labels[topk_idx[:, j]].unsqueeze(1), topk_sim[:, j].unsqueeze(1))
        
        predicted = pred.argmax(dim=1)
        actual = test_labels[i:i+batch_size].to(device)
        total_correct += (predicted == actual).sum().item()
        total += actual.size(0)
    
    return total_correct / total

print("Extracting training features (for kNN database)...")
train_features, train_labels = extract_features(dataloader_train_knn)

print("Extracting test features...")
test_features, test_labels = extract_features(dataloader_test_knn)

print(f"Running kNN evaluation (k={knn_k})...")
accuracy = knn_predict(train_features, train_labels, test_features, test_labels, k=knn_k)

print("\n" + "="*60)
print("kNN Evaluation Complete!")
print(f"Model: {model_type} (ResNet-18) on Imagenette")
print(f"Top-1 kNN Accuracy: {accuracy * 100:.2f}%")
print("="*60)