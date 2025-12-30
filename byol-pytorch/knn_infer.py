import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import os
import argparse

IMAGE_SIZE = 32
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext. lower() in IMAGE_EXTS:
                self.paths.append(path)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)


class LabeledDataset(Dataset):
    """Dataset with labels for k-NN evaluation"""
    def __init__(self, folder, image_size):
        super().__init__()
        self.paths = []
        self.labels = []
        
        # Assumes folder structure: folder/class_name/image. jpg
        class_folders = sorted([d for d in Path(folder).iterdir() if d.is_dir()])
        self.class_to_idx = {cls. name: idx for idx, cls in enumerate(class_folders)}
        
        for class_folder in class_folders: 
            for path in class_folder.glob('*'):
                _, ext = os. path.splitext(path)
                if ext.lower() in IMAGE_EXTS:
                    self.paths.append(path)
                    self.labels.append(self.class_to_idx[class_folder.name])

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms. CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        label = self.labels[index]
        return self.transform(img), label


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract features from the encoder"""
    model.eval()
    features = []
    labels = []
    
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            images, label = batch
            labels.append(label)
        else:
            images = batch
        
        images = images.to(device)
        # Get features before the final FC layer
        feat = model(images)
        features.append(feat. cpu())
    
    features = torch.cat(features, dim=0)
    if labels:
        labels = torch.cat(labels, dim=0)
        return features, labels
    return features


@torch.no_grad()
def knn_evaluate(train_features, train_labels, test_features, test_labels, k=200):
    """
    k-NN evaluation without temperature (hard voting)
    """
    train_features = nn.functional.normalize(train_features, dim=1)
    test_features = nn.functional.normalize(test_features, dim=1)
    
    num_classes = train_labels.max().item() + 1
    num_test = test_features.size(0)
    
    # Compute similarity (cosine similarity since normalized)
    similarity = test_features @ train_features.T  # [num_test, num_train]
    
    # Get top-k nearest neighbors
    _, indices = similarity.topk(k, dim=1, largest=True, sorted=True)
    
    # Get labels of k nearest neighbors
    retrieved_labels = train_labels[indices]  # [num_test, k]
    
    # Hard voting (no temperature) - majority vote
    predictions = []
    for i in range(num_test):
        # Count votes for each class
        votes = torch.bincount(retrieved_labels[i], minlength=num_classes)
        predictions.append(votes.argmax().item())
    
    predictions = torch.tensor(predictions)
    
    # Calculate accuracy
    correct = (predictions == test_labels).sum().item()
    accuracy = correct / num_test * 100
    
    return accuracy


def load_encoder(checkpoint_path, device):
    """Load the pretrained ResNet encoder"""
    # Create the same architecture
    net = models.resnet34(pretrained=False)

    # Load the checkpoint and extract the full state_dict
    # checkpoint_path = '/mnt/disk4/baodq/byol-pytorch/lightning_logs/version_4/checkpoints/epoch=199-step=93800.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # use 'cpu' or your device
    full_state_dict = checkpoint['state_dict']  # this is a dict with prefixed keys

    prefix = "learner.online_encoder.net."
    new_state_dict = {
        key[len(prefix):]: value
        for key, value in full_state_dict.items()
        if key.startswith(prefix)
    }

    # Load the stripped state_dict into your clean backbone
    net.load_state_dict(new_state_dict)
    
    return net.to('cuda')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='k-NN Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to saved model checkpoint')
    parser.add_argument('--train_folder', type=str, required=True,
                        help='path to training images (with class subfolders)')
    parser.add_argument('--test_folder', type=str, required=True,
                        help='path to test images (with class subfolders)')
    parser.add_argument('--k', type=int, default=200, help='number of neighbors')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load encoder
    encoder = load_encoder(args.checkpoint, device)
    print(f"Loaded encoder from {args.checkpoint}")
    
    # Create datasets
    train_dataset = LabeledDataset(args.train_folder, IMAGE_SIZE)
    test_dataset = LabeledDataset(args.test_folder, IMAGE_SIZE)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_to_idx)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args. batch_size, 
                              shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=8)
    
    # Extract features
    print("Extracting training features...")
    train_features, train_labels = extract_features(encoder, train_loader, device)
    
    print("Extracting test features...")
    test_features, test_labels = extract_features(encoder, test_loader, device)
    
    # k-NN evaluation
    print(f"\nRunning {args.k}-NN evaluation (no temperature, hard voting)...")
    accuracy = knn_evaluate(train_features, train_labels, 
                           test_features, test_labels, k=args.k)
    
    print(f"Top-1 Accuracy: {accuracy:.2f}%")



