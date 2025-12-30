import hydra
from omegaconf import DictConfig
import logging

import numpy as np
from PIL import Image
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34
from torchvision.datasets import ImageFolder
from torchvision import transforms

from models import SimCLR
from tqdm import tqdm


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ImageNettePair(ImageFolder):
    """Generate mini-batch pairs on ImageNette training set."""
    def __getitem__(self, idx):
        path, target = self. samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair



def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
# Color distortion for ImageNette (use stronger augmentation s=1.0 for larger images)
def get_color_distortion(s=1.0):  # 1.0 for ImageNette (224x224 images)
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms. Compose([rnd_color_jitter, rnd_gray])
    return color_distort

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ===================== KNN Evaluation =====================

@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract features from the online encoder for all samples."""
    model.eval()
    features = []
    labels = []
    
    for images, targets in tqdm(dataloader, desc="Extracting features"):
        # Handle both paired and single image datasets
        if images.dim() == 5:  # Paired images [B, 2, C, H, W]
            images = images[: , 0]  # Take first view only
        images = images.to(device)
        
        # Get features from online encoder
        feat = model.enc(images)
        feat = F.normalize(feat, dim=1)
        
        features.append(feat.cpu())
        labels.append(targets)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


@torch.no_grad()
def knn_evaluate(model, train_loader, test_loader, device, k=200, num_classes=10):
    """
    Evaluate the model using KNN classifier with simple majority voting (unweighted).
    
    Args:
        model: BYOL model (or any model with .online_encoder)
        train_loader: DataLoader for training set (used as memory bank)
        test_loader: DataLoader for test set
        device: torch device
        k: number of nearest neighbors
        num_classes: number of classes
    
    Returns:
        top1_accuracy: KNN top-1 accuracy
    """
    model.eval()
    
    # Extract features from training set (memory bank)
    logger.info("Extracting training features for KNN...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    # Extract features from test set
    logger.info("Extracting test features for KNN...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Move to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # Normalize features for cosine similarity (important for BYOL/SimCLR representations)
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    total_correct = 0
    total_samples = test_features.size(0)
    batch_size = 256  # Process in batches to avoid OOM
    
    for idx in tqdm(range(0, total_samples, batch_size), desc="KNN evaluation (majority vote)"):
        batch_features = test_features[idx:idx + batch_size]
        batch_labels = test_labels[idx:idx + batch_size]
        
        # Cosine similarity: [batch_size, num_train_samples]
        sim = torch.mm(batch_features, train_features.t())
        
        # Get indices of top-k nearest neighbors
        _, indices_topk = sim.topk(k=k, dim=1, largest=True)  # [batch_size, k]
        
        # Get labels of the k nearest neighbors
        neighbors_labels = train_labels[indices_topk]  # [batch_size, k]
        
        # Simple majority voting: count votes per class
        # Create one-hot votes (1 for each neighbor)
        votes = torch.zeros(batch_features.size(0), num_classes, device=device)
        votes.scatter_add_(1, neighbors_labels, torch.ones_like(neighbors_labels, dtype=votes.dtype))
        
        # Predicted class is the one with most votes
        _, predictions = votes.max(dim=1)
        
        total_correct += (predictions == batch_labels).sum().item()
    
    top1_accuracy = 100.0 * total_correct / total_samples
    return top1_accuracy

@hydra.main(
    version_base="1.1",
    config_path=".",
    config_name="simclr_config_18_img.yml"
    )
def train(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    image_size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms. RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # Convert to tensor BEFORE color jitter to avoid uint8 overflow
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))  # Gaussian blur for larger images
        ], p=0.5),
        transforms. Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # Transform for KNN evaluation (no augmentation)
    eval_transform = transforms.Compose([
        transforms. Resize(256),  # Resize shorter side to 256
        transforms.CenterCrop(image_size),  # Center crop to 224x224
        transforms. ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    
    train_set = ImageNettePair(
        root=train_dir,
        transform=train_transform
    )
    
    # Training dataset with augmentation pairs
    train_loader = DataLoader(
        train_set,
        batch_size=args. batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True
    )
    
    # Datasets for KNN evaluation (no augmentation)
    knn_train_set = ImageFolder(
        root=train_dir,
        transform=eval_transform
    )
    
    knn_test_set = ImageFolder(
        root=val_dir,
        transform=eval_transform
    )
    
    # print(f"KNN train samples: {len(knn_train_set)}")
    # print(f"KNN test samples: {len(knn_test_set)}")
    
    knn_train_loader = DataLoader(
        knn_train_set,
        batch_size=args. batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    knn_test_loader = DataLoader(
        knn_test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-3))

    # SimCLR training
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_meter = AverageMeter("SimCLR_loss")
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)

            optimizer.zero_grad()
            feature, rep = model(x)
            loss = nt_xent(rep, args.temperature)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))

        # save checkpoint very log_interval epochs
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            logger.info("==> Save checkpoint. Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
            torch.save(model.state_dict(), 'simclr_{}_epoch{}.pt'.format(args.backbone, epoch))

    
    
    logger.info("=" * 50)
    logger.info("Starting KNN evaluation...")
    
        
    # # Load the checkpoint for KNN eval immediately
    # state_dict = torch.load("/mnt/disk4/baodq/SimCLR-CIFAR10/logs/SimCLR/cifar10/simclr_resnet34_epoch200.pt", map_location='cpu')
    # device = torch.device('cuda')
    # model.load_state_dict(state_dict, strict=True)
    # model = model.to(device)
    # model.eval()
    
    device = torch.device('cuda')
    # KNN EVAL
    # Evaluate with different k values
    # for k in [1, 5, 20, 200]:
    for k in [200]:
        top1_acc = knn_evaluate(
            model=model,
            train_loader=knn_train_loader,
            test_loader=knn_test_loader,
            device=device,
            k=k,
            num_classes=10
        )
        logger.info(f"KNN (k={k}) Top-1 Accuracy: {top1_acc:.2f}%")
    
    logger.info("=" * 50)
    logger.info("Training and evaluation completed!")

if __name__ == '__main__':
    train()


