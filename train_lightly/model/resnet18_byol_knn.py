import copy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision import transforms

from lightly.loss import NegativeCosineSimilarity
from lightly.models. modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms. byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.utils.scheduler import cosine_schedule

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


# ─────────────────────────────────────────────────────────────────────────────
# kNN Evaluation Callback
# ─────────────────────────────────────────────────────────────────────────────
class KNNEvaluationCallback(Callback):
    """
    Callback to perform kNN evaluation at the end of each epoch.
    Embeds all training images and evaluates on test set with k=200. 
    """
    
    def __init__(self, train_loader, test_loader, k=200, feature_dim=512):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.k = k
        self.feature_dim = feature_dim
        self.best_knn_accuracy = 0.0
    
    @torch.no_grad()
    def extract_features(self, backbone, dataloader, device):
        """Extract features from all images in dataloader."""
        backbone.eval()
        features = []
        labels = []
        
        for images, targets in dataloader: 
            images = images. to(device)
            feats = backbone(images).flatten(start_dim=1)
            features.append(feats. cpu())
            labels.append(targets)
        
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # L2 normalize features
        features = nn.functional.normalize(features, dim=1)
        
        return features, labels
    
    @torch.no_grad()
    def knn_predict(self, train_features, train_labels, test_features, test_labels, k, num_classes=10):
        """
        Perform kNN classification. 
        """
        # Compute similarity (cosine similarity since features are normalized)
        # Do this in batches to avoid memory issues
        batch_size = 256
        total_correct = 0
        total_samples = 0
        
        for i in range(0, len(test_features), batch_size):
            batch_features = test_features[i:i + batch_size]
            batch_labels = test_labels[i:i + batch_size]
            
            # Compute cosine similarity
            similarity = torch.mm(batch_features, train_features.T)
            
            # Get top-k neighbors
            _, indices = similarity.topk(k, dim=1, largest=True, sorted=True)
            
            # Get labels of k nearest neighbors
            retrieved_labels = train_labels[indices]
            
            # Weighted voting (using similarity as weight)
            top_k_similarities = torch.gather(similarity, 1, indices)
            
            # Vote for each class
            predictions = []
            for j in range(len(batch_features)):
                votes = torch.zeros(num_classes)
                for neighbor_idx in range(k):
                    label = retrieved_labels[j, neighbor_idx]. item()
                    weight = top_k_similarities[j, neighbor_idx].item()
                    votes[label] += weight
                predictions.append(votes. argmax().item())
            
            predictions = torch.tensor(predictions)
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += len(batch_labels)
        
        accuracy = total_correct / total_samples
        return accuracy
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Run kNN evaluation at the end of each epoch."""
        device = pl_module.device
        
        # Extract features from train and test sets
        train_features, train_labels = self.extract_features(
            pl_module.backbone, self.train_loader, device
        )
        test_features, test_labels = self.extract_features(
            pl_module.backbone, self.test_loader, device
        )
        
        # Perform kNN classification
        knn_accuracy = self.knn_predict(
            train_features, train_labels,
            test_features, test_labels,
            k=self.k
        )
        
        # Track best accuracy
        if knn_accuracy > self.best_knn_accuracy: 
            self.best_knn_accuracy = knn_accuracy
        
        # Log to wandb
        pl_module.log('knn_accuracy', knn_accuracy, on_epoch=True, logger=True)
        pl_module.log('best_knn_accuracy', self.best_knn_accuracy, on_epoch=True, logger=True)
        
        print(f"\nEpoch {trainer. current_epoch} | "
              f"kNN Top-1 Accuracy: {knn_accuracy*100:.2f}% | "
              f"Best: {self.best_knn_accuracy*100:.2f}%\n")


# ─────────────────────────────────────────────────────────────────────────────
# BYOL Model
# ─────────────────────────────────────────────────────────────────────────────
class BYOL(pl. LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models. resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self. prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 200, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1) = batch[0]
        p0 = self. forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        
        # Log loss to wandb
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        return torch. optim.SGD(self.parameters(), lr=0.06)


# ─────────────────────────────────────────────────────────────────────────────
# Data Setup
# ─────────────────────────────────────────────────────────────────────────────

# Transform for kNN evaluation (no augmentation, just normalize)
knn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    ),
])

# Training dataset with BYOL augmentations
byol_transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
)

train_dataset_byol = torchvision.datasets.CIFAR10(
    "datasets/cifar10", train=True, download=False, transform=byol_transform
)

# Datasets for kNN evaluation (no augmentation)
train_dataset_knn = torchvision.datasets.CIFAR10(
    "datasets/cifar10", train=True, download=False, transform=knn_transform
)
test_dataset_knn = torchvision.datasets.CIFAR10(
    "datasets/cifar10", train=False, download=False, transform=knn_transform
)

# DataLoaders
train_loader_byol = torch.utils.data.DataLoader(
    train_dataset_byol,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

train_loader_knn = torch.utils.data.DataLoader(
    train_dataset_knn,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)

test_loader_knn = torch.utils.data.DataLoader(
    test_dataset_knn,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)

# ─────────────────────────────────────────────────────────────────────────────
# Training Setup
# ─────────────────────────────────────────────────────────────────────────────
model = BYOL()

wandb_logger = WandbLogger(
    project="SSL_Comp",
    name="byol-resnet18-run",
    log_model=False,
)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/byol",
    filename="byol-resnet18-last",
    save_last=True,
    save_top_k=0,
)

# kNN evaluation callback with k=200
knn_callback = KNNEvaluationCallback(
    train_loader=train_loader_knn,
    test_loader=test_loader_knn,
    k=200,
    feature_dim=512,
)

accelerator = "gpu" if torch.cuda. is_available() else "cpu"

trainer = pl.Trainer(
    max_epochs=200,
    devices=1,
    accelerator=accelerator,
    log_every_n_steps=10,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, knn_callback],
)

trainer.fit(model=model, train_dataloaders=train_loader_byol)

# Print final results
print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"{'='*60}")
print(f"Best kNN Top-1 Accuracy: {knn_callback.best_knn_accuracy * 100:.2f}%")
print(f"{'='*60}")

wandb_logger.experiment.finish()