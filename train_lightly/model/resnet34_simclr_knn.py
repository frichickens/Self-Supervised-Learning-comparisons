import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision import transforms

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

# ADD
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
            images = images.to(device)
            feats = backbone(images).flatten(start_dim=1)
            features.append(feats.cpu())
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
                    label = retrieved_labels[j, neighbor_idx].item()
                    weight = top_k_similarities[j, neighbor_idx].item()
                    votes[label] += weight
                predictions.append(votes.argmax().item())
            
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
        
        print(f"\nEpoch {trainer.current_epoch} | "
              f"kNN Top-1 Accuracy: {knn_accuracy*100:.2f}% | "
              f"Best: {self.best_knn_accuracy*100:.2f}%\n")

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet34()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        
        # Log loss to wandb
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
    
model = SimCLR()

transform = SimCLRTransform(input_size=32)

knn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    ),
])

# Dataset for training with augmentation
dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10", download=True, transform=transform
)

# Datasets for kNN evaluation (no augmentation)
train_dataset_knn = torchvision.datasets.CIFAR10(
    "datasets/cifar10", train=True, download=False, transform=knn_transform
)
test_dataset_knn = torchvision.datasets.CIFAR10(
    "datasets/cifar10", train=False, download=False, transform=knn_transform
)

# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"



# CHANGE
wandb_logger = WandbLogger(
    project="SSL_Comp",  # Change to your project name
    name="simclr-resnet34-run",  # Change to your run name
    log_model=False,
)

# CHANGE
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/simclr",  # Directory to save checkpoints
    filename="simclr-resnet34-last",   # Filename for the checkpoint
    save_last=True,         # Save the last checkpoint
    save_top_k=0,           # Don't save based on metric, only last
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

knn_callback = KNNEvaluationCallback(
    train_loader=train_loader_knn,
    test_loader=test_loader_knn,
    k=200,
    feature_dim=512,
)


trainer = pl.Trainer(
    max_epochs=200,
    devices=1,
    accelerator=accelerator,
    log_every_n_steps=10,
    logger=wandb_logger,              
    callbacks=[checkpoint_callback, knn_callback],  
)


trainer.fit(model=model, train_dataloaders=dataloader)

# Print final results
print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"{'='*60}")
print(f"Best kNN Top-1 Accuracy: {knn_callback.best_knn_accuracy * 100:.2f}%")
print(f"{'='*60}")

wandb_logger.experiment.finish()