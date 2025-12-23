import copy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision import transforms

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule

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
        
        
class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        input_dim = 512
        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        # input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.backbone = self.teacher_backbone
        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 200, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        
        # Log loss to wandb
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim
    
model = DINO()
dino_transform = DINOTransform()
# we ignore object detection annotations by setting target_transform to return 0

def target_transform(t):
    return 0


knn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    ),
])


# Datasets for kNN evaluation (no augmentation)
train_dataset_knn = torchvision.datasets.CIFAR10(
    "datasets/cifar10", train=True, download=False, transform=knn_transform
)
test_dataset_knn = torchvision.datasets.CIFAR10(
    "datasets/cifar10", train=False, download=False, transform=knn_transform
)


# CHANGE
wandb_logger = WandbLogger(
    project="SSL_Comp",  # Change to your project name
    name="dino-resnet18-knn-run",  # Change to your run name
    log_model=False,
)

# CHANGE
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/dino",  # Directory to save checkpoints
    filename="dino-resnet18-last",   # Filename for the checkpoint
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

train_dataset_dino = torchvision.datasets.CIFAR10(
    "datasets/cifar10",
    train=True,
    download=False,
    transform=dino_transform,
    target_transform=target_transform,
)

train_loader_dino = torch.utils.data.DataLoader(
    train_dataset_dino,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)




accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(
    max_epochs=200,
    devices=1,
    accelerator=accelerator,
    log_every_n_steps=10,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, knn_callback],
)

trainer.fit(model=model, train_dataloaders=train_loader_dino)

# Print final results
print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"{'='*60}")
print(f"Best kNN Top-1 Accuracy: {knn_callback.best_knn_accuracy * 100:.2f}%")
print(f"{'='*60}")


wandb_logger.experiment.finish()