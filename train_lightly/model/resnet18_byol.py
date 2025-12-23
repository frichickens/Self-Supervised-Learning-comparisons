import copy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.utils. scheduler import cosine_schedule

# ADD
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class BYOL(pl.LightningModule):
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
        y = self. backbone(x).flatten(start_dim=1)
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


model = BYOL()

# CHANGE
wandb_logger = WandbLogger(
    project="SSL_Comp",  # Change to your project name
    name="byol-resnet18-run",  # Change to your run name
    log_model=False,
)

# CHANGE
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/byol",  # Directory to save checkpoints
    filename="byol-resnet18-last",   # Filename for the checkpoint
    save_last=True,         # Save the last checkpoint
    save_top_k=0,           # Don't save based on metric, only last
)

# We disable resizing and gaussian blur for cifar10.
transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
)
dataset = torchvision.datasets. CIFAR10(
    "datasets/cifar10", download=False, transform=transform
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda. is_available() else "cpu"

# CHANGE
trainer = pl.Trainer(
    max_epochs=200,
    devices=1,
    accelerator=accelerator,
    log_every_n_steps=10,
    logger=wandb_logger,              
    callbacks=[checkpoint_callback],  
)

trainer.fit(model=model, train_dataloaders=dataloader)

wandb_logger.experiment.finish()