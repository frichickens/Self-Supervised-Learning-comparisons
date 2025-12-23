import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

# ADD
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
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
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
    
model = SimCLR()

transform = SimCLRTransform(input_size=32)
dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10", download=False, transform=transform
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
    name="simclr-resnet18-run",  # Change to your run name
    log_model=False,
)

# CHANGE
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/simclr",  # Directory to save checkpoints
    filename="simclr-resnet18-last",   # Filename for the checkpoint
    save_last=True,         # Save the last checkpoint
    save_top_k=0,           # Don't save based on metric, only last
)


trainer = pl.Trainer(
    max_epochs=200,
    devices=1,
    accelerator=accelerator,
    log_every_n_steps=10,
    logger=wandb_logger,              
    callbacks=[checkpoint_callback],  
)


trainer.fit(model=model, train_dataloaders=dataloader)