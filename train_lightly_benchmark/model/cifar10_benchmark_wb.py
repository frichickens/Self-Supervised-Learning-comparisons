# -*- coding: utf-8 -*-
"""
Benchmark Results

Updated: 27.03.2023 (42a6a924b1b6d5b6cc89a6b2a0a0942cc4af93ab)

------------------------------------------------------------------------------------------
| Model         | Batch Size | Epochs |  KNN Test Accuracy |       Time | Peak GPU Usage |
------------------------------------------------------------------------------------------
| BarlowTwins   |        128 |    200 |              0.842 |  375.9 Min |      1.7 GByte |
| BYOL          |        128 |    200 |              0.869 |  121.9 Min |      1.6 GByte |
| DCL           |        128 |    200 |              0.844 |  102.2 Min |      1.5 GByte |
| DCLW          |        128 |    200 |              0.833 |  100.4 Min |      1.5 GByte |
| DINO          |        128 |    200 |              0.840 |  120.3 Min |      1.6 GByte |
| FastSiam      |        128 |    200 |              0.906 |  164.0 Min |      2.7 GByte |
| Moco          |        128 |    200 |              0.838 |  128.8 Min |      1.7 GByte |
| NNCLR         |        128 |    200 |              0.834 |  101.5 Min |      1.5 GByte |
| SimCLR        |        128 |    200 |              0.847 |   97.7 Min |      1.5 GByte |
| SimSiam       |        128 |    200 |              0.819 |   97.3 Min |      1.6 GByte |
| SwaV          |        128 |    200 |              0.812 |   99.6 Min |      1.5 GByte |
| SMoG          |        128 |    200 |              0.743 |  192.2 Min |      1.2 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        512 |    200 |              0.819 |  153.3 Min |      5.1 GByte |
| BYOL          |        512 |    200 |              0.868 |  108.3 Min |      5.6 GByte |
| DCL           |        512 |    200 |              0.840 |   88.2 Min |      4.9 GByte |
| DCLW          |        512 |    200 |              0.824 |   87.9 Min |      4.9 GByte |
| DINO          |        512 |    200 |              0.813 |  108.6 Min |      5.0 GByte |
| FastSiam      |        512 |    200 |              0.788 |  146.9 Min |      9.5 GByte |
| Moco (*)      |        512 |    200 |              0.847 |  112.2 Min |      5.6 GByte |
| NNCLR (*)     |        512 |    200 |              0.815 |   88.1 Min |      5.0 GByte |
| SimCLR        |        512 |    200 |              0.848 |   87.1 Min |      4.9 GByte |
| SimSiam       |        512 |    200 |              0.764 |   87.8 Min |      5.0 GByte |
| SwaV          |        512 |    200 |              0.842 |   88.7 Min |      4.9 GByte |
| SMoG          |        512 |    200 |              0.686 |  110.0 Min |      3.4 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        512 |    800 |              0.859 |  517.5 Min |      7.9 GByte |
| BYOL          |        512 |    800 |              0.910 |  400.9 Min |      5.4 GByte |
| DCL           |        512 |    800 |              0.874 |  334.6 Min |      4.9 GByte |
| DCLW          |        512 |    800 |              0.871 |  333.3 Min |      4.9 GByte |
| DINO          |        512 |    800 |              0.848 |  405.2 Min |      5.0 GByte |
| FastSiam      |        512 |    800 |              0.902 |  582.0 Min |      9.5 GByte |
| Moco (*)      |        512 |    800 |              0.899 |  417.8 Min |      5.4 GByte |
| NNCLR (*)     |        512 |    800 |              0.892 |  335.0 Min |      5.0 GByte |
| SimCLR        |        512 |    800 |              0.879 |  331.1 Min |      4.9 GByte |
| SimSiam       |        512 |    800 |              0.904 |  333.7 Min |      5.1 GByte |
| SwaV          |        512 |    800 |              0.884 |  330.5 Min |      5.0 GByte |
| SMoG          |        512 |    800 |              0.800 |  415.6 Min |      3.2 GByte |
------------------------------------------------------------------------------------------

(*): Increased size of memory bank from 4096 to 8192 to avoid too quickly 
changing memory bank due to larger batch size.

The benchmarks were created on a single NVIDIA RTX A6000.

Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)

"""
import copy
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb

from lightly.data import LightlyDataset
from lightly.loss import (
    DCLLoss,
    DCLWLoss,
    DINOLoss,
    NegativeCosineSimilarity,
    NTXentLoss,
)
from lightly.models import ResNetGenerator, modules, utils
from lightly.models.modules import heads, memory_bank
from lightly.transforms import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
    DINOTransform,
    SimCLRTransform,
)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import BenchmarkModule

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
num_workers = 8
knn_k = 200
knn_t = 0.1
classes = 10

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False

# benchmark
resnet_version = 34
n_runs = 1  # optional, increase to create multiple runs and report mean + std
batch_size = 32
lr_factor = batch_size / 128  # scales the learning rate linearly with batch size

# Number of devices and hardware to use for training.
devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

if distributed:
    strategy = "ddp"
    # reduce batch size for distributed training
    batch_size = batch_size // devices
else:
    strategy = "auto"  # Set to "auto" if using PyTorch Lightning >= 2.0
    # limit to single device if not using distributed training
    devices = min(devices, 1)


# ================== WANDB CONFIGURATION ==================
# The dataset structure should be like this:
WANDB_PROJECT = "ssl-benchmark"  # Change to your project name
WANDB_ENTITY = None  # Change to your entity/team name, or None for personal
USE_WANDB = True  # Set to False to disable wandb logging
# =========================================================


# Adapted from our MoCo Tutorial on CIFAR-10
#
# Replace the path with the location of your CIFAR-10 dataset.
# We assume we have a train folder with subfolders
# for each class and .png images inside.
#
# You can download `CIFAR-10 in folders from kaggle
# <https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders>`_.

# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/
path_to_train = "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10/train/"
path_to_test = "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10/test/"

# Use BYOL augmentations
byol_transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
)

# Use SimCLR augmentations
simclr_transform = SimCLRTransform(
    input_size=32,
    cj_strength=0.5,
    gaussian_blur=0.0,
)


# Multi crop augmentation for DINO, additionally, disable blur for cifar10
dino_transform = DINOTransform(
    global_crop_size=32,
    n_local_views=0,
    cj_strength=0.5,
    gaussian_blur=(0, 0, 0),
)


# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ]
)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = LightlyDataset(input_dir=path_to_train, transform=test_transforms)

dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)


def create_dataset_train_ssl(model):
    """Helper method to apply the correct transform for ssl.

    Args:
        model:
            Model class for which to select the transform.
    """
    model_to_transform = {
        BYOLModel: byol_transform,
        DINOModel: dino_transform,
        SimCLRModel: simclr_transform,
    }
    transform = model_to_transform[model]
    return LightlyDataset(input_dir=path_to_train, transform=transform)


def get_data_loaders(batch_size: int, dataset_train_ssl):
    """Helper method to create dataloaders for ssl, kNN train and kNN test.

    Args:
        batch_size: Desired batch size for all dataloaders.
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-34")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss
    
    def on_train_epoch_end(self):
        # Log epoch-level metrics
        self.log("epoch", float(self.current_epoch), on_epoch=True)
        self.log("max_accuracy", self.max_accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-34")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.projection_head = heads.BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = heads.BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        utils.update_momentum(
            self.projection_head, self.projection_head_momentum, m=0.99
        )
        (x0, x1), _, _ = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("epoch", float(self.current_epoch), on_epoch=True)
        self.log("max_accuracy", self.max_accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DINOModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        resnet = ResNetGenerator("resnet-34")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.head = self._build_projection_head()
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = self._build_projection_head()

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048)

    def _build_projection_head(self):
        head = heads.DINOProjectionHead(512, 2048, 256, 2048, batch_norm=True)
        # use only 2 layers for cifar10
        head.layers = heads.ProjectionHead(
            [
                (512, 2048, nn.BatchNorm1d(2048), nn.GELU()),
                (2048, 256, None, None),
            ]
        ).layers
        return head

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self.teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("epoch", float(self.current_epoch), on_epoch=True)
        self.log("max_accuracy", self. max_accuracy, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        param = list(self.backbone.parameters()) + list(self.head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]



models = [
    BYOLModel,
    DINOModel,
    SimCLRModel,
]
bench_results = dict()

experiment_version = None
# loop through configurations and train models
for BenchmarkModel in models:
    runs = []
    model_name = BenchmarkModel.__name__.replace("Model", "")
    for seed in range(n_runs):
        pl.seed_everything(seed)
        dataset_train_ssl = create_dataset_train_ssl(BenchmarkModel)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, dataset_train_ssl=dataset_train_ssl
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

        # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, "cifar10"),
            name="",
            sub_dir=sub_dir,
            version=experiment_version,
        )
        if experiment_version is None:
            # Save results of all models under same version directory
            experiment_version = tb_logger.version
            
        loggers = [tb_logger]
        # WandB Logger
        if USE_WANDB:
            wandb_logger = WandbLogger(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{model_name}_run{seed}_resnet{resnet_version}",
                group=model_name,  # Group runs by model type
                job_type="train",
                tags=[model_name, f"seed_{seed}", "cifar10"],
                config={
                    "model": model_name,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "learning_rate": 6e-2 * lr_factor,
                    # "input_size": input_size,
                    "seed": seed,
                    "num_classes": classes,
                    "dataset": "cifar10",
                    "knn_k": knn_k,
                    "knn_t": knn_t,
                },
                save_dir=os. path.join(logs_root_dir, "wandb"),
                reinit=True,  
            )
            loggers.append(wandb_logger)    
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints")
        )
        
        class WandbMetricsCallback(pl. Callback):
            """Callback to log additional metrics to wandb at epoch end."""
            
            def on_train_epoch_end(self, trainer, pl_module):
                if USE_WANDB:
                    # Log GPU memory usage
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
                        wandb.log({
                            "gpu_memory_gb": gpu_memory,
                            "epoch": trainer.current_epoch,
                        })
            
            def on_validation_epoch_end(self, trainer, pl_module):
                if USE_WANDB and hasattr(pl_module, 'max_accuracy'):
                    wandb.log({
                        "knn_accuracy": pl_module. max_accuracy,
                        "epoch":  trainer.current_epoch,
                    })
        callbacks = [checkpoint_callback]
        if USE_WANDB:
            callbacks.append(WandbMetricsCallback())
        
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=devices,
            accelerator=accelerator,
            default_root_dir=logs_root_dir,
            strategy=strategy,
            sync_batchnorm=sync_batchnorm,
            logger=loggers,
            callbacks=[checkpoint_callback],
        )
        start = time.time()
        trainer.fit(
            benchmark_model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test,
        )
        end = time.time()
        run = {
            "model": model_name,
            "batch_size": batch_size,
            "epochs": max_epochs,
            "max_accuracy": benchmark_model.max_accuracy,
            "runtime": end - start,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(),
            "seed": seed,
        }
        runs.append(run)
        print(run)

        if USE_WANDB:
            wandb. log({
                "final_accuracy": benchmark_model. max_accuracy,
                "total_runtime_min": (end - start) / 60,
                "peak_gpu_memory_gb": torch. cuda.max_memory_allocated() / (1024**3),
            })
            # Finish the current wandb run
            wandb.finish()
        
        # delete model and trainer + free up cuda memory
        del benchmark_model
        del trainer
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    bench_results[model_name] = runs

# print results table
header = (
    f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
    f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
)
print("-" * len(header))
print(header)
print("-" * len(header))
for model, results in bench_results.items():
    runtime = np.array([result["runtime"] for result in results])
    runtime = runtime.mean() / 60  # convert to min
    accuracy = np.array([result["max_accuracy"] for result in results])
    gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3)  # convert to gbyte

    if len(accuracy) > 1:
        accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
    else:
        accuracy_msg = f"{accuracy.mean():>18.3f}"

    print(
        f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
        f"| {accuracy_msg} | {runtime:>6.1f} Min "
        f"| {gpu_memory_usage:>8.1f} GByte |",
        flush=True,
    )
print("-" * len(header))


if USE_WANDB:
    # Create a new summary run
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="benchmark_summary",
        job_type="summary",
        tags=["summary", "imagenette"],
    )
    
    # Create summary table
    summary_data = []
    for model, results in bench_results.items():
        runtime = np.array([result["runtime"] for result in results]).mean() / 60
        accuracy = np.array([result["max_accuracy"] for result in results])
        gpu_memory = np.array([result["gpu_memory_usage"] for result in results]).max() / (1024**3)
        
        summary_data.append([
            model,
            batch_size,
            max_epochs,
            accuracy. mean(),
            accuracy.std() if len(accuracy) > 1 else 0,
            runtime,
            gpu_memory,
        ])
    
    summary_table = wandb.Table(
        columns=["Model", "Batch Size", "Epochs", "Mean Accuracy", "Std Accuracy", "Runtime (min)", "Peak GPU (GB)"],
        data=summary_data
    )
    wandb.log({"benchmark_summary": summary_table})
    wandb.finish()
    
