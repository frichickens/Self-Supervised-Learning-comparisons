# -*- coding: utf-8 -*-
"""
Note that this benchmark also supports a multi-GPU setup.  If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application.  Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR : )
You can download the ImageNette dataset from here:  https://github.com/fastai/imagenette

Code has been tested on a A6000 GPU with 48GBytes of memory.
"""
import copy
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch. nn as nn
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from timm.models.vision_transformer import vit_base_patch32_224
from torchvision.models. vision_transformer import VisionTransformer

import wandb
from lightly.data import LightlyDataset
from lightly.loss import (
    BarlowTwinsLoss,
    DCLLoss,
    DCLWLoss,
    DINOLoss,
    MSNLoss,
    NegativeCosineSimilarity,
    NTXentLoss,
    PMSNLoss,
    SwaVLoss,
    TiCoLoss,
    VICRegLLoss,
    VICRegLoss,
)
from lightly. models import modules, utils
from lightly. models.modules import (
    MAEDecoderTIMM,
    MaskedVisionTransformerTIMM,
    MaskedVisionTransformerTorchvision,
    heads,
    memory_bank,
)
from lightly. transforms import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
    DINOTransform,
    SimCLRTransform,
)
from lightly. transforms.utils import IMAGENET_NORMALIZE
from lightly.utils import scheduler
from lightly. utils.benchmarking import BenchmarkModule
from lightly.utils.lars import LARS

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

num_workers = 12
memory_bank_size = 4096

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
knn_k = 200
knn_t = 0.1
classes = 10
input_size = 128

# Set to True to enable Distributed Data Parallel training. 
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating the loss
gather_distributed = False

# benchmark
n_runs = 1  # optional, increase to create multiple runs and report mean + std
batch_size = 128
lr_factor = batch_size / 256  # scales the learning rate linearly with batch size

# Number of devices and hardware to use for training. 
devices = torch.cuda.device_count() if torch.cuda. is_available() else 1
accelerator = "gpu" if torch.cuda. is_available() else "cpu"

if distributed:
    strategy = "ddp"
    batch_size = batch_size // devices
else:
    strategy = None
    devices = min(devices, 1)

# ================== WANDB CONFIGURATION ==================
WANDB_PROJECT = "ssl-benchmark"  # Change to your project name
WANDB_ENTITY = None  # Change to your entity/team name, or None for personal
USE_WANDB = True  # Set to False to disable wandb logging
# =========================================================

path_to_train = "/datasets/imagenette2-160/train/"
path_to_test = "/datasets/imagenette2-160/val/"

# Use BYOL augmentations
byol_transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=input_size),
    view_2_transform=BYOLView2Transform(input_size=input_size),
)

# Use SimCLR augmentations
simclr_transform = SimCLRTransform(
    input_size=input_size,
    cj_strength=0.5,
)

# Multi crop augmentation for DINO
dino_transform = DINOTransform(
    global_crop_size=128,
    local_crop_size=64,
    cj_strength=0.5,
)

normalize_transform = torchvision.transforms.Normalize(
    mean=IMAGENET_NORMALIZE["mean"],
    std=IMAGENET_NORMALIZE["std"],
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(input_size),
        torchvision. transforms.CenterCrop(128),
        torchvision.transforms. ToTensor(),
        normalize_transform,
    ]
)

dataset_train_kNN = LightlyDataset(input_dir=path_to_train, transform=test_transforms)
dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)


def create_dataset_train_ssl(model):
    """Helper method to apply the correct transform for ssl."""
    model_to_transform = {
        BYOLModel: byol_transform,
        DINOModel: dino_transform,
        SimCLRModel:  simclr_transform,
    }
    transform = model_to_transform[model]
    return LightlyDataset(input_dir=path_to_train, transform=transform)


def get_data_loaders(batch_size:  int, dataset_train_ssl):
    """Helper method to create dataloaders for ssl, kNN train and kNN test."""
    dataloader_train_ssl = torch.utils. data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_train_kNN = torch.utils.data. DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils. data.DataLoader(
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
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(*list(resnet. children())[:-1])
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
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
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Log epoch-level metrics
        self.log("epoch", float(self.current_epoch), on_epoch=True)
        self.log("max_accuracy", self.max_accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim. SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [cosine_scheduler]


class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn. Sequential(*list(resnet.children())[:-1])

        self.projection_head = heads.BYOLProjectionHead(feature_dim, 4096, 256)
        self.prediction_head = heads. BYOLPredictionHead(256, 4096, 256)

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
        p0 = self. forward(x0)
        z0 = self. forward_momentum(x0)
        p1 = self. forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("epoch", float(self.current_epoch), on_epoch=True)
        self.log("max_accuracy", self.max_accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        params = (
            list(self.backbone. parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head. parameters())
        )
        optim = torch.optim. SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch. optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [cosine_scheduler]


class DINOModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(*list(resnet. children())[:-1])
        self.head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
        )
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
        )

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self. head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self. teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[: 2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("epoch", float(self.current_epoch), on_epoch=True)
        self.log("max_accuracy", self. max_accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        param = list(self.backbone. parameters()) + list(self.head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler. CosineAnnealingLR(optim, max_epochs)
        return [optim], [cosine_scheduler]


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
    model_name = BenchmarkModel.__name__. replace("Model", "")
    
    for seed in range(n_runs):
        pl.seed_everything(seed)
        dataset_train_ssl = create_dataset_train_ssl(BenchmarkModel)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, dataset_train_ssl=dataset_train_ssl
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

        # Setup loggers
        sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
        
        # TensorBoard Logger
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, "imagenette"),
            name="",
            sub_dir=sub_dir,
            version=experiment_version,
        )
        
        if experiment_version is None:
            experiment_version = tb_logger.version

        # List of loggers to use
        loggers = [tb_logger]

        # WandB Logger
        if USE_WANDB:
            wandb_logger = WandbLogger(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{model_name}_run{seed}",
                group=model_name,  # Group runs by model type
                job_type="train",
                tags=[model_name, f"seed_{seed}", "imagenette"],
                config={
                    "model": model_name,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "learning_rate": 6e-2 * lr_factor,
                    "input_size": input_size,
                    "seed": seed,
                    "num_classes": classes,
                    "dataset": "imagenette",
                    "knn_k": knn_k,
                    "knn_t": knn_t,
                },
                save_dir=os. path.join(logs_root_dir, "wandb"),
                reinit=True,  # Allow multiple wandb. init() calls
            )
            loggers.append(wandb_logger)

        checkpoint_callback = pl. callbacks.ModelCheckpoint(
            dirpath=os.path. join(tb_logger.log_dir, "checkpoints")
        )
        
        # Custom callback for additional wandb logging
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
            logger=loggers,  # Pass list of loggers
            callbacks=callbacks,
        )
        
        start = time. time()
        trainer.fit(
            benchmark_model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test,
        )
        end = time.time()
        
        run = {
            "model": model_name,
            "batch_size":  batch_size,
            "epochs": max_epochs,
            "max_accuracy":  benchmark_model.max_accuracy,
            "runtime": end - start,
            "gpu_memory_usage":  torch.cuda.max_memory_allocated(),
            "seed":  seed,
        }
        runs.append(run)
        print(run)

        # Log final results to wandb
        if USE_WANDB: 
            wandb. log({
                "final_accuracy": benchmark_model. max_accuracy,
                "total_runtime_min": (end - start) / 60,
                "peak_gpu_memory_gb": torch. cuda.max_memory_allocated() / (1024**3),
            })
            # Finish the current wandb run
            wandb.finish()

        # Cleanup
        del benchmark_model
        del trainer
        torch. cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    bench_results[model_name] = runs

# Print results table
header = (
    f"| {'Model': <13} | {'Batch Size':>10} | {'Epochs':>6} "
    f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
)
print("-" * len(header))
print(header)
print("-" * len(header))
for model, results in bench_results.items():
    runtime = np.array([result["runtime"] for result in results])
    runtime = runtime.mean() / 60
    accuracy = np.array([result["max_accuracy"] for result in results])
    gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3)

    if len(accuracy) > 1:
        accuracy_msg = f"{accuracy. mean():>8.3f} +- {accuracy.std():>4.3f}"
    else:
        accuracy_msg = f"{accuracy. mean():>18.3f}"

    print(
        f"| {model: <13} | {batch_size:>10} | {max_epochs:>6} "
        f"| {accuracy_msg} | {runtime:>6.1f} Min "
        f"| {gpu_memory_usage:>8.1f} GByte |",
        flush=True,
    )
print("-" * len(header))

# Log summary table to wandb
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
    
