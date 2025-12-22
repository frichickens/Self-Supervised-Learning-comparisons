import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10
from PIL import ImageFilter
import random


class GaussianBlur:
    """Gaussian blur augmentation"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DINOAugmentation:
    """
    DINO multi-crop augmentation strategy.
    Creates 2 global views + N local views
    """
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                n_local_crops=8, global_crop_size=224, local_crop_size=96):
        
        # Color jittering
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                    saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # Normalization 
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
        
        # Global crop 1
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, 
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(sigma=[0.1, 2.0]),
            normalize,
        ])
        
        # Global crop 2
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale,
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(sigma=[0.1, 2.0]),
            transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.1),
            normalize,
        ])
        
        # Local crops
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crops_scale,
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(sigma=[0.1, 2.0]),
            normalize,
        ])
        
        self.n_local_crops = n_local_crops
    
    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(image))
        
        return crops


class DINODataset(torch.utils.data.Dataset):
    """Wrapper dataset that applies DINO augmentation"""
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.transform = DINOAugmentation(
            global_crops_scale=config['training']['global_crops_scale'],
            local_crops_scale=config['training']['local_crops_scale'],
            n_local_crops=config['training']['n_local_crops'],
            global_crop_size=config['training']['global_crop_size'],
            local_crop_size=config['training']['local_crop_size']
        )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        crops = self.transform(image)
        return crops


def get_dino_dataloader(config):
    """
    Create DINO dataloader from config
    
    This creates a dataloader with DINO's multi-crop augmentation strategy.
    Each batch returns a list of tensors: [global1, global2, local1, ..., localN]
    """
    dataset_name = config['data']['dataset']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    n_local_crops = config['training']['n_local_crops']
    data_dir = config['data'].get('data_dir', './data')
    
    if dataset_name == 'cifar10':
        base_dataset = CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )
    elif dataset_name == 'stl10':
        base_dataset = STL10(
            root=data_dir,
            split='train+unlabeled',  # Use both labeled and unlabeled for pretraining
            download=True,
            transform=None,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Base dataset loaded: {len(base_dataset)} samples")
    
    # Wrap with DINO augmentation
    dino_dataset = DINODataset(base_dataset, config)
    
    def collate_fn(batch):
        """
        Collate function for multi-crop batches.
        
        Input: batch is a list of crops, where each element is:
               [global1, global2, local1, local2, ..., localN]
        
        Output: list of tensors, where each tensor is a batch of one crop type:
                [batch_global1, batch_global2, batch_local1, ..., batch_localN]
        """
        n_crops = 2 + n_local_crops
        return [torch.stack([crops[i] for crops in batch]) for i in range(n_crops)]
    
    dataloader = torch.utils.data.DataLoader(
        dino_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"DINO dataloader created: {len(dataloader)} batches, "
          f"{2 + n_local_crops} crops per sample (2 global + {n_local_crops} local)")
    
    return dataloader


def get_dino_eval_dataloader(config, split='test'):
    """
    Create standard dataloader for evaluation (no multi-crop, just center crop)
    
    Args:
        config: configuration dictionary
        split: 'test' or 'val'
    
    Returns:
        DataLoader for evaluation
    """
    dataset_name = config['data']['dataset']
    batch_size = config['training'].get('eval_batch_size', config['training']['batch_size'])
    num_workers = config['training']['num_workers']
    data_dir = config['data'].get('data_dir', './data')
    
    # Standard evaluation transforms (no augmentation)
    if dataset_name == 'cifar10':
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
        dataset = CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=eval_transform
        )
    elif dataset_name == 'stl10':
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
        dataset = STL10(
            root=data_dir,
            split='test',
            download=True,
            transform=eval_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader