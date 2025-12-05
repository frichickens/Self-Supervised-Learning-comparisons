"""create dataset and dataloader"""
import torch
import torch.utils.data

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None, collate_fn=None):
    phase = dataset_opt['phase']
    batch_size = dataset_opt.get('batch_size', 128)
    
    if phase == 'train':
        shuffle = True
        num_workers = dataset_opt.get('n_workers', 4) * max(len(opt.get('gpu_ids', [])), 1)
    else:
        shuffle = False
        num_workers = dataset_opt.get('n_workers', 4)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == 'train'),
        sampler=sampler
    )


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']

    if mode == 'cifar10':
        # Old way (your previous custom dataset)
        # from data.cifar10 import BaseDataset as D

        # New way: use clean ImageFolder version
        from data.cifar10 import CIFAR10ImageFolderDataset as D

    else:
        raise NotImplementedError(f'Dataset [{mode}] is not recognized.')

    dataset = D(dataset_opt)
    print(f'[{mode}] dataset {dataset_opt["phase"]} phase: {len(dataset)} samples')
    return dataset