import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD  = (0.2470, 0.2435, 0.2616)

class CIFAR10ImageFolderDataset:

    def __init__(self, dataset_opt):
        phase = dataset_opt['phase']           
        data_root = dataset_opt['dataroot']    

        assert phase in ['train', 'val', 'test'], f"phase must be train/val/test, got {phase}"

        # Define transforms based on phase
        if phase == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(NORM_MEAN, NORM_STD),
            ])
        else:  # val or test
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(NORM_MEAN, NORM_STD),
            ])

        folder_path = os.path.join(data_root, phase)
        self.dataset = datasets.ImageFolder(folder_path, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]