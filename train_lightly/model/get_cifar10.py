import torchvision
import os
from PIL import Image

# Download CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10', train=True, download=False)
testset = torchvision.datasets.CIFAR10(root='/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10', train=False, download=False)

# Create folders
os.makedirs('/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10/train', exist_ok=True)
os.makedirs('/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10/test', exist_ok=True)

# Save train images
for i, (img, label) in enumerate(trainset):
    img.save(f'/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10/train/image_{i}.png')
# Save test images
for i, (img, label) in enumerate(testset):
    img.save(f'/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10/test/image_{i}.png')

print("CIFAR-10 train and test images saved to 'cifar10/train' and 'cifar10/test' folders.")