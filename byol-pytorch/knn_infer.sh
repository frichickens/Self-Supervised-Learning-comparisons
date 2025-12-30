#!/bin/bash

python knn_infer.py \
    --checkpoint "/mnt/disk4/baodq/byol-pytorch/lightning_logs/version_7/checkpoints/epoch=199-step=375000.ckpt" \
    --train_folder "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/src/datasets/cifar10_images/train" \
    --test_folder "/mnt/disk4/baodq/Self-Supervised-Learning-comparisons/src/datasets/cifar10_images/test" \
    --k 200