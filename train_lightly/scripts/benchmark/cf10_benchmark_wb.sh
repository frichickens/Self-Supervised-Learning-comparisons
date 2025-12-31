export WANDB_API_KEY="9c233a1274e348c64884fa361aac455906cf6a0e"
wandb login --relogin $WANDB_API_KEY
python /mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/model/cifar10_benchmark_wb.py