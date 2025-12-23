#!/bin/bash
#SBATCH --job-name=resnet34_dino_knn        # Job name
#SBATCH --output=./log_slurm/result/resnet34_dino_knn.txt      # Output file
#SBATCH --error=./log_slurm/error/resnet34_dino_knn.txt       # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1               # Số node yêu cầu
#SBATCH --cpus-per-task=10       # Số CPU cho mỗi task

export WANDB_API_KEY="9c233a1274e348c64884fa361aac455906cf6a0e"
wandb login --relogin $WANDB_API_KEY
python /home/user01/aiotlab/baodq/Self-Supervised-Learning-comparisons/train_lightly/model/resnet34_byol_knn.py