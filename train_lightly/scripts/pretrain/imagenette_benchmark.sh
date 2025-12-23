#!/bin/bash
#SBATCH --job-name=imagenette_benchmark        # Job name
#SBATCH --output=./log_slurm/result/imagenette_benchmark.txt      # Output file
#SBATCH --error=./log_slurm/error/imagenette_benchmark.txt       # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1               # Số node yêu cầu
#SBATCH --cpus-per-task=20       # Số CPU cho mỗi task

export WANDB_API_KEY="9c233a1274e348c64884fa361aac455906cf6a0e"
wandb login --relogin $WANDB_API_KEY
python /home/user01/aiotlab/baodq/Self-Supervised-Learning-comparisons/train_lightly/model/imagenette_benchmark.py