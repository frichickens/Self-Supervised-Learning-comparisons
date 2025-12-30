wandb_key="9c233a1274e348c64884fa361aac455906cf6a0e"
cd ..
wandb login --relogin $wandb_key
wandb login

python normal.py