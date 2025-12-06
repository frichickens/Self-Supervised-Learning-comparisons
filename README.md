# How to use this github?

```
.
├── Checkpoints/        # folder saves checkpoints
├── datasets/           
│   └── cifar10-images/ # images folder (.png with labels)
│       ├── train/      
│       ├── test/       
│       └── valid/
│    
├── .env                # private API keys (add yours here, DO NOT replace)
│
├── .gitignore          # Ignored large/unimportant files
│
├── src/                # Main source code
│   ├── model/          # Model architectures
│   ├── scripts/        # Reproduction scripts
│   ├── utils/          # Metrics, loaders, helpers
│   ├── options/        # Training configs (hyperparameters, model type)
│   └── train.py        # Main training pipeline
│
└── env.yml             # env
│
└── README.md
```

# Setup env:

```
conda env create -f env.yml
conda activate ssl_comp
```

# Our processed dataset is public at:

```
Cifar-10: https://www.kaggle.com/datasets/quocbaohust/cifar10
Cifar-100: 
```
