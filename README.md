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
└── README.md
```
