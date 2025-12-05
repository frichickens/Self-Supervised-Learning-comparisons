# How to use this github?

.
├── Checkpoints/        # Directory to save model checkpoints
├── .env                # Private keys (HuggingFace, WandB)
├── .gitignore          # Ignore large or unimportant files
├── src/                # Main source code
│   ├── model/          # Model architectures only
│   ├── scripts/        # Scripts for reproducing results
│   ├── utils/          # Utilities (metrics, loaders, helpers)
│   ├── options/        # Configuration files (model type, hyperparams, ...)
│   └── train.py        # Main training pipeline
└── README.md
