# How to use this github?

- Checkpoints: path to save checkpoints of models
- .env: private keys for HF and WandB
- .gitignore: unimportant large files
- src:
- *model: ONLY model architecture
- *scripts: script to reproduce results
- *utils: utils for training (calculate metrics, load model, ...)
- *options: config of the training model (model type, hyperparameters, ...)
- *train.py: main training code
