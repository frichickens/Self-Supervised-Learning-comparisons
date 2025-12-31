# How to use this github?

This is a github for benchmarking SSL (Self-supervised Learning methods)
This Github implemented:
- Supervised Learning
- SimCLR
- BYOL
- DINO

Dependencies:
```
conda env create --file environment.yml 
```

Cifar10:
```
python /mnt/disk4/baodq/Self-Supervised-Learning-comparisons/train_lightly/model/get_cifar10.py
```

ImageNette: we downloaded the .tar file from here then extract
```
https://github.com/fastai/imagenette
```

Scripts to reproduce results are provided in the folder:
```
Self-Supervised-Learning-comparisons/train_lightly/scripts
```
