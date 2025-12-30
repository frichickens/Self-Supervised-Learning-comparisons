# import torch
# from torchvision import models  # or wherever your net comes from

# # Recreate the exact same backbone architecture
# net = models.resnet18(pretrained=False)  # or your custom net
# keys = list(net.state_dict().keys())
# for key in keys:
#     need_to_load_key = "learner.target_encoder.net."+str(key)
#     print(need_to_load_key)


# # Load the checkpoint
# checkpoint_path = '/mnt/disk4/baodq/byol-pytorch/lightning_logs/version_4/checkpoints/epoch=199-step=93800.ckpt'  # choose your checkpoint
# net.load_state_dict(torch.load(checkpoint_path)['state_dict'])

# net.eval()


import torch
from torchvision import models

# Recreate the exact same backbone architecture
net = models.resnet18(pretrained=False)

# Load the checkpoint and extract the full state_dict
checkpoint_path = '/mnt/disk4/baodq/byol-pytorch/lightning_logs/version_4/checkpoints/epoch=199-step=93800.ckpt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # use 'cpu' or your device
full_state_dict = checkpoint['state_dict']  # this is a dict with prefixed keys

prefix = "learner.target_encoder.net."
new_state_dict = {
    key[len(prefix):]: value
    for key, value in full_state_dict.items()
    if key.startswith(prefix)
}

# Load the stripped state_dict into your clean backbone
net.load_state_dict(new_state_dict)

net.eval()  # ready for inference or downstream tasks