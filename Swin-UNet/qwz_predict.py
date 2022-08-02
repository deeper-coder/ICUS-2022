import os
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets.spine_dataset import Spine_Dataset
import scipy
from PIL import Image

from torch.utils.data import DataLoader, random_split
from configs import parse_args
from custom_loss.dice_score import dice_loss
import matplotlib.pyplot as plt
from train_spine import YoneModel

path = "./spine_proj_final_results/222cgv7k/checkpoints"
file_name = os.listdir(path)
file_path = os.path.join(path, file_name[0])

args = parse_args()

model = YoneModel.load_from_checkpoint(
    file_path,
    arch=args.arch,
    encoder_name=args.backbone,
    encoder_weights=None,
    in_channels=1,
    out_classes=1,
    lr=args.learning_rate,
)

# 1. create dataset
dataset = Spine_Dataset(
    images_dir="/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/qwz_img2/s5",
    masks_dir="/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/qwz_mask2/s5",
    augmentation=args.aug,
)

# 2. split dataset
n_val = int(len(dataset) * args.val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# 3. generate dataloader
loader_args = dict(batch_size=args.batch_size, num_workers=16, pin_memory=True)
train_dataloader = DataLoader(dataset=train_set, shuffle=True, **loader_args)  # type: ignore os.cpu_count()
#val_loader = DataLoader(dataset=val_set, shuffle=True, **loader_args)
train_dataloader = DataLoader(dataset=dataset, shuffle=True, **loader_args)

batch = next(iter(train_dataloader))

with torch.no_grad():
    model.eval()
    logits = model(batch["img"])

temp = np.array(logits.numpy(), dtype=np.float64)[0].squeeze()
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(temp)
plt.savefig(r"/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/qwz_predict2/50d5/666_50d.jpg")
print(temp[0])
#im = Image.fromarray(temp).convert('RGB')
#im.save(r"/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/qwz_predict/100.jpg")
#scipy.misc.imsave('/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/qwz_predict/100.jpg', logits)
#print(temp)