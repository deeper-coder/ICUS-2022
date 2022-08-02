import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import segmentation_models_pytorch as smp
from datasets.dataset_spine import Spine_Dataset
from config import get_config
from parser_config import get_parser

from train_pl import YasuoModel
from torch.utils.data import DataLoader, random_split
from networks.vision_transformer import SwinUnet as ViT_seg
from utils.custom_loss.dice_score import dice_loss
import matplotlib.pyplot as plt

print("******************************** start predicting ********************************")

path = "./lightning_logs/version_6/checkpoints/"
file_name = os.listdir(path)
file_path = os.path.join(path, file_name[0])

# 1. generate config file
args = get_parser()    
config = get_config(args)

# 2. initialize model
net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
# model = YasuoModel(
#     config = config,
#     args = args,
#     net = net
# )

# 3. load ckpt
model = YasuoModel.load_from_checkpoint(
    file_path,
    config = config,
    args = args,
    net = net
)

# 4. create dataset
images_dir = args.root_path + 'imgs/'
labels_dir = args.root_path + 'masks/'

dataset = Spine_Dataset(
    images_dir=images_dir,
    labels_dir=labels_dir,
    augmentation=True,
)

# 5. split dataset
n_val = int(len(dataset) * args.val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# 6. generate dataloader
loader_args = dict(batch_size=args.batch_size, pin_memory=True)
train_dataloader = DataLoader(dataset=train_set, shuffle=True, **loader_args)
val_dataloader = DataLoader(dataset=val_set, shuffle=False, **loader_args)

# 7. load val_set and pred
batch = next(iter(val_dataloader))
threshold = 0.5
with torch.no_grad():
    model.eval()
    logits = model(batch["img"])
pred_masks = torch.sigmoid(logits)
pred_masks = (pred_masks > threshold).type(torch.uint8)

# 8. visualize
cnt = 0
for image, gt_mask, pred_mask in zip(batch["img"], batch["label"], pred_masks):
    cnt = cnt + 1
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")
    plt.savefig(f"./predicted_results/compare{cnt}.png")
    plt.show()

print("******************************** finish predicting ********************************")
