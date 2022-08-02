import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from hubconf import unet_carvana
from unet_old import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.unified_focal_loss import SymmetricUnifiedFocalLoss


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=3e-4,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False, help="Use data parallel"
    )
    parser.add_argument(
        "--multi-class",
        dest="multi_class",
        action="store_true",
        default=True,
        help="Use multi-class in two-class tasks",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    net = UNet(
        n_channels=1, n_classes=2 if args.multi_class else 1, bilinear=args.bilinear
    )

    # 加载模型参数
    net.load_state_dict(torch.load("./important_pth/old_unet.pth"))
    net.eval()
    dummy_input = torch.randn(1, 1, 256, 256)
    torch.onnx.export(
        net, dummy_input, "./unet_model.onnx", opset_version=11,
    )
    # torch.save(net, "./unet_model.pth")
    """
    # Test
    from PIL import Image
    import numpy as np

    I = Image.open("/home/pose3d/projs/UNet_Spine_Proj/UNet_Spine/data/imgs/0.png")
    I_array = np.array(I)
    I_array = np.expand_dims(I_array, 0)  # channel dim
    I_array = np.expand_dims(I_array, 0)  # batch dim
    I_array = I_array.astype(np.float32)
    I_array = torch.tensor(I_array)
    Image.fromarray(net(I_array).numpy()).save("./onnx_out.png")
    """
