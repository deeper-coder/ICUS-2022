import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .augmentation import Compose, RandomFlip_LR, RandomFlip_UD, RandomRotate


class Spine_Dataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, augmentation: bool = True):
        self.augmentation = augmentation
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith(".")]
        if not self.ids:
            raise RuntimeError(f"No input file found in {images_dir}, make sure you put your images there")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img):
        
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        # img.shape (1, w, h)  mask.shape (1, w, h)
        img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in [".npz", ".npy"]:
            return Image.fromarray(np.load(filename))
        elif ext in [".pt", ".pth"]:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)  # return (w, h)

    @staticmethod
    def transform(img, mask):
        data_transforms = Compose([RandomFlip_LR(prob=0.5), RandomFlip_UD(prob=0.5), RandomRotate()])
        return data_transforms(img, mask)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(f"{name}.*"))
        label_file = list(self.labels_dir.glob(f"{name}.*"))

        img = self.load(img_file[0])
        label = self.load(label_file[0])
        
        # resize the img and label to (224, 224)
        resize = transforms.Resize([224,224])
        img_ = resize(img)
        label_ = resize(label)
        
        img = self.preprocess(img_)
        label = self.preprocess(label_)

        img = torch.as_tensor(img.copy()).float().contiguous()
        label = torch.as_tensor(label.copy()).long().contiguous()
        if self.augmentation:
            img, label = self.transform(img, label)

        return {"img": img, "label": label}