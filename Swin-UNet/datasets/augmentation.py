from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob, is_mask=False):
        if prob <= self.prob:
                img = img.flip([2])
        return img

    def __call__(self, img, mask):
        prob = random.uniform(0, 1)
        return self._flip(img, prob), self._flip(mask, prob)


class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob, is_mask=False):
        if prob <= self.prob:
                img = img.flip([1])
        return img

    def __call__(self, img, mask):
        prob = random.uniform(0, 1)
        return self._flip(img, prob), self._flip(mask, prob)


class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt, is_mask=False):
        img = torch.rot90(img, cnt, [1, 2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0, self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask