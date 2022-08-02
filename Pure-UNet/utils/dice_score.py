import torch
import torch.nn as nn
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()

    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target)
    if sets_sum.item() == 0:
        return torch.tensor(1, dtype=torch.float32).to(input.device)
    else:
        return (2 * inter + epsilon) / (sets_sum + epsilon)


def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = torch.tensor(0, dtype=torch.float32).to(input.device)
    for channel in range(input.shape[1]):
        dice += dice_coeff(
            input[:, channel, ...], target[:, channel, ...].float(), epsilon
        )
    return dice / input.shape[1]


class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        return 1 - multiclass_dice_coeff(input, target)


def compute_pre(input: Tensor, target: Tensor):
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(input)
    if total.item() == 0:
        return 0
    result = inter.item() / total.item()
    return result


def compute_rec(input: Tensor, target: Tensor):
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(target)
    if total.item() == 0:
        return 1
    result = inter.item() / total.item()
    return result


def compute_miou(input: Tensor, target: Tensor):
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(input) + torch.sum(target) - inter
    if total.item() == 0:
        return 0
    result = inter.item() / total.item()
    return result


def compute_pre_rec_miou(input: Tensor, target: Tensor, multi_class: bool = False):
    assert input.size() == target.size()
    if multi_class:
        return (
            compute_pre(input[:, 1, ...], target[:, 1, ...].float()),
            compute_rec(input[:, 1, ...], target[:, 1, ...].float()),
            compute_miou(input[:, 1, ...], target[:, 1, ...].float()),
        )  # only channel 1
    else:
        return (
            compute_pre(input, target.float()),
            compute_rec(input, target.float()),
            compute_miou(input, target.float()),  # not 'm'IoU actually TODO
        )


def recall_fucking_loss(input: Tensor, target: Tensor, epsilon=1e-6, weight=10):
    assert input.size() == target.size()

    inter = torch.dot(input.reshape(-1), target.reshape(-1).float())
    sets_sum = torch.sum(target)
    if sets_sum.item() == 0:
        return torch.tensor(1, dtype=torch.float32).to(input.device)
    else:
        return (1 - (inter + epsilon) / (sets_sum + epsilon)) * weight
