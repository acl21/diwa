import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class ScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 1] range

    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.float().div(255)


class UnNormalizeImageTensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    """
    Args:
        array (np.ndarray): image of size (H, W, C) to be unnormalized and unscaled.
    Returns:
        array: uint8 image.
    """

    def __call__(self, array: np.ndarray) -> np.ndarray:
        assert isinstance(array, np.ndarray)

        arr = ((array * self.std) + self.mean) * 255
        image = np.clip(arr, 0, 255).astype("uint8")
        image = image.transpose((1, 2, 0))
        return image


class UnNormalizeImageTensorTorch(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    """
    Args:
        array (torch.Tensor): image of size (H, W, C) to be unnormalized and unscaled.
    Returns:
        array: uint8 image.
    """

    def __call__(self, array: torch.Tensor) -> torch.Tensor:
        assert isinstance(array, torch.Tensor)

        arr = ((array * self.std) + self.mean) * 255
        image = torch.clip(arr, 0, 255).type(torch.uint8)
        return image


class NormalizeVector(object):
    """Normalize a tensor vector with mean and standard deviation."""

    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.as_tensor(std)
        self.std[self.std == 0.0] = 1.0
        self.mean = torch.as_tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class NormalizeVectorMinMax(object):
    """Normalize a tensor vector with max and min values."""

    def __init__(self, min, max):
        self.min = torch.as_tensor(min)
        self.max = torch.as_tensor(max)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return (tensor - self.min) / (self.max - self.min)

    def __repr__(self):
        return self.__class__.__name__ + "(min={0}, max={1})".format(self.min, self.max)


class UnnormalizeVectorMinMax(object):
    """Unnormalize a tensor vector with max and min values."""

    def __init__(self, min, max):
        self.min = torch.as_tensor(min)
        self.max = torch.as_tensor(max)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor * (self.max - self.min) + self.min

    def __repr__(self):
        return self.__class__.__name__ + "(min={0}, max={1})".format(self.min, self.max)
