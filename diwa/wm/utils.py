import torch
from torchvision.transforms import Compose, Normalize
import yaml

from diwa.utils.transforms import (
    NormalizeVector,
    ScaleImageTensor,
    UnNormalizeImageTensorTorch,
)


def transpose_tensor(tensor):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    return torch.transpose(tensor, 0, 1)


def get_rgb_normalizer(device):
    rgb_mean = torch.tensor(
        [
            0.5,
            0.5,
            0.5,
        ]
    ).to(device)
    rgb_std = torch.tensor(
        [
            0.5,
            0.5,
            0.5,
        ]
    ).to(device)
    rgb_obs_normalizer = Compose(
        [
            ScaleImageTensor(),
            Normalize(rgb_mean, rgb_std),
        ]
    )
    return rgb_obs_normalizer


def get_rgb_unnormalizer(device):
    rgb_mean = torch.tensor(
        [
            0.5,
            0.5,
            0.5,
        ]
    ).to(device)
    rgb_std = torch.tensor(
        [
            0.5,
            0.5,
            0.5,
        ]
    ).to(device)
    rgb_obs_unnormalizer = UnNormalizeImageTensorTorch(rgb_mean, rgb_std)
    return rgb_obs_unnormalizer


def get_robot_obs_normalizer(stats_path, device):
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
        robot_obs_mean = torch.tensor(stats["robot_obs"][0]["mean"]).to(device)[:7]
        robot_obs_std = torch.tensor(stats["robot_obs"][0]["std"]).to(device)[:7]

    robot_obs_normalizer = NormalizeVector(robot_obs_mean, robot_obs_std)
    return robot_obs_normalizer


def get_scene_obs_normalizer(stats_path, device):
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
        scene_obs_mean = torch.tensor(stats["scene_obs"][0]["mean"]).to(device)
        scene_obs_std = torch.tensor(stats["scene_obs"][0]["std"]).to(device)

    scene_obs_normalizer = NormalizeVector(scene_obs_mean, scene_obs_std)
    return scene_obs_normalizer
