import torch
from torchvision.transforms import Compose, Normalize
import yaml

from diwa.utils.transforms import (
    NormalizeVectorMinMax,
    ScaleImageTensor,
    UnNormalizeImageTensorTorch,
    UnnormalizeVectorMinMax,
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
        robot_obs_min = torch.tensor(stats["robot_obs"][0]["min"]).to(device)
        robot_obs_max = torch.tensor(stats["robot_obs"][0]["max"]).to(device)

    robot_obs_normalizer = NormalizeVectorMinMax(robot_obs_min, robot_obs_max)
    return robot_obs_normalizer


def get_robot_obs_unnormalizer(stats_path, device):
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
        robot_obs_min = torch.tensor(stats["robot_obs"][0]["min"]).to(device)
        robot_obs_max = torch.tensor(stats["robot_obs"][0]["max"]).to(device)

    robot_obs_unnormalizer = UnnormalizeVectorMinMax(robot_obs_min, robot_obs_max)
    return robot_obs_unnormalizer


def get_scene_obs_normalizer(stats_path, device):
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
        scene_obs_min = torch.tensor(stats["scene_obs"][0]["min"]).to(device)
        scene_obs_max = torch.tensor(stats["scene_obs"][0]["max"]).to(device)

    scene_obs_normalizer = NormalizeVectorMinMax(scene_obs_min, scene_obs_max)
    return scene_obs_normalizer


def get_scene_obs_unnormalizer(stats_path, device):
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
        scene_obs_min = torch.tensor(stats["scene_obs"][0]["min"]).to(device)
        scene_obs_max = torch.tensor(stats["scene_obs"][0]["max"]).to(device)

    scene_obs_unnormalizer = UnnormalizeVectorMinMax(scene_obs_min, scene_obs_max)
    return scene_obs_unnormalizer
