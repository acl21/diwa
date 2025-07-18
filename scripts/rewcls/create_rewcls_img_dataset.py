import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from diwa.wm.wrapper.base import BaseWMWrapper


def make_rewcls_img_dataset(cfg, split):
    """
    Creates a dataset to train an image-based reward classifier.
    This function only saves the images with the corresponding binary labels.
    """
    print(f"Creating reward classifier dataset for {split} split...")

    for skill_name in cfg.skills_list:
        print(f"Processing skill: {skill_name}")

        img_data_path = Path(cfg.img_data_dir) / skill_name / f"{split}.npz"
        if not img_data_path.exists():
            print(f"Image data for {skill_name} in {split} split does not exist. Skipping.")
            continue
        img_data = np.load(img_data_path, allow_pickle=True)

        rewards_data_path = Path(cfg.rewards_data_dir) / skill_name / f"{split}_rewards.npz"
        rewards_data = np.load(rewards_data_path, allow_pickle=True)

        ep_ends = np.cumsum(img_data["traj_lengths"])
        ep_starts = np.concatenate(([0], ep_ends[:-1]))

        # Save the latent states with the corresponding binary labels
        X = img_data["images"]
        y = rewards_data["rewards"]

        assert len(X) == len(y), "Length of images and rewards must match."

        np.savez(
            Path(cfg.data_out_dir) / skill_name / f"{cfg.data_out_prefix}_{split}.npz",
            X=X,
            y=y,
            ep_starts=ep_starts,
        )


@hydra.main(version_base="1.3", config_path="../../config/rewcls", config_name="rewcls_img_dataset")
def main(cfg: DictConfig):
    img_data_dir = Path(cfg.img_data_dir)
    rewards_data_dir = Path(cfg.rewards_data_dir)
    assert img_data_dir.exists(), f"Image data directory {img_data_dir} does not exist."
    assert rewards_data_dir.exists(), f"Rewards data directory {rewards_data_dir} does not exist."

    data_out_dir = Path(cfg.data_out_dir)
    if not data_out_dir.exists():
        os.makedirs(data_out_dir, exist_ok=True)

    make_rewcls_img_dataset(cfg, "train")
    make_rewcls_img_dataset(cfg, "val")


if __name__ == "__main__":
    main()
