"""
Unmodified from https://github.com/nematoli/lumos
"""

import logging
import os
from pathlib import Path
import pickle
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import CombinedLoader, rank_zero_only
import torch
from tqdm import tqdm

from lumos.utils.info_utils import print_system_env_info

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, parent_path.as_posix())
os.chdir(cwd_path)

logger = logging.getLogger(__name__)

BZ = 10000
EPOCH = 4


def is_combined_loader(loader):
    return isinstance(loader, CombinedLoader)


@hydra.main(version_base="1.3", config_path="../config", config_name="featurizer")
def featurizer(cfg: DictConfig) -> None:
    """
    This is called to calculate features for a dataset under a loaded world model
    Args:
        cfg: hydra config
    """
    if cfg.exp_dir is None:
        raise ValueError("The path to the world model is not given.")
    chk = Path(cfg.exp_dir)
    log_rank_0(f"checkpoint:{chk.as_posix()}")
    log_rank_0(f"Extracting features with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())

    seed_everything(cfg.seed, workers=True)

    if chk is None:
        raise ValueError("World model's checkpoint was not found.")
    else:
        if cfg.world_model.name == "dreamer_v2":
            from lumos.world_models.dreamer_v2 import DreamerV2

            world_model = DreamerV2.load_from_checkpoint(chk.as_posix()).to(cfg.device)
            world_model.eval()
        else:
            raise NotImplementedError(f"Unknown model: {cfg.world_model.name}")

    # longest episode on calvin is 37683 frames long.
    # so we can set the batch size to 10000 and the run the feature extraction loop 4 times

    cfg.datamodule.batch_sampler.init_idx = 0
    cfg.datamodule.seq_len = 1  # Force T to be 1
    cfg.datamodule.train_batch_size = BZ
    cfg.datamodule.val_batch_size = BZ
    cfg.datamodule.reset_prob = 0.0
    cfg.datamodule.datasets.wm_disk_dataset.num_workers = 2
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    logger.info("Extracting features from the training dataset.")
    extract_features(
        world_model,
        datamodule.train_dataloader(),
        datamodule.train_datasets["vis"],
        cfg,
    )

    logger.info("Extracting features from the validation dataset.")
    extract_features(
        world_model,
        datamodule.val_dataloader(),
        datamodule.val_datasets["vis"],
        cfg,
    )


def extract_features(world_model, data_loader, dataset, cfg):
    """
    Extracts features from the given model and data_loader
    Args:
        world_model: The world model
        data_loader: The data loader for the dataset
        dataset: The dataset to extract features from
        cfg: The configuration for the extraction
    """

    dim_features = world_model.decoder.in_dim
    feats = np.zeros((len(dataset), dim_features), dtype=np.float32)
    zfeats = np.zeros((len(dataset), dim_features), dtype=np.float32)
    rel_acts = np.zeros((len(dataset), cfg.datamodule.action_space), dtype=np.float32)
    resets = np.zeros((len(dataset), 1), dtype=bool)
    frames = np.zeros((len(dataset), 1), dtype=int)
    robot_obs = np.zeros((len(dataset), cfg.world_model.robot_dim), dtype=np.float32)

    data_dict = {}

    for epoch in range(EPOCH):
        if epoch == 0:
            in_state = world_model.rssm_core.init_state(cfg.datamodule.train_batch_size)
        else:
            in_state = [torch.roll(x, 1, 0) for x in in_state]
            in_state[0][0] = torch.zeros_like(in_state[0][0])
            in_state[1][0] = torch.zeros_like(in_state[1][0])

        combined = is_combined_loader(data_loader)
        if combined:
            loader = tqdm(data_loader)
        else:
            loader = tqdm(data_loader["vis"])

        for i, batch in enumerate(loader):
            if combined:
                batch = batch["vis"]

            features, out_state = world_model.infer_features(
                batch["rgb_obs"]["rgb_static"],
                batch["robot_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                in_state,
                batch["rgb_obs"]["rgb_gripper"],
            )
            in_state = out_state

            idxs = batch["idx"].cpu().numpy()
            feats[idxs] = features.cpu().numpy().squeeze(0)
            zfeats[idxs] = (
                obs_to_zero_feature(
                    world_model,
                    batch["rgb_obs"]["rgb_static"],
                    batch["rgb_obs"]["rgb_gripper"],
                    batch["robot_obs"],
                )
                .cpu()
                .numpy()
                .squeeze(0)
            )
            rel_acts[idxs] = batch["actions"]["rel_actions"].cpu().numpy().squeeze(0)
            resets[idxs] = batch["reset"].cpu().numpy().squeeze(0)
            frames[idxs] = batch["frame"].cpu().numpy().squeeze(0)
            robot_obs[idxs] = batch["robot_obs"].cpu().numpy().squeeze(0)

            for idx in idxs:
                data_dict[int(frames[idx])] = {
                    "features": feats[idx],
                    "zero_features": zfeats[idx],
                    "rel_actions": rel_acts[idx],
                    "reset": resets[idx],
                    "robot_obs": robot_obs[idx],
                }

    cached_feats_path = dataset.abs_datasets_dir / cfg.output_file
    with open(str(cached_feats_path), "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def obs_to_zero_feature(wm, rgb_s, rgb_g, proprio):
    bz = rgb_s.size(1)
    zero_action = torch.zeros((1, bz, 7)).to(wm.device)
    zero_action[:, :, -1] = 1.0
    true_reset = torch.ones((1, bz, 1), dtype=torch.bool).to(wm.device)

    features, _ = wm.infer_features(
        rgb_s,
        proprio,
        zero_action,
        true_reset,
        wm.rssm_core.init_state(bz),
        rgb_g,
    )
    return features


@rank_zero_only
def log_rank_0(*args, **kwargs):
    """
    Log the information using the logger at rank 0.
    """
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    featurizer()
