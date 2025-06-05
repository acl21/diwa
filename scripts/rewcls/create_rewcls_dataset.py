import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from diwa.wm.wrapper.base import BaseWMWrapper


def make_rewcls_dataset(cfg, wmw, split):
    """
    Creates a dataset to train the reward classifier.
    This function only saves the latent states with the corresponding binary labels.
    The labels are automatically generated based on the privileged state.
    """
    print(f"Creating reward classifier dataset for {split} split...")

    for skill_name in cfg.skills_list:
        print(f"Processing skill: {skill_name}")
        wmw.skill = skill_name

        feat_data_path = Path(cfg.feat_data_dir) / skill_name / f"{split}.npz"
        feat_data = np.load(feat_data_path, allow_pickle=True)

        rewards_data_path = Path(cfg.rewards_data_dir) / skill_name / f"{split}_rewards.npz"
        rewards_data = np.load(rewards_data_path, allow_pickle=True)

        traj_lengths = feat_data["traj_lengths"]
        ep_ends = np.cumsum(feat_data["traj_lengths"])
        ep_starts = np.concatenate(([0], ep_ends[:-1]))

        # Save the latent states with the corresponding binary labels
        X = feat_data["states"]
        y = rewards_data["rewards"]

        assert len(X) == len(y), "Length of states and rewards must match."

        np.savez(
            Path(cfg.data_out_dir) / skill_name / f"{cfg.data_out_prefix}_{split}.npz",
            X=X,
            y=y,
            ep_starts=ep_starts,
        )

        if split == "val":
            print("Skipping imagined latents for validation split.")
            continue
        # We add imagined latents to the dataset by rolling out inside the world model.
        # We do this since the reward classifier would see imagined latents during training.
        # Ignoring this does not really affect the performance, but it is better to have a more realistic dataset.
        imagined_X = []
        imagined_X_starts = []
        labels = []
        imagined_ep_starts = []
        idx = 0
        for episode_idx in tqdm(range(len(traj_lengths)), desc="Imagining Latents: "):
            start_idx = ep_starts[episode_idx]
            end_idx = ep_ends[episode_idx]

            obs_feat = feat_data["states"][start_idx:end_idx]
            obs_feat = torch.tensor(obs_feat, dtype=torch.float32).to(cfg.device)
            actions = feat_data["actions"][start_idx:end_idx]
            actions = torch.tensor(actions, dtype=torch.float32).to(cfg.device)
            true_y = rewards_data["rewards"][start_idx:end_idx]

            latent = obs_feat[0].unsqueeze(0)
            imagined_X.append(latent.squeeze().cpu().numpy())
            imagined_X_starts.append(latent.squeeze().cpu().numpy())
            labels.append(true_y[0])
            imagined_ep_starts.append(idx)
            idx += 1
            for i in range(len(obs_feat) - 1):
                action = actions[i]
                next_latent = wmw.wm_step(latent, action.unsqueeze(0))

                imagined_X.append(next_latent.squeeze().cpu().numpy())
                imagined_X_starts.append(obs_feat[0].squeeze().cpu().numpy())
                labels.append(true_y[i + 1])

                latent = next_latent
                idx += 1

        imagined_X = np.array(imagined_X)
        imagined_X_starts = np.array(imagined_X_starts)
        labels = np.array(labels)
        imagined_ep_starts = np.array(imagined_ep_starts)
        new_ep_starts = len(X) + imagined_ep_starts
        new_ep_starts = np.concatenate((ep_starts, new_ep_starts), axis=0)

        # Concatenate the imagined latent states with the true latent states
        X_mix = np.concatenate((X, imagined_X), axis=0)
        y_mix = np.concatenate((rewards_data["rewards"], labels), axis=0)
        np.savez(
            Path(cfg.data_out_dir) / skill_name / f"{cfg.data_out_prefix}_{split}_mix.npz",
            X=X_mix,
            y=y_mix,
            ep_starts=new_ep_starts,
        )


@hydra.main(version_base="1.3", config_path="../../config/rewcls", config_name="rewcls_dataset_libero")
def main(cfg: DictConfig):
    feat_data_dir = Path(cfg.feat_data_dir)
    rewards_data_dir = Path(cfg.rewards_data_dir)
    assert feat_data_dir.exists(), f"Feature data directory {feat_data_dir} does not exist."
    assert rewards_data_dir.exists(), f"Rewards data directory {rewards_data_dir} does not exist."

    data_out_dir = Path(cfg.data_out_dir)
    if not data_out_dir.exists():
        os.makedirs(data_out_dir, exist_ok=True)

    wm_ckpt_path = Path(cfg.wm.ckpt_path)
    assert wm_ckpt_path.exists(), f"WM checkpoint {wm_ckpt_path} does not exist."
    stats_path = Path(cfg.wm.stats_path)
    assert stats_path.exists(), f"Statistics file {stats_path} does not exist."

    if cfg.wm.type == "vis":
        from diwa.wm.encoder.visionwm import VisionWMObsEncoder as WMObsEncoder
    elif cfg.wm.type == "state":
        from diwa.wm.encoder.statewm import StateWMObsEncoder as WMObsEncoder
    elif cfg.wm.type == "hybrid":
        from diwa.wm.encoder.hybridwm import HybridWMObsEncoder as WMObsEncoder
    else:
        raise ValueError(f"Unknown world model type: {cfg.wm.type}")

    wm = WMObsEncoder(
        ckpt_path=wm_ckpt_path,
        device=cfg.device,
        stats_path=stats_path,
    )
    wmw = BaseWMWrapper(
        temperature=cfg.wm.temperature,
        skill="",
        device=cfg.device,
    )
    wmw.set_wm(wm.wm)
    make_rewcls_dataset(cfg, wmw, "train")
    make_rewcls_dataset(cfg, wmw, "val")


if __name__ == "__main__":
    main()
