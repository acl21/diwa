import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


def balance_rewcls_dataset(cfg):
    """
    Balances the rewcls dataset (by subsampling from each episode) for the reward classifier.
    Note: The dataset is already sorted (seq. of episode trajectories).
    """
    for skill_name in cfg.skills_list:
        data_path = Path(cfg.rewcls_data_dir) / skill_name / f"{cfg.rewcls_data_name}.npz"
        data = np.load(data_path, allow_pickle=True)

        ep_starts = data["ep_starts"]

        filtered_X = []
        filtered_y = []
        for episode_idx in range(1, len(ep_starts)):
            start_idx = ep_starts[episode_idx - 1]
            end_idx = ep_starts[episode_idx]

            X = data["X"][start_idx:end_idx]
            y = data["y"][start_idx:end_idx]

            # Give me the indices where the class is 0
            class_0_indices = np.where(y == 0)[0]
            # Choose the n_random frames from the episode (Class 0)
            if len(X) > cfg.n_random:
                if cfg.n_random > len(class_0_indices):
                    n_random = len(class_0_indices)
                else:
                    n_random = cfg.n_random
                idxs = np.random.choice(len(class_0_indices), n_random, replace=False)
                idxs = np.sort(idxs)
                filtered_X.extend(X[idxs])
                filtered_y.extend(y[idxs])

            # Keep all the frame from the episode (Class 1)
            class_1_indices = np.where(y == 1)[0]
            filtered_X.extend(X[class_1_indices])
            filtered_y.extend(y[class_1_indices])

        start_idx = ep_starts[-1]
        end_idx = len(data["X"])
        X = data["X"][start_idx:end_idx]
        y = data["y"][start_idx:end_idx]

        # Give me the indices where the class is 0
        class_0_indices = np.where(y == 0)[0]
        # Choose the n_random frames from the episode (Class 0)
        if len(X) > cfg.n_random:
            if cfg.n_random > len(class_0_indices):
                n_random = len(class_0_indices)
            else:
                n_random = cfg.n_random
            idxs = np.random.choice(len(class_0_indices), n_random, replace=False)
            idxs = np.sort(idxs)
            filtered_X.extend(X[idxs])
            filtered_y.extend(y[idxs])
        # Keep all the frame from the episode (Class 1)
        class_1_indices = np.where(y == 1)[0]
        filtered_X.extend(X[class_1_indices])
        filtered_y.extend(y[class_1_indices])
        # Convert to numpy arrays

        np.savez(
            Path(cfg.data_out_dir) / skill_name / f"{cfg.data_out_name}.npz",
            X=filtered_X,
            y=filtered_y,
        )
        print(f"Saved balanced dataset for {skill_name}.")
    print("Done.")


@hydra.main(version_base="1.3", config_path="../../config/rewcls", config_name="balance_rewcls_dataset")
def main(cfg: DictConfig):
    rewcls_data_dir = Path(cfg.rewcls_data_dir)
    assert rewcls_data_dir.exists(), f"RewCls data directory {rewcls_data_dir} does not exist."

    data_out_dir = Path(cfg.data_out_dir)
    if not data_out_dir.exists():
        os.makedirs(data_out_dir, exist_ok=True)

    balance_rewcls_dataset(cfg)


if __name__ == "__main__":
    main()
