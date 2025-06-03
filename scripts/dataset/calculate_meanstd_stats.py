import argparse
from pathlib import Path
import shutil

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="calculate_stats")
def calculate_dataset_statistics(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Do mean-std normalization
    running_mean_robot_obs = np.zeros(cfg.robot_obs_size)
    running_var_robot_obs = np.zeros(cfg.robot_obs_size)
    if cfg.scene_obs_included:
        running_mean_scene_obs = np.zeros(cfg.scene_obs_size)
        running_var_scene_obs = np.zeros(cfg.scene_obs_size)

    act_min = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    act_max = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    counter = 0
    """Parse through the whole CALVIN-style data i.e., training and validation."""
    for split in ["training", "validation"]:
        split_path = Path(input_dir) / split

        # Load episode start and end ids if needed for processing
        ep_start_end_ids = np.load(split_path / "ep_start_end_ids.npy", allow_pickle=True)
        for ep_start, ep_end in tqdm(ep_start_end_ids):
            for ep_id in tqdm(range(ep_start, ep_end + 1)):
                ep_id_str = str(ep_id).zfill(cfg.episode_id_length)
                npz_file = split_path / f"episode_{ep_id_str}.npz"

                data = np.load(npz_file)

                counter += 1

                robot_obs = data["robot_obs"]
                delta = robot_obs - running_mean_robot_obs
                running_mean_robot_obs += delta / counter
                running_var_robot_obs += delta * (robot_obs - running_mean_robot_obs)

                if cfg.scene_obs_included:
                    scene_obs = data["scene_obs"]
                    delta = scene_obs - running_mean_scene_obs
                    running_mean_scene_obs += delta / counter
                    running_var_scene_obs += delta * (scene_obs - running_mean_scene_obs)

                act_min = np.minimum(act_min, data["rel_actions"].min(axis=0))
                act_max = np.maximum(act_max, data["rel_actions"].max(axis=0))

    # Calculate the running std
    running_std_robot_obs = np.sqrt(running_var_robot_obs / counter)
    if cfg.scene_obs_included:
        running_std_scene_obs = np.sqrt(running_var_scene_obs / counter)

    # Save the running mean and std for normalization
    if cfg.scene_obs_included:
        np.savez_compressed(
            output_dir / "statistics.npz",
            robot_obs_mean=running_mean_robot_obs,
            robot_obs_std=running_std_robot_obs,
            scene_obs_mean=running_mean_scene_obs,
            scene_obs_std=running_std_scene_obs,
            act_min_bound=act_min,
            act_max_bound=act_max,
        )
    else:
        np.savez_compressed(
            output_dir / "statistics.npz",
            robot_obs_mean=running_mean_robot_obs,
            robot_obs_std=running_std_robot_obs,
            act_min_bound=act_min,
            act_max_bound=act_max,
        )
    print(f"Statistics saved to {output_dir / 'statistics.npz'}")
    # Write the mean and std to a yaml file
    stats_dict = {
        "robot_obs": {
            "mean": np.round(running_mean_robot_obs, 6).tolist(),
            "std": np.round(running_std_robot_obs, 6).tolist(),
        },
        "act_min_bound": np.round(act_min, 6).tolist(),
        "act_max_bound": np.round(act_max, 6).tolist(),
    }
    if cfg.scene_obs_included:
        stats_dict["scene_obs"] = {
            "mean": np.round(running_mean_scene_obs, 6).tolist(),
            "std": np.round(running_std_scene_obs, 6).tolist(),
        }
    stats_file = output_dir / "statistics.yaml"
    with open(stats_file, "w") as f:
        for key, value in stats_dict.items():
            if key in ["robot_obs", "scene_obs"]:
                f.write(f"{key}:\n")
                f.write("  - _target_: calvin_agent.utils.transforms.NormalizeVector\n")
                for sub_key, sub_value in value.items():
                    f.write(f"    {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
    # Copy the statistics.yaml file to two different locations
    shutil.copy(stats_file, output_dir / "training" / "statistics.yaml")
    shutil.copy(stats_file, output_dir / "validation" / "statistics.yaml")
    print(f"Statistics saved to {output_dir / 'statistics.yaml'}")
    print(f"Statistics saved to {output_dir / 'training' / 'statistics.yaml'}")
    print(f"Statistics saved to {output_dir / 'validation' / 'statistics.yaml'}")


if __name__ == "__main__":
    calculate_dataset_statistics()
