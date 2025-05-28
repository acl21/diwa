import argparse
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm


def calculate_dataset_statistics(args) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_robot_obs = np.array([np.inf] * args.robot_obs_size)
    max_robot_obs = np.array([-np.inf] * args.robot_obs_size)
    if args.scene_obs_included:
        min_scene_obs = np.array([np.inf] * args.scene_obs_size)
        max_scene_obs = np.array([-np.inf] * args.scene_obs_size)

    act_min = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    act_max = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    """Parse through the whole CALVIN-style data i.e., training and validation."""
    for split in ["training", "validation"]:
        split_path = Path(input_dir) / split

        # Load episode start and end ids if needed for processing
        ep_start_end_ids = np.load(split_path / "ep_start_end_ids.npy", allow_pickle=True)
        for ep_start, ep_end in tqdm(ep_start_end_ids):
            for ep_id in tqdm(range(ep_start, ep_end + 1)):
                ep_id_str = str(ep_id).zfill(6)
                npz_file = split_path / f"episode_{ep_id_str}.npz"

                data = np.load(npz_file)

                min_robot_obs = np.minimum(min_robot_obs, data["robot_obs"])
                max_robot_obs = np.maximum(max_robot_obs, data["robot_obs"])
                if args.scene_obs_included:
                    min_scene_obs = np.minimum(min_scene_obs, data["scene_obs"])
                    max_scene_obs = np.maximum(max_scene_obs, data["scene_obs"])

                act_min = np.minimum(act_min, data["rel_actions"].min(axis=0))
                act_max = np.maximum(act_max, data["rel_actions"].max(axis=0))

    # Save the min and max for normalization
    if args.scene_obs_included:
        np.savez_compressed(
            output_dir / "statistics_minmax.npz",
            robot_obs_min=min_robot_obs,
            robot_obs_max=max_robot_obs,
            scene_obs_min=min_scene_obs,
            scene_obs_max=max_scene_obs,
            act_min_bound=act_min,
            act_max_bound=act_max,
        )
    else:
        np.savez_compressed(
            output_dir / "statistics_minmax.npz",
            robot_obs_min=min_robot_obs,
            robot_obs_max=max_robot_obs,
            act_min_bound=act_min,
            act_max_bound=act_max,
        )
    print(f"Statistics saved to {output_dir / 'statistics_minmax.npz'}")
    # Write the mean and std to a yaml file
    stats_dict = {
        "robot_obs": {
            "min": np.round(min_robot_obs, 6).tolist(),
            "max": np.round(max_robot_obs, 6).tolist(),
        },
        "act_min_bound": np.round(act_min, 6).tolist(),
        "act_max_bound": np.round(act_max, 6).tolist(),
    }
    if args.scene_obs_included:
        stats_dict["scene_obs"] = {
            "min": np.round(min_scene_obs, 6).tolist(),
            "max": np.round(max_scene_obs, 6).tolist(),
        }
    stats_file = output_dir / "statistics_minmax.yaml"
    with open(stats_file, "w") as f:
        for key, value in stats_dict.items():
            if key in ["robot_obs", "scene_obs"]:
                f.write(f"{key}:\n")
                f.write("  - _target_: calvin_agent.utils.transforms.NormalizeVectorMinMax\n")
                for sub_key, sub_value in value.items():
                    f.write(f"    {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
    # Copy the statistics_minmax.yaml file to two different locations
    shutil.copy(stats_file, output_dir / "training" / "statistics_minmax.yaml")
    shutil.copy(stats_file, output_dir / "validation" / "statistics_minmax.yaml")
    print(f"Statistics saved to {output_dir / 'statistics_minmax.yaml'}")
    print(f"Statistics saved to {output_dir / 'training' / 'statistics_minmax.yaml'}")
    print(f"Statistics saved to {output_dir / 'validation' / 'statistics_minmax.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data2/ws1/nematoli-MORSE/libero/libero_90_calvin_fmt_64/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data2/ws1/nematoli-MORSE/libero/libero_90_calvin_fmt_64/",
    )
    parser.add_argument(
        "--robot-obs-size",
        type=int,
        default=15,
        help="Size of the robot observation vector",
    )
    parser.add_argument(
        "--scene-obs-size",
        type=int,
        default=33,
        help="Size of the scene observation vector",
    )
    parser.add_argument(
        "--scene-obs-included",
        action="store_true",
        help="Include scene observation in the statistics",
        default=False,
    )
    args = parser.parse_args()
    calculate_dataset_statistics(args)
