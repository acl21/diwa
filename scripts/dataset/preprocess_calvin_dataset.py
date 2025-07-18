from pathlib import Path
import shutil

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from diwa.env.utils.calvin_helpers import resize_image


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="preprocess_calvin")
def process_dataset(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    keys = cfg.desired_keys
    if "rgb_static" in keys or "rgb_gripper" in keys:
        img_size = cfg.desired_resolution
        if cfg.rgb_interpolation == "Area":
            rgb_interpolation = cv2.INTER_AREA
        elif cfg.rgb_interpolation == "Cubic":
            rgb_interpolation = cv2.INTER_CUBIC
        else:
            raise ValueError(f"Invalid interpolation method: {cfg.rgb_interpolation}")
    if cfg.rot6d:
        from lumos.utils.rotation_transformer import RotationTransformer

        rot_transform = RotationTransformer(from_rep="euler_angles", to_rep="rotation_6d", from_convention="XYZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    """Process the Calvin dataset and create a smaller version of it."""
    for split in ["training", "validation"]:
        split_path = Path(input_dir) / split
        output_split_path = Path(output_dir) / split
        output_split_path.mkdir(parents=True, exist_ok=True)

        # Copy "lang_annotations" directory if it exists
        lang_annotations_path = split_path / "lang_annotations"
        if lang_annotations_path.exists():
            new_lang_annotations_path = output_split_path / "lang_annotations"
            shutil.copytree(lang_annotations_path, new_lang_annotations_path, dirs_exist_ok=True)
        else:
            print(f"Warning: 'lang_annotations' directory not found in {split_path}. Skipping copy.")

        # Copy ep_start_end_ids.npy from original folder to new folder
        orig_ep_start_end_ids = split_path / "ep_start_end_ids.npy"
        new_ep_start_end_ids = output_split_path / "ep_start_end_ids.npy"

        shutil.copy(orig_ep_start_end_ids, new_ep_start_end_ids)

        # Copy statistics.yaml from original folder to new folder
        orig_stats = split_path / "statistics.yaml"
        if orig_stats.exists():
            new_stats = output_split_path / "statistics.yaml"
            shutil.copy(orig_stats, new_stats)
        else:
            print(f"Warning: 'statistics.yaml' not found in {split_path}. Skipping copy.")

        # Iterate over .npz files in the directory
        for npz_file in tqdm(split_path.glob("episode_*.npz"), desc=f"Processing {split} data"):
            data = np.load(npz_file)
            extracted_data = {}

            for key in keys:
                if key in data.keys():
                    if key.startswith("rgb"):
                        extracted_data[key] = resize_image(data[key], intp=rgb_interpolation, resolution=img_size)
                    elif key.startswith("depth"):
                        extracted_data[key] = resize_image(data[key], intp=cv2.INTER_NEAREST, resolution=img_size)
                    elif key == "robot_obs":
                        if cfg.rot6d:
                            robot_obs = data[key]
                            euler_angles = robot_obs[3:6]
                            rotation_6d = rot_transform.forward(euler_angles)
                            extracted_data[key] = np.concatenate([robot_obs[:3], rotation_6d, robot_obs[6:]])
                        else:
                            extracted_data[key] = data[key]
                    elif key == "scene_obs":
                        if cfg.rot6d:
                            scene_obs = data[key]
                            # red, blue, pink blocks respectively
                            indices = [[9, 10, 11], [15, 16, 17], [21, 22, 23]]
                            rotation_6ds = []
                            for idx in indices:
                                euler_angles = scene_obs[idx]
                                rotation_6d = rot_transform.forward(euler_angles)
                                rotation_6ds.append(rotation_6d)
                            extracted_data[key] = np.concatenate(
                                [
                                    scene_obs[:9],
                                    rotation_6ds[0],
                                    scene_obs[12:15],
                                    rotation_6ds[1],
                                    scene_obs[18:21],
                                    rotation_6ds[2],
                                ],
                            )
                        else:
                            extracted_data[key] = data[key]
                    else:
                        extracted_data[key] = data[key]
                else:
                    print(f"Key {key} not found in {npz_file}")

            # Prepare the filename for the output file
            output_file = output_split_path / npz_file.name
            np.savez_compressed(output_file, **extracted_data)


if __name__ == "__main__":
    process_dataset()
