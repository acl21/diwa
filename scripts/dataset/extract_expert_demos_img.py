if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


split_to_file = {
    "training": "train",
    "validation": "val",
}


class CALVINSkillExtractor:
    """
    This class is used to extract skills from the raw CALVIN dataset.
    An object of this is iterable and returns one episode of the chosen skill
    as a dictionary.
    """

    def __init__(
        self,
        data_dir: str,
        skill_name: str,
        data_to_extract: list,
        step_len: int,
    ):
        self.data_dir = Path(data_dir)
        self.skill_name = skill_name
        self.data_to_extract = data_to_extract
        self.episode_lookup = self.load_file_indices(self.data_dir, self.skill_name)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()
        self.step_len = step_len

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        return self.get_sequences(idx)

    def lookup_naming_pattern(self):
        it = os.scandir(self.data_dir)
        while True:
            filename = Path(next(it))
            if "npz" in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def zip_sequence(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame
        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        episodes = [
            load_npz(self.get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx + 1, self.step_len)
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key, _ in episodes[0].items()}
        return episode

    def get_sequences(self, idx: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """
        info_indx = self.episode_lookup[idx]
        start_file_indx = info_indx[0]
        end_file_indx = info_indx[1]

        episode = self.zip_sequence(start_file_indx, end_file_indx)

        batch = {}
        if "robot_obs" in self.data_to_extract:
            batch.update({"robot_obs": episode["robot_obs"]})

        if "scene_obs" in self.data_to_extract:
            batch.update({"scene_obs": episode["scene_obs"]})

        if "rel_actions" in self.data_to_extract:
            batch.update({"rel_actions": episode["rel_actions"]})

        if "rgb_gripper" in self.data_to_extract:
            batch.update({"rgb_gripper": episode["rgb_gripper"].astype(np.int64)})

        if "rgb_static" in self.data_to_extract:
            batch.update({"rgb_static": episode["rgb_static"].astype(np.int64)})

        return batch

    def load_file_indices(self, data_dir: Path, skill: str) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used for loading the episodes
        parameters
        ----------
        data_dir:               absolute path of the directory containing the datasets
        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert data_dir.is_dir()
        skill_name = skill

        episode_lookup = []

        file_name = data_dir / "lang_annotations" / "auto_lang_ann.npy"
        data = np.load(file_name, allow_pickle=True).reshape(-1)[0]

        all_eps_idx_part_task = [i for (i, v) in enumerate(data["language"]["task"]) if v == skill_name]
        all_eps_start_end_part_task = [data["info"]["indx"][i] for i in all_eps_idx_part_task]

        for i in range(len(all_eps_start_end_part_task)):
            episode_lookup.append(all_eps_start_end_part_task[i])

        logger.info(f"Found {len(episode_lookup)} demonstrations of skill {skill_name}.")
        return episode_lookup


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="extract_expert_demos_img")
def make_dataset(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    skills_list = cfg.skills_list
    data_to_extract = cfg.fields_to_extract
    assert "robot_obs" in data_to_extract, "robot_obs must be in fields_to_extract"
    assert "rel_actions" in data_to_extract, "rel_actions must be in fields_to_extract"
    assert (
        "rgb_static" in data_to_extract or "rgb_gripper" in data_to_extract
    ), "At least one of rgb_static or rgb_gripper must be in fields_to_extract"
    np.random.seed(cfg.seed)
    for split in ["training", "validation"]:
        split_dir = os.path.join(cfg.input_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Input directory {split_dir} does not exist.")
        file_name = split_to_file[split]
        logger.info(f"Processing split: {split}")
        for skill in tqdm(skills_list):
            logger.info(f"Skill: {skill}")

            output_dir = os.path.join(cfg.output_dir, skill)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            extractor = CALVINSkillExtractor(
                data_dir=split_dir,
                skill_name=skill,
                data_to_extract=data_to_extract,
                step_len=cfg.step_len,
            )

            if len(extractor) == 0:
                logger.warning(f"No episodes found for skill {skill} in split {split}. Skipping...")
                continue

            obs_dim = extractor[0]["robot_obs"].shape[1]
            if "scene_obs" in data_to_extract:
                obs_dim += extractor[0]["scene_obs"].shape[1]
            action_dim = extractor[0]["rel_actions"].shape[1]

            images = np.array([])
            states = np.array([])
            actions = np.array([])
            traj_lengths = np.array([])

            obs_min = np.inf * np.ones((obs_dim))
            obs_max = -np.inf * np.ones((obs_dim))
            action_min = np.inf * np.ones((action_dim))
            action_max = -np.inf * np.ones((action_dim))

            if cfg.save_starts:
                robot_obs_starts = np.array([])
                scene_obs_starts = np.array([])
                save_start_length = cfg.save_start_length

            if cfg.n_episodes < len(extractor) and cfg.n_episodes > 0 and "val" not in split:
                # randomly sample a subset of episodes
                ep_indices = np.random.choice(len(extractor), cfg.n_episodes, replace=False)
                logger.info(
                    f"Extracting only {cfg.n_episodes} episodes for skill {skill} from {len(extractor)} total episodes."
                )
                # save episode indices for in a text file for quick access
                with open(os.path.join(output_dir, f"{file_name}_ep_indices.txt"), "w") as f:
                    for idx in sorted(ep_indices):
                        f.write(f"{idx}\n")
            else:
                # use all episodes
                ep_indices = np.arange(len(extractor))
                logger.info(f"Extracting all {len(extractor)} episodes for skill {skill}.")

            for idx in tqdm(ep_indices, desc=f"Processing skill: {skill}", total=len(ep_indices)):
                episode = extractor[idx]
                eps_len = int(episode["robot_obs"].shape[0])

                # Image observations
                rgb_static = episode["rgb_static"]
                if "rgb_gripper" in data_to_extract:
                    rgb_gripper = episode["rgb_gripper"]
                    # Concatenate images on the last axis
                    image = np.concatenate((rgb_static, rgb_gripper), axis=-1)
                    # image = rgb_static
                    image = np.transpose(image, (0, 3, 1, 2))  # (T, C, H, W)
                else:
                    image = np.transpose(rgb_static, (0, 3, 1, 2))

                # Low-dim observations
                robot_obs = episode["robot_obs"]
                if "scene_obs" in data_to_extract:
                    scene_obs = episode["scene_obs"]

                    # Concatenate robot and scene observations to form "state" observations
                    state_obs = np.concatenate((robot_obs, scene_obs), axis=1)
                else:
                    state_obs = robot_obs

                # Actions
                rel_actions = episode["rel_actions"]

                obs_min = np.minimum(obs_min, np.min(state_obs, axis=0))
                obs_max = np.maximum(obs_max, np.max(state_obs, axis=0))
                action_min = np.minimum(action_min, np.min(rel_actions, axis=0))
                action_max = np.maximum(action_max, np.max(rel_actions, axis=0))

                # Append to arrays
                images = np.vstack((images, image)) if images.size else image
                states = np.vstack((states, state_obs)) if states.size else state_obs
                actions = np.vstack((actions, rel_actions)) if actions.size else rel_actions
                traj_lengths = np.append(traj_lengths, eps_len)

                if cfg.save_starts:
                    robot_obs_starts = (
                        np.vstack((robot_obs_starts, robot_obs[:save_start_length]))
                        if robot_obs_starts.size
                        else robot_obs[:save_start_length]
                    )
                    if "scene_obs" in data_to_extract:
                        scene_obs_starts = (
                            np.vstack((scene_obs_starts, scene_obs[:save_start_length]))
                            if scene_obs_starts.size
                            else scene_obs[:save_start_length]
                        )

            # Normalize data
            if cfg.normalize:
                if "val" in split:
                    logger.info("Loading normalization stats from the output directory...")
                    # load normalization stats from the save directory
                    # assuming those stats are calculated on the training data
                    normalization_stats = np.load(os.path.join(output_dir, "normalization.npz"))
                    obs_min = normalization_stats["obs_min"]
                    obs_max = normalization_stats["obs_max"]
                    action_min = normalization_stats["action_min"]
                    action_max = normalization_stats["action_max"]
                else:
                    np.savez(
                        os.path.join(output_dir, "normalization.npz"),
                        obs_min=obs_min,
                        obs_max=obs_max,
                        action_min=action_min,
                        action_max=action_max,
                    )
                logger.info("Normalizing data...")
                states = 2 * (states - obs_min) / (obs_max - obs_min + 1e-6) - 1
                actions = 2 * (actions - action_min) / (action_max - action_min + 1e-6) - 1

            # saving in robomimic format
            np.savez(
                os.path.join(output_dir, f"{file_name}.npz"),
                images=images,
                states=states,
                actions=actions,
                traj_lengths=traj_lengths.astype(int),
            )

            # Saving start states helps with environment initialization
            if cfg.save_starts:
                prefix = [x for x in cfg.fields_to_extract if x in ["robot_obs", "scene_obs"]]
                prefix = "_".join(prefix)
                dataset_type_str = f"{file_name}_{prefix}_starts_{cfg.save_start_length}"
                # save robot and scene obs
                np.savez(
                    os.path.join(output_dir, f"{dataset_type_str}.npz"),
                    robot_obs=robot_obs_starts,
                    scene_obs=scene_obs_starts,
                )
    # Save the config file for reproducibility
    OmegaConf.save(
        cfg,
        os.path.join(cfg.output_dir, "config.yaml"),
    )


if __name__ == "__main__":
    make_dataset()
