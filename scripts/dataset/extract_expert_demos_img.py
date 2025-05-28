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

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


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


def make_dataset(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    skill_list = args.skills_list
    data_to_extract = args.fields_to_extract
    assert "robot_obs" in data_to_extract, "robot_obs must be in fields_to_extract"
    assert "rel_actions" in data_to_extract, "rel_actions must be in fields_to_extract"
    assert (
        "rgb_static" in data_to_extract or "rgb_gripper" in data_to_extract
    ), "At least one of rgb_static or rgb_gripper must be in fields_to_extract"
    np.random.seed(args.seed)
    for skill in tqdm(skill_list):
        logger.info(f"Extracting data for skill: {skill}")

        save_dir = os.path.join(args.save_dir, skill)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        extractor = CALVINSkillExtractor(
            data_dir=args.load_path,
            skill_name=skill,
            data_to_extract=data_to_extract,
            step_len=args.step_len,
        )

        obs_dim = extractor[0]["robot_obs"].shape[1] + extractor[0]["scene_obs"].shape[1]
        action_dim = extractor[0]["rel_actions"].shape[1]

        states = np.array([])
        actions = np.array([])
        traj_lengths = np.array([])

        obs_min = np.inf * np.ones((obs_dim))
        obs_max = -np.inf * np.ones((obs_dim))
        action_min = np.inf * np.ones((action_dim))
        action_max = -np.inf * np.ones((action_dim))

        if args.save_starts:
            robot_obs_starts = np.array([])
            scene_obs_starts = np.array([])
            save_start_length = args.save_start_length

        if args.n_episodes < len(extractor):
            # randomly sample a subset of episodes
            ep_indices = np.random.choice(len(extractor), args.n_episodes, replace=False)
        else:
            # use all episodes
            ep_indices = np.arange(len(extractor))
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
            states = np.vstack((states, state_obs)) if states.size else state_obs
            actions = np.vstack((actions, rel_actions)) if actions.size else rel_actions
            traj_lengths = np.append(traj_lengths, eps_len)

            if args.save_starts:
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
        if args.normalize:
            if args.is_val:
                # load normalization stats from the save directory
                # assuming those stats are calculated on the training data
                normalization_stats = np.load(os.path.join(save_dir, "normalization.npz"))
                obs_min = normalization_stats["obs_min"]
                obs_max = normalization_stats["obs_max"]
                action_min = normalization_stats["action_min"]
                action_max = normalization_stats["action_max"]
            else:
                np.savez(
                    os.path.join(save_dir, "normalization.npz"),
                    obs_min=obs_min,
                    obs_max=obs_max,
                    action_min=action_min,
                    action_max=action_max,
                )
            states = 2 * (states - obs_min) / (obs_max - obs_min + 1e-6) - 1
            actions = 2 * (actions - action_min) / (action_max - action_min + 1e-6) - 1

        np.savez(
            os.path.join(save_dir, args.file_name),
            states=states,
            actions=actions,
            traj_lengths=traj_lengths.astype(int),
        )

        # Saving start states helps with environment initialization
        if args.save_starts:
            dataset_type_str = (
                f"robot_scene_obs_starts{save_start_length}"
                if not args.is_val
                else f"robot_scene_obs_starts_val{save_start_length}"
            )
            # save robot and scene obs
            np.savez(
                os.path.join(save_dir, f"{dataset_type_str}.npz"),
                robot_obs=robot_obs_starts,
                scene_obs=scene_obs_starts,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        type=str,
        default="./data/calvin/task_D_D_rgb64_rot6d/training/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/expert/calvin-feat-vision-2M-25/",
    )
    parser.add_argument(
        "--skills_list",
        type=str,
        nargs="+",
        default=[
            "open_drawer",
            "move_slider_left",
            "push_pink_block_right",
            "lift_pink_block_table",
            "close_drawer",
            "turn_on_lightbulb",
            "turn_off_lightbulb",
            "move_slider_right",
            "turn_on_led",
            "turn_off_led",
        ],
    )
    parser.add_argument(
        "--fields_to_extract",
        type=str,
        nargs="+",
        default=["robot_obs", "rel_actions", "rgb_static", "rgb_gripper"],
    )
    parser.add_argument("--n_episodes", type=int, default=50)  # use -1 for all episodes
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step_len", type=int, default=1)
    parser.add_argument("--file_name", type=str, default="train.npz")  # with extension
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--is_val", action="store_true", default=False)
    args = parser.parse_args()

    print(args)

    make_dataset(args)
