if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import logging
import os
from pathlib import Path
import pickle
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
    This class is used to extract skills from the featurised CALVIN dataset
    (but it also accesses the raw CALVIN dataset). An object of this is iterable
    and returns one episode of the chosen skill as a dictionary.
    """

    def __init__(
        self,
        data_dir: str,
        skill_name: str,
        data_to_extract: list,
        step_len: int,
        features_file_name: str,
    ):
        self.data_dir = Path(data_dir)
        self.skill_name = skill_name
        self.data_to_extract = data_to_extract
        self.episode_lookup = self.load_file_indices(self.data_dir, self.skill_name)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()
        self.step_len = step_len
        feature_pkl = self.data_dir / Path(features_file_name)
        assert feature_pkl.exists(), f"Feature file {feature_pkl} does not exist."
        self.features = pickle.load(open(feature_pkl, "rb"))

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

    def zip_sequence_feat(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
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
        episodes = [self.features[file_idx] for file_idx in range(start_idx, end_idx + 1, self.step_len)]
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

        episode = self.zip_sequence_feat(start_file_indx, end_file_indx)

        batch = {}
        if "features" in self.data_to_extract:
            batch.update({"features": episode["features"]})

        if "rel_actions" in self.data_to_extract:
            batch.update({"rel_actions": episode["rel_actions"]})

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


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="extract_expert_demos_feat")
def make_dataset(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    skill_list = cfg.skills_list
    data_to_extract = cfg.fields_to_extract
    assert "features" in data_to_extract, "features must be included in the fields to extract."
    assert "rel_actions" in data_to_extract, "rel_actions must be included in the fields to extract."
    np.random.seed(cfg.seed)
    for split in ["training", "validation"]:
        split_dir = os.path.join(cfg.input_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Input directory {split_dir} does not exist.")
        file_name = split_to_file[split]
        logger.info(f"Processing split: {split}")
        for skill in tqdm(skill_list):
            logger.info(f"Skill: {skill}")

            output_dir = os.path.join(cfg.output_dir, skill)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            extractor = CALVINSkillExtractor(
                data_dir=split_dir,
                skill_name=skill,
                data_to_extract=data_to_extract,
                step_len=cfg.step_len,
                features_file_name=cfg.features_file_name,
            )

            states = np.array([])
            actions = np.array([])
            traj_lengths = np.array([])

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

            for idx in tqdm(ep_indices, desc=f"Processing skill: {skill}"):
                episode = extractor[idx]
                eps_len = int(episode["rel_actions"].shape[0])

                # WM Features
                features = episode["features"]

                # Actions
                rel_actions = episode["rel_actions"]

                # Append to arrays
                states = np.vstack((states, features)) if states.size else features
                actions = np.vstack((actions, rel_actions)) if actions.size else rel_actions
                traj_lengths = np.append(traj_lengths, eps_len)

            # save in robomimic format
            np.savez(
                os.path.join(output_dir, f"{file_name}.npz"),
                states=states,
                actions=actions,
                traj_lengths=traj_lengths.astype(int),
            )
    # Save the config file for reproducibility
    OmegaConf.save(
        cfg,
        os.path.join(cfg.output_dir, "config.yaml"),
    )


if __name__ == "__main__":
    make_dataset()
