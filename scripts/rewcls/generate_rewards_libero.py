import os

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from diwa.env.wrapper.libero_lowdim import LIBEROLowDimWrapper


def get_env_rewards(env, data):
    traj_lengths = data["traj_lengths"]
    ep_ends = np.cumsum(traj_lengths)
    ep_starts = np.concatenate(([0], ep_ends[:-1]))
    states = data["states"][:, 15:]  # ignore robot obs

    rewards = []
    for start, end in tqdm(zip(ep_starts, ep_ends), desc="Episodes"):
        env.reset()
        env.set_init_state(states[start])
        for i in range(start, end):
            action = data["actions"][i]
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
    rewards = np.array(rewards)
    return rewards


@hydra.main(config_path="../../config/rewcls", config_name="generate_rewards_libero")
def main(cfg: DictConfig):
    for split in tqdm(["train", "val"], desc="Split"):
        print(f"Processing split: {split}")
        for task_name in tqdm(cfg.tasks_list, desc="Task"):
            print(f"Processing task: {task_name}")
            env = LIBEROLowDimWrapper(task_name=task_name, max_episode_steps=1000)
            if not os.path.exists(f"{cfg.input_dir}/{task_name}/{split}.npz"):
                print(f"Input file {cfg.input_dir}/{task_name}/{split}.npz does not exist. Skipping.")
                continue
            data = np.load(
                f"{cfg.input_dir}/{task_name}/{split}.npz",
                allow_pickle=True,
            )
            rewards = get_env_rewards(env, data)
            np.savez(
                f"{cfg.output_dir}/{task_name}/{split}_rewards.npz",
                rewards=rewards,
            )


if __name__ == "__main__":
    main()
