import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from diwa.env.wrapper.calvin_lowdim import CALVINLowDimWrapper
from diwa.utils.config_fix import calvin_config_fixer


def get_env_rewards(env, data):
    traj_lengths = data["traj_lengths"]
    ep_ends = np.cumsum(traj_lengths)
    ep_starts = np.concatenate(([0], ep_ends[:-1]))
    robot_obs = data["states"][:, :15]  # robot obs
    scene_obs = data["states"][:, 15:]  # ignore robot obs
    scene_obs[:, 4] = np.round(scene_obs[:, 4]).astype(int).astype(float)
    scene_obs[:, 5] = np.round(scene_obs[:, 5]).astype(int).astype(float)
    rewards = []
    for start, end in tqdm(zip(ep_starts, ep_ends), desc="Episodes"):
        env.reset(robot_obs=robot_obs[start], scene_obs=scene_obs[start])
        for i in range(start, end):
            action = data["actions"][i]
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
    rewards = np.array(rewards)
    return rewards


@hydra.main(config_path="../../config/rewcls", config_name="generate_rewards")
def main(cfg: DictConfig):
    cfg = calvin_config_fixer(cfg, env_cfg_name="env_cfg")
    env = CALVINLowDimWrapper(
        cfg.env_cfg,
        skill_name="",
        max_episode_steps=200,  # Set a reasonable max episode length
        load_scene_from_dataset=None,
    )
    for split in tqdm(["train", "val"], desc="Split"):
        print(f"Processing split: {split}")
        for skill_name in tqdm(cfg.skills_list, desc="Skill"):
            print(f"Processing skill: {skill_name}")
            env.skill_name = skill_name
            data = np.load(
                f"{cfg.input_dir}/{skill_name}/{split}.npz",
                allow_pickle=True,
            )
            rewards = get_env_rewards(env, data)
            np.savez(
                f"{cfg.output_dir}/{skill_name}/{split}_rewards.npz",
                rewards=rewards,
            )


if __name__ == "__main__":
    main()
