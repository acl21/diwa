import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from diwa.env.wrapper.libero_image import LIBEROImageWrapper


def get_init_obs(env):
    init_states = env.init_states
    rgb_statics = []
    rgb_grippers = []
    robot_obs = []

    for state in tqdm(init_states, desc="Init States"):
        env.reset()
        env.set_init_state(state)
        obs = env.get_obs()
        robot_obs.append(obs["robot_obs"])
        rgb_statics.append(obs["rgb_static"])
        rgb_grippers.append(obs["rgb_gripper"])
    robot_obs = np.array(robot_obs)
    rgb_statics = np.array(rgb_statics)
    rgb_grippers = np.array(rgb_grippers)
    return robot_obs, rgb_statics, rgb_grippers


@hydra.main(config_path="../../config/dataset", config_name="generate_init_obs_libero")
def main(cfg: DictConfig):
    for task_name in tqdm(cfg.tasks_list, desc="Task"):
        print(f"Processing task: {task_name}")
        env = LIBEROImageWrapper(task_name=task_name, max_episode_steps=1000)
        robot_obs, rgb_statics, rgb_grippers = get_init_obs(env)
        np.savez(
            f"{cfg.output_dir}/{task_name}/init_obs.npz",
            robot_obs=robot_obs,
            rgb_statics=rgb_statics,
            rgb_grippers=rgb_grippers,
        )


if __name__ == "__main__":
    main()
