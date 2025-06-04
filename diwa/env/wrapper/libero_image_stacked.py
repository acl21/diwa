import logging

import gym
import numpy as np

from diwa.env.wrapper.base_libero import LIBEROBaseWrapper

logger = logging.getLogger(__name__)


class LIBEROImageWrapper(LIBEROBaseWrapper):
    def __init__(
        self,
        task_name=None,
        normalization_path=None,
        render_hw=(192, 192),  # divisble by 16
        render_camera_name="rgb_static",
        max_episode_steps=None,
        seed=42,
    ):
        super().__init__(
            task_name=task_name,
            normalization_path=normalization_path,
            render_hw=render_hw,
            render_camera_name=render_camera_name,
            max_episode_steps=max_episode_steps,
            seed=seed,
        )

        self.observation_space = self.get_observation_space()

    def get_observation_space(self):
        """Returns the observation space for the environment based on the skill"""
        obs_dim = 15
        rgb_dim = 64 * 64 * 6
        return gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
                "rgb": gym.spaces.Box(low=0, high=255, shape=(rgb_dim,)),
            }
        )

    def get_obs(self):
        state_obs = self.get_state_obs()
        robot_obs = state_obs["robot_obs"]
        if self.normalize:
            robot_obs = self.normalize_obs(robot_obs)

        rgb_static, rgb_gripper = self.get_rgb_obs(size=64)
        image = np.concatenate([rgb_static, rgb_gripper], axis=0)  # C x H x W

        obs = {}
        obs["state"] = robot_obs
        obs["rgb"] = image

        return obs
