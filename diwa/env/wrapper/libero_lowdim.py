import logging

import gym
import numpy as np

from diwa.env.wrapper.base_libero import LIBEROBaseWrapper

logger = logging.getLogger(__name__)


class LIBEROLowDimWrapper(LIBEROBaseWrapper):
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
        """
        Returns the observation space of the environment.
        """
        obs_dim = 66
        return gym.spaces.Box(low=-1, high=1, shape=(obs_dim,))

    def get_obs(self):
        """
        Returns the state observation of the robot and scene.
        """
        obs = self.get_state_obs()
        robot_obs = obs["robot_obs"]
        scene_obs = obs["scene_obs"]
        obs = np.concatenate([robot_obs, scene_obs])

        if self.normalize:
            obs = self.normalize_obs(obs)

        return obs
