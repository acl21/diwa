import logging

import gym
import numpy as np

from diwa.env.utils.calvin_helpers import replace_euler_with_rot6d
from diwa.env.wrapper.base_calvin import CALVINBaseWrapper

logger = logging.getLogger(__name__)


class CALVINLowDimWrapper(CALVINBaseWrapper):
    def __init__(
        self,
        cfg,
        skill_name=None,
        clamp_obs=False,
        normalization_path=None,
        render_hw=(192, 192),  # divisble by 16
        render_camera_name="rgb_static",
        max_episode_steps=None,
        load_scene_from_dataset="",  # path to npz file
        rand_sample_size=200,
        device="cuda:0",
    ):
        super().__init__(
            cfg,
            skill_name=skill_name,
            clamp_obs=clamp_obs,
            normalization_path=normalization_path,
            render_hw=render_hw,
            render_camera_name=render_camera_name,
            max_episode_steps=max_episode_steps,
            load_scene_from_dataset=load_scene_from_dataset,
            rand_sample_size=rand_sample_size,
            device=device,
        )

        self.observation_space = self.get_observation_space()

    def get_observation_space(self):
        """Returns the observation space for the environment based on the skill"""
        obs_dim = 51  # 18 + 33
        return gym.spaces.Box(low=-1, high=1, shape=(obs_dim,))

    def get_obs(self):
        obs = self.get_state_obs()
        robot_obs = obs["robot_obs"]
        scene_obs = obs["scene_obs"]
        robot_obs = replace_euler_with_rot6d(self.rot_transformer, robot_obs, type="robot")
        scene_obs = replace_euler_with_rot6d(self.rot_transformer, scene_obs, type="scene")

        obs = np.concatenate([robot_obs, scene_obs])
        if self.normalize:
            obs = self.normalize_obs(obs)
        return obs
