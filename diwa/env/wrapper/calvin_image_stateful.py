import logging

import gym
import numpy as np

from diwa.env.utils.calvin_helpers import replace_euler_with_rot6d
from diwa.env.wrapper.base_calvin import CALVINBaseWrapper

logger = logging.getLogger(__name__)


class CALVINImageWrapper(CALVINBaseWrapper):
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
        rgb_dim = 64 * 64 * 3
        return gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
                "rgb_static": gym.spaces.Box(low=0, high=255, shape=(rgb_dim,)),
                "rgb_gripper": gym.spaces.Box(low=0, high=255, shape=(rgb_dim,)),
            }
        )

    def get_obs(self):
        state_obs = self.get_state_obs()
        robot_obs = state_obs["robot_obs"]
        scene_obs = state_obs["scene_obs"]
        robot_obs = replace_euler_with_rot6d(self.rot_transformer, robot_obs, type="robot")
        scene_obs = replace_euler_with_rot6d(self.rot_transformer, scene_obs, type="scene")
        state_obs = np.concatenate([robot_obs, scene_obs])
        if self.normalize:
            state_obs = self.normalize_obs(state_obs)

        rgb_static, rgb_gripper = self.get_rgb_obs(size=64)

        obs = {}
        obs["state"] = state_obs
        obs["rgb_static"] = rgb_static
        obs["rgb_gripper"] = rgb_gripper

        return obs
