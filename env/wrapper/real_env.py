import logging
import os
from typing import Any, Dict, Tuple, Union

import gym
import gym.spaces
import numpy as np
from env.robot_io_dev.robot_io.envs.robot_env import RobotEnv
from env.robot_io_dev.robot_io.utils.utils import quat_to_euler
import torch
import hydra
from util.rotation_transformer import RotationTransformer
import imageio
import cv2

logger = logging.getLogger(__name__)


def resize_image(image, intp, resolution=64):
    """Resize an image to the target size using INTER_AREA interpolation."""
    target_size = (resolution, resolution)
    return cv2.resize(image, target_size, interpolation=intp)

def obs_dict_to_np(robot_obs):
    tcp_pos = robot_obs["tcp_pos"]
    tcp_orn = quat_to_euler(robot_obs["tcp_orn"])
    gripper_width = robot_obs["gripper_opening_width"]
    gripper_action = 1 if gripper_width > 0.06 else -1
    joint_positions = robot_obs["joint_positions"]

    return np.concatenate([tcp_pos, tcp_orn, [gripper_width], joint_positions, [gripper_action]])


class RealEnvWrapper(gym.Env):
    """
    A simple wrapper for the real robot environment.
    """

    def __init__(
        self,
        cfg = None,
        skill_name: str = "",
        normalization_path: str = "",
        max_episode_steps: int = 300,
        device: str = "cuda:0",
        load_scene_from_dataset: str = "",
        rgb_obs: bool = False,
        # Default stuff
        relative_action: bool = True,
        max_rel_pos: float = 0.08,
        max_rel_orn: float = 0.2,
        **kwargs: Any,
    ) -> None:
        
        robot = hydra.utils.instantiate(cfg.robot)
        self.env = hydra.utils.instantiate(cfg.robot_env, robot=robot)

        self.euler_to_rot6d = RotationTransformer(from_rep="euler_angles", to_rep="rotation_6d", from_convention="XYZ")
        self.skill_name = skill_name
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.relative_actions = relative_action
        self.load_scene_from_dataset = load_scene_from_dataset
        if self.load_scene_from_dataset is not None:
            file = np.load(self.load_scene_from_dataset)
            self.robot_obs = file["robot_obs"]
        
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]
        
        self.rgb_obs = rgb_obs
        
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(self.rgb_obs)

        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn

        self._t = 0
        self.video_writer = None

        logger.info(f"Initialized RealImageWrapper for device {self.device}")
        logger.info(f"Relative actions: {self.relative_actions}")

        self.frames = []
        self.gripper_frames = []
    
    @staticmethod
    def get_action_space():
        return gym.spaces.Box(
            low=-1,
            high=1,
            shape=(7,),
            dtype=np.float32,
        )
    
    @staticmethod
    def get_observation_space(rgb_obs: bool = False):
        """Returns the observation space for the environment based on the skill"""
        obs_dim = 18
        rgb_dim = 64 * 64 * 3
        return gym.spaces.Dict(
            {
                "robot_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
                "rgb_static": gym.spaces.Box(low=0, high=255, shape=(rgb_dim,)),
                "rgb_gripper": gym.spaces.Box(low=0, high=255, shape=(rgb_dim,)),
            }
        )
    
    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min
    
    def step(self, action_tensor):
        if self.normalize:
            action_tensor = self.unnormalize_action(action_tensor)
        if self.relative_actions:
            action_tensor = np.clip(action_tensor, -1, 1)
        action = np.split(action_tensor, [3, 6])
        if self.relative_actions:
            # scale actions to metric values
            action[0] *= self.max_rel_pos
            action[1] *= self.max_rel_orn
        action[2] = 1 if action[-1] > 0 else -1
        action_dict = {"motion": action, "ref": "rel" if self.relative_actions else "abs"}
        o, r, terminated, i = self.env.step(action_dict)
        
        truncated = False
        self._t += 1
        if self._t >= self.max_episode_steps:
            truncated = True
            self._t = 0
        
        if self.video_writer is not None:
            self.video_writer.append_data(self.env.render(cam="rgb_static",mode="rgb_array"))

        return self._get_obs(), r, terminated, truncated, i

    def reset(self, episode=None, robot_obs=None, target_pos=None, target_orn=None, gripper_state="open", options = {}):
        
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        
        if options is not None:
            if "video_path" in options:
                self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        if episode is not None:
            robot_obs = episode["state_info"]["robot_obs"][0]

        if robot_obs is not None:
            target_pos = robot_obs[:3]
            # From rot6d to euler
            target_orn = self.euler_to_rot6d.inverse(robot_obs[3:9])
            gripper_state = "open" if robot_obs[-1] == 1 else "closed"
            obs = self.env.reset(target_pos=target_pos, target_orn=target_orn, gripper_state=gripper_state)
        elif target_pos is not None and target_orn is not None:
            obs = self.env.reset(target_pos=target_pos, target_orn=target_orn, gripper_state=gripper_state)
        else:
            obs = self.env.reset()

        return self._get_obs(), None

    def _get_obs(self):
        obs = {}
        obs.update(self.env._get_state_obs())

        # Replace robot orientation in euler angles with rot6d
        robot_obs = obs["robot_obs"]
        new_robot_obs = np.concatenate(
            [
                robot_obs[:3],
                self.euler_to_rot6d.forward(robot_obs[3:6]),
                robot_obs[6:],
            ]
        )
        if self.normalize:
            new_robot_obs = self.normalize_obs(new_robot_obs)
        
        new_obs = {}
        new_obs["robot_obs"] = new_robot_obs

        obs = self.env._get_rgb_obs()
        # Resize the rgb_static and rgb_gripper images
        rgb_static = obs["rgb_static"]
        rgb_static = resize_image(rgb_static, cv2.INTER_AREA, resolution=64)
        self.frames.append(rgb_static)
        # H x W x C -> C x H x W
        rgb_static = np.transpose(rgb_static, (2, 0, 1))
        new_obs["rgb_static"] = rgb_static

        rgb_gripper = obs["rgb_gripper"]
        rgb_gripper = resize_image(rgb_gripper, cv2.INTER_AREA, resolution=64)
        self.gripper_frames.append(rgb_gripper)
        # H x W x C -> C x H x W
        rgb_gripper = np.transpose(rgb_gripper, (2, 0, 1))
        new_obs["rgb_gripper"] = rgb_gripper

        return new_obs