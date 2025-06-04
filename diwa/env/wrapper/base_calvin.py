if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import logging
import sys

import cv2
import gym
import hydra
import imageio
import numpy as np
import pybullet as p

from calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
from calvin_env.calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
from diwa.env.utils.calvin_helpers import (
    replace_rot6d_with_euler,
    sample_random_block_orn,
    sample_random_block_pos,
    sample_random_robot_orn,
    sample_random_robot_pos,
    sample_random_scene_obs,
)
from diwa.utils.rotation_transformer import RotationTransformer

logger = logging.getLogger(__name__)


def resize_image(image, intp, resolution=64):
    """Resize an image to the target size using INTER_AREA interpolation."""
    target_size = (resolution, resolution)
    return cv2.resize(image, target_size, interpolation=intp)


class CALVINBaseWrapper(PlayTableSimEnv):
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
        pt_cfg = {**cfg.env}
        pt_cfg.pop("_target_", None)
        pt_cfg.pop("_recursive_", None)
        self.robot_cfg = cfg["robot"]
        self.scene_cfg = cfg["scene"]
        self.cameras_c = cfg["cameras"]
        if "cuda" in device:
            self.set_egl_device(device)
        super().__init__(**pt_cfg)

        self.skill_name = skill_name

        self.action_space = self.get_action_space()
        self.tasks = hydra.utils.instantiate(cfg.tasks)
        self.max_episode_steps = max_episode_steps
        self.video_writer = None
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.render_mode = "rgb_array"
        self.load_scene_from_dataset = load_scene_from_dataset
        self.rand_sample_size = rand_sample_size
        if self.load_scene_from_dataset is not None:
            file = np.load(self.load_scene_from_dataset)
            self.robot_obs = file["robot_obs"]
            self.scene_obs = file["scene_obs"]
        else:
            self.robot_obs = self.random_sample_robot_obs(self.rand_sample_size)
            self.scene_obs = self.random_sample_scene_obs(self.rand_sample_size)

        self.clamp_obs = clamp_obs
        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        self._t = 0

        self.rot_transformer = RotationTransformer(from_rep="euler_angles", to_rep="rotation_6d", from_convention="XYZ")

    @staticmethod
    def get_action_space():
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    def random_sample_scene_obs(self, size):
        rand_block_pos = sample_random_block_pos(size)
        rand_block_orn = sample_random_block_orn(size)

        scene_obs = []
        for i in range(size):
            self.scene.reset()
            scene_obs.append(self.scene.get_obs())
        scene_obs = np.array(scene_obs)

        # random sample scene obs
        scene_obs = sample_random_scene_obs(scene_obs, self.skill_name, size, rand_block_pos, rand_block_orn)

        return scene_obs

    def random_sample_robot_obs(self, size):
        rand_robot_pos = sample_random_robot_pos(size)
        rand_robot_orn = sample_random_robot_orn(size)
        # Gripper is always open
        robot_gripper_width = 0.08
        robot_gripper_action = 1

        robot_obs = []
        for i in range(size):
            robot_jnt_pos = self.robot.mixed_ik.get_ik(rand_robot_pos[1], p.getQuaternionFromEuler(rand_robot_orn[i]))
            robot_obs.append(
                np.concatenate(
                    [
                        rand_robot_pos[i],
                        rand_robot_orn[i],
                        [robot_gripper_width],
                        robot_jnt_pos,
                        [robot_gripper_action],
                    ]
                )
            )
        return np.array(robot_obs)

    def normalize_obs(self, obs):
        obs = 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()
        task_filter = [self.skill_name]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return self.skill_name in task_info

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        reward = int(self._success()) * 1
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self, reward):
        """Indicates if the robot has reached a terminal state"""
        success = reward > 0
        d_info = {"success": success}
        return success, d_info

    def _truncation(self):
        """Indicates if the robot has reached a terminal state"""
        truncated = self._t >= self.max_episode_steps
        return truncated, {"truncated": truncated}

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, terminated, truncated, info
        """
        env_action = action.copy()
        if self.normalize:
            env_action = self.unnormalize_action(env_action)
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1
        self.robot.apply_action(env_action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)

        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        self._t += 1
        terminated, d_info = self._termination(reward)
        truncated, t_info = self._truncation()
        info.update(r_info)
        info.update(d_info)
        info.update(t_info)

        if self.video_writer is not None:
            self.video_writer.append_data(self.render())

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        robot_obs=None,
        scene_obs=None,
        options={},
        seed=None,
        return_info=False,
    ):
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if options is not None:
            if "video_path" in options:
                self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        if seed is not None:
            self.seed(seed=seed)
        if robot_obs is None and scene_obs is None:
            rand_scene_idx = np.random.randint(0, len(self.robot_obs))
            robot_obs = self.robot_obs[rand_scene_idx]
            scene_obs = self.scene_obs[rand_scene_idx]
        if robot_obs.shape[0] > 15 or scene_obs.shape[0] > 24:
            robot_obs = replace_rot6d_with_euler(self.rot_transformer, robot_obs, type="robot")
            scene_obs = replace_rot6d_with_euler(self.rot_transformer, scene_obs, type="scene")

        super().reset(robot_obs, scene_obs)

        self.start_info = self.get_info()

        self._t = 0
        obs = self.get_obs()
        return obs, None

    def get_rgb_obs(self, size=64):
        rgb_obs, _ = self.get_camera_obs()
        rgb_static = rgb_obs["rgb_static"]
        rgb_static = resize_image(rgb_static, cv2.INTER_AREA, resolution=size)
        # H x W x C -> C x H x W
        rgb_static = np.transpose(rgb_static, (2, 0, 1))

        rgb_gripper = rgb_obs["rgb_gripper"]
        rgb_gripper = resize_image(rgb_gripper, cv2.INTER_AREA, resolution=size)
        # H x W x C -> C x H x W
        rgb_gripper = np.transpose(rgb_gripper, (2, 0, 1))

        return rgb_static, rgb_gripper

    def render(self):
        rgb_obs, depth_obs = self.get_camera_obs()
        if "rgb" in self.render_camera_name:
            frame = rgb_obs[self.render_camera_name]
        else:
            frame = depth_obs[self.render_camera_name]
        frame = resize_image(frame, cv2.INTER_AREA, resolution=self.render_hw[0])
        return frame

    @staticmethod
    def set_egl_device(device):
        import os

        if "EGL_VISIBLE_DEVICES" in os.environ:
            logger.warning("Environment variable EGL_VISIBLE_DEVICES is already set. Is this intended?")
        cuda_id = int(device.split(":")[-1])
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to calvin env README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")
