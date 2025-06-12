from pathlib import Path

import cv2
import gym
import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import robosuite.utils.transform_utils as T
import torch

from diwa.env.utils.calvin_helpers import resize_image

benchmark_name = "libero_90"


class LIBEROBaseWrapper(OffScreenRenderEnv):
    def __init__(
        self,
        task_name=None,
        normalization_path=None,
        render_hw=(256, 256),  # divisble by 16
        render_camera_name="rgb_static",
        max_episode_steps=None,
        seed=42,
    ):
        benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark_instance = benchmark_dict[benchmark_name]()

        task_names = self.benchmark_instance.get_task_names()
        self.task_id = task_names.index(task_name)
        self.task = self.benchmark_instance.get_task(self.task_id)

        self.task_name = task_name

        bddl_files_default_path = get_libero_path("bddl_files")
        env_args = {
            "bddl_file_name": Path(bddl_files_default_path) / self.task.problem_folder / self.task.bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        super().__init__(**env_args)
        self.seed(seed)
        self.action_space = self.get_action_space()
        self.max_episode_steps = max_episode_steps
        self.video_writer = None
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.init_states = self.get_task_init_states()
        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization_data = np.load(normalization_path)
            self.obs_min = normalization_data["obs_min"]
            self.obs_max = normalization_data["obs_max"]
            self.action_min = normalization_data["action_min"]
            self.action_max = normalization_data["action_max"]

        self._t = 0

    def get_task_init_states(self):
        """
        This function exists here because libero.libero.benchmark.__init__.py
        has a method get_task_init_states that expects PyTorch version older than 2.6.
        """
        init_states_path = Path(
            get_libero_path("init_states"),
            self.task.problem_folder,
            self.task.init_states_file,
        )
        init_states = torch.load(init_states_path, weights_only=False)
        return init_states

    def get_action_space(self):
        """
        Returns the action space of the environment.
        """
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_dim,),
            dtype=np.float32,
        )

    def normalize_obs(self, obs):
        obs = 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)  # -> [-1, 1]
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

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
        if self.normalize:
            action = self.unnormalize_action(action)
        _, reward, _, info = self.env.step(action.tolist())
        obs = self.get_obs()

        terminated, d_info = self._termination(reward)
        truncated, t_info = self._truncation()
        info.update(d_info)
        info.update(t_info)

        self._t += 1

        if self.video_writer is not None:
            self.video_writer.append_data(self.render())

        return obs, reward, terminated, truncated, info

    def reset(self, init_state=None, options={}, seed=None, return_info=False):
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
        self.env.reset()
        if init_state is not None:
            self.set_init_state(init_state)
        obs = self.get_obs()
        self._t = 0
        return obs, None

    def get_state_obs(self):
        obs = self.env._get_observations()
        robot_obs = np.concatenate(
            [
                np.array(obs["robot0_eef_pos"]),
                np.array(T.quat2axisangle(obs["robot0_eef_quat"])),
                np.array(obs["robot0_gripper_qpos"]),
                np.array(obs["robot0_joint_pos"]),
            ],
            axis=-1,
        )

        scene_obs = self.env.sim.get_state().flatten()
        obs = {
            "robot_obs": robot_obs,
            "scene_obs": scene_obs,
        }
        return obs

    def get_rgb_obs(self, size=64):
        obs = self.env._get_observations()

        rgb_static = np.flipud(obs["agentview_image"])
        rgb_static = resize_image(rgb_static, cv2.INTER_AREA, size)
        # H x W x C -> C x H x W
        rgb_static = np.transpose(rgb_static, (2, 0, 1))

        rgb_gripper = np.flipud(obs["robot0_eye_in_hand_image"])
        rgb_gripper = resize_image(rgb_gripper, cv2.INTER_AREA, size)
        # H x W x C -> C x H x W
        rgb_gripper = np.transpose(rgb_gripper, (2, 0, 1))

        return rgb_static, rgb_gripper

    def get_camera_obs(self):
        """
        Only used for rendering.
        """
        obs = self.env._get_observations()

        rgb_static = np.flipud(obs["agentview_image"])
        rgb_gripper = np.flipud(obs["robot0_eye_in_hand_image"])

        rgb_obs = {
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
        }
        return rgb_obs

    def render(self):
        rgb_obs = self.get_camera_obs()
        frame = rgb_obs[self.render_camera_name]
        frame = resize_image(
            frame,
            cv2.INTER_AREA,
            resolution=self.render_hw[0],
        )
        return frame


if __name__ == "__main__":
    # Example usage
    task_name = "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet"  # Replace with an actual task name
    seed = 42  # Replace with a desired seed
    env = LIBEROBaseWrapper(task_name=task_name, seed=seed)
    print(f"Initialized Libero environment for task: {env.task_name} with ID: {env.task_id}")
