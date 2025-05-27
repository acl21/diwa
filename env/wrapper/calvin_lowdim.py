if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import logging
import hydra
import gc
import sys
import gym
import cv2
import imageio
import numpy as np

from env.calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
from env.calvin_env.calvin_env.utils.utils import (
    EglDeviceNotFoundError,
    get_egl_device_id,
)

logger = logging.getLogger(__name__)


class CALVINLowDimWrapper(PlayTableSimEnv):
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
        device="cuda:0",
    ):
        pt_cfg = {**cfg.env}
        pt_cfg.pop("_target_", None)
        pt_cfg.pop("_recursive_", None)
        self.robot_cfg = cfg["robot"]
        self.scene_cfg = cfg["scene"]
        self.cameras_c = cfg["cameras"]
        self.set_egl_device(device)
        super().__init__(**pt_cfg)

        self.skill_name = skill_name

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        self.tasks = hydra.utils.instantiate(cfg.tasks)
        self.max_episode_steps = max_episode_steps
        self.video_writer = None
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.render_mode = "rgb_array"
        self.load_scene_from_dataset = load_scene_from_dataset
        if self.load_scene_from_dataset is not None:
            file = np.load(self.load_scene_from_dataset)
            self.robot_obs = file["robot_obs"]
            self.scene_obs = file["scene_obs"]

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

    @staticmethod
    def get_action_space():
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    def get_observation_space(self):
        """Returns the observation space for the environment based on the skill"""
        obs_dim = 39  # 15 + 24
        return gym.spaces.Box(low=-1, high=1, shape=(obs_dim,))

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_obs(self):
        obs = super().get_state_obs()
        obs = np.concatenate([obs["robot_obs"], obs["scene_obs"]])
        if self.normalize:
            obs = self.normalize_obs(obs)
        return obs

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
            if self.load_scene_from_dataset:
                rand_scene_idx = np.random.randint(0, len(self.robot_obs))
                robot_obs = self.robot_obs[rand_scene_idx]
                scene_obs = self.scene_obs[rand_scene_idx]

        super().reset(robot_obs, scene_obs)

        # if not self.load_scene_from_dataset:
        #     self.calibrate_scene(self.skill_name)
        self.start_info = self.get_info()

        self._t = 0
        return self.get_obs(), None

    def calibrate_scene_for_close_drawer(self):
        """Calibrate the scene for the close_drawer skill"""
        self.scene.doors[1].reset(0.2)
        self.scene.doors[1].initial_state = 0.2

    def calibrate_scene_for_turn_off_lightbulb(self):
        """Calibrate the scene for the turn_off_lightbulb skill"""
        self.scene.lights[0].reset(1)
        self.scene.switches[0].reset(0.08)

    def calibrate_scene_for_move_slider_right(self):
        """Calibrate the scene for the move_slider_right skill"""
        self.scene.doors[0].reset(0.2)
        self.scene.doors[0].initial_state = 0.2

    def calibrate_scene_for_turn_off_led(self):
        """Calibrate the scene for the turn_off_led skill"""
        self.scene.lights[1].reset(1)
        self.scene.buttons[0].reset(0)

    def reset_close_drawer_scene(self):
        """Reset the scene for the close_drawer skill"""
        self.scene.doors[1].reset(0)
        self.scene.doors[1].initial_state = 0

    def reset_turn_off_lightbulb_scene(self):
        """Reset the scene for the turn_off_lightbulb skill"""
        self.scene.lights[0].reset(0)
        self.scene.switches[0].reset(0)

    def reset_move_slider_right_scene(self):
        """Reset the scene for the move_slider_right skill"""
        self.scene.doors[0].reset(0)
        self.scene.doors[0].initial_state = 0

    def reset_turn_off_led_scene(self):
        """Reset the scene for the turn_off_led skill"""
        self.scene.lights[1].reset(0)
        self.scene.buttons[0].reset(0)

    def calibrate_scene(self, skill):
        """
        Change scene based on the skill to be performed.

        Logic: Set scene for one scene but reset others
        """
        if skill == "close_drawer":
            self.calibrate_scene_for_close_drawer()
            self.reset_turn_off_lightbulb_scene()
            self.reset_move_slider_right_scene()
            self.reset_turn_off_led_scene()
        elif skill == "turn_off_lightbulb":
            self.calibrate_scene_for_turn_off_lightbulb()
            self.reset_close_drawer_scene()
            self.reset_move_slider_right_scene()
            self.reset_turn_off_led_scene()
        elif skill == "move_slider_right":
            self.calibrate_scene_for_move_slider_right()
            self.reset_close_drawer_scene()
            self.reset_turn_off_lightbulb_scene()
            self.reset_turn_off_led_scene()
        elif skill == "turn_off_led":
            self.calibrate_scene_for_turn_off_led()
            self.reset_close_drawer_scene()
            self.reset_turn_off_lightbulb_scene()
            self.reset_move_slider_right_scene()
        else:
            # reset all
            self.reset_close_drawer_scene()
            self.reset_turn_off_lightbulb_scene()
            self.reset_move_slider_right_scene()
            self.reset_turn_off_led_scene()

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()
        task_filter = [self.skill_name]
        task_info = self.tasks.get_task_info_for_set(
            self.start_info, current_info, task_filter
        )
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

    def render(self):
        rgb_obs, depth_obs = self.get_camera_obs()
        if "rgb" in self.render_camera_name:
            frame = rgb_obs[self.render_camera_name]
        else:
            frame = depth_obs[self.render_camera_name]
        return cv2.resize(frame, self.render_hw, interpolation=cv2.INTER_AREA)

    @staticmethod
    def set_egl_device(device):
        import os

        if "EGL_VISIBLE_DEVICES" in os.environ:
            logger.warning(
                "Environment variable EGL_VISIBLE_DEVICES is already set. Is this intended?"
            )
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


@hydra.main(
    config_path="/home/lagandua/projects/dppo/cfg/calvin/calvin_env",
    config_name="new_default",
)
def main(cfg):
    env = CALVINLowDimWrapper(cfg)
    obs = env.reset()
    print(obs)
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
    env.close()
    gc.collect()


if __name__ == "__main__":
    main()
