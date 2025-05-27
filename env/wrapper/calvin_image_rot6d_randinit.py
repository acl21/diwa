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
import pybullet as p
from env.calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
from env.calvin_env.calvin_env.utils.utils import (
    EglDeviceNotFoundError,
    get_egl_device_id,
)
from util.rotation_transformer import RotationTransformer

logger = logging.getLogger(__name__)


def resize_image(image, intp, resolution=64):
    """Resize an image to the target size using INTER_AREA interpolation."""
    target_size = (resolution, resolution)
    return cv2.resize(image, target_size, interpolation=intp)


def sample_random_robot_pos(size):
    """Sample random robot positions"""
    chunk_size = int(0.75 * size)
    robot_x = np.concatenate(
        [
            np.random.uniform(-0.3, 0.3, chunk_size),
            np.random.uniform(-0.1, 0.25, size - chunk_size),
        ]
    )
    robot_y = np.concatenate(
        [
            np.random.uniform(-0.15, 0.0, chunk_size),
            np.random.uniform(-0.3, -0.15, size - chunk_size),
        ]
    )
    robot_z = np.random.uniform(0.50, 0.575, size)
    return np.array([robot_x, robot_y, robot_z]).T


def sample_random_robot_orn(size):
    """Sample random robot orientations"""
    chunk_size = int(0.8 * size)
    robot_roll = np.concatenate(
        [
            np.random.uniform(3, 3.14, chunk_size),
            np.random.uniform(-3.14, -3, size - chunk_size),
        ]
    )
    robot_pitch = np.random.normal(-0.1, 0.1, size)
    robot_yaw = np.random.normal(1.57, 0.1, size)
    return np.array([robot_roll, robot_pitch, robot_yaw]).T


def sample_random_block_pos(size):
    """Sample random block positions"""
    chunk_size = int(0.75 * size)

    block_x = np.concatenate(
        [
            np.random.uniform(-0.05, 0.3, chunk_size),  # Right to the button
            np.random.uniform(-0.3, -0.2, size - chunk_size),  # Left to the button
        ]
    )
    block_y = np.random.uniform(-0.12, -0.05, size)
    block_z = np.random.uniform(0.46, 0.46, size)
    return np.array([block_x, block_y, block_z]).T


def sample_random_block_orn(size):
    """Sample random block orientations"""
    rand_roll = np.random.choice(
        [-3.14, -1.57, 0, 1.57, 3.14], p=[0.1, 0.1, 0.5, 0.2, 0.1], size=size
    )
    rand_pitch = np.zeros(size)
    chunk_size = int(0.5 * size)
    rand_yaw = np.concatenate(
        [
            np.random.uniform(-1.75, -1.4, chunk_size),
            np.random.uniform(1.4, 1.75, size - chunk_size),
        ]
    )
    return np.array([rand_roll, rand_pitch, rand_yaw]).T


class CALVINImageWrapper(PlayTableSimEnv):
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
        self.observation_space = self.get_observation_space()
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

        self.rot_transformer = RotationTransformer(
            from_rep="euler_angles", to_rep="rotation_6d", from_convention="XYZ"
        )

    def random_sample_scene_obs(self, size):
        rand_block_pos = sample_random_block_pos(size)
        rand_block_orn = sample_random_block_orn(size)

        scene_obs = []
        for i in range(size):
            self.scene.reset()
            scene_obs.append(self.scene.get_obs())
        scene_obs = np.array(scene_obs)

        # Block positions and orientations
        if "pink" in self.skill_name:
            scene_obs[:, [-6, -5, -4]] = rand_block_pos
            scene_obs[:, [-3, -2, -1]] = rand_block_orn
        elif "blue" in self.skill_name:
            scene_obs[:, [-12, -11, -10]] = rand_block_pos
            scene_obs[:, [-9, -8, -7]] = rand_block_orn
        elif "red" in self.skill_name:
            scene_obs[:, [-18, -17, -16]] = rand_block_pos
            scene_obs[:, [-15, -14, -13]] = rand_block_orn

        # Slider positions
        if "slider" not in self.skill_name:
            # randomly open/close the slider
            scene_obs[:, 0] = np.random.choice(
                [0, 0.25, 0.28], size=size, p=[0.5, 0.1, 0.4]
            )
        else:
            if "right" in self.skill_name:
                scene_obs[:, 0] = np.random.choice(
                    [0.25, 0.26, 0.28], size=size, p=[0.2, 0.2, 0.6]
                )
            elif "left" in self.skill_name:
                scene_obs[:, 0] = np.random.choice([0, 0.02], size=size, p=[0.8, 0.2])

        # Drawer positions
        if "drawer" not in self.skill_name:
            # randomly open/close the drawer
            scene_obs[:, 1] = np.random.choice(
                [0, 0.19, 0.22], size=size, p=[0.2, 0.4, 0.4]
            )
        else:
            if "open" in self.skill_name:
                scene_obs[:, 1] = 0.0
            elif "close" in self.skill_name:
                scene_obs[:, 1] = np.random.uniform(0.19, 0.22, size)

        # Light and switch positions
        if "lightbulb" not in self.skill_name:
            # randomly turn on the lightbulb
            scene_obs[:, 4] = np.random.choice([0, 1], size=size, p=[0.5, 0.5])
            scene_obs[scene_obs[:, 4] == 0, 3] = 0
            scene_obs[scene_obs[:, 4] == 1, 3] = 0.088
        else:
            if "turn_on" in self.skill_name:
                scene_obs[:, 4] = 0
                scene_obs[:, 3] = 0
            elif "turn_off" in self.skill_name:
                scene_obs[:, 4] = 1
                scene_obs[:, 3] = 0.088

        # LED
        if "led" not in self.skill_name:
            # randonmly turn on the led
            scene_obs[:, 5] = np.random.choice([0, 1], size=size, p=[0.5, 0.5])
        else:
            if "turn_on" in self.skill_name:
                scene_obs[:, 5] = 0
            elif "turn_off" in self.skill_name:
                scene_obs[:, 5] = 1

        # When the drawer is open, place the blocks inside
        if sum(scene_obs[:, 1]) > 0:
            drawer_x = np.random.uniform(0.15, 0.25, size)
            drawer_y = np.random.uniform(-0.3, -0.25, size)
            drawer_z = np.random.uniform(0.37, 0.37, size)

            drawer_xyz = np.array([drawer_x, drawer_y, drawer_z]).T

            pink_indices = [-6, -5, -4]
            blue_indices = [-12, -11, -10]
            red_indices = [-18, -17, -16]
            option1 = None
            option2 = None

            if "block" in self.skill_name:
                if "pink" in self.skill_name:
                    option1 = blue_indices
                    option2 = red_indices
                elif "blue" in self.skill_name:
                    option1 = pink_indices
                    option2 = red_indices
                elif "red" in self.skill_name:
                    option1 = pink_indices
                    option2 = blue_indices

                # Randomly choose between the two options
                open_filter = scene_obs[:, 1] > 0
                option1_filter = np.copy(open_filter)
                option2_filter = np.copy(open_filter)
                # Create two filters for the two options and randomly choose between them
                block_filter = np.random.choice([True, False], size=sum(open_filter))
                option1_filter[open_filter == True] = block_filter
                option2_filter[open_filter == True] = ~block_filter
                scene_obs[option1_filter, option1[0]] = drawer_xyz[option1_filter, 0]
                scene_obs[option1_filter, option1[1]] = drawer_xyz[option1_filter, 1]
                scene_obs[option1_filter, option1[2]] = drawer_xyz[option1_filter, 2]

                scene_obs[option2_filter, option2[0]] = drawer_xyz[option2_filter, 0]
                scene_obs[option2_filter, option2[1]] = drawer_xyz[option2_filter, 1]
                scene_obs[option2_filter, option2[2]] = drawer_xyz[option2_filter, 2]
            else:
                # If the skill is not block related, we get one extra option
                option1 = pink_indices
                option2 = blue_indices
                option3 = red_indices

                # Randomly choose between the three options
                open_filter = scene_obs[:, 1] > 0
                option1_filter = np.copy(open_filter)
                option2_filter = np.copy(open_filter)
                option3_filter = np.copy(open_filter)
                # Create two filters for the two options and randomly choose between them
                block_filter = np.random.choice([1, 2, 3], size=sum(open_filter))
                option1_filter[open_filter == True] = block_filter == 1
                option2_filter[open_filter == True] = block_filter == 2
                option3_filter[open_filter == True] = block_filter == 3

                scene_obs[option1_filter, option1[0]] = drawer_xyz[option1_filter, 0]
                scene_obs[option1_filter, option1[1]] = drawer_xyz[option1_filter, 1]
                scene_obs[option1_filter, option1[2]] = drawer_xyz[option1_filter, 2]

                scene_obs[option2_filter, option2[0]] = drawer_xyz[option2_filter, 0]
                scene_obs[option2_filter, option2[1]] = drawer_xyz[option2_filter, 1]
                scene_obs[option2_filter, option2[2]] = drawer_xyz[option2_filter, 2]

                scene_obs[option3_filter, option3[0]] = drawer_xyz[option3_filter, 0]
                scene_obs[option3_filter, option3[1]] = drawer_xyz[option3_filter, 1]
                scene_obs[option3_filter, option3[2]] = drawer_xyz[option3_filter, 2]

        return scene_obs

    def random_sample_robot_obs(self, size):
        rand_robot_pos = sample_random_robot_pos(size)
        rand_robot_orn = sample_random_robot_orn(size)
        # Gripper is always open
        robot_gripper_width = 0.08
        robot_gripper_action = 1

        robot_obs = []
        for i in range(size):
            robot_jnt_pos = self.robot.mixed_ik.get_ik(
                rand_robot_pos[1], p.getQuaternionFromEuler(rand_robot_orn[i])
            )
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

    @staticmethod
    def get_action_space():
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

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

    def replace_euler_with_rot6d(self, obs, type="robot"):
        """Replace the euler angles in robot and scene obs with rotation_6d"""
        if type == "robot":
            obs = np.concatenate(
                [
                    obs[:3],
                    self.rot_transformer.forward(obs[3:6]),
                    obs[6:],
                ]
            )
        elif type == "scene":
            # red, blue, pink blocks respectively
            indices = [[9, 10, 11], [15, 16, 17], [21, 22, 23]]
            rotation_6ds = []
            for idx in indices:
                euler_angles = obs[idx]
                rotation_6d = self.rot_transformer.forward(euler_angles)
                rotation_6ds.append(rotation_6d)
            obs = np.concatenate(
                [
                    obs[:9],
                    rotation_6ds[0],
                    obs[12:15],
                    rotation_6ds[1],
                    obs[18:21],
                    rotation_6ds[2],
                ]
            )
        return obs

    def replace_rot6d_with_euler(self, obs, type="robot"):
        """Replace the euler angles in robot and scene obs with rotation_6d"""
        if type == "robot":
            obs = np.concatenate(
                [
                    obs[:3],
                    self.rot_transformer.inverse(obs[3:9]),
                    obs[9:],
                ]
            )
        elif type == "scene":
            # red, blue, pink blocks respectively
            indices = [
                [9, 10, 11, 12, 13, 14],
                [18, 19, 20, 21, 22, 23],
                [27, 28, 29, 30, 31, 32],
            ]
            euler_angles_arr = []
            for idx in indices:
                rotation_6d = obs[idx]
                euler_angles = self.rot_transformer.inverse(rotation_6d)
                euler_angles_arr.append(euler_angles)
            obs = np.concatenate(
                [
                    obs[:9],
                    euler_angles_arr[0],
                    obs[15:18],
                    euler_angles_arr[1],
                    obs[24:27],
                    euler_angles_arr[2],
                ]
            )
        return obs

    def get_obs(self):
        state_obs = super().get_state_obs()
        robot_obs = state_obs["robot_obs"]
        scene_obs = state_obs["scene_obs"]
        robot_obs = self.replace_euler_with_rot6d(robot_obs, type="robot")
        scene_obs = self.replace_euler_with_rot6d(scene_obs, type="scene")
        state_obs = np.concatenate([robot_obs, scene_obs])
        if self.normalize:
            state_obs = self.normalize_obs(state_obs)

        rgb_static, rgb_gripper = self.get_rgb_obs(size=64)

        obs = {}
        obs["state"] = state_obs
        obs["rgb_static"] = rgb_static
        obs["rgb_gripper"] = rgb_gripper

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
            rand_scene_idx = np.random.randint(0, len(self.robot_obs))
            robot_obs = self.robot_obs[rand_scene_idx]
            scene_obs = self.scene_obs[rand_scene_idx]
        if robot_obs.shape[0] > 15 or scene_obs.shape[0] > 24:
            robot_obs = self.replace_rot6d_with_euler(robot_obs, type="robot")
            scene_obs = self.replace_rot6d_with_euler(scene_obs, type="scene")

        super().reset(robot_obs, scene_obs)

        self.start_info = self.get_info()

        self._t = 0
        return self.get_obs(), None

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
    env = CALVINImageWrapper(cfg)
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
