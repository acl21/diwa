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
        render_hw=(192, 192),  # divisble by 16
        render_cams=["rgb_static", "rgb_gripper"],
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

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.tasks = hydra.utils.instantiate(cfg.tasks)
        self.render_hw = render_hw
        self.render_cams = render_cams

    @staticmethod
    def get_action_space():
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    def get_observation_space(self):
        """Returns the observation space for the environment based on the skill"""
        obs_dim = 39  # 15 + 24
        return gym.spaces.Box(low=-1, high=1, shape=(obs_dim,))

    def get_obs(self):
        obs = super().get_state_obs()
        obs = np.concatenate([obs["robot_obs"], obs["scene_obs"]])
        return obs

    def reset(self, robot_obs=None, scene_obs=None):

        super().reset(robot_obs, scene_obs)

        return self.get_obs()

    def render(self):
        rgb_obs, depth_obs = self.get_camera_obs()
        renders = []
        for cam_name in self.render_cams:
            if "rgb" in cam_name:
                frame = rgb_obs[cam_name]
            else:
                frame = depth_obs[cam_name]
            renders.append(
                cv2.resize(frame, self.render_hw, interpolation=cv2.INTER_AREA)
            )
        return renders

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
    config_path="/home/lagandua/projects/lumos/config/calvin_env",
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
