import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from diwa.wm.encoder.statewm import StateWMObsEncoder
from diwa.wm.wrapper.base import BaseWMWrapper


class StateWMWrapper(BaseWMWrapper, StateWMObsEncoder):
    def __init__(
        self,
        ckpt_path,
        device,
        stats_path,
        temperature=1.0,
        skill="open_drawer",
    ):
        super(StateWMObsEncoder).__init__(ckpt_path, device, stats_path)
        super(BaseWMWrapper, self).__init__(temperature, skill, device)

    def decode_latent(self, latent):
        dcd_state_obs = self.wm.decoder(latent)

        r_unnorm, s_unnorm = self.get_unnormalizers(self.device)

        robot_obs = r_unnorm(dcd_state_obs[:, :18].detach())
        scene_obs = s_unnorm(dcd_state_obs[:, 18:].detach())

        scene_obs[:, 4] = torch.round(scene_obs[:, 4]).int().float()
        scene_obs[:, 5] = torch.round(scene_obs[:, 5]).int().float()

        return robot_obs, scene_obs

    def multi_step(self, latent, action, start_scene):
        for i in range(action.shape[1]):
            latent = self.wm_step(latent, action[:, i, :])

        dcd_r_obs, dcd_s_obs = self.decode_latent(latent)

        reward = torch.zeros(action.shape[0]).to(self.device)
        start_scene = torch.from_numpy(start_scene).to(self.device)
        if "open_drawer" in self.skill:
            delta = dcd_s_obs[:, 1] - start_scene[:, 1]
            reward[delta >= 0.12] = 1
        elif "close_drawer" in self.skill:
            delta = dcd_s_obs[:, 1] - start_scene[:, 1]
            reward[delta <= -0.12] = 1
        elif "move_slider_left" in self.skill:
            delta = dcd_s_obs[:, 0] - start_scene[:, 0]
            reward[delta >= 0.15] = 1
        elif "move_slider_right" in self.skill:
            delta = dcd_s_obs[:, 0] - start_scene[:, 0]
            reward[delta <= -0.15] = 1
        elif "turn_on_lightbulb" in self.skill:
            # Check if lightbulb is on
            reward[dcd_s_obs[:, 4] == 1] = 1
        elif "turn_off_lightbulb" in self.skill:
            # Check if lightbulb is off
            reward[dcd_s_obs[:, 4] == 0] = 1
        elif "turn_on_led" in self.skill:
            # Check if led is on
            reward[dcd_s_obs[:, 5] == 1] = 1
        elif "turn_off_led" in self.skill:
            # Check if led is off
            reward[dcd_s_obs[:, 5] == 0] = 1
        elif "push_pink_block_right" in self.skill:
            # Check x-coordinate of pink block
            pink_x = 24
            # get euclidean distance
            delta = dcd_s_obs[:, pink_x] - start_scene[:, pink_x]
            reward[delta >= 0.1] = 1
        elif "lift_pink_block_table" in self.skill:
            # Check z-coordinate of pink block
            pink_z = 26
            # get euclidean distance
            delta = dcd_s_obs[:, pink_z] - start_scene[:, pink_z]
            reward[delta >= 0.05] = 1
        elif "rotate_pink_block_right" in self.skill:
            # The logic is similar to `rotate_object`
            # from calvin_env/envs/tasks.py

            # Check rotation of pink block
            pink_pos = [24, 25, 26]
            pink_orn = [27, 28, 29, 30, 31, 32]

            start_orn = self.rot6d_to_quat.forward(start_scene[:, pink_orn])
            end_orn = self.rot6d_to_quat.forward(dcd_s_obs[:, pink_orn])

            start_orn = R.from_quat(start_orn.cpu().numpy())
            end_orn = R.from_quat(end_orn.cpu().numpy())

            diff = start_orn * end_orn.inv()
            xyz = diff.as_euler("xyz", degrees=True)
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]

            z_degrees = -60
            z_threshold = 180
            x_y_threshold = 30
            movement_threshold = 0.1

            reward_filter = (
                (z > -z_threshold) & (z_degrees > z) & (np.abs(x) < x_y_threshold) & (np.abs(y) < x_y_threshold)
            )

            # Make sure the pink block has not moved too much
            start_pos = start_scene[:, pink_pos]
            end_pos = dcd_s_obs[:, pink_pos]
            pos_diff = end_pos - start_pos

            reward_filter = reward_filter & ~(np.linalg.norm(pos_diff.cpu().numpy(), axis=1) > movement_threshold)

            reward[reward_filter] = 1

        return (
            latent,
            reward.cpu().numpy(),
            dcd_r_obs.cpu().numpy(),
            dcd_s_obs.cpu().numpy(),
        )
