import copy

import numpy as np
import torch

from diwa.wm.encoder.base import BaseWMObsEncoder
from diwa.wm.utils import (
    get_robot_obs_normalizer,
    get_robot_obs_unnormalizer,
    get_scene_obs_normalizer,
    get_scene_obs_unnormalizer,
    transpose_tensor,
)


class StateWMObsEncoder(BaseWMObsEncoder):
    def __init__(self, ckpt_path, device, stats_path):
        super().__init__(ckpt_path, device, stats_path, "state")

    def get_normalizers(self, device):
        r_obs_norm = get_robot_obs_normalizer(self.stats_path, device)
        s_obs_norm = get_scene_obs_normalizer(self.stats_path, device)
        return r_obs_norm, s_obs_norm

    def get_unnormalizers(self, device):
        r_obs_unnorm = get_robot_obs_unnormalizer(self.stats_path, device)
        s_obs_unnorm = get_scene_obs_unnormalizer(self.stats_path, device)
        return r_obs_unnorm, s_obs_unnorm

    def get_zero_wm_features(self, obs, device):
        B = obs.shape[0]
        r_norm, s_norm = self.get_normalizers(device)

        obs = torch.from_numpy(obs).float().to(device).squeeze()
        if B == 1:
            obs = obs.unsqueeze(0)
        robot_obs, scene_obs = obs.split([18, 33], dim=-1)
        robot_obs = r_norm(robot_obs)
        scene_obs = s_norm(scene_obs)
        obs = torch.cat((robot_obs, scene_obs), dim=-1).unsqueeze(1)
        obs = transpose_tensor(obs)

        reset = torch.ones(B, 1, 1).bool().to(device)
        reset = transpose_tensor(reset)

        zero_action = torch.zeros(B, 1, 7).to(device)
        zero_action = transpose_tensor(zero_action)

        features, out_state = self.wm.infer_features(
            obs,
            zero_action,
            reset,
            self.wm.rssm_core.init_state(B),
        )
        return transpose_tensor(features), out_state

    def get_hist_wm_features(self, obs, action, prev_done, in_state, device):
        B = obs.shape[0]
        r_norm, s_norm = self.get_normalizers(device)

        if np.sum(prev_done) > 0:
            in_state_copy = copy.deepcopy(in_state)
            for j in range(B):
                if prev_done[j]:
                    (h, w) = self.wm.rssm_core.init_state(1)
                    in_state_copy[0][j] = h
                    in_state_copy[1][j] = w
            in_state = in_state_copy

        reset = torch.from_numpy(prev_done).reshape(B, 1, 1).bool().to(device)
        reset = transpose_tensor(reset)

        for i in range(obs.shape[1]):
            prev_obs = torch.from_numpy(obs[:, i, :]).float().to(device)
            robot_obs, scene_obs = prev_obs.split([18, 33], dim=-1)
            robot_obs = r_norm(robot_obs)
            scene_obs = s_norm(scene_obs)
            prev_obs = torch.cat((robot_obs, scene_obs), dim=-1).unsqueeze(1)
            prev_obs = transpose_tensor(prev_obs)

            pre_action = (torch.from_numpy(action[:, i, :]).float().to(device)).unsqueeze(1)
            pre_action = transpose_tensor(pre_action)

            features, out_state = self.wm.infer_features(
                prev_obs,
                pre_action,
                reset,
                in_state,
            )
            in_state = out_state
            reset = torch.zeros(B, 1, 1).bool().to(device)
            reset = transpose_tensor(reset)
        return transpose_tensor(features), out_state
