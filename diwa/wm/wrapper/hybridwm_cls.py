from pathlib import Path

import torch

from diwa.model.rewcls.contrastive import ContrastiveModel as RewardClassifier
from diwa.wm.encoder.hybridwm import HybridWMObsEncoder
from diwa.wm.wrapper.base import BaseWMWrapper


class HybridWMWrapper(BaseWMWrapper, HybridWMObsEncoder):
    def __init__(
        self,
        ckpt_path,
        device,
        stats_path,
        temperature=1.0,
        skill="open_drawer",
        rew_cls_path="",
        rew_cls_input_dim=2048,
    ):
        HybridWMObsEncoder.__init__(self, ckpt_path, device, stats_path)
        BaseWMWrapper.__init__(self, temperature, skill, device)
        self.set_wm(self.wm)

        if rew_cls_path == "":
            self.rewcls = None
            print("WARNING: Using VisionWMWrapper without a reward classifier.\n")
        elif Path(rew_cls_path).exists():
            self.rewcls = RewardClassifier(input_dim=rew_cls_input_dim)
            self.rewcls.load_state_dict(torch.load(rew_cls_path, map_location="cpu"))
            self.rewcls.to(device)
        else:
            raise FileNotFoundError(f"Reward classifier path {rew_cls_path} does not exist.")

    def decode_latent(self, latent):
        dcd_img_s, dcd_img_g, dcd_state_obs = self.wm.decoder(latent)

        r_unnorm, s_unnorm, rgb_unnorm = self.get_unnormalizers(self.device)

        robot_obs = r_unnorm(dcd_state_obs[:, :18].detach())
        scene_obs = s_unnorm(dcd_state_obs[:, 18:].detach())

        scene_obs[:, 4] = torch.round(scene_obs[:, 4]).int().float()
        scene_obs[:, 5] = torch.round(scene_obs[:, 5]).int().float()

        # RGB images
        rgb_static = rgb_unnorm(dcd_img_s.permute(0, 2, 3, 1))
        rgb_gripper = rgb_unnorm(dcd_img_g.permute(0, 2, 3, 1))

        return robot_obs, scene_obs, rgb_static, rgb_gripper

    def multi_step(self, latent, action, start_scene, next_latent=None):
        for i in range(action.shape[1]):
            latent = self.wm_step(latent, action[:, i, :])

        dcd_r_obs, dcd_s_obs, dcd_rgb_s, dcd_rgb_g = self.decode_latent(latent)

        if self.rewcls is None:
            reward = torch.zeros(latent.shape[0], device=self.device)
        else:
            reward = self.rewcls(latent)
            reward = reward.argmax(dim=1)

        return (
            latent,
            reward.cpu().numpy(),
            dcd_r_obs.cpu().numpy(),
            dcd_s_obs.cpu().numpy(),
            dcd_rgb_s.cpu().numpy(),
            dcd_rgb_g.cpu().numpy(),
        )
