import torch

from diwa.model.rewcls.contrastive import ContrastiveModel as RewardClassifier
from diwa.wm.encoder.visionwm import VisionWMObsEncoder
from diwa.wm.wrapper.base import BaseWMWrapper


class VisionWMWrapper(BaseWMWrapper, VisionWMObsEncoder):
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
        VisionWMObsEncoder.__init__(self, ckpt_path, device, stats_path)
        BaseWMWrapper.__init__(self, temperature, skill, device)
        self.set_wm(self.wm)
        self.rewcls = RewardClassifier(input_dim=rew_cls_input_dim)
        self.rewcls.load_state_dict(torch.load(rew_cls_path))
        self.rewcls.to(device)

    def decode_latent(self, latent):
        dcd_img_s, dcd_img_g = self.wm.decoder(latent)
        _, rgb_unnorm = self.get_unnormalizers(self.device)
        # RGB images
        rgb_static = rgb_unnorm(dcd_img_s.permute(0, 2, 3, 1))
        rgb_gripper = rgb_unnorm(dcd_img_g.permute(0, 2, 3, 1))

        return rgb_static, rgb_gripper

    def multi_step(self, latent, action):
        for i in range(action.shape[1]):
            latent = self.wm_step(latent, action[:, i, :])

        dcd_rgb_s, dcd_rgb_g = self.decode_latent(latent)

        reward = self.rewcls(latent)
        reward = reward.argmax(dim=1)

        return (
            latent,
            reward.cpu().numpy(),
            dcd_rgb_s.cpu().numpy(),
            dcd_rgb_g.cpu().numpy(),
        )
