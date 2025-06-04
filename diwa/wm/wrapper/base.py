import torch


class BaseWMWrapper:
    def __init__(self, temperature, skill, device):
        self.temperature = temperature
        self.skill = skill
        self.device = device

    def set_wm(self, wm):
        self.wm = wm
        self.latent_half = self.wm.rssm_core.cell.deter_dim

    def wm_step(self, latent, action):
        h, z = latent.split([self.latent_half, self.latent_half], -1)
        pp, (h, z) = self.wm.dream(action, (h, z), temperature=self.temperature)
        next_latent = torch.cat((h, z), dim=-1)
        return next_latent

    def decode_latent(self, latent):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def multi_step(self, latent, action, start_scene_wm_features):
        raise NotImplementedError("This method should be implemented in the subclass.")
