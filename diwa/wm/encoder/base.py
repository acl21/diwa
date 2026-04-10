from pathlib import Path


class BaseWMObsEncoder:
    def __init__(self, ckpt_path, device, stats_path, wm_type):
        assert wm_type in ["state", "hybrid", "vis"], "Invalid world model type specified."
        assert Path(ckpt_path).exists(), f"Checkpoint path {ckpt_path} does not exist."
        assert Path(stats_path).exists(), f"Stats path {stats_path} does not exist."

        if wm_type == "state":
            from lumos.world_models.dreamer_v2_state import DreamerV2
        elif wm_type == "hybrid":
            from lumos.world_models.dreamer_v2_hybrid import DreamerV2
        elif wm_type == "vis":
            from lumos.world_models.dreamer_v2 import DreamerV2

        self.wm = DreamerV2.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
        self.wm.requires_grad_(False)
        self.wm.eval()
        self.wm.to(device)

        self.stats_path = stats_path

    def get_normalizers(self, device):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_unnormalizers(self, device):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_zero_wm_features(self, obs, device):
        """
        Given environment's initial observations, get features from the world model.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_hist_wm_features(self, obs, action, prev_done, in_state, device):
        """
        Given environment's history of observations, actions taken and done flags,
        get features from the world model.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")
