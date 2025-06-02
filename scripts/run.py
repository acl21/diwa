"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import logging
import math
import os
import sys

import hydra
from omegaconf import OmegaConf

from diwa.utils.config_fix import calvin_config_fixer

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # A small function to replace values in the config with their string representation.
    if cfg.get("env_type") == "calvin":
        cfg = calvin_config_fixer(cfg, env_cfg_name="env_cfg")

    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # For pre-training: download dataset if needed
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        # raise error that the dataset path does not exist
        raise ValueError(f"Dataset path {cfg.train_dataset_path} does not exist. Please specify a valid path.")

    # For for-tuning: download normalization if needed
    if "normalization_path" in cfg and cfg.normalization_path is not None:
        if not os.path.exists(cfg.normalization_path):
            raise ValueError(
                f"Normalization path {cfg.normalization_path} does not exist. Please specify a valid path."
            )

    # For for-tuning: download checkpoint if needed
    if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
        raise ValueError(f"Base policy path {cfg.base_policy_path} does not exist. Please specify a valid path.")

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
