"""
Parent eval agent class.

"""

import logging
import os
import random

import hydra
import numpy as np
import torch

log = logging.getLogger(__name__)
from diwa.env.utils import make_calvin_env, make_libero_env


class EvalAgent:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        DEVICE = cfg.get("device", "cuda:0")

        # WM wrapper
        if "wm" in cfg:
            self.wme = hydra.utils.instantiate(cfg.wm)
        else:
            self.wme = None

        self.env_name = cfg.get("env_name", None)
        self.env_type = cfg.get("env_type", None)

        env_type = cfg.env.get("env_type", None)

        if self.env_type == "calvin":
            _, self.env = make_calvin_env(
                cfg.env.name,
                env_type=self.env_type,
                num_envs=1,
                max_episode_steps=cfg.env.max_episode_steps,
                calvin_env_cfg=cfg.get("env_cfg", None),
                normalization_path=cfg.get("normalization_path", None),
                device=cfg.get("device", "cuda:0"),
                n_obs_steps=cfg.env.get("n_obs_steps", 1),
                n_action_steps=cfg.env.get("n_action_steps", 4),
                load_scene_from_dataset=cfg.env.get("load_scene_from_dataset", False),
                eval_scene_from_dataset=cfg.env.get("eval_scene_from_dataset", False),
                offline_method=cfg.get("offline_method", False),
                rgb_obs=cfg.get("rgb_obs", False),
                hybrid_obs=cfg.get("hybrid_obs", False),
                stacked_obs=cfg.get("stacked_obs", False),
            )
        elif self.env_type == "libero":
            _, self.env = make_libero_env(
                cfg.env.name,
                normalization_path=cfg.get("normalization_path", None),
                max_episode_steps=cfg.env.max_episode_steps,
                num_envs=1,
                n_obs_steps=cfg.env.get("n_obs_steps", 1),
                n_action_steps=cfg.env.get("n_action_steps", 4),
                offline_method=cfg.get("offline_method", False),
                rgb_obs=cfg.get("rgb_obs", False),
                stacked_obs=cfg.get("stacked_obs", False),
            )
        elif self.env_type == "real":
            # For real world environments, one could use the world model to simulate
            self.venv = None
            if "env_runner" in cfg:
                self.env = cfg.env
            else:
                self.env = None

        self.n_envs = 1
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.n_steps = self.max_episode_steps

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Logging, rendering
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.n_render = cfg.env.n_render
        self.render_video = cfg.env.save_video
        if self.env_type != "real":
            assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
            assert not (self.n_render <= 0 and self.render_video), "Need to set n_render > 0 if saving video"

    def run(self):
        pass

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [{k: v for k, v in kwargs.items()} for _ in range(self.n_envs)]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {key: np.stack([obs_venv[i][key] for i in range(self.n_envs)]) for key in obs_venv[0].keys()}
        if verbose:
            for index in range(self.n_envs):
                logging.info(f"<-- Reset environment {index} with options {options_venv[index]}")
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
