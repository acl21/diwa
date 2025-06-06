class BaseEnvRunner:
    """
    Base class for environment runners.
    This class should be extended by specific environment runners.
    """

    def __init__(self):
        self.n_envs = None
        self.n_render = None
        self.n_steps = None
        self.use_wandb = None
        self.render_dir = None
        self.reset_at_iteration = None
        self.act_steps = None
        self.render_video = None
        self.best_reward_threshold_for_success = None

    def init_values(self, cfg):
        self.n_envs = cfg.env.n_envs
        self.n_render = cfg.env.n_render
        self.n_steps = cfg.env.max_episode_steps
        self.use_wandb = cfg.wandb is not None
        self.act_steps = cfg.act_steps
        self.render_video = cfg.env.save_video
        self.best_reward_threshold_for_success = cfg.env.best_reward_threshold_for_success
        self.env_type = cfg.env_type
