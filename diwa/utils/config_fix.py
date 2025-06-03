from omegaconf import OmegaConf


def insert_str(base_string, idx, new_string):
    """
    Inserts a string value at a specific index in a string.
    """
    return base_string[:idx] + new_string + base_string[idx:]


def calvin_config_fixer(cfg: OmegaConf, env_cfg_name: str) -> OmegaConf:
    """
    Fixes the config by replacing some values with env_cfg_name
    to avoid interpolation errors.
    E.g.:
    Originally, env_cfg.env.cameras = "${cameras}"
    After fixing, env_cfg.env.cameras = "${env_cfg_name}.cameras"
    """
    env_cfg_dict = OmegaConf.to_container(cfg.env_cfg, resolve=False)
    cfg.env_cfg.env.cameras = insert_str(env_cfg_dict["env"]["cameras"], 2, f"{env_cfg_name}.")
    cfg.env_cfg.env.use_vr = insert_str(env_cfg_dict["env"]["use_vr"], 2, f"{env_cfg_name}.")
    cfg.env_cfg.env.robot_cfg = insert_str(env_cfg_dict["env"]["robot_cfg"], 2, f"{env_cfg_name}.")
    cfg.env_cfg.env.scene_cfg = insert_str(env_cfg_dict["env"]["scene_cfg"], 2, f"{env_cfg_name}.")

    cfg.env_cfg.robot.base_position = insert_str(env_cfg_dict["robot"]["base_position"], 2, f"{env_cfg_name}.")
    cfg.env_cfg.robot.base_orientation = insert_str(env_cfg_dict["robot"]["base_orientation"], 2, f"{env_cfg_name}.")
    cfg.env_cfg.robot.initial_joint_positions = insert_str(
        env_cfg_dict["robot"]["initial_joint_positions"], 2, f"{env_cfg_name}."
    )

    cfg.env_cfg.scene.data_path = insert_str(env_cfg_dict["scene"]["data_path"], 2, f"{env_cfg_name}.")
    cfg.env_cfg.scene.euler_obs = insert_str(env_cfg_dict["scene"]["euler_obs"], 2, f"{env_cfg_name}.")
    return cfg
