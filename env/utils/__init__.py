def make_async_calvin(
    id,
    num_envs=1,
    asynchronous=True,
    env_type=None,
    max_episode_steps=None,
    # below for calvin only
    calvin_env_cfg=None,
    normalization_path=None,
    load_scene_from_dataset=False,
    device="cuda:0",
    n_obs_steps=1,
    n_action_steps=4,
    eval_scene_from_dataset=False,
    offline_method=False,
    rgb_obs=False,
    hybrid_obs=False,
    stacked_obs=False,
):
    assert env_type == "calvin", f"env_type must be calvin, got {env_type}"
    assert calvin_env_cfg is not None, "calvin_env_cfg must be provided"
    assert asynchronous, "asynchronous must be True"

    from env.utils.custom_subprocvec import CustomSubprocVecEnv
    from env.wrapper.calvin_multistep import CALVINMultiStep

    if rgb_obs:
        if hybrid_obs:
            from env.wrapper.calvin_image_rot6d_randinit import (
                CALVINImageWrapper as CALVINWrapper,
            )
        else:
            if stacked_obs:
                from env.wrapper.calvin_image_only_stacked_rot6d_randinit import (
                    CALVINImageWrapper as CALVINWrapper,
                )
            else:
                from env.wrapper.calvin_image_only_rot6d_randinit import (
                    CALVINImageWrapper as CALVINWrapper,
                )
    else:
        from env.wrapper.calvin_lowdim_rot6d_randinit import (
            CALVINLowDimWrapper as CALVINWrapper,
        )

    # There is always only one evaluation environment
    eval_env = CALVINWrapper(
        calvin_env_cfg,
        skill_name=id,
        normalization_path=normalization_path,
        max_episode_steps=max_episode_steps,
        load_scene_from_dataset=eval_scene_from_dataset,
        device=device,
        # device="cpu",
    )
    eval_env_ms = CALVINMultiStep(
        env=eval_env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    if num_envs == 1:  # a special case for eval only (pretrain)
        return None, eval_env_ms
    if offline_method:
        # for methods that are offline (e.g., DiWA)
        # a single environment is enough to generate the environment start states
        train_env = CALVINWrapper(
            calvin_env_cfg,
            skill_name=id,
            normalization_path=normalization_path,
            max_episode_steps=max_episode_steps,
            load_scene_from_dataset=load_scene_from_dataset,
            device=device,
            # device="cpu",
        )
        train_env_ms = CALVINMultiStep(
            env=train_env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )
        return train_env_ms, eval_env_ms
    else:
        # for methods that are online (e.g., DPPO, DPPO(Vision WM Encoder), DPPO(State), etc.)
        def _make_env():
            env = CALVINWrapper(
                calvin_env_cfg,
                skill_name=id,
                normalization_path=normalization_path,
                max_episode_steps=max_episode_steps,
                load_scene_from_dataset=load_scene_from_dataset,
                device=device,
            )
            seed = env.seed()[0]
            import numpy as np

            env.seed(seed + np.random.randint(1e6))
            return CALVINMultiStep(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
            )

        env_fns = [_make_env for _ in range(num_envs)]
        return CustomSubprocVecEnv(env_fns), eval_env_ms


def make_real_env(
    id,
    real_env_cfg,
    normalization_path=None,
    max_episode_steps=None,
    device="cuda:0",
    n_obs_steps=1,
    n_action_steps=4,
    eval_scene_from_dataset=False,
    rgb_obs=False,
    stacked_obs=False,
):
    from env.wrapper.real_multistep import RealMultiStep

    if stacked_obs:
        from env.wrapper.real_env_image_stacked import RealEnvWrapper
    else:
        from env.wrapper.real_env import RealEnvWrapper

    eval_env = RealEnvWrapper(
        real_env_cfg,
        skill_name=id,
        normalization_path=normalization_path,
        max_episode_steps=max_episode_steps,
        load_scene_from_dataset=eval_scene_from_dataset,
        device=device,
        rgb_obs=rgb_obs,
    )
    eval_env_ms = RealMultiStep(
        env=eval_env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    return eval_env_ms
