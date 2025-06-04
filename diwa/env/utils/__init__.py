def make_calvin_env(
    skill_name,
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

    from diwa.env.utils.custom_subprocvec import CustomSubprocVecEnv
    from diwa.env.wrapper.multistep import MultiStepWrapper

    if rgb_obs:
        if hybrid_obs:
            from diwa.env.wrapper.calvin_image_stateful import CALVINImageWrapper as CALVINWrapper
        else:
            if stacked_obs:
                from diwa.env.wrapper.calvin_image_stacked import CALVINImageWrapper as CALVINWrapper
            else:
                from diwa.env.wrapper.calvin_image import CALVINImageWrapper as CALVINWrapper
    else:
        from diwa.env.wrapper.calvin_lowdim import CALVINLowDimWrapper as CALVINWrapper

    # There is always only one evaluation environment
    eval_env = CALVINWrapper(
        calvin_env_cfg,
        skill_name=skill_name,
        normalization_path=normalization_path,
        max_episode_steps=max_episode_steps,
        load_scene_from_dataset=eval_scene_from_dataset,
        device=device,
        # device="cpu",
    )
    eval_env_ms = MultiStepWrapper(
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
        train_env_ms = MultiStepWrapper(
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
            return MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
            )

        env_fns = [_make_env for _ in range(num_envs)]
        return CustomSubprocVecEnv(env_fns), eval_env_ms


def make_real_env(
    skill_name,
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
    from diwa.env.wrapper.real_multistep import RealMultiStep

    if stacked_obs:
        from diwa.env.wrapper.real_env_image_stacked import RealEnvWrapper
    else:
        from diwa.env.wrapper.real_env import RealEnvWrapper

    eval_env = RealEnvWrapper(
        real_env_cfg,
        skill_name=skill_name,
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


def make_libero_env(
    task_name,
    normalization_path=None,
    max_episode_steps=None,
    num_envs=1,
    n_obs_steps=1,
    n_action_steps=4,
    offline_method=False,
    rgb_obs=False,
    hybrid_obs=False,
    stacked_obs=False,
):
    from diwa.env.utils.custom_subprocvec import CustomSubprocVecEnv
    from diwa.env.wrapper.multistep import MultiStepWrapper

    if rgb_obs:
        if hybrid_obs:
            from diwa.env.wrapper.libero_image_stateful import LIBEROImageWrapper as LIBEROEnvWrapper
        else:
            if stacked_obs:
                from diwa.env.wrapper.libero_image_stacked import LIBEROEnvWrapper
            else:
                from diwa.env.wrapper.libero_image import LIBEROImageWrapper as LIBEROEnvWrapper
    else:
        from diwa.env.wrapper.libero_lowdim import LIBEROLowDimWrapper as LIBEROEnvWrapper

    eval_env = LIBEROEnvWrapper(
        task_name=task_name,
        normalization_path=normalization_path,
        max_episode_steps=max_episode_steps,
    )
    eval_env_ms = MultiStepWrapper(
        env=eval_env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    if num_envs == 1:  # a special case for eval only (pretrain)
        return None, eval_env_ms

    if offline_method:
        # for methods that are offline (e.g., DiWA)
        # a single environment is enough to generate the environment start states
        train_env = LIBEROEnvWrapper(
            task_name=task_name,
            normalization_path=normalization_path,
            max_episode_steps=max_episode_steps,
        )
        train_env_ms = MultiStepWrapper(
            env=train_env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )
        return train_env_ms, eval_env_ms
    else:
        # for methods that are online (e.g., DPPO, DPPO(Vision WM Encoder), DPPO(State), etc.)
        def _make_env():
            env = LIBEROEnvWrapper(
                task_name=task_name,
                normalization_path=normalization_path,
                max_episode_steps=max_episode_steps,
            )
            seed = env.seed()[0]
            import numpy as np

            env.seed(seed + np.random.randint(1e6))
            return MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
            )

        env_fns = [_make_env for _ in range(num_envs)]
        return CustomSubprocVecEnv(env_fns), eval_env_ms
