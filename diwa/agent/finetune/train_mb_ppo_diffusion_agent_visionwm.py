"""
DiWA fine-tuning.

"""

import logging
import math
import os
import pickle

import einops
import imageio
import numpy as np
import torch

import wandb

log = logging.getLogger(__name__)
from diwa.agent.finetune.train_ppo_agent import TrainPPOAgent
from diwa.utils.scheduler import CosineAnnealingWarmupRestarts
from diwa.utils.timer import Timer


class TrainMBPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

        assert self.logprob_batch_size % self.n_envs == 0, "logprob_batch_size must be divisible by n_envs"

    def get_init_obs(self, n_envs):
        """
        Get initial observations for the environment.
        This function is used to reset a dummy environment and get the initial observations.
        """
        if self.env_type == "calvin":
            obs = self.get_init_obs_calvin(n_envs)
        elif self.env_type == "libero":
            obs = self.get_init_obs_libero(n_envs)
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")
        return obs

    def get_init_obs_calvin(self, n_envs):
        obs = [self.venv.reset()[0] for _ in range(n_envs)]
        robot_obs = np.array([ob["robot_obs"][-1] for ob in obs])
        rgb_static = np.array([ob["rgb_static"][-1] for ob in obs])
        rgb_gripper = np.array([ob["rgb_gripper"][-1] for ob in obs])

        obs = {
            "robot_obs": robot_obs,
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
        }
        return obs

    def get_init_obs_libero(self, n_envs):
        # generate n_envs random indices
        rand_indices = np.random.choice(np.arange(self.env_init_obs["robot_obs"].shape[0]), n_envs, replace=True)
        robot_obs = self.env_init_obs["robot_obs"][rand_indices]
        rgb_statics = self.env_init_obs["rgb_statics"][rand_indices]
        rgb_grippers = self.env_init_obs["rgb_grippers"][rand_indices]

        obs = {
            "robot_obs": robot_obs,
            "rgb_static": rgb_statics,
            "rgb_gripper": rgb_grippers,
        }
        return obs

    def get_terminated_from_reward(self, reward):
        terminated = np.zeros_like(reward)
        terminated[reward > 0] = 1
        return terminated.astype(bool)

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_wm_train_step = 0
        self.video_writer = None
        self.itr = 0

        while self.itr <= self.n_train_itr:
            # Define train or eval - all envs restart
            if self.itr == 0 or self.itr == self.n_train_itr:
                eval_mode = True
            else:
                eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()

            if not eval_mode:
                done_venv = np.zeros((1, self.n_envs))
                obs_trajs = {
                    "state": np.zeros(
                        (
                            self.n_steps,
                            self.n_envs,
                            self.n_cond_step,
                            self.obs_dim,
                        )
                    )
                }
                chains_trajs = np.zeros(
                    (
                        self.n_steps,
                        self.n_envs,
                        self.model.ft_denoising_steps + 1,
                        self.horizon_steps,
                        self.action_dim,
                    )
                )
                terminated_trajs = np.zeros((self.n_steps, self.n_envs))
                reward_trajs = np.zeros((self.n_steps, self.n_envs))
                firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
                rgb_static_trajs = np.zeros((self.n_steps, self.n_envs, 64, 64, 3))
                rgb_gripper_trajs = np.zeros((self.n_steps, self.n_envs, 64, 64, 3))

                # Initialize environment
                prev_obs_venv = self.get_init_obs(self.n_envs)
                wm_step_counter = np.zeros((self.n_envs)).astype(int)
                firsts_trajs[0] = 1

                if self.wme is not None:
                    for key in prev_obs_venv:
                        prev_obs_venv[key] = np.expand_dims(prev_obs_venv[key], 1)

                    wm_features, out_state = self.wme.get_zero_wm_features(prev_obs_venv, self.device)
                    prev_obs_venv = wm_features.cpu().numpy()
                    in_state = out_state

                if self.save_full_observations:  # state-only
                    obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                    obs_full_trajs = np.vstack((obs_full_trajs, prev_obs_venv[:, -1][None]))

                for step in range(self.n_steps):
                    if step % 10 == 0:
                        print(f"Processed env step {step} of {self.n_steps}")

                    with torch.no_grad():
                        cond = {"state": torch.from_numpy(prev_obs_venv).float().to(self.device)}
                        samples = self.model(
                            cond=cond,
                            deterministic=eval_mode,
                            return_chain=True,
                        )
                        output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                        chains_venv = samples.chains.cpu().numpy()  # n_env x denoising x horizon x act
                    action_venv = output_venv[:, : self.act_steps]
                    # Apply multi-step in wm action
                    (
                        obs_venv,
                        reward_venv,
                        dcd_rgb_static,
                        dcd_rgb_gripper,
                    ) = self.wme.multi_step(
                        torch.from_numpy(prev_obs_venv).squeeze().float().to(self.device),
                        torch.from_numpy(action_venv).reshape(self.n_envs, -1).float().to(self.device),
                    )
                    obs_venv = obs_venv.cpu().numpy()
                    wm_step_counter += self.act_steps
                    truncated_venv = wm_step_counter >= self.env.env.max_episode_steps
                    terminated_venv = self.get_terminated_from_reward(reward_venv)
                    done_venv = terminated_venv | truncated_venv

                    obs_trajs["state"][step] = prev_obs_venv
                    chains_trajs[step] = chains_venv
                    reward_trajs[step] = reward_venv
                    terminated_trajs[step] = terminated_venv
                    firsts_trajs[step + 1] = done_venv

                    # To render videos later
                    rgb_static_trajs[step] = dcd_rgb_static
                    rgb_gripper_trajs[step] = dcd_rgb_gripper

                    cnt_wm_train_step += self.n_envs * self.act_steps

                    # If an episode is done, reset environment for that env
                    if sum(done_venv) > 0:
                        new_obs_venv = self.get_init_obs(int(sum(done_venv)))

                        # WM Encoder
                        if self.wme is not None:
                            for key in new_obs_venv:
                                new_obs_venv[key] = np.expand_dims(new_obs_venv[key], 1)
                            wm_features, out_state = self.wme.get_zero_wm_features(new_obs_venv, self.device)
                            new_obs_venv = wm_features.squeeze().cpu().numpy()

                        obs_venv[done_venv.astype(bool)] = new_obs_venv
                        wm_step_counter[done_venv.astype(bool)] = 0

                    prev_obs_venv = np.expand_dims(obs_venv, 1)

                avg_episode_reward = np.sum(reward_trajs) / np.sum(firsts_trajs[1:, :])
                num_episode_finished = np.sum(firsts_trajs[1:, :])
                obs_venv = {"state": obs_venv}

                # Randomly choose an episode and render it
                if self.itr % self.render_freq == 0:
                    train_rgb_static_path = os.path.join(self.render_dir, f"train-rgb-static-{self.itr}.mp4")
                    train_rgb_gripper_path = os.path.join(self.render_dir, f"train-rgb-gripper-{self.itr}.mp4")

                    if self.video_writer is None:
                        self.video_writer_rgb_static = imageio.get_writer(
                            train_rgb_static_path, fps=int(30 / self.act_steps)
                        )
                        self.video_writer_rgb_gripper = imageio.get_writer(
                            train_rgb_gripper_path, fps=int(30 / self.act_steps)
                        )
                    train_env_id = np.random.randint(0, self.n_envs)
                    ep_ends = np.where(firsts_trajs[1:, train_env_id])[0]
                    c_ = 0
                    failed = False
                    while len(ep_ends) == 0:
                        train_env_id = np.random.randint(0, self.n_envs)
                        ep_ends = np.where(firsts_trajs[1:, train_env_id])[0]
                        c_ += 1
                        if c_ > self.n_envs:
                            log.info("No terminated episode found for rendering")
                            failed = True
                            break
                    if not failed and len(ep_ends) > 1:
                        rand_ind = np.random.randint(1, len(ep_ends))
                        start_ind = ep_ends[rand_ind - 1] + 1
                        end_ind = ep_ends[rand_ind] + 1
                        for idx in range(start_ind, end_ind):
                            self.video_writer_rgb_static.append_data(
                                rgb_static_trajs[idx, train_env_id].astype("uint8")
                            )
                            self.video_writer_rgb_gripper.append_data(
                                rgb_gripper_trajs[idx, train_env_id].astype("uint8")
                            )
                        self.video_writer_rgb_static.close()
                        self.video_writer_rgb_gripper.close()
                        self.video_writer_rgb_static = None
                        self.video_writer_rgb_gripper = None
                    else:
                        self.video_writer_rgb_static = None
                        self.video_writer_rgb_gripper = None

            else:
                episode_rewards = []
                episode_lengths = []
                if self.env_type == "calvin":
                    robot_obs = self.env.robot_obs
                    scene_obs = self.env.scene_obs
                    total_episodes_eval = robot_obs.shape[0]
                elif self.env_type == "libero":
                    init_states = self.env.init_states
                    total_episodes_eval = init_states.shape[0]
                options_venv = [{} for _ in range(total_episodes_eval)]
                if self.n_render == 1:
                    rand_ind = np.random.randint(0, total_episodes_eval)
                    options_venv[rand_ind] = {
                        "video_path": os.path.join(
                            self.render_dir,
                            f"itr-{self.itr}_trial-{rand_ind}.mp4",
                        )
                    }
                elif self.n_render == -1:
                    for i in range(total_episodes_eval):
                        options_venv[i] = {
                            "video_path": os.path.join(
                                self.render_dir,
                                f"itr-{self.itr}_trial-{i}.mp4",
                            )
                        }

                for i in range(total_episodes_eval):
                    if i % 10 == 0:
                        print(f"Processed episode {i} of {total_episodes_eval}")
                    prev_obs_venv = {}
                    if self.env_type == "calvin":
                        prev_obs_venv["state"], _ = self.env.reset(
                            robot_obs=robot_obs[i],
                            scene_obs=scene_obs[i],
                            options=options_venv[i],
                        )
                    elif self.env_type == "libero":
                        prev_obs_venv["state"], _ = self.env.reset(
                            init_states=init_states[i],
                            options=options_venv[i],
                        )

                    # WM Encoder
                    if self.wme is not None:
                        for key in prev_obs_venv["state"]:
                            prev_obs_venv["state"][key] = np.expand_dims(
                                np.expand_dims(prev_obs_venv["state"][key][-1, :], 0), 0
                            )

                        wm_features, out_state = self.wme.get_zero_wm_features(prev_obs_venv["state"], self.device)
                        prev_obs_venv["state"] = wm_features.cpu().numpy()
                        in_state = out_state
                        prev_done_venv = np.zeros((1)).astype(bool)

                    for step in range(self.n_steps):
                        # Select action
                        with torch.no_grad():
                            cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                            samples = self.model(
                                cond=cond,
                                deterministic=eval_mode,
                                return_chain=True,
                            )
                            output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                        action_venv = output_venv[:, : self.act_steps]
                        # Apply multi-step action
                        (
                            obs_venv,
                            reward_venv,
                            terminated_venv,
                            truncated_venv,
                            info_venv,
                        ) = self.env.step(np.squeeze(action_venv))
                        done_venv = terminated_venv | truncated_venv

                        if reward_venv > 0:
                            episode_rewards.append(reward_venv)
                        if done_venv:
                            episode_lengths.append(step)
                            break

                        # WM Encoder
                        if self.wme is not None:
                            for key in obs_venv:
                                obs_venv[key] = np.expand_dims(obs_venv[key], 0)
                            wm_features, out_state = self.wme.get_hist_wm_features(
                                obs_venv,
                                action_venv,
                                prev_done_venv,
                                in_state,
                                self.device,
                            )
                            prev_done_venv = np.array(done_venv).astype(bool)
                            obs_venv = wm_features.cpu().numpy()
                            in_state = out_state

                        prev_obs_venv = {"state": obs_venv}

                avg_episode_reward = np.sum(episode_rewards) / total_episodes_eval
                avg_episode_length = np.sum(episode_lengths) / total_episodes_eval
                success_rate = avg_episode_reward
                num_episode_finished = len(episode_rewards)
                avg_best_reward = avg_episode_reward

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    obs_trajs["state"] = torch.from_numpy(obs_trajs["state"]).float().to(self.device)

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(self.n_envs * self.n_steps / self.logprob_batch_size)
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack((values_trajs, values.reshape(-1, self.n_envs)))
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {"state": torch.from_numpy(obs_venv["state"]).float().to(self.device)}
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = self.model.critic(obs_venv_ts).reshape(1, -1).cpu().numpy()
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs).float().to(self.device),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = torch.tensor(returns_trajs).float().to(self.device).reshape(-1)
                values_k = torch.tensor(values_trajs).float().to(self.device).reshape(-1)
                advantages_k = torch.tensor(advantages_trajs).float().to(self.device).reshape(-1)
                logprobs_k = torch.tensor(logprobs_trajs).float().to(self.device)

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs
                inds_k = np.arange(total_steps)
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    # for each epoch, go through all data in batches
                    flag_break = False
                    np.random.shuffle(inds_k)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        obs_b = {"state": obs_k["state"][inds_b]}
                        chains_b = chains_k[inds_b]
                        returns_b = returns_k[inds_b]
                        values_b = values_k[inds_b]
                        advantages_b = advantages_k[inds_b]
                        logprobs_b = logprobs_k[inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(),
                                    self.max_grad_norm,
                                )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}")

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Plot state trajectories (only in D3IL)
            if self.itr % self.render_freq == 0 and self.n_render > 0 and self.traj_plotter is not None:
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_wm_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                                "avg episode length - eval": avg_episode_length,
                                "total env steps - eval": cnt_wm_train_step,
                                "itr - eval": self.itr,
                            },
                            step=self.itr,
                            commit=False,
                        )
                        if self.n_render > 0:
                            if self.n_render == 1:
                                wandb.log(
                                    {"video - eval": wandb.Video(options_venv[rand_ind]["video_path"], format="mp4")},
                                    step=self.itr,
                                    commit=False,
                                )
                            else:
                                # worry about this case later
                                for env_ind in range(self.n_render):
                                    wandb.log(
                                        {
                                            f"video - {env_ind}": wandb.Video(
                                                options_venv[env_ind]["video_path"], format="mp4"
                                            )
                                        },
                                        step=self.itr,
                                        commit=False,
                                    )
                        elif self.n_render == -1:
                            for env_ind in range(len(options_venv)):
                                wandb.log(
                                    {
                                        f"video - {env_ind}": wandb.Video(
                                            options_venv[env_ind]["video_path"], format="mp4"
                                        )
                                    },
                                    step=self.itr,
                                    commit=False,
                                )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_wm_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        if self.itr % self.render_freq == 0:
                            if os.path.exists(train_rgb_static_path):
                                wandb.log(
                                    {"video - train rgb static": wandb.Video(train_rgb_static_path, format="mp4")},
                                    step=self.itr,
                                    commit=False,
                                )
                            if os.path.exists(train_rgb_gripper_path):
                                wandb.log(
                                    {"video - train rgb gripper": wandb.Video(train_rgb_gripper_path, format="mp4")},
                                    step=self.itr,
                                    commit=False,
                                )
                        wandb.log(
                            {
                                "total env step": cnt_wm_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                                "itr": self.itr,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
