"""
Modified from https://github.com/irom-princeton/dppo
"""

"""
DPPO fine-tuning for pixel observations.

"""

import logging
import math
import os
import pickle

import einops
import numpy as np
import torch

import wandb

log = logging.getLogger(__name__)
from diwa.agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from diwa.model.common.modules import RandomShiftsAug
from diwa.utils.timer import Timer


class TrainPPOImgDiffusionAgent(TrainPPODiffusionAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr <= self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            if self.itr == 0 or self.itr == self.n_train_itr:
                eval_mode = True
            else:
                eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            if not eval_mode:
                # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
                firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
                self.venv._options = options_venv
                prev_obs_venv = self.venv.reset()
                firsts_trajs[0] = 1

                # Holder
                obs_trajs = {
                    k: np.zeros(
                        (
                            self.n_steps,
                            self.n_envs,
                            self.n_cond_step,
                            *self.obs_dims[k],
                        )
                    )
                    for k in self.obs_dims
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

                # Collect a set of trajectories from env
                for step in range(self.n_steps):
                    if step % 10 == 0:
                        print(f"Processed step {step} of {self.n_steps}")

                    # Select action
                    with torch.no_grad():
                        cond = {
                            key: torch.from_numpy(prev_obs_venv[key]).float().to(self.device) for key in self.obs_dims
                        }
                        samples = self.model(
                            cond=cond,
                            deterministic=eval_mode,
                            return_chain=True,
                        )
                        output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                        chains_venv = samples.chains.cpu().numpy()  # n_env x denoising x horizon x act
                    action_venv = output_venv[:, : self.act_steps]

                    # Apply multi-step action
                    (
                        obs_venv,
                        reward_venv,
                        terminated_venv,
                        truncated_venv,
                        info_venv,
                    ) = self.venv.step(action_venv)
                    done_venv = terminated_venv | truncated_venv

                    for k in obs_trajs:
                        obs_trajs[k][step] = prev_obs_venv[k]
                    chains_trajs[step] = chains_venv
                    reward_trajs[step] = reward_venv
                    terminated_trajs[step] = terminated_venv
                    firsts_trajs[step + 1] = done_venv

                    # update for next step
                    prev_obs_venv = obs_venv

                    # count steps --- not acounting for done within action chunk
                    cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

                # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
                episodes_start_end = []
                for env_ind in range(self.n_envs):
                    env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                    for i in range(len(env_steps) - 1):
                        start = env_steps[i]
                        end = env_steps[i + 1]
                        if end - start > 1:
                            episodes_start_end.append((env_ind, start, end - 1))
                if len(episodes_start_end) > 0:
                    reward_trajs_split = [
                        reward_trajs[start : end + 1, env_ind] for env_ind, start, end in episodes_start_end
                    ]
                    num_episode_finished = len(reward_trajs_split)
                    episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
                    episode_best_reward = episode_reward
                    avg_episode_reward = np.mean(episode_reward)
                    avg_best_reward = np.mean(episode_best_reward)
                    success_rate = np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
                    episodes_start_end = np.array(episodes_start_end)
                    avg_episode_length = np.mean(episodes_start_end[:, 2] - episodes_start_end[:, 1] + 1)
                else:
                    episode_reward = np.array([])
                    num_episode_finished = 0
                    avg_episode_reward = 0
                    avg_best_reward = 0
                    success_rate = 0
                    avg_episode_length = 0
                    log.info("[WARNING] No episode completed within the iteration!")
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
                        prev_obs_venv, _ = self.env.reset(
                            robot_obs=robot_obs[i],
                            scene_obs=scene_obs[i],
                            options=options_venv[i],
                        )
                    elif self.env_type == "libero":
                        prev_obs_venv, _ = self.env.reset(
                            init_states=init_states[i],
                            options=options_venv[i],
                        )

                    for step in range(self.n_steps):
                        with torch.no_grad():
                            cond = {
                                key: torch.from_numpy(prev_obs_venv[key]).unsqueeze(0).float().to(self.device)
                                for key in self.obs_dims
                            }
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

                        prev_obs_venv = obs_venv

                avg_episode_reward = np.sum(episode_rewards) / total_episodes_eval
                avg_episode_length = np.sum(episode_lengths) / total_episodes_eval
                success_rate = avg_episode_reward
                num_episode_finished = len(episode_rewards)
                avg_best_reward = avg_episode_reward

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    # apply image randomization
                    obs_trajs["rgb"] = torch.from_numpy(obs_trajs["rgb"]).float().to(self.device)
                    obs_trajs["state"] = torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    if self.augment:
                        rgb = einops.rearrange(
                            obs_trajs["rgb"],
                            "s e t c h w -> (s e t) c h w",
                        )
                        rgb = self.aug(rgb)
                        obs_trajs["rgb"] = einops.rearrange(
                            rgb,
                            "(s e t) c h w -> s e t c h w",
                            s=self.n_steps,
                            e=self.n_envs,
                        )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(self.n_envs * self.n_steps / self.logprob_batch_size)
                    obs_ts = [{} for _ in range(num_split)]
                    for k in obs_trajs:
                        obs_k = einops.rearrange(
                            obs_trajs[k],
                            "s e ... -> (s e) ...",
                        )
                        obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                        for i, obs_t in enumerate(obs_ts_k):
                            obs_ts[i][k] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs, no_augment=True).cpu().numpy().flatten()
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
                    obs_venv_ts = {
                        key: torch.from_numpy(obs_venv[key]).float().to(self.device) for key in self.obs_dims
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = self.model.critic(obs_venv_ts, no_augment=True).reshape(1, -1).cpu().numpy()
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
                    k: einops.rearrange(
                        obs_trajs[k],
                        "s e ... -> (s e) ...",
                    )
                    for k in obs_trajs
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
                        obs_b = {k: obs_k[k][inds_b] for k in obs_k}
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
                        loss.backward()
                        # if (batch + 1) % self.grad_accumulate == 0:
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
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        log.info(f"run grad update at batch {batch}")
                        log.info(f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}")

                        # Stop gradient update if KL difference reaches target
                        if (
                            self.target_kl is not None
                            and approx_kl > self.target_kl
                            and self.itr >= self.n_critic_warmup_itr
                        ):
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
                    "step": cnt_train_step,
                }
            )
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
                                "avg episode length - eval": avg_episode_length,
                                "num episode - eval": num_episode_finished,
                                "total env steps - eval": cnt_train_step,
                            },
                            step=self.itr,
                            commit=False,
                        )
                        if self.n_render > 0:
                            if self.n_render == 1:
                                wandb.log(
                                    {"video - eval": wandb.Video(options_venv[rand_ind]["video_path"])},
                                    step=self.itr,
                                    commit=False,
                                )
                            else:
                                # worry about this case later
                                for env_ind in range(self.n_render):
                                    wandb.log(
                                        {f"video - {env_ind}": wandb.Video(options_venv[env_ind]["video_path"])},
                                        step=self.itr,
                                        commit=False,
                                    )
                        elif self.n_render == -1:
                            for env_ind in range(len(options_venv)):
                                wandb.log(
                                    {f"video - {env_ind}": wandb.Video(options_venv[env_ind]["video_path"])},
                                    step=self.itr,
                                    commit=False,
                                )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
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
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
