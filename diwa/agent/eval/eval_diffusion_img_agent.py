"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)
from diwa.agent.eval.eval_agent import EvalAgent
from diwa.utils.timer import Timer


class EvalImgDiffusionAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

    def run(self):
        # Start training loop
        timer = Timer()

        episode_rewards = []
        episode_lengths = []
        if self.env_type == "calvin":
            robot_obs = self.env.robot_obs
            scene_obs = self.env.scene_obs
            total_episodes_eval = robot_obs.shape[0]
        elif self.env_type == "libero":
            init_states = self.env.init_states
            total_episodes_eval = init_states.shape[0]
        options = [{} for _ in range(total_episodes_eval)]
        if self.n_render == 1:
            rand_ind = np.random.randint(0, total_episodes_eval)
            options[rand_ind] = {
                "video_path": os.path.join(
                    self.render_dir,
                    f"trial-{rand_ind}.mp4",
                )
            }
        elif self.n_render == -1:
            for i in range(total_episodes_eval):
                options[i] = {
                    "video_path": os.path.join(
                        self.render_dir,
                        f"trial-{i}.mp4",
                    )
                }

        # Reset env before iteration starts
        self.model.eval()

        for i in range(total_episodes_eval):
            if i % 10 == 0:
                print(f"Processed episode {i} of {total_episodes_eval}")
            if self.env_type == "calvin":
                prev_obs_venv, _ = self.env.reset(
                    robot_obs=robot_obs[i],
                    scene_obs=scene_obs[i],
                    options=options[i],
                )
            elif self.env_type == "libero":
                prev_obs_venv = self.env.reset(
                    init_states=init_states[i],
                    options=options[i],
                )
            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                # Select action
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(prev_obs_venv[key]).float().to(self.device) for key in self.obs_dims
                    }  # batch each type of obs and put into dict
                    cond["rgb"] = cond["rgb"].unsqueeze(0)  # (1, T, C, H, W)
                    samples = self.model(cond=cond, deterministic=True)
                    output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.env.step(
                    np.squeeze(action_venv)
                )
                done_venv = terminated_venv | truncated_venv

                if reward_venv > 0:
                    episode_rewards.append(reward_venv)
                if done_venv:
                    episode_lengths.append(step)
                    break

                # update for next step
                prev_obs_venv = obs_venv

        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        avg_episode_reward = np.sum(episode_rewards) / total_episodes_eval
        avg_episode_length = np.sum(episode_lengths) / total_episodes_eval
        success_rate = avg_episode_reward
        num_episode_finished = len(episode_rewards)
        avg_best_reward = avg_episode_reward

        # Log loss and save metrics
        time = timer()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            eval_episode_length=avg_episode_length,
            time=time,
        )
