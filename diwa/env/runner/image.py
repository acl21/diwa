import logging
import os

import numpy as np
import torch

from diwa.env.runner.base_runner import BaseEnvRunner
import wandb

log = logging.getLogger(__name__)


class EnvRunner(BaseEnvRunner):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def run(self, epoch, model, venv, wme):
        model.eval()
        device = model.device

        episode_rewards = []
        episode_lengths = []
        if self.env_type == "calvin":
            robot_obs = venv.robot_obs
            scene_obs = venv.scene_obs
            total_episodes_eval = robot_obs.shape[0]
        elif self.env_type == "libero":
            init_states = venv.init_states
            total_episodes_eval = init_states.shape[0]
        options_venv = [{} for _ in range(total_episodes_eval)]
        rand_ind = np.random.randint(0, total_episodes_eval)
        options_venv[rand_ind] = {
            "video_path": os.path.join(
                self.render_dir,
                f"itr-trial-{rand_ind}.mp4",
            )
        }

        for i in range(total_episodes_eval):
            if i % 10 == 0:
                print(f"Processed episode {i} of {total_episodes_eval}")
            prev_obs_venv = {}
            if self.env_type == "calvin":
                prev_obs_venv, _ = venv.reset(
                    robot_obs=robot_obs[i],
                    scene_obs=scene_obs[i],
                    options=options_venv[i],
                )
            elif self.env_type == "libero":
                prev_obs_venv, _ = venv.reset(
                    init_states=init_states[i],
                    options=options_venv[i],
                )

            for step in range(self.n_steps):
                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).unsqueeze(0).float().to(device),
                        "rgb": torch.from_numpy(prev_obs_venv["rgb"]).unsqueeze(0).float().to(device),
                    }
                    samples = model(cond=cond, deterministic=True)
                    output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                action_venv = output_venv[:, : self.act_steps]
                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = venv.step(np.squeeze(action_venv))
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

        if self.use_wandb:
            wandb.log(
                {
                    "success rate - eval": success_rate,
                    "avg episode reward - eval": avg_episode_reward,
                    "avg best reward - eval": avg_best_reward,
                    "num episode - eval": num_episode_finished,
                    "avg episode length - eval": avg_episode_length,
                },
                step=epoch,
                commit=False,
            )
            if self.n_render > 0:
                if self.n_render == 1:
                    wandb.log(
                        {"video": wandb.Video(options_venv[rand_ind]["video_path"], format="mp4")},
                        step=epoch,
                        commit=False,
                    )
                else:
                    for env_ind in range(self.n_render):
                        wandb.log(
                            {f"video - {env_ind}": wandb.Video(options_venv[env_ind]["video_path"], format="mp4")},
                            step=epoch,
                            commit=False,
                        )
