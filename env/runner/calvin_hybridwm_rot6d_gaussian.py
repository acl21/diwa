import os
import logging
import wandb
import torch
import numpy as np
import yaml
from util.transforms import (
    NormalizeVectorMinMax,
    UnnormalizeVectorMinMax,
    ScaleImageTensor,
    UnNormalizeImageTensor,
)
from torchvision.transforms import Normalize, Compose
from env.runner.calvin_hybridwm_rot6d import CALVINEnvRunner as ParentCALVINEnvRunner

log = logging.getLogger(__name__)


def transpose_tensor(tensor):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    return torch.transpose(tensor, 0, 1)


class CALVINEnvRunner(ParentCALVINEnvRunner):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def run(self, epoch, model, venv, wme):
        model.eval()
        device = model.device

        episode_rewards = []
        episode_lengths = []
        robot_obs = venv.robot_obs
        scene_obs = venv.scene_obs
        total_episodes_eval = robot_obs.shape[0]
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
            prev_obs_venv["state"], _ = venv.reset(
                robot_obs=robot_obs[i],
                scene_obs=scene_obs[i],
                options=options_venv[i],
            )

            # WM Encoder
            if wme is not None:
                if type(prev_obs_venv["state"]) is not dict:
                    prev_obs_venv["state"] = np.expand_dims(
                        np.expand_dims(prev_obs_venv["state"][-1, :], 0), 0
                    )
                else:
                    for key in prev_obs_venv["state"]:
                        prev_obs_venv["state"][key] = np.expand_dims(
                            np.expand_dims(prev_obs_venv["state"][key][-1, :], 0), 0
                        )

                wm_features, out_state = wme.get_zero_wm_features(
                    prev_obs_venv["state"], device
                )
                prev_obs_venv["state"] = wm_features.cpu().numpy()
                in_state = out_state
                # prev_done_venv = np.zeros((self.n_envs)).astype(bool)
                prev_done_venv = np.zeros((1)).astype(bool)

            for step in range(self.n_steps):
                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(device)
                    }
                    samples = model(cond=cond, deterministic=True)
                    output_venv = samples.cpu().numpy()  # n_env x horizon x act
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

                # WM Encoder
                if wme is not None:
                    if type(obs_venv) is not dict:
                        obs_venv = np.expand_dims(obs_venv, 0)
                    else:
                        for key in obs_venv:
                            obs_venv[key] = np.expand_dims(obs_venv[key], 0)
                    wm_features, out_state = wme.get_hist_wm_features(
                        obs_venv,
                        action_venv,
                        prev_done_venv,
                        in_state,
                        device,
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
                        {f"video": wandb.Video(options_venv[rand_ind]["video_path"])},
                        step=epoch,
                        commit=False,
                    )
                else:
                    for env_ind in range(self.n_render):
                        wandb.log(
                            {
                                f"video - {env_ind}": wandb.Video(
                                    options_venv[env_ind]["video_path"]
                                )
                            },
                            step=epoch,
                            commit=False,
                        )
