import os

import hydra
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from diwa.env.wrapper.calvin_image import CALVINImageWrapper
from diwa.wm.wrapper.visionwm_cls import VisionWMWrapper

""" This script visually tests the quality of a given world model, rendering the results as GIFs.
For each skill, it randomly samples an episode and performs the following:
1. It renders the true observations from the environment.
2. It decodes the world model latent representations to RGB images.
3. It performs an imagination rollout with true actions and decodes the imagined latents to RGB images.
"""


@hydra.main(config_path=f"{os.environ['DIWA_ROOT_DIR']}/config/env/", config_name="calvin")
def main(cfg):
    wm_path = f"{os.environ['DIWA_LOG_DIR']}/wm/ckpts/vision64-ac5.ckpt"
    stats_path = f"{os.environ['DIWA_DATA_DIR']}/task_D_D_64/validation/statistics.yaml"
    device = "cuda:0"
    skills = [
        "open_drawer",
        "close_drawer",
        "move_slider_left",
        "move_slider_right",
        "turn_on_lightbulb",
        "turn_off_lightbulb",
        "turn_on_led",
        "turn_off_led",
    ]
    comment = "one_step"

    outdir = f"{os.environ['DIWA_ROOT_DIR']}/scripts/tests/gifs/temp/"
    os.makedirs(outdir, exist_ok=True)
    action_chunk_size = 5
    env = CALVINImageWrapper(
        cfg,
        skill_name="open_drawer",
        normalization_path=None,
        max_episode_steps=128,
        load_scene_from_dataset=None,
        device="cpu",
    )
    wmw = VisionWMWrapper(ckpt_path=wm_path, device=device, stats_path=stats_path)
    for skill_name in tqdm(skills, desc="Skills"):
        env.skill_name = skill_name

        # Load data
        data_path_feat = f"{os.environ['DIWA_DATA_DIR']}/expert/calvin-feat-50/{skill_name}/train.npz"
        data_path = f"{os.environ['DIWA_DATA_DIR']}/expert/calvin-state-raw-50/{skill_name}/train.npz"
        data_feat = np.load(data_path_feat, allow_pickle=True)
        data = np.load(data_path, allow_pickle=True)

        traj_lengths = data["traj_lengths"]

        # Get an episode based on the trajectory lengths
        episode_idx = np.random.randint(len(traj_lengths))
        while traj_lengths[episode_idx] < 60:
            episode_idx = np.random.randint(len(traj_lengths))
        start_idx = int(np.sum(traj_lengths[:episode_idx]))
        end_idx = int(start_idx + traj_lengths[episode_idx])

        obs = data["states"][start_idx:end_idx]
        actions = data["actions"][start_idx:end_idx]
        obs_feat = data_feat["states"][start_idx:end_idx]

        # 1. TRUE EPISODE
        # Render TRUE observation episodes
        robot_obs, scene_obs = np.split(obs[0], [15])
        env.reset(robot_obs, scene_obs)
        render = env.render()
        true_frames = [Image.fromarray(render.astype(np.uint8))]
        for i in range(0, len(obs), action_chunk_size):
            for act in actions[i:i+action_chunk_size]:
                env.step(act)
            render = env.render()
            true_frames.append(Image.fromarray(render.astype(np.uint8)))

        # Create GIF of TRUE episode
        true_frames[0].save(
            f"{outdir}/{skill_name}_{comment}_true.gif",
            save_all=True,
            append_images=true_frames[1:],
            duration=60,
        )

        # 2. DECODED EPISODE
        # Run through features and decode
        robot_obs, scene_obs = np.split(obs[0], [15])
        env.reset(robot_obs, scene_obs)
        render = env.render()
        dcd_frames = []
        dcd_frames_gripper = []
        for i in range(0, len(obs_feat), action_chunk_size):
            latent = torch.from_numpy(obs_feat[i]).to(device).unsqueeze(0).float()
            rgb_s, rgb_g = wmw.decode_latent(latent)
            dcd_frames.append(Image.fromarray(rgb_s.squeeze().cpu().numpy().astype(np.uint8)))
            dcd_frames_gripper.append(Image.fromarray(rgb_g.squeeze().cpu().numpy().astype(np.uint8)))

        # Create GIF of decoded episode
        dcd_frames[0].save(
            f"{outdir}/{skill_name}_{comment}_decoded.gif",
            save_all=True,
            append_images=dcd_frames[1:],
            duration=60,
        )

        dcd_frames_gripper[0].save(
            f"{outdir}/{skill_name}_{comment}_g_decoded.gif",
            save_all=True,
            append_images=dcd_frames_gripper[1:],
            duration=60,
        )

        # 3. IMAGINED EPISODE
        # Get zero wm features
        wm_features = torch.from_numpy(obs_feat[0]).to(device).unsqueeze(0).float()

        # Run through the episode
        robot_obs, scene_obs = np.split(obs[0], [15])
        env.reset(robot_obs, scene_obs)
        render = env.render()
        imagined_frames = []
        imagined_frames_gripper = []
        for i in range(0, len(obs_feat), action_chunk_size):
            pred_next_latent = wmw.wm_step(
                wm_features,
                torch.from_numpy(actions[i:i+action_chunk_size].reshape(1, -1)).to(device).float(),
            )
            rgb_s, rgb_g = wmw.decode_latent(pred_next_latent)
            imagined_frames.append(Image.fromarray(rgb_s.squeeze().cpu().numpy().astype(np.uint8)))
            imagined_frames_gripper.append(Image.fromarray(rgb_g.squeeze().cpu().numpy().astype(np.uint8)))
            wm_features = pred_next_latent

        # Create GIF of imagined episode
        imagined_frames[0].save(
            f"{outdir}/{skill_name}_{comment}_imagined.gif",
            save_all=True,
            append_images=imagined_frames[1:],
            duration=60,
        )

        imagined_frames_gripper[0].save(
            f"{outdir}/{skill_name}_{comment}_g_imagined.gif",
            save_all=True,
            append_images=imagined_frames_gripper[1:],
            duration=60,
        )

    print("Saved gifs at ", outdir)


if __name__ == "__main__":
    main()
