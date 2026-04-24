import os

import hydra
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import time

from diwa.env.wrapper.calvin_image import CALVINImageWrapper
from diwa.wm.wrapper.visionwm_cls import VisionWMWrapper
# from lat2img.dual_cond_diffusion import DualCondDiffusionModel
from lat2img.dual_cond_shortcut_fm import DualCondShortcutFlowMatching

""" This script visually tests the quality of a given world model, rendering the results as GIFs.
For each skill, it randomly samples an episode and performs the following:
1. It renders the true observations from the environment.
2. It decodes the world model latent representations to RGB images.
3. It performs an imagination rollout with true actions and decodes the imagined latents to RGB images.
"""


@hydra.main(config_path=f"{os.environ['DIWA_ROOT_DIR']}/config/env/", config_name="calvin")
def main(cfg):
    wm_path = f"{os.environ['DIWA_LOG_DIR']}/wm/ckpts/vision64-ac5.ckpt"
    stats_path = f"{os.environ['DIWA_DATA_DIR']}/statistics.yaml"
    device = "cuda:3"
    device2 = "cuda:3"
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

    outdir = f"{os.environ['DIWA_ROOT_DIR']}/scripts/tests/gifs/lat2img/60M-ShortcutFM-1/"
    os.makedirs(outdir, exist_ok=True)

    env = CALVINImageWrapper(
        cfg,
        skill_name="open_drawer",
        normalization_path=None,
        max_episode_steps=128,
        load_scene_from_dataset=None,
        device="cpu",
    )
    wmw = VisionWMWrapper(ckpt_path=wm_path, device=device, stats_path=stats_path)

    lat2img_ckpt = f"{os.environ['DIWA_ROOT_DIR']}/lat2img/logs/2026-04-19/11-51-30/checkpoints/last.ckpt"
    # lat2img_ckpt = f"{DIWA_ROOT_DIR}/lat2img/logs/2026-03-13/12-17-54/checkpoints/last.ckpt"
    lat2img_model = DualCondShortcutFlowMatching.load_from_checkpoint(lat2img_ckpt).to(device2)
    times = []
    for skill_name in tqdm(skills, desc="Skills"):
        env.skill_name = skill_name

        # Load data
        data_path_feat = f"{os.environ['DIWA_DATA_DIR']}/expert/calvin-feat-50/{skill_name}/train.npz"
        data_path = f"{os.environ['DIWA_DATA_DIR']}/expert/calvin-state-raw-50/{skill_name}/train.npz"
        data_feat = np.load(data_path_feat, allow_pickle=True)
        data = np.load(data_path, allow_pickle=True)

        traj_lengths = data["traj_lengths"]

        # Get an episode based on the trajectory lengths
        # episode_idx = np.random.randint(len(traj_lengths))
        episode_idx = 21
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
        action_chunk_size = 5
        for i in range(len(obs)-1):
            env.step(actions[i])
            if (i + 1) % action_chunk_size == 0:
                render = env.render()
                true_frames.append(Image.fromarray(render.astype(np.uint8)))

        # Create GIF of TRUE episode
        true_frames[0].save(
            f"{outdir}/{skill_name}_{comment}_true.gif",
            save_all=True,
            append_images=true_frames[1:],
            duration=30,
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
        rgb_statics = []
        rgb_grippers = []
        latents = []
        action_chunk_size = 5
        action_chunk = []
        if actions.shape[0] % action_chunk_size != 0:
            actions = np.concatenate((actions, np.zeros((action_chunk_size - actions.shape[0] % action_chunk_size, 35))))
        actions = actions.reshape(-1, 35)
        for i in range(len(actions)):
            action_chunk = torch.from_numpy(actions[i]).to(device).unsqueeze(0).float()
            pred_next_latent = wmw.wm_step(
                wm_features,
                action_chunk,
            )
            ##### Decoded images in [-1, 1]
            # rgb_s, rgb_g = wmw.wm.decoder(pred_next_latent)
            # rgb_statics.append(rgb_s)
            # rgb_grippers.append(rgb_g)
            # latents.append(pred_next_latent)
            #####
            rgb_s, rgb_g = wmw.decode_latent(pred_next_latent)
            imagined_frames.append(Image.fromarray(rgb_s.squeeze().cpu().numpy().astype(np.uint8)))
            imagined_frames_gripper.append(Image.fromarray(rgb_g.squeeze().cpu().numpy().astype(np.uint8)))

            wm_features = pred_next_latent

            rgb_s = Image.fromarray(rgb_s.squeeze().cpu().numpy().astype(np.uint8))
            rgb_s = rgb_s.convert("RGB")
            rgb_s = np.asarray(rgb_s, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]
            rgb_s = torch.from_numpy(rgb_s).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            rgb_statics.append(rgb_s)
            rgb_g = Image.fromarray(rgb_g.squeeze().cpu().numpy().astype(np.uint8))
            rgb_g = rgb_g.convert("RGB")
            rgb_g = np.asarray(rgb_g, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]
            rgb_g = torch.from_numpy(rgb_g).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            rgb_grippers.append(rgb_g)
            latents.append(pred_next_latent)

        # Create GIF of imagined episode
        imagined_frames[0].save(
            f"{outdir}/{skill_name}_{comment}_imagined.gif",
            save_all=True,
            append_images=imagined_frames[1:],
            duration=30,
        )

        imagined_frames_gripper[0].save(
            f"{outdir}/{skill_name}_{comment}_g_imagined.gif",
            save_all=True,
            append_images=imagined_frames_gripper[1:],
            duration=30,
        )

        # Using Lat2Img to decode latents to high-res images
        latents = torch.cat(latents, dim=0)  # B x latent_dim
        latents_tensor = latents.to(device2).float()

        rgb_statics = torch.cat(rgb_statics, dim=0)
        rgb_grippers = torch.cat(rgb_grippers, dim=0)

        cond_img = torch.cat([rgb_statics, rgb_grippers], dim=1).to(device2).float()  # B x 2C x H x W

        # Split into batches to avoid OOM
        batch_size = 35
        all_samples = []
        
        for i in range(0, latents_tensor.size(0), batch_size):
            latents_batch = latents_tensor[i : i + batch_size]
            cond_img_batch = cond_img[i : i + batch_size]
            time_start = time.time()
            samples_batch, _ = lat2img_model.sample(latents_batch, cond_img=cond_img_batch, num_inference_steps=1)
            time_end = time.time()
            times.append(time_end - time_start)
            print(f"Time taken for batch {i} to {i + batch_size} Lat2Img: {time_end - time_start} seconds")
            all_samples.append(samples_batch)
            print(f"Processed latents {i} to {i + batch_size} / {latents_tensor.size(0)}")
        samples = torch.cat(all_samples, dim=0)
        print("samples shape: ", samples.shape)

        highres_frames = []
        highres_frames_gripper = []
        for i in range(samples.shape[0]):
            sample = samples[i]
            static_img = sample[:3, :, :].detach().cpu().numpy().transpose(1, 2, 0)
            gripper_img = sample[3:6, :, :].detach().cpu().numpy().transpose(1, 2, 0)

            static_img = (static_img * 255).astype(np.uint8)
            gripper_img = (gripper_img * 255).astype(np.uint8)

            highres_frames.append(Image.fromarray(static_img))
            highres_frames_gripper.append(Image.fromarray(gripper_img))

        highres_frames[0].save(
            f"{outdir}/{skill_name}_{comment}_highres_imagined.gif",
            save_all=True,
            append_images=highres_frames[1:],
            duration=30,
        )
        highres_frames_gripper[0].save(
            f"{outdir}/{skill_name}_{comment}_g_highres_imagined.gif",
            save_all=True,
            append_images=highres_frames_gripper[1:],
            duration=30,
        )
    
    print(f"Time taken for all skills: {np.sum(times)} seconds for {len(skills)} skills")

    print("Saved gifs at ", outdir)


if __name__ == "__main__":
    main()
