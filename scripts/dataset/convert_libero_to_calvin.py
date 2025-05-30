import os
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="../../config/dataset", config_name="from_libero_to_calvin")
def from_libero_to_calvin(cfg: DictConfig) -> None:
    """Convert the LIBERO-X dataset to CALVIN format."""

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    # Check if the input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    input_dir = input_dir / cfg.benchmark
    output_dir = output_dir / cfg.benchmark_calvin_fmt

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "lang_annotations", exist_ok=True)

    # Get numbers of tasks in the benchmark (counting the .hdf5 files)
    tasks = list(input_dir.glob("*.hdf5"))
    num_tasks = len(tasks)
    print(f"Number of tasks: {num_tasks}")

    ep_start_end_ids = []
    ep_lens = []
    idx = 0
    ann = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }
    for task in tqdm(tasks):
        with h5py.File(task, "r") as f:
            # Get the task name
            # The following line only works with the real LIBERO dataset
            # task_name = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
            # task_name = task_name.replace(" ", "_")
            # For the regenerated dataset, we need to extract the task name from the filename (it is ugly but works)
            task_name = str(task).split("/")[-1].split(".")[0].split("_SCENE")[1][2:-5]
            task_ann = task_name.replace("_", " ")

            # Get number of episodes for the task
            num_episodes = len(f["data"].keys())
            print(f"Number of episodes for task {task_name}: {num_episodes}")

            for ep_name in tqdm(f["data"].keys()):
                # Get the episode data
                ep_data = f[f"data/{ep_name}/"]

                rgb_static = np.array(ep_data["obs/agentview_rgb"])
                rgb_gripper = np.array(ep_data["obs/eye_in_hand_rgb"])
                robot_obs = np.concatenate(
                    [
                        np.array(ep_data["obs/ee_pos"]),
                        np.array(ep_data["obs/ee_ori"]),
                        np.array(ep_data["obs/gripper_states"]),
                        np.array(ep_data["obs/joint_states"]),
                    ],
                    axis=-1,
                )
                scene_obs = np.array(ep_data["states"])
                actions = np.array(ep_data["actions"])

                start_idx = idx
                for i in range(len(robot_obs)):
                    output_file = output_dir / f"episode_{idx:06d}.npz"
                    np.savez_compressed(
                        output_file,
                        rgb_static=np.flipud(rgb_static[i]),
                        rgb_gripper=np.flipud(rgb_gripper[i]),
                        robot_obs=robot_obs[i],
                        scene_obs=scene_obs[i],
                        rel_actions=actions[i],
                    )
                    idx += 1
                ep_start_end_ids.append([start_idx, idx - 1])
                ep_lens.append(idx - start_idx)
                ann["language"]["task"].append(task_name)
                ann["language"]["ann"].append(task_ann)
                ann["info"]["indx"].append([start_idx, idx - 1])

    # Save the annotation file at the end
    np.save(output_dir / "lang_annotations" / "auto_lang_ann.npy", ann)
    np.save(output_dir / "ep_start_end_ids.npy", np.array(ep_start_end_ids))
    np.save(output_dir / "ep_lens.npy", np.array(ep_lens))


if __name__ == "__main__":
    from_libero_to_calvin()
