# DiWA: Diffusion Policy Adaptation with World Models

[[Paper]()]&nbsp;&nbsp;[[Website]()]

[Akshay L Chandra](https://akshaychandra.com/)<sup>1</sup>, [Iman Nematollahi](https://imanema.com/)<sup>1</sup>, [Chenguang Huang](https://scholar.google.com/citations?user=_rcR8TAAAAAJ&hl=en)<sup>2</sup>, [Tim Welschehold](https://rl.uni-freiburg.de/people/welschehold)<sup>1</sup> [Wolfram Burgard](https://scholar.google.com/citations?user=zj6FavAAAAAJ&hl=en)<sup>2</sup>, [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)<sup>1</sup>

<sup>1</sup>University of Freiburg, <sup>2</sup>University of Technology Nürnberg

<img src="https://github.com/acl21/diwa/blob/main/docs/diwa_overview.png" alt="drawing" width="100%"/>

> DiWA is an algorithmic framework for fine-tuning diffusion-based policies entirely inside frozen world models (learned from large play data).


## Installation
1. To begin, clone this repository locally
```bash
git clone --recurse-submodules https://github.com/acl21/diwa.git
cd diwa
```

2. Set environment variables for datasets and logging directory (default is `dataset/` and `logs/`), and set WandB entity (username or team name)
```bash
source scripts/set_path.sh
```

3. ⚠️ If you've already cloned the repo without `--recurse-submodules`, run:
```bash
git submodule update --init --recursive
```
- **Submodule:** `lumos`

    This repository includes `https://github.com/nematoli/lumos/` as a submodule for all things related to world model training and featurizing, tracking its `diwa` branch. If you want to inspect or update the submodule manually:
    ```bash
    cd DIWA_ROOT_DIR/lumos
    git checkout diwa
    git pull origin diwa
    ```
- **Submodule:** `calvin_env`

    This repository inclues `https://github.com/mees/calvin_env/` as a submodule for simulation experiments, tracking its `main` branch.

- **Submoudle:** `LIBERO`
    
    This repository includes `https://github.com/Lifelong-Robot-Learning/LIBERO` as a submodule for simulation experiments, tracking its `master` branch. 

- **(Optional) Submodule**: `robot_io`
    
    This repository inclues `https://github.com/acl21/robot_io/` as a submodule for real-world experiments, tracking its `main` branch.

4. Create and activate the conda environment, then install the dependencies:

```bash
cd DIWA_ROOT_DIR
conda create -n diwa python=3.10
conda activate diwa
sh install.sh
```

## Usage
### 0. Dataset
To download and preprocess datasets for DiWA, please follow [this guide](dataset/README.md).

### 1. World Model
**Note**: (TODO) You may skip world model training if you would like to use the default checkpoints (available for download).

#### 1.1 Training
Before running the following, see A.1 [here](dataset/README.md#1-play-data-for-world-model-training).

```bash
python scripts/train_wm.py trainer.devices=[<GPU-ID>]
```
#### 1.2 Featurizer
```bash
python scripts/featurizer.py device=<GPU-ID>
```
#### 1.3 (Optional) World Model Tests
```bash
(TODO)
```
### 2. Diffusion Policy Training
**Note**: (TODO) You may skip pre-training if you would like to use the default checkpoint (available for download) for fine-tuning. Before running the following, see A.2 [here](dataset/README.md#2-expert-data-for-diffusion-policy-training). All the configs for pre-training can be found under `config/<env>/pretrain/`.
```bash
python scripts/run.py --config-name=pre_diffusion_mlp_feat_vision --config-dir=config/calvin/pretrain/close_drawer
```
### 3. Reward Estimation
Before running the following, see A.3 [here](dataset/README.md#3-class-balanced-data-for-reward-classifier-training).
```bash
python scripts/rewcls/train_contrastive.py
```
### 4. Fine-tuning inside World Model
All the configs can be found under `config/<env>/finetune/`.
```bash
python scripts/run.py --config-name=ft_mb_ppo_diffusion_mlp_feat_vision --config-dir=cfg/calvin/finetune/close_drawer
```

## Citation
If you find DiWA useful in your work, please leave a ⭐ and consider citing our work with:
```
@article{chandra2025diwa,
    title={DiWA: Diffusion Policy Adaptation with World Models},
    author={Chandra, Akshay L and Nematollahi, Iman and Huang, Chenguang and Welschehold, Tim and Burgard, Wolfram and Valada, Abhinav},
    journal={},
    year={2025},
}
```

## License
This repository is released under the GPL-3.0 license. See [LICENSE](LICENSE).

## Acknowledgement
* [DPPO, Zen et al.](https://github.com/irom-princeton/dppo): Code base on top of which DiWA was built. Specifically, `sequence.py` in `diwa/dataset`, DDPM, DDIM, Gaussian, MLP/U-Net, ViT implementation in `diwa/model/`, PPO implementation in `diwa/agent/`.
* [LUMOS, Nematollahi et al.](https://github.com/nematoli/lumos): World model training.
* [CALVIN, Mees et al.](https://github.com/mees/calvin_env): Simulation experiments.
* [LIBERO, Liu et al.](https://github.com/Lifelong-Robot-Learning/LIBERO): Simulation experiments.