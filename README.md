# DiWA: Diffusion Policy Adaptation with World Models

[[Paper]()]&nbsp;&nbsp;[[Website](https://diwa.cs.uni-freiburg.de/)]

[Akshay L Chandra](https://akshaychandra.com/)<sup>1</sup>, [Iman Nematollahi](https://imanema.com/)<sup>1</sup>, [Chenguang Huang](https://scholar.google.com/citations?user=_rcR8TAAAAAJ&hl=en)<sup>2</sup>, [Tim Welschehold](https://rl.uni-freiburg.de/people/welschehold)<sup>1</sup> [Wolfram Burgard](https://scholar.google.com/citations?user=zj6FavAAAAAJ&hl=en)<sup>2</sup>, [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)<sup>1</sup>

<sup>1</sup>University of Freiburg, <sup>2</sup>University of Technology Nürnberg

<img src="https://github.com/acl21/diwa/blob/main/docs/diwa_cover.png" alt="drawing" width="100%"/>

> DiWA is an algorithmic framework for fine-tuning diffusion-based policies entirely inside frozen world models (learned from large play data).

## Overview
<img src="https://github.com/acl21/diwa/blob/main/docs/diwa_overview.png" alt="drawing" width="100%"/>

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
To download and preprocess datasets for DiWA, please follow A.0 and A.1 [here](dataset/README.md#a-calvin).

### 1. World Model
**Note**: You may skip world model training if you would like to use the default checkpoints (available for download [here](https://diwa.cs.uni-freiburg.de/download/ckpts/vision.ckpt)).

#### 1.1 Training
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

**Note**: Before pre-training, please extract the featurized expert data with A.2 [here](dataset/README.md#a2-extract-expert-data-for-diffusion-policy-training) or you can download [here](https://diwa.cs.uni-freiburg.de/download/data/expert.zip).

All configs relevant for pre-training can be found under `config/<env>/pretrain/<skill-name>`. To pretrain CALVIN's `close_drawer` skill, run:
```bash
python scripts/run.py --config-name=pre_diffusion_mlp_feat_vision --config-dir=config/calvin/pretrain/close_drawer
```

### 3. Reward Estimation
Before training the reward classifier, please generate the class-balanced classification data with A.3 [here](dataset/README.md#a3-generate-class-balanced-data-for-reward-classifier-training).
```bash
python scripts/rewcls/train_contrastive.py
```

### 4. Fine-tuning inside World Model
All configs relevant for fine-tuning can be found under `config/<env>/finetune/<skill-name>`. Set `base_policy_path` to the relevant pretrained policy checkpoint. To fine-tune CALVIN's `close_drawer` skill, run:
```bash
python scripts/run.py --config-name=ft_mb_ppo_diffusion_mlp_feat_vision --config-dir=cfg/calvin/finetune/close_drawer
```

## Known Issues

1. To solve the `TypeError` you may face with line 72 in `calvin_env/calvin_env/envs/play_table_env.py`, replace line 20 with `from calvin_env import calvin_env`. 


## Citation
If you find DiWA useful in your work, please leave a ⭐ and consider citing our work with:
```
@article{chandra2025diwa,
    title={DiWA: Diffusion Policy Adaptation with World Models},
    author={Chandra, Akshay L and Nematollahi, Iman and Huang, Chenguang and Welschehold, Tim and Burgard, Wolfram and Valada, Abhinav},
    journal={Preprint},
    year={2025},
}
```

## License
This repository is released under the GPL-3.0 license. See [LICENSE](LICENSE).


## Acknowledgement
* [DPPO, Zen et al.](https://github.com/irom-princeton/dppo): Code base on top of which DiWA was built. Specifically, `sequence.py` in `diwa/dataset`, DDPM, DDIM, Gaussian, MLP/U-Net, ViT implementations in `diwa/model/`, PPO implementation in `diwa/agent/` are all borrowed.
* [LUMOS, Nematollahi et al.](https://github.com/nematoli/lumos): World model training.
* [CALVIN, Mees et al.](https://github.com/mees/calvin_env): Simulation experiments.
* [LIBERO, Liu et al.](https://github.com/Lifelong-Robot-Learning/LIBERO): Simulation experiments.