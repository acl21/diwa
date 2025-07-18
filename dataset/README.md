# Dataset

## A. CALVIN

### A.0 Download

To download the CALVIN task_D_D dataset:
```bash
cd $DIWA_ROOT/dataset
sh download_data.sh calvin
```

### A.1. Preprocess CALVIN Play Data for World Model Training

To have a faster dataloading and training, we preprocess the raw CALVIN dataset by  
- extracting only the essential fields
- replacing Euler angles with 6D rotations
- resizing pixel observations to `64 x 64`
```bash
python scripts/dataset/preprocess_calvin_dataset.py
```

For better world model training, we normalise the data with either mean-std or min-max. Run the following script to generate the `.yaml` file with the statistics:
```bash
python scripts/dataset/calculate_<meanstd|minmax>_stats.py 
```

### A.2. Extract Expert Data for Diffusion Policy Training
After featurizing the whole CALVIN dataset (i.e., after 1.2 [here](../README.md#12-featurizer)), extract featurized expert demonstrations from the play data with:
```bash
python scripts/dataset/extract_expert_demos_feat.py
```

You can extract expert demonstrations with privilege state and/or images, run:
```bash
python scripts/dataset/extract_expert_demos_state.py
(Optional) python scripts/dataset/extract_expert_demos_img.py
```
Note: Please use the same seed across all the above extraction runs to index the exact same episodes.

### A.3. Generate Class-Balanced Data for Reward Classifier Training
For Step 3, we first generate task-completion reward labels by rolling out expert demos in the env and then curate the featurized expert demonstrations to have equal class-distribution since data points with task-completion rewards are comparatively low. To get this, we run the following in order:
```bash
python scripts/rewcls/generate_rewards_calvin.py
python scripts/rewcls/create_rewcls_dataset.py
python scripts/rewcls/balance_rewcls_dataset.py
```

**Note:** Each of these scripts are linked to a separate config file. Please familirize with them before running these scripts.

## B. LIBERO-90

To download the LIBERO-90 dataset:
```bash
python LIBERO/benchmark_scripts/download_libero_datasets.py --download-dir /path/to/libero/data/dir/ --datasets libero_90
```

### B.1. Regenerate LIBERO-90 with `256 x 256` Observations

We regenerate LIBERO-90 data by rendering the environment at `256 x 256` size (this is a bit useless as we resize them back to `64 x 64` for world model training) first. We did this for reasons we cannot explain here. To regenerate, run:
```bash
python scripts/dataset/regenerate_libero_dataset.py
```

### B.2. Convert LIBERO-90 to CALVIN format
For simplicity, we convert LIBERO-90 dataset to CALVIN format. This way, all other steps works seamlessly with the same code. To convert, run:
```bash
python scripts/dataset/convert_libero_to_calvin.py
```

Then, steps 1 and 2 follow exactly similar to CALVIN. In step 3, use appropriate reward data generator:
```bash
python scripts/rewcls/generate_rewards_libero.py
```
Note that we did not train hybrid or state-based world model on LIBERO-90 because of the varying privilege state vector sizes (`scene_obs` equivalent in CALVIN) for different LIBERO-90 scenes and layouts.

### B.3. Generate Initial Observations
In our fine-tuning experiments, we sample env initial states (observations) from an environment. However, `env.reset()` for LIBERO was expensive compared to CALVIN so we load the initial states from a pre-generated `.npz` file. To get the same, run:
```bash
python scripts/dataset/generate_init_obs_libero.py
```

### C. Real-World

To download the Real-World dataset:
```bash
cd $DIWA_ROOT/dataset
sh download_data.sh real
```
The above command downloads the raw play data. Step 1 is the same for the real-world data. (TODO)