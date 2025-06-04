# Dataset

## CALVIN

To download the CALVIN task_D_D dataset:
```bash
cd $DIWA_ROOT/dataset
sh download_data.sh calvin
```

### 1. Play Data for World Model Training

To have a faster dataloading, first preprocess the dataset to extract the essential fields:
```bash
python scripts/dataset/preprocess_calvin_dataset.py
```

For world model training, we must normalise data with either mean-std or min-max, run the following scripts (see file for argument list):
```bash
python scripts/dataset/calculate_<meanstd/minmax>_stats.py 
```

### 2. Expert Data for Diffusion Policy Training
After featurizing the whole CALVIN dataset (Step 1.2 in [here](../README.md)), extract featurized expert demonstrations with:
```bash
python scripts/dataset/extract_expert_demos_feat.py
```

Also, to reproduce hybrid or state world model baseline experiments, first extract expert demonstration with images/privilege state, run:
```bash
python scripts/dataset/extract_expert_demos_<img/state>.py
```

### 3. Class-Balanced Data for Reward Classifier Training
For Step 3, we curate the featurized expert demonstrations to have equal class-distribution since data points with success states are comparatively low. We run the following in order:
```bash
python scripts/rewcls/create_rewcls_dataset.py
python scripts/rewcls/balance_rewcls_dataset.py
```

## LIBERO-90

(TODO) To download the LIBERO-90 dataset:
```bash
cd $DIWA_ROOT/dataset
sh download_data.sh libero_90
```

### 0.1. Regenerate LIBERO-90 with `256 x 256` Observations

(TODO) 

### 0.2. Convert LIBERO-90 to CALVIN format
For simplicity, we convert LIBERO-90 dataset to CALVIN format. This way, all other steps works seamlessly with the same code. To convert, run:
```bash
python scripts/dataset/convert_libero_to_calvin.py
```

Then, steps 1 and 2 follow exactly similar to CALVIN. Note that we did not train hybrid or state-based world model on LIBERO-90 because of the varying privilege state vector sizes (`scene_obs` equivalent in CALVIN) for different LIBERO-90 scenes and layouts. (TODO) For step 3,