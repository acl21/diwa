# Dataset

## CALVIN

To download the CALVIN task_D_D dataset:
```bash
cd $DIWA_ROOT/dataset
sh download_data.sh calvin
```

### Play Data for World Model Training

To have a faster dataloading, first preprocess the dataset to extract the essential fields:
```bash
python scripts/dataset/preprocess_calvin_dataset.py
```

For world model training, we must normalise data with either mean-std or min-max, run the following scripts (see file for argument list):
```bash
python scripts/dataset/calculate_<meanstd/minmax>_stats.py 
```

### Expert Data for Diffusion Policy Training
After featurizing the whole CALVIN dataset (Step 1.2 in [here](../README.md)), extract featurized expert demonstrations with:
```bash
python scripts/dataset/extract_expert_demos_feat.py
```

Also, to extract expert demonstration with images/privilege state, run:
```bash
python scripts/dataset/extract_expert_demos_<img/state>.py
```

### Class-Balanced Data for Reward Classifier Training
For Step 3, we curate the featurized expert demonstrations to have equal class-distribution since data points with success states are comparatively low. We run the following in order:
```bash
python scripts/rewcls/create_rewcls_dataset.py
python scripts/rewcls/balance_rewcls_dataset.py
```
