# Dataset

## CALVIN

To download the CALVIN task_D_D dataset:
```bash
cd $DIWA_ROOT/dataset
sh download_data.sh calvin
```

To have a faster dataloading, first preprocess the dataset to extract the essential fields:
```bash
python scripts/dataset/preprocess_calvin_dataset.py
```

To normalise data with either mean-std or min-max, run the following scripts (see file for argument list):
```bash
python scripts/dataset/calculate_meanstd_stats.py 
```
or
```bash
python scripts/dataset/calculate_minmax_stats.py
```