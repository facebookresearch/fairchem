## Open Catalyst 2020 (OC20)
### Run scripts to process OC20 dataset
```
python src/oc20_preprocessing/01_download_metadata.py
python src/oc20_preprocessing/02_filter_tio2_systems.py
python src/oc20_preprocessing/03_download_s2ef_data.py
python src/oc20_preprocessing/04_create_tio2_lmdb.py
python src/oc20_preprocessing/05_compute_statistics.py
```

### Folder structure of OC20 dataset
```
oc20_data/
├── metadata/
├── raw/
│   ├── 200k
│   ├── 2M
│   ├── 20M
│   └── all
├── uncompressed_data/
│   ├── 200k
│   ├── 2M
│   ├── 20M
│   └── all
└── tio2_lmddb/
    ├── 200k
    ├── 2M
    ├── 20M
    └── all
```

## Use to convert extxyz.txt and xyz.txt into lmdb (OC20 dataset)
```
python -m scripts.preprocess_ef --data-path "D:/extxyz.txt_xyz.txt_dir" --out-path "D:/lmdb_dir" --num-workers 8
```
```
python -m scripts.preprocess_ef --data-path "D:/Data44" --out-path "D:/Data44_processed" --num-workers 8
```

## Open Catalyst 2022 (OC22)
### Run scripts to process OC22 dataset
```
python src/oc22_preprocessing/01_download_metadata.py
python src/oc22_preprocessing/02_filter_tio2_systems.py
python src/oc22_preprocessing/03_download_extract_oc22_lmdb.py
python src/oc22_preprocessing/04_filter_tio2_lmdb.py
```

### Folder structure of OC22 dataset
```
oc22_data/
├── s2ef_total_train_val_test_lmdbs.tar.gz
├── s2ef-total/
│   ├── train/
│   ├── val_id/
│   ├── val_ood/
│   ├── test_id/
│   └── test_ood/
├── tio2_filtered/
│   ├── train/
│   ├── val_id/
│   ├── val_ood/
│   ├── test_id/
│   └── test_ood/
└── metadata/
```

---
## Helper
### Installation
```
conda env create -f env.yml
```

### Git Bash
```
source /c/Users/Admin/miniconda3/etc/profile.d/conda.sh
```

### Powershell/cmd
```
conda activate tio2-s2ef
```

### If you want to update any packages in env.yml
```
conda env update -f env.yml -n tio2-s2ef
```

### Uninstall conda env
```
conda env remove --name tio2-s2ef-v1
```


```
conda env list
```

An Environment that works for fairchem v1:
https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.7.0/packages/env.cpu.yml