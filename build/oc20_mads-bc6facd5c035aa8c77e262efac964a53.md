# Open Catalyst 2020 Multi-Adsorbate (mAds) Dataset

:::{card} Dataset Overview

| Property | Value |
|----------|-------|
| **Size** | 21.8M structures |
| **Max Adsorbates** | Up to 5 adsorbates per surface |
| **Purpose** | Multi-adsorbate and coverage effects |
| **Paper** | [UMA Paper](https://arxiv.org/pdf/2506.23971) |
| **License** | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode) |
:::

## Overview

The OC20-mAds dataset is a training set expanding the original OC20 dataset to include multi-adsorbate and coverage effects on catalyst surfaces. Adsorbates are randomly sampled from the list of OC20 adsorbates, up to 5 maximum adsorbates. For a small fraction of the dataset, all adsorbates on the surface may be identical. OC20-mAds is introduced in the [UMA paper](https://arxiv.org/pdf/2506.23971).

## File Contents and Download
|Splits |Size | MD5 checksum (download link)   |
|---     |---    |---     |
|Train   |   21,804,758   | [6435960ba5ad1a7c949bd2f2b51825bc](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20mAds/oc20_multiads_train.tar.gz)   |

The following metadata can be accessed in the respective `atoms.info` entry:

- `bulk_id`: Bulk identifier
- `millers`: 3-tuple of integers indicating the Miller indices of the surface.
- `shift`: C-direction shift used to determine cutoff for the surface (c-direction is following the nomenclature from Pymatgen).
- `top`: Boolean indicating whether the chosen surface was at the top or bottom of the originally enumerated surface.
- `adsorbates`: List of adsorbates sampled and their respective placements.
- `sid`: Unique system identifier.
- `fid`: Frame index along the relaxation/AIMD trajectory.
- `results_path`: Internal results location.
- `fmax`: Max per-atom force.
