# Open Catalyst 2020 Surfaces Dataset

:::{card} Dataset Overview

| Property | Value |
|----------|-------|
| **Size** | 18.3M structures |
| **Purpose** | Clean catalyst surfaces |
| **Paper** | [UMA Paper](https://arxiv.org/pdf/2506.23971) |
| **License** | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode) |
:::

## Overview

This dataset contains the clean OC20 catalyst surfaces used to train UMA. Surfaces appearing in any OC20 test split were excluded to avoid data leakage. This filtering is strict — even surfaces in the in-domain test split are excluded to ensure fair S2EF evaluations.

## File Contents and Download

| Splits    | Size       | MD5 Checksum (Download Link) |
|-----------|------------|------------------------------|
| Train+Val | 18,323,074 | [f047cbd515f213d0b0925704abbb7ae5](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20/oc20_slabs.tar.gz) |

The following metadata can be accessed in the `atoms.info` entry:

- `sid`: Unique surface system identifier
- `fid`: Frame index along the relaxation trajectory
- `fmax`: Maximum per-atom force
