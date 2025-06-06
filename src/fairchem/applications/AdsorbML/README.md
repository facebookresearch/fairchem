## AdsorbML: Accelerating Adsorption Energy Calculations with Machine Learning

![adsorbml](https://user-images.githubusercontent.com/45150244/213025581-498b459e-9077-42ac-84e1-65ef331555d2.png)

`AdsorbML` is an algorithm to calculating the minima adsorbate binding energy (adsorption energy) for a unique adsorbate+surface combination. All ML models are obtained from [`fairchem`](https://fair-chem.github.io/) to perform corresponding structure relaxations.

This repository holds the dataset, scripts, and downloads for the accompanying [paper](https://arxiv.org/abs/2211.16486).

### OC20-Dense Dataset (OC20-Dense)

OC20-Dense contains a dense sampling of adsorbate configurations on ~1,000 randomly selected adsorbate+surface materials from the [OC20](https://arxiv.org/abs/2010.09990) dataset. The dataset comprises a total of 85,658 unique input configurations.

The dataset is stored in an LMDB file and ready to be used in `ocp` upon download. Additionally, ground truth DFT relaxations are store in ASE trajectories and provided for all converged systems used for evaluation.

NOTE - ASE trajectories exclude systems that were not converged or had invalid configurations as defined by the constraints in the `AdsorbML` manuscript. This resulted in 65,073 relaxations available for evaluation and are provided here.


|Splits |Size of compressed version (in bytes)  |Size of uncompressed version (in bytes)    | MD5 checksum (download link)   |
|---    |---    |---    |---    |
|LMDB    |654M   |9.8G   | [0163b0e8c4df6d9c426b875a28d9178a](https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_data.tar.gz)   |
|ASE Trajectories    |29G    |112G   | [ee937e5290f8f720c914dc9a56e0281f](https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_trajectories.tar.gz)   |

The following files are also provided to be used for evaluation and general information:
* `oc20dense_mapping.pkl` : Mapping of the LMDB `sid` to general metadata information. If this file is not present, run the command `python src/fairchem/core/scripts/download_large_files.py adsorbml` from the root of the fairchem repo to download it. -
  * `system_id`: Unique system identifier for an adsorbate, bulk, surface combination.
  * `config_id`: Unique configuration identifier, where `rand` and `heur` correspond to random and heuristic initial configurations, respectively.
  * `mpid`: Materials Project bulk identifier.
  * `miller_idx`: 3-tuple of integers indicating the Miller indices of the surface.
  * `shift`: C-direction shift used to determine cutoff for the surface (c-direction is following the nomenclature from Pymatgen).
  * `top`: Boolean indicating whether the chosen surface was at the top or bottom of the originally enumerated surface.
  * `adsorbate`: Chemical composition of the adsorbate.
  * `adsorption_site`: A tuple of 3-tuples containing the Cartesian coordinates of each binding adsorbate atom
* `oc20dense_targets.pkl` :  DFT adsorption energies across different system and placement ids.
* `oc20dense_compute.pkl` :  DFT compute as measured in the number of ionic and scf steps for each evaluated relaxation.
* `oc20dense_ref_energies.pkl` : Reference energy used for a specified `system_id`. This energy includes the relaxed clean surface and the gas phase adsorbate energy to ensure consistency across calculations.
* `oc20dense_tags.pkl` : Tag information used for a specified `system_id`. Where 0 = subsurface, 1 = surface, 2 = adsorbate.

All mappings can be obtained at the following downloadable link: https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_mappings.tar.gz

MD5 checksums:
```
c18735c405ce6ce5761432b07287d8d9  oc20_dense_mappings.tar.gz
3e26c3bcef01ccfc9b001931065ea6e6  oc20dense_mapping.pkl
fd589b013b72e62e11a6b2a5bd1d323c  oc20dense_targets.pkl
78d25997e0aaf754df526ab37276bb89  oc20dense_compute.pkl
b07c64158e4bfa5f7b9bf6263753ecc5  oc20dense_ref_energies.pkl
1ba0bc266130f186850f5faa547b6a02  oc20dense_tags.pkl
```

### Running `AdsorbML`

Please see the [README](adsorbml/scripts/README.md) inside the `scripts` directory for instructions.

### Citing `AdsorbML`

If you use this codebase in your work, please consider citing:

```bibtex
@article{lan2022adsorbml,
  title={AdsorbML: Accelerating Adsorption Energy Calculations with Machine Learning},
  author={Lan*, Janice and Palizhati*, Aini and Shuaibi*, Muhammed and Wood*, Brandon M and Wander, Brook and Das, Abhishek and Uyttendaele, Matt and Zitnick, C Lawrence and Ulissi, Zachary W},
  journal={arXiv preprint arXiv:2211.16486},
  year={2022}
}
```
