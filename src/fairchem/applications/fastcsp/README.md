## FastCSP: Accelerated Molecular Crystal Structure Prediction with Universal Model for Atoms

### Overview of FastCSP workflow

Starting from a molecular conformer, [`Genarris 3.0`](https://github.com/Yi5817/Genarris) generates a diverse set of random crystal structures. The generated structures undergo optimization with Rigid Press and deduplication is performed to remove similar structures. The remaining structures are fully relaxed using the UMA model from [`fairchem`](https://fair-chem.github.io/) and undergo another round of deduplication. The final ranking may be based on UMA lattice energy, or optionally, on Helmholtz or Gibbs free energies at finite temperature and pressure, also calculated using UMA.

### Getting started
Configured for use:
1. Install `fairchem-core`: [instructions](https://fair-chem.github.io/core/install.html)
2. Pip install fairchem-applications-fastcsp `pip install fairchem-applications-fastcsp`


Configured for local development:
1. Clone the [fairchem repo](https://github.com/facebookresearch/fairchem/tree/main)
2. Install `fairchem-core`: [instructions](https://fair-chem.github.io/core/install.html)
3. Install this repository `pip install -e packages/fairchem-applications-fastcsp`

External dependencies:
1. [`Genarris 3.0`](https://github.com/Yi5817/Genarris) should be installed separately in the current or separate environment.
2. Optionally, if using [`CSD Python API`](https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html) for the final evaluations, install it in the current or separate environment.

### Running `FastCSP`

Entire FastCSP workflow is controlled with a configuration file. An example file can be found in [configs](core/configs).

### Citing `FastCSP`

If you use this workflow in your work, please consider citing:

```bibtex
@misc{gharakhanyan2025fastcsp,
  title={FastCSP: Accelerated Molecular Crystal Structure Prediction with Universal Model for Atoms},
  author={Gharakhanyan, Vahe and Yang, Yi and Barroso-Luque, Luis and Shuaibi, Muhammed and Levine, Daniel S and Michel, Kyle and Bernat, Viachaslau and Dzamba, Misko and Fu, Xiang and Gao, Meng and others},
  year={2025},
  eprint={2508.02641},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2508.02641},
}
```
