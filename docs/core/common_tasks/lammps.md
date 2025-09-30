---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

LAMMPs Integration
------------------
We provide an integration with the [LAMMPs](https://www.lammps.org) Molecular Simulator through the [`fix external`](https://docs.lammps.org/fix_external.html) command. This simple integration hands control of the neighborlist (graph) generation, parallelism, energy, force, stress calculations all to UMA. The main advantage is that we can optimize UMA for distributed parallel inference directly without modifying Lammps. The user would also not need to deal with building Lammps from source (see conda install option below) nor [Kokkos](https://docs.lammps.org/Speed_kokkos.html), which is notorious difficult to build correctly. However, it does incurs some python overhead. For very fast emperical force fields, python would be a limiting factor but at the speeds of current MLIPs (10s - 100s) of ms per step regimes, python overhead is negligible. (This is the same reason nearly all modern LLM inference uses python engines). In addition, to easily scale to multi-node parallelism regimes, we designed the architecture using a client-server interface so Lammps would only see the client and the server code running inference can be optimized completely independently later.

Since the `fix external` integration simply wraps the UMA predictor interface, the way inference is run is identical to using the [MLIPPredictUnit, ASE Calculator or ParallelMLIPPredictUnit for Multi-GPU inference](https://fair-chem.github.io/core/common_tasks/ase_calculator.html).

## Usage notes that differ from regular lammps workflows:
* We currently only support `metal` [units](https://docs.lammps.org/units.html), ie: energy in `ev` and forces in `ev/A`
* User can write lammps scripts in the usual way (see lammps_in_example.file)
* User should *NOT* define other types of forces such as "pair_style", "bond_style" in their scripts. These forces will get added together with UMA forces and most likely produce false results
* UMA uses atomic numbers so we try to guess the atomic number from the provided atomic masses in your Lammps scripts. Just make sure you provide the right masses for your atom types - this makes it easy so that you don't need to redefine atomic element mappings with Lammps. *This assumption fails if you use isotopes or non standard atomic masses, but we don't expect our models to work in those cases anyways*

## Install and run
User can install lammps however they like but the simplest is to install via conda (https://docs.lammps.org/Install_conda.html) if you don't need any bells and whistles.

For conda install, simple activate the conda env with lammps and install fairchem into it. For manual Lammps installs, you need to provide python paths so Lammps can find fairchem. We separate the lammps integration code into a standalone package (fairchem-lammps). Please note fairchem-lammps uses the GnuV2 License as is required by any code that uses Lammps, instead of the MIT License used by the Fairchem repository. Note the "extras" is required to install for multi-GPU inference.

```
# first install conda and lammps following the instructions above
# then activate the environment and install fairchem
conda activate lammps-env
pip install fairchem-core[extras]
pip install fairchem-lammps
```

Assuming you have a classic lammps .in script, to run it, make the following changes to your script
1. Remove all other forces that you normally from your lammps script (ie: pair_style etc.)
2. Make sure the units are in "metal"
3. Make sure there is only 1 run command at the bottom of the script

To run, use the python entrypoint `lmp_fc` (shortcut name for the [python lammps_fc.py script](https://github.com/facebookresearch/fairchem/pull/1454))
```
lmp_fc lmp_in="lammps_in_example.file"  task_name="omol"
```

To try running with multiple gpus in parallel (this will only benefit large inputs/models, for small systems this might even run slower due to the communication bottleneck)
```
lmp_fc lmp_in="lammps_in_example.file" task_name="omol" predict_unit='${parallel_predict_unit}'
```
