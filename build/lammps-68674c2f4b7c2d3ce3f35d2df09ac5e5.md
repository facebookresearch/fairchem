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

# LAMMPS Integration

We provide an integration with the [LAMMPS](https://www.lammps.org) Molecular Simulator through the [`fix external`](https://docs.lammps.org/fix_external.html) command. This simple integration hands control of the neighborlist (graph) generation, parallelism, energy, force, and stress calculations all to UMA.

:::{tip}
The main advantage is that we can optimize UMA for distributed parallel inference directly without modifying LAMMPS. The user would also not need to deal with building LAMMPS from source (see conda install option below) nor [Kokkos](https://docs.lammps.org/Speed_kokkos.html), which is notoriously difficult to build correctly.
:::

There is some Python overhead, but for very fast empirical force fields where Python would be a limiting factor, this is negligible at the speeds of current MLIPs (10s - 100s of ms per step). This is the same reason nearly all modern LLM inference uses Python engines. Additionally, to easily scale to multi-node parallelism regimes, we designed the architecture using a client-server interface so LAMMPS would only see the client and the server code running inference can be optimized completely independently later.

Since the `fix external` integration simply wraps the UMA predictor interface, the way inference is run is identical to using the [MLIPPredictUnit, ASE Calculator or ParallelMLIPPredictUnit for Multi-GPU inference](https://fair-chem.github.io/core/common_tasks/ase_calculator.html).

## Usage Notes

:::{warning}
Please note the following differences from regular LAMMPS workflows:
:::

- We currently only support `metal` [units](https://docs.lammps.org/units.html), i.e., energy in `eV` and forces in `eV/A`
- Users can write LAMMPS scripts in the usual way (see lammps_in_example.file)
- Users should **NOT** define other types of forces such as "pair_style", "bond_style" in their scripts. These forces will get added together with UMA forces and most likely produce false results
- UMA uses atomic numbers so we try to guess the atomic number from the provided atomic masses in your LAMMPS scripts. Just make sure you provide the right masses for your atom types - this makes it easy so that you don't need to redefine atomic element mappings with LAMMPS

:::{note}
This assumption fails if you use isotopes or non-standard atomic masses, but we don't expect our models to work in those cases anyway.
:::

## Install and Run

Users can install LAMMPS however they like, but the simplest is to install via conda ([https://docs.lammps.org/Install_conda.html](https://docs.lammps.org/Install_conda.html)) if you don't need any bells and whistles.

For conda install, activate the conda env with LAMMPS and install fairchem into it. For manual LAMMPS installs, you need to provide python paths so LAMMPS can find fairchem.

:::{note}
We separate the LAMMPS integration code into a standalone package (`fairchem-lammps`). Please note fairchem-lammps uses the GnuV2 License as is required by any code that uses LAMMPS, instead of the MIT License used by the FAIRChem repository. The "extras" is required for multi-GPU inference.
:::

```bash
# first install conda and lammps following the instructions above
# then activate the environment and install fairchem
conda activate lammps-env
pip install fairchem-core[extras]
pip install fairchem-lammps
```

Assuming you have a classic LAMMPS .in script, make the following changes:

1. Remove all other forces from your LAMMPS script (e.g., pair_style, etc.)
2. Make sure the units are in "metal"
3. Make sure there is only 1 run command at the bottom of the script

To run, use the Python entrypoint `lmp_fc` (shortcut name for the [python lammps_fc.py script](https://github.com/facebookresearch/fairchem/pull/1454)):

```bash
lmp_fc lmp_in="lammps_in_example.file" task_name="omol"
```

## Multi-GPU Parallelism

Our LAMMPS integration is fully compatible out of the box with our Multi-GPU inference API.

:::{tip}
The only change required is to pass the `ParallelMLIPPredictUnit` [here](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/lammps/lammps_fc_config.yaml#L20) instead of the regular predict unit when initializing the LAMMPS fairchem script. No need to install anything new such as Kokkos or add communication code.
:::

For example:

```bash
lmp_fc lmp_in="lammps_in_example.file" task_name="omol" predict_unit='${parallel_predict_unit}'
```
