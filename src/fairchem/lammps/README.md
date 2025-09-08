Copyright (c) Meta Platforms, Inc. and affiliates.

This directory provides an interface to use FAIR Chemistry models in conjuction with the [LAMMPs](https://github.com/lammps/lammps) software package to run molecular simulations.

The source under sub-repository (src/fairchem/lammps) is licensed under the GPL-2.0 License, the same as in the LAMMPs software package. Please refer to the LICENSE file in this same directory. ***It is NOT the same as the license for rest of this repository, which is licensed under the MIT license.***

This lammps integration uses the lammps "[fix external](https://docs.lammps.org/fix_external.html)" command to run the external MLIP: UMA and compute forces and potential energy of the system. This hands control of the parallelism to UMA instead of integrating with directly with LAMMPS neighborlist, domain decomp and forward + backward pass communication algorithms as well as converting to-from per-atom forces/pair-wise forces.


## Usage notes that differ from regular lammps workflows:
* We currently only support `metal` [units](https://docs.lammps.org/units.html), ie: energy in `ev` and forces in `ev/A`
* User can write lammps scripts in the usual way (see lammps_in_example.file)
* User should *NOT* define other types of forces such as "pair_style", "bond_style" in their scripts. These forces will get added together with UMA forces and most likely produce false results
* UMA uses atomic numbers so we try to guess the atomic number from the provided atomic masses, make sure you provide the right masses for your atom types - this makes it easy so that you don't need to redefine atomic element mappings with lammps

## Install and run
User can install lammps however they like but the simplest is to install via conda (https://docs.lammps.org/Install_conda.html). Next install fairchem into the same conda env and then you can run like so:

Activate the conda env with lammps and install fairchem into it
```
conda activate lammps-env
pip install fairchem/packages/fairchem-core[extras]
pip install fairchem/packages/lammps
```

Assuming you have a classic lammps .in script, to run it, do the following
```
1. Remove all other forces that you normally from your lammps script (ie: pair_style etc.)
2. Make sure the units are in`metal`
3. Make sure there is only 1 run command at the bottom of the script
```

Finally to run with `luma` (shorthand for python lammps_uma.py script)
```
luma lmp_in="lammps_in_example.file"  task_name="omc"
```
