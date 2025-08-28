Copyright (c) Meta Platforms, Inc. and affiliates.

This directory provides an interface to use FAIR Chemistry models in conjuction with the [LAMMPs](https://github.com/lammps/lammps) software package to run molecular simulations.

The source under sub-repository (src/external/lammps) is licensed under the GPL-2.0 License, the same as in the LAMMPs software package. Please refer to the LICENSE file in this same directory. ***It is NOT the same as the license for rest of this repository, which is licensed under the MIT license.***

This lammps integration uses the lammps "fix external" command to run the external MLIP: UMA and compute forces and potential energy of the system. This hands control of the parallelism to UMA instead of integrating with directly with LAMMPS neighborlist, domain decomp and forward + backward pass communication algorithms as well as converting to-from per-atom forces/pair-wise forces.


## Usage notes that differ from regular lammps workflows:
* User can write lammps scripts in the usual way (see lammps_in_example.file)
* User should *NOT* define other types of forces such as "pair_style", "bond_style" in their scripts. These forces will get added together with UMA forces and most likely produce false results
* UMA uses atomic numbers so we try to guess the atomic number from the provided atomic masses, make sure you provide the right masses for your atom types

## Install and run
User can install lammps however they like but the simplest is to install via conda (https://docs.lammps.org/Install_conda.html). Next install fairchem into the same conda env and then you can run like so:

```
conda activate lammps-env
python lammps_uma.py lmp_in = "lammps_in_example.file"
```
