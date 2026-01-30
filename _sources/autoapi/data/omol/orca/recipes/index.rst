data.omol.orca.recipes
======================

.. py:module:: data.omol.orca.recipes


Functions
---------

.. autoapisummary::

   data.omol.orca.recipes.single_point_calculation
   data.omol.orca.recipes.ase_relaxation


Module Contents
---------------

.. py:function:: single_point_calculation(atoms, charge, spin_multiplicity, xc=ORCA_FUNCTIONAL, basis=ORCA_BASIS, orcasimpleinput=None, orcablocks=None, nprocs=12, outputdir=os.getcwd(), vertical=Vertical.Default, nbo=False, copy_files=None, **calc_kwargs)

   Wrapper around QUACC's static job to standardize single-point calculations.
   See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
   for more details.

   :param atoms: Atoms object
   :type atoms: Atoms
   :param charge: Charge of system
   :type charge: int
   :param spin_multiplicity: Multiplicity of the system
   :type spin_multiplicity: int
   :param xc: Exchange-correlaction functional
   :type xc: str
   :param basis: Basis set
   :type basis: str
   :param orcasimpleinput: List of `orcasimpleinput` settings for the calculator
   :type orcasimpleinput: list
   :param orcablocks: List of `orcablocks` swaps for the calculator
   :type orcablocks: list
   :param nprocs: Number of processes to parallelize across
   :type nprocs: int
   :param nbo: Run NBO as part of the Orca calculation
   :type nbo: bool
   :param outputdir: Directory to move results to upon completion
   :type outputdir: str
   :param calc_kwargs: Additional kwargs for the custom Orca calculator


.. py:function:: ase_relaxation(atoms, charge, spin_multiplicity, xc=ORCA_FUNCTIONAL, basis=ORCA_BASIS, orcasimpleinput=None, orcablocks=None, nprocs=12, opt_params=None, outputdir=os.getcwd(), vertical=Vertical.Default, copy_files=None, nbo=False, step_counter_start=0, **calc_kwargs)

   Wrapper around QUACC's ase_relax_job to standardize geometry optimizations.
   See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
   for more details.

   :param atoms: Atoms object
   :type atoms: Atoms
   :param charge: Charge of system
   :type charge: int
   :param spin_multiplicity: Multiplicity of the system
   :type spin_multiplicity: int
   :param xc: Exchange-correlaction functional
   :type xc: str
   :param basis: Basis set
   :type basis: str
   :param orcasimpleinput: List of `orcasimpleinput` settings for the calculator
   :type orcasimpleinput: list
   :param orcablocks: List of `orcablocks` swaps for the calculator
   :type orcablocks: list
   :param nprocs: Number of processes to parallelize across
   :type nprocs: int
   :param opt_params: Dictionary of optimizer parameters
   :type opt_params: dict
   :param nbo: Run NBO as part of the Orca calculation
   :type nbo: bool
   :param step_counter_start: Index to start step counter from (used for optimization restarts)
   :type step_counter_start: int
   :param outputdir: Directory to move results to upon completion
   :type outputdir: str
   :param calc_kwargs: Additional kwargs for the custom Orca calculator


