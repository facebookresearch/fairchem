core.components.calculate.recipes.omol
======================================

.. py:module:: core.components.calculate.recipes.omol

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   OMol Evaluation Recipes
   ========================

   This module provides evaluation recipes for various molecular property tasks proposed in the OMol25 paper.

   The module includes functions for:
   - Geometry optimizations and conformer generation
   - Protonation state energetics
   - Ionization energies and electron affinities
   - Spin gap calculations
   - Protein-ligand interactions
   - Ligand strain energies
   - Distance scaling behavior
   - Single-point energy and force calculations

   Each function follows a consistent pattern of taking input data and any ASE calculator,
   performing the required calculations, and returning results in a standardized format
   suitable for downstream evaluation on the OMol leaderboard - #TODO: add link.



Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.omol.relax_job
   core.components.calculate.recipes.omol.single_point_job
   core.components.calculate.recipes.omol.conformers
   core.components.calculate.recipes.omol.protonation
   core.components.calculate.recipes.omol.ieea
   core.components.calculate.recipes.omol.spin_gap
   core.components.calculate.recipes.omol.ligand_pocket
   core.components.calculate.recipes.omol.ligand_strain
   core.components.calculate.recipes.omol.distance_scaling
   core.components.calculate.recipes.omol.singlepoint


Module Contents
---------------

.. py:function:: relax_job(atoms: ase.Atoms, calculator: ase.calculators.calculator.Calculator, opt_flags: dict[str, Any]) -> dict[str, Any]

   Perform a geometry optimization job on an atomic structure.

   This function optimizes the geometry of an atomic structure using the provided
   calculator and optimization parameters. It captures both the initial and final
   states (energy, forces, and atomic positions) for comparison.

   :param atoms: ASE Atoms object representing the initial structure
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations
   :param opt_flags: Dictionary containing optimization parameters including:
                     - optimizer: ASE optimizer class (e.g., BFGS, FIRE)
                     - optimizer_kwargs: Additional kwargs for the optimizer
                     - fmax: Force convergence criterion (eV/Å)
                     - max_steps: Maximum number of optimization steps
   :type opt_flags: dict

   :returns: Results organized in the following form -
             {
                 "initial": {
                     "atoms": MSONAtoms dictionary of initial structure,
                     "energy": Initial total energy (eV),
                     "forces": Initial forces as a list (eV/Å),
                 },
                 "final": {
                     "atoms": MSONAtoms dictionary of optimized structure,
                     "energy": Final total energy (eV),
                     "forces": Final forces as a list (eV/Å),
                 }
             }
   :rtype: dict

   .. note::

      If optimization fails, the function logs the error and returns the last
      valid state rather than crashing.


.. py:function:: single_point_job(atoms: ase.Atoms, calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Perform a single-point energy and force calculation.

   This function calculates the energy and forces for a given atomic structure.

   :param atoms: ASE Atoms object representing the structure
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "atoms": MSONAtoms dictionary of the structure,
                 "energy": Total energy (eV),
                 "forces": Forces as a list (eV/Å),
             }
   :rtype: dict


.. py:function:: conformers(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Calculate conformer energies and geometries.

   This function performs geometry optimizations on molecular conformers.

   :param input_data: Input data organized by molecule families, where each
                      entry contains conformer information with initial and final structures
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "molecule_family_1": {
                     "conformer_id_1": {
                         "initial": {
                             "atoms": MSONAtoms dictionary of initial structure,
                             "energy": Initial total energy (eV),
                             "forces": Initial forces as a list (eV/Å),
                         },
                         "final": {
                             "atoms": MSONAtoms dictionary of optimized structure,
                             "energy": Final total energy (eV),
                             "forces": Final forces as a list (eV/Å),
                         },
                     },
                     "conformer_id_2": { ... },
                     ...
                 },
                 "molecule_family_2": { ... },
                 ...
             }
   :rtype: dict


.. py:function:: protonation(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Calculate protonation state energies and geometries.

   This function calculates the energies and geometries of different protonation
   states of molecules.

   :param input_data: Input data organized by molecule families, where each
                      entry contains different protonation states with initial and final structures
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "molecule_family_1": {
                     "protonation_state_1": {
                         "initial": {
                             "atoms": MSONAtoms dictionary of initial structure,
                             "energy": Initial total energy (eV),
                             "forces": Initial forces as a list (eV/Å),
                         },
                         "final": {
                             "atoms": MSONAtoms dictionary of optimized structure,
                             "energy": Final total energy (eV),
                             "forces": Final forces as a list (eV/Å),
                         },
                     },
                     "protonation_state_2": { ... },
                     ...
                 },
                 "molecule_family_2": { ... },
                 ...
             }
   :rtype: dict


.. py:function:: ieea(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Calculate unoptimized ionization energies and electron affinities.

   This function performs single-point calculations on structures at different
   charge states to evaluate ionization energies (IE) and electron affinities (EA).
   No geometry optimization is performed, testing the MLIP's ability to predict
   energetics of charged species at fixed geometries.

   :param input_data: Input data organized by system identifier, with each
                      entry containing structures at different charge and spin states
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "identifier_1": {
                     "charge_state_1": {
                         "spin_state_1": {
                             "atoms": MSONAtoms dictionary of the structure,
                             "energy": Total energy (eV),
                             "forces": Forces as a list (eV/Å),
                         },
                         "spin_state_2": { ... },
                         ...
                     },
                     "charge_state_2": { ... },
                     ...
                 },
                 "identifier_2": { ... },
             }
   :rtype: dict


.. py:function:: spin_gap(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Calculate unoptimized spin gap energies.

   This function performs single-point calculations on structures at different
   spin states to evaluate spin gaps (energy differences between different
   spin multiplicities). No geometry optimization is performed.

   :param input_data: Input data organized by system identifier, with each
                      entry containing structures at different spin states
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "identifier_1": {
                     "spin_state_1": {
                         "atoms": MSONAtoms dictionary of the structure,
                         "energy": Total energy (eV),
                         "forces": Forces as a list (eV/Å),
                     },
                     "spin_state_2": { ... },
                 },
                 "identifier_2": { ... },
             }
   :rtype: dict


.. py:function:: ligand_pocket(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Calculate protein-ligand interaction energies and forces.

   This function performs single-point calculations on protein-ligand systems,
   calculating energies and forces for the complex and individual components
   (ligand, pocket, ligand_pocket). This enables evaluation of interaction
   energies and binding affinity predictions.

   :param input_data: Input data organized by system identifier, with each
                      entry containing ASE Atoms objects for ligand, pocket, and complex
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "identifier_1": {
                     "ligand": {
                         "atoms": MSONAtoms dictionary of the structure,
                         "energy": Total energy (eV),
                         "forces": Forces as a list (eV/Å),
                     },
                     "pocket": {
                         "atoms": MSONAtoms dictionary of the structure,
                         "energy": Total energy (eV),
                         "forces": Forces as a list (eV/Å),
                     },
                     "ligand_pocket": {
                         "atoms": MSONAtoms dictionary of the structure,
                         "energy": Total energy (eV),
                         "forces": Forces as a list (eV/Å),
                     },
                 },
                 "identifier_2": { ... },
             }
   :rtype: dict


.. py:function:: ligand_strain(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Calculate ligand strain energies in protein-bound conformations.

   This function calculates strain energies by comparing the energy of a ligand
   in its bioactive (protein-bound) conformation with its global minimum energy
   conformation in the gas phase.


   :param input_data: Input data organized by system identifier, with each
                      entry containing:
                      - bioactive_conf: Ligand in bioactive conformation
                      - conformers: List of (initial, final) conformer pairs for gas phase
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "identifier_1": {
                     "bioactive": {
                         "atoms": MSONAtoms dictionary of bioactive conformation,
                         "energy": Total energy (eV),
                         "forces": Forces as a list (eV/Å),
                     },
                     "gas_phase": {
                         "0": {  # conformer index
                             "initial": {
                                 "atoms": MSONAtoms dictionary of initial structure,
                                 "energy": Initial total energy (eV),
                                 "forces": Initial forces as a list (eV/Å),
                             },
                             "final": {
                                 "atoms": MSONAtoms dictionary of optimized structure,
                                 "energy": Final total energy (eV),
                                 "forces": Final forces as a list (eV/Å),
                             }
                         },
                         "1": { ... },
                         ...
                     },
                 },
                 "identifier_2": { ...
                 }
             }
   :rtype: dict


.. py:function:: distance_scaling(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Calculate energies and forces at different inter-molecular distances.

   This function performs single-point calculations on molecular systems where
   inter-molecular distances have been systematically varied. This tests the
   MLIP's ability to capture both short-range repulsion and long-range attraction
   in potential energy surfaces.

   :param input_data: Input data organized by domain type (vertical), then
                      by system identifier, then by distance scale factor, containing
                      ASE Atoms objects at different inter-molecular separations
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "vertical_1": {
                     "identifier_1": {
                         "short_range_scaled_complex_X": {
                             "atoms": MSONAtoms dictionary of the structure,
                             "energy": Total energy (eV),
                             "forces": Forces as a list (eV/Å),
                         },
                         "short_range_scaled_complex_Y": { ... },
                         "long_range_scaled_complex_Z": { ... },
                         ...
                     },
                     "identifier_2": { ... },
                     ...
                 },
                 "vertical_2": { ... },
                 ...
   :rtype: dict


.. py:function:: singlepoint(input_data: dict[str, Any], calculator: ase.calculators.calculator.Calculator) -> dict[str, Any]

   Perform general single-point energy and force calculations.

   This is a general-purpose function for performing single-point calculations
   on arbitrary molecular structures.

   :param input_data: Input data organized by system identifier, with each
                      entry containing an ASE Atoms object
   :type input_data: dict
   :param calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

   :returns: Results organized in the following form -
             {
                 "identifier_1": {
                     "atoms": MSONAtoms dictionary of the structure,
                     "energy": Total energy (eV),
                     "forces": Forces as a list (eV/Å),
                 },
                 "identifier_2": { ... },
             }
   :rtype: dict


