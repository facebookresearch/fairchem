core.calculate.ase_calculator
=============================

.. py:module:: core.calculate.ase_calculator

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Exceptions
----------

.. autoapisummary::

   core.calculate.ase_calculator.MixedPBCError
   core.calculate.ase_calculator.AllZeroUnitCellError


Classes
-------

.. autoapisummary::

   core.calculate.ase_calculator.FAIRChemCalculator


Module Contents
---------------

.. py:class:: FAIRChemCalculator(predict_unit: fairchem.core.units.mlip_unit.MLIPPredictUnit, task_name: fairchem.core.units.mlip_unit.api.inference.UMATask | str | None = None, seed: int | None = None)

   Bases: :py:obj:`ase.calculators.calculator.Calculator`


   Base-class for all ASE calculators.

   A calculator must raise PropertyNotImplementedError if asked for a
   property that it can't calculate.  So, if calculation of the
   stress tensor has not been implemented, get_stress(atoms) should
   raise PropertyNotImplementedError.  This can be achieved simply by not
   including the string 'stress' in the list implemented_properties
   which is a class member.  These are the names of the standard
   properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
   'magmom' and 'magmoms'.


   .. py:attribute:: implemented_properties

      Properties calculator can handle (energy, forces, ...)


   .. py:attribute:: predictor


   .. py:attribute:: a2g


   .. py:property:: task_name
      :type: str



   .. py:method:: from_model_checkpoint(name_or_path: str, task_name: fairchem.core.units.mlip_unit.api.inference.UMATask | None = None, inference_settings: fairchem.core.units.mlip_unit.api.inference.InferenceSettings | str = 'default', overrides: dict | None = None, device: Literal['cuda', 'cpu'] | None = None, seed: int = 41) -> FAIRChemCalculator
      :classmethod:


      Instantiate a FAIRChemCalculator from a checkpoint file.

      :param cls: The class reference
      :param name_or_path: A model name from fairchem.core.pretrained.available_models or a path to the checkpoint
                           file
      :param task_name: Task name
      :param inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
                                 (optimized for speed but requires fixed atomic composition). Advanced use cases can
                                 use a custom InferenceSettings object.
      :param overrides: Optional dictionary of settings to override default inference settings.
      :param device: Optional torch device to load the model onto.
      :param seed: Random seed for reproducibility.



   .. py:method:: check_state(atoms: ase.Atoms, tol: float = 1e-15) -> list

      Check for any system changes since the last calculation.

      :param atoms: The atomic structure to check.
      :type atoms: ase.Atoms
      :param tol: Tolerance for detecting changes.
      :type tol: float

      :returns: A list of changes detected in the system.
      :rtype: list



   .. py:method:: calculate(atoms: ase.Atoms, properties: list[str], system_changes: list[str]) -> None

      Perform the calculation for the given atomic structure.

      :param atoms: The atomic structure to calculate properties for.
      :type atoms: Atoms
      :param properties: The list of properties to calculate.
      :type properties: list[str]
      :param system_changes: The list of changes in the system.
      :type system_changes: list[str]

      .. rubric:: Notes

      - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
      - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
      - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to `0`.
      - `charge` and `spin` are currently only used for the `omol` head.
      - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy.
        It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.



   .. py:method:: _get_single_atom_energies(atoms) -> dict

      Populate output with single atom energies



   .. py:method:: _check_atoms_pbc(atoms) -> None

      Check for invalid PBC conditions

      :param atoms: The atomic structure to check.
      :type atoms: ase.Atoms



   .. py:method:: _validate_charge_and_spin(atoms: ase.Atoms) -> None

      Validate and set default values for charge and spin.

      :param atoms: The atomic structure containing charge and spin information.
      :type atoms: Atoms



.. py:exception:: MixedPBCError(message='Attempted to guess PBC for an atoms object, but the atoms object has PBC set to True for somedimensions but not others. Please ensure that the atoms object has PBC set to True for all dimensions.')

   Bases: :py:obj:`ValueError`


   Specific exception example.


   .. py:attribute:: message


.. py:exception:: AllZeroUnitCellError(message='Atoms object claims to have PBC set, but the unit cell is identically 0. Please ensure that the atomsobject has a non-zero unit cell.')

   Bases: :py:obj:`ValueError`


   Specific exception example.


   .. py:attribute:: message


