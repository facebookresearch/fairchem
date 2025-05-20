core.components.calculate.recipes.phonons
=========================================

.. py:module:: core.components.calculate.recipes.phonons

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   Helper scripts to run phonon calculations

   - Compute phonon frequencies at commensurate points
   - Compute thermal properties with Fourier interpolation
   - Optionally compute and plot band-structures and DOS

   Needs phonopy installed



Attributes
----------

.. autoapisummary::

   core.components.calculate.recipes.phonons.THz_to_K


Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.phonons.run_mdr_phonon_benchmark
   core.components.calculate.recipes.phonons.get_phonopy_object
   core.components.calculate.recipes.phonons.produce_force_constants
   core.components.calculate.recipes.phonons.calculate_phonon_frequencies
   core.components.calculate.recipes.phonons.calculate_thermal_properties


Module Contents
---------------

.. py:data:: THz_to_K

.. py:function:: run_mdr_phonon_benchmark(mdr_phonon: phonopy.Phonopy, calculator: ase.calculators.calculator.Calculator, displacement: float = 0.01, run_relax: bool = True, fix_symm_relax: bool = False, symprec: int = 0.0001, symmetrize_fc: bool = False) -> dict

   Run a phonon calculation for a single datapoint of the MDR PBE dataset

   Properties computed for benchmark:
       - maximum frequency from phonon frequencies computed at supercell commensurate points
       - vibrational free energy, entropy and heat capacity computed with a [20, 20, 20] mesh

   :param mdr_phonon: the baseline MDR Phonopy object
   :param calculator: an Ase Calculator
   :param displacement: displacement step to compute forces (A)
   :param run_relax: run a structural relaxation
   :param fix_symm_relax: wether to fix symmetry in relaxation
   :param symprec: symmetry precision used by phonopy
   :param symmetrize_fc: symmetrize force constants

   :returns: dictionary of computed properties
   :rtype: dict


.. py:function:: get_phonopy_object(atoms: phonopy.structure.atoms.PhonopyAtoms | ase.Atoms | pymatgen.core.Structure, displacement: float = 0.01, supercell_matrix: numpy.typing.ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)), primitive_matrix: numpy.typing.ArrayLike | None = None, symprec: int = 1e-05, **phonopy_kwargs) -> phonopy.Phonopy

   Create a Phonopy api object from ase Atoms.

   :param atoms: Phonopy atoms, ASE atoms object or a pmg Structure
   :param displacement: displacement step to compute forces (A)
   :param supercell_matrix: transformation matrix to super cell from unit cell.
   :param primitive_matrix: transformation matrix to primitive cell from unit cell.
   :param symprec: symmetry precision
   :param phonopy_kwargs: additional keyword arguments to initialize Phonopy API object

   :returns: api object
   :rtype: Phonopy


.. py:function:: produce_force_constants(phonon: phonopy.Phonopy, calculator: ase.calculators.calculator.Calculator, symmetrize: bool = False) -> None

   Run force calculations and produce force constants with Phonopy

   :param phonon: a Phonopy API object
   :param calculator: an ASE Calculator
   :param symmetrize: symmetrize force constants


.. py:function:: calculate_phonon_frequencies(phonon: phonopy.Phonopy, qpoints: numpy.typing.ArrayLike | None = None) -> numpy.typing.NDArray

   Calculate phonon frequencies at a given set of qpoints.

   :param phonon: a Phonopy api object with displacements generated
   :param qpoints: ndarray of qpoints to calculate phonon frequencies at. If none are given, the supercell commensurate
                   points will be used

   :returns: ndarray of phonon frequencies in THz, (qpoints, frequencies)
   :rtype: NDArray


.. py:function:: calculate_thermal_properties(phonon: phonopy.Phonopy, t_min, t_max, t_step, mesh: numpy.typing.ArrayLike = (20, 20, 20)) -> dict[str, float]

   Calculate thermal properties from initialized phonopy object

   Thermal properties include: vibrational free energy, entropy and heat capacity

   :param phonon: a Phonopy api object with displacements generated
   :param t_min: minimum temperature
   :param t_max: max temperature
   :param t_step: temperature step between min and max
   :param mesh: qpoint mesh to compute properties using Fourier interpolation

   :returns: dictionary of computed properties
   :rtype: dict


