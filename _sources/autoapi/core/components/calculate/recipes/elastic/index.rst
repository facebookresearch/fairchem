core.components.calculate.recipes.elastic
=========================================

.. py:module:: core.components.calculate.recipes.elastic

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   Functions to run elasticity calculations using ASE + pymatgen

   - Compute elasticity tensor
   - Compute Bulk modulus
   - Compute Shear modulus



Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.elastic.calculate_elasticity


Module Contents
---------------

.. py:function:: calculate_elasticity(atoms: ase.Atoms, calculator: ase.calculators.calculator.Calculator, norm_strains: collections.abc.Sequence[float] | float = (-0.01, -0.005, 0.005, 0.01), shear_strains: collections.abc.Sequence[float] | float = (-0.06, -0.03, 0.03, 0.06), relax_initial: bool = True, relax_strained: bool = True, use_equilibrium_stress: bool = True, **relax_kwargs) -> dict[str, Any]

   Calculate elastic tensor, bulk, shear moduli following MP workflow

   Will not run a relaxation. We do that outside in order to be able to have more control.

   :param atoms: ASE atoms object
   :param calculator: an ASE Calculator
   :param norm_strains: sequence of normal strains
   :param shear_strains: sequence of shear strains
   :param relax_initial: relax the initial structure. Default is True
   :param relax_strained: relax the atomic positions of strained structure. Default True.
   :param use_equilibrium_stress: use equilibrium stress in calculation. For relaxed structures this
                                  should be very small

   :returns: dict of elasticity results


