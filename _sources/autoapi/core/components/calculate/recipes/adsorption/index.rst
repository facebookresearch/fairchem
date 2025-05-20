core.components.calculate.recipes.adsorption
============================================

.. py:module:: core.components.calculate.recipes.adsorption

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.adsorption.adsorb_atoms


Module Contents
---------------

.. py:function:: adsorb_atoms(adslab_atoms: ase.Atoms, calculator: ase.calculators.calculator.Calculator, optimizer_cls: ase.optimize.optimize.Optimizer, steps: int = 500, fmax: float = 0.02, relax_surface: bool = False, save_relaxed_atoms: bool = False) -> ase.Atoms

   Simple helper function to run relaxations and compute the adsorption energy
   of a given adsorbate+surface atoms object.

   :param atoms: ASE atoms object
   :param calculator: ASE calculator
   :param optimizer_cls: ASE optimizer. Default LBFGS
   :param steps: max number of relaxation steps
   :param fmax: force convergence threshold
   :param relax_surface: Whether to relax the bare surface
   :param save_relaxed_atoms: Whether to save the relaxed atoms

   :returns: dict of adsorption results


