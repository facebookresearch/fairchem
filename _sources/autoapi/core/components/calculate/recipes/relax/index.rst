core.components.calculate.recipes.relax
=======================================

.. py:module:: core.components.calculate.recipes.relax

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.relax.relax_atoms


Module Contents
---------------

.. py:function:: relax_atoms(atoms: ase.Atoms, steps: int = 500, fmax: float = 0.02, optimizer_cls: type[ase.optimize.Optimizer] | None = None, fix_symmetry: bool = False, cell_filter_cls: type[ase.filters.Filter] | None = None) -> ase.Atoms

   Simple helper function to run relaxations and return the relaxed Atoms

   :param atoms: ASE atoms with a calculator
   :param steps: max number of relaxation steps
   :param fmax: force convergence threshold
   :param optimizer_cls: ASE optimizer. Default FIRE
   :param fix_symmetry: fix structure symmetry in relaxation: Default False
   :param cell_filter_cls: An instance of an ASE filter.

   :returns: relaxed atoms
   :rtype: Atoms


