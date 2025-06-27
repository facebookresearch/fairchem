"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.constraints import FixSymmetry
from ase.optimize import FIRE

if TYPE_CHECKING:
    from ase import Atoms
    from ase.filters import Filter
    from ase.optimize import Optimizer


def relax_atoms(
    atoms: Atoms,
    steps: int = 500,
    fmax: float = 0.02,
    optimizer_cls: type[Optimizer] | None = None,
    fix_symmetry: bool = False,
    cell_filter_cls: type[Filter] | None = None,
) -> Atoms:
    """Simple helper function to run relaxations and return the relaxed Atoms

    Args:
        atoms: ASE atoms with a calculator
        steps: max number of relaxation steps
        fmax: force convergence threshold
        optimizer_cls: ASE optimizer. Default FIRE
        fix_symmetry: fix structure symmetry in relaxation: Default False
        cell_filter_cls: An instance of an ASE filter.

    Returns:
        Atoms: relaxed atoms
    """

    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    if cell_filter_cls is not None:
        _atoms = cell_filter_cls(atoms)
    else:
        _atoms = atoms

    optimizer_cls = FIRE if optimizer_cls is None else optimizer_cls
    opt = optimizer_cls(_atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    atoms.info |= {"opt_nsteps": opt.nsteps, "opt_converged": opt.converged()}
    return atoms
