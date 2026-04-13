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


def relax_atoms_w_maxstep(
    atoms: Atoms,
    steps: int = 500,
    fmax: float = 0.02,
    maxstep: float | None = None,
    optimizer_cls: type[Optimizer] | None = None,
    fix_symmetry: bool = False,
    cell_filter_cls: type[Filter] | None = None,
) -> Atoms:
    """Run a relaxation on ASE atoms and return the relaxed structure.

    Args:
        atoms: ASE Atoms object with a calculator
        steps: Maximum number of relaxation steps
        fmax: Force convergence threshold
        maxstep: Maximum atomic displacement per iteration
        optimizer_cls: ASE optimizer class. Defaults to FIRE
        fix_symmetry: Whether to fix structure symmetry during relaxation
        cell_filter_cls: Optional ASE filter to modify the atoms before optimization

    Returns:
        Atoms: Relaxed ASE atoms object
    """

    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    if cell_filter_cls is not None:
        _atoms = cell_filter_cls(atoms)
    else:
        _atoms = atoms

    optimizer_cls = FIRE if optimizer_cls is None else optimizer_cls
    opt = optimizer_cls(_atoms, maxstep=maxstep, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    # Update atoms.info with optimization metadata
    atoms.info |= {"opt_nsteps": opt.nsteps, "opt_converged": opt.converged()}
    return atoms
