from __future__ import annotations

import numpy as np
from ase.build import bulk
from ase.lattice.cubic import FaceCenteredCubic


def get_fcc_carbon_xtal(
    num_atoms: int,
    lattice_constant: float = 3.8,
):
    # lattice_constant = 3.8, fcc generates a supercell with ~50 edges/atom, used for benchmarking
    atoms = bulk("C", "fcc", a=lattice_constant)
    n_cells = int(np.ceil(np.cbrt(num_atoms)))
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    indices = np.random.choice(len(atoms), num_atoms, replace=False)
    sampled_atoms = atoms[indices]
    return sampled_atoms


def get_copper_fcc(
    n_cells: int,
):
    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbol="Cu",
        size=(n_cells, n_cells, n_cells),
        pbc=True,
    )
    atoms.info = {"charge": 0, "spin": 0}
    return atoms
