from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.lattice.cubic import FaceCenteredCubic


def get_fcc_crystal_by_num_atoms(
    num_atoms: int,
    lattice_constant: float = 3.8,
    atom_type: str = "C",
) -> Atoms:
    # lattice_constant = 3.8, fcc generates a supercell with ~50 edges/atom, used for benchmarking
    atoms = bulk(atom_type, "fcc", a=lattice_constant)
    n_cells = int(np.ceil(np.cbrt(num_atoms)))
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    indices = np.random.choice(len(atoms), num_atoms, replace=False)
    sampled_atoms = atoms[indices]
    sampled_atoms.info = {"charge": 0, "spin": 0}
    return sampled_atoms


def get_fcc_crystal_by_num_cells(
    n_cells: int,
    atom_type: str = "Cu",
    lattice_constant: float = 3.61,
) -> Atoms:
    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbol=atom_type,
        size=(n_cells, n_cells, n_cells),
        pbc=True,
        latticeconstant=lattice_constant,
    )
    atoms.info = {"charge": 0, "spin": 0}
    return atoms


def get_water_box(num_molecules=20, box_size=10.0, seed=42) -> Atoms:
    """Create a random box of water molecules."""

    rng = np.random.default_rng(seed)
    water = molecule("H2O")

    all_positions = []
    all_symbols = []

    for _ in range(num_molecules):
        # Random position and rotation for each water molecule
        offset = rng.random(3) * box_size
        positions = water.get_positions() + offset
        all_positions.extend(positions)
        all_symbols.extend(water.get_chemical_symbols())

    atoms = Atoms(
        symbols=all_symbols, positions=all_positions, cell=[box_size] * 3, pbc=True
    )
    return atoms
