"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.build.surface import add_adsorbate
from ase.constraints import FixAtoms
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


def get_slab_adsorbate(
    slab_symbol: str = "Cu",
    size: tuple[int, int, int] = (3, 3, 4),
    adsorbate: str = "CO",
    vacuum: float = 10.0,
) -> Atoms:
    """
    Create a slab+adsorbate system with OC20-style tags and FixAtoms.

    Tags follow OC20 convention: subsurface=0, surface=1, adsorbate=2.
    Subsurface atoms are fixed via FixAtoms constraints.

    Args:
        slab_symbol: Element for the FCC slab.
        size: (x, y, layers) repetitions for fcc111.
        adsorbate: Molecule name for the adsorbate (ASE molecule db).
        vacuum: Vacuum spacing in Angstroms.

    Returns:
        Atoms with tags, FixAtoms constraints, and charge/spin in info.
    """
    slab = fcc111(slab_symbol, size=size, vacuum=vacuum, periodic=True)

    # OC20-style tags: subsurface=0, surface=1
    # ASE fcc111 tags layers 1 (top) through N (bottom)
    tags = np.zeros(len(slab), dtype=int)
    tags[slab.get_tags() == 1] = 1  # top layer = surface
    slab.set_tags(tags)

    # Fix subsurface atoms
    fixed_indices = np.where(tags == 0)[0]
    slab.constraints = [FixAtoms(indices=fixed_indices)]

    # Add adsorbate molecule
    ads = molecule(adsorbate)
    add_adsorbate(slab, ads, height=2.0, position="ontop")

    # Tag adsorbate atoms
    new_tags = slab.get_tags()
    new_tags[-len(ads) :] = 2
    slab.set_tags(new_tags)

    slab.info = {"charge": 0, "spin": 0}
    return slab
