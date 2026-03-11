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


# Common FCC metals with approximate lattice constants (Angstroms)
FCC_ELEMENTS = {
    "Cu": 3.61,
    "Ag": 4.09,
    "Au": 4.08,
    "Al": 4.05,
    "Ni": 3.52,
    "Pd": 3.89,
    "Pt": 3.92,
    "Pb": 4.95,
    "Ca": 5.58,
    "Sr": 6.08,
}


def get_mixed_batch_systems(
    natoms_list: list[int],
    elements: list[str] | None = None,
    seed: int = 42,
    pbc: bool = False,
) -> list[Atoms]:
    """Generate a list of FCC systems with varying sizes and different elements.

    This is useful for testing batched inference where each system has a different
    number of atoms and element type, which exercises segmented operations like
    segment_mm in fairchem_cpp.

    Args:
        natoms_list: List of atom counts, one per system (e.g., [500, 1000, 2000])
        elements: List of element symbols to use. If None, cycles through FCC_ELEMENTS.
                  If provided, must have same length as natoms_list.
        seed: Random seed for reproducibility
        pbc: Whether to enable periodic boundary conditions

    Returns:
        List of ASE Atoms objects, each with different element and size
    """
    if elements is None:
        element_keys = list(FCC_ELEMENTS.keys())
        elements = [element_keys[i % len(element_keys)] for i in range(len(natoms_list))]
    else:
        if len(elements) != len(natoms_list):
            raise ValueError(
                f"elements list length ({len(elements)}) must match "
                f"natoms_list length ({len(natoms_list)})"
            )

    systems = []
    for i, (natoms, elem) in enumerate(zip(natoms_list, elements)):
        np.random.seed(seed + i)  # deterministic per-system
        lattice_const = FCC_ELEMENTS.get(elem, 3.8)  # fallback for unknown elements
        atoms = get_fcc_crystal_by_num_atoms(
            natoms, lattice_constant=lattice_const, atom_type=elem
        )
        atoms.pbc = [pbc, pbc, pbc]
        systems.append(atoms)

    return systems
