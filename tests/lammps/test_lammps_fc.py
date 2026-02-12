from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ase import Atoms
from fairchem.lammps.lammps_fc import restricted_cell_from_lammps_box


def create_lammps_data_file(filepath, positions, cell, atom_types, masses):
    """
    Create a LAMMPS data file for a triclinic box.

    Args:
        filepath: Path to write the data file
        positions: Nx3 array of atom positions in Cartesian coordinates
        cell: 3x3 cell matrix (rows are lattice vectors)
        atom_types: List of atom type IDs (1-indexed for LAMMPS)
        masses: Dict mapping atom type ID to mass
    """
    n_atoms = len(positions)
    n_types = len(masses)

    # Extract cell parameters for LAMMPS triclinic box
    # Cell rows are: a = cell[0], b = cell[1], c = cell[2]
    a_vec = cell[0]
    b_vec = cell[1]
    c_vec = cell[2]

    # LAMMPS restricted triclinic parameters
    # See: https://docs.lammps.org/Howto_triclinic.html
    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi = a_vec[0]  # lx
    xy = b_vec[0]
    yhi = b_vec[1]  # ly
    xz = c_vec[0]
    yz = c_vec[1]
    zhi = c_vec[2]  # lz

    with open(filepath, "w") as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_types} atom types\n\n")

        f.write(f"{xlo} {xhi} xlo xhi\n")
        f.write(f"{ylo} {yhi} ylo yhi\n")
        f.write(f"{zlo} {zhi} zlo zhi\n")
        f.write(f"{xy} {xz} {yz} xy xz yz\n\n")

        f.write("Masses\n\n")
        for type_id, mass in masses.items():
            f.write(f"{type_id} {mass}\n")
        f.write("\n")

        f.write("Atoms\n\n")
        for i, (pos, atype) in enumerate(zip(positions, atom_types), start=1):
            f.write(f"{i} {atype} {pos[0]} {pos[1]} {pos[2]}\n")


def test_scaled_positions_lammps_vs_ase():
    """
    Test that scaled atomic positions computed by ASE match those from LAMMPS.

    This test:
    1. Creates a LAMMPS simulation with atoms in a triclinic box
    2. Extracts box parameters and Cartesian positions from LAMMPS
    3. Uses restricted_cell_from_lammps_box to get the ASE cell
    4. Creates an ASE Atoms object with the positions and cell
    5. Verifies that scaled (fractional) positions match
    """
    lammps = pytest.importorskip("lammps")

    # Define a general triclinic cell (rows are lattice vectors)
    # This is already in restricted triclinic form for simplicity
    cell = np.array(
        [
            [5.0, 0.0, 0.0],  # a vector
            [1.0, 4.0, 0.0],  # b vector (tilted in xy)
            [0.5, 0.3, 6.0],  # c vector (tilted in xz and yz)
        ],
        dtype=np.float64,
    )

    # Define atom positions in Cartesian coordinates
    # Place atoms at various fractional coordinates
    fractional_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
            [0.25, 0.75, 0.25],
        ],
        dtype=np.float64,
    )

    # Convert fractional to Cartesian: pos = frac @ cell
    cartesian_positions = fractional_positions @ cell

    # Atom types (all type 1 = Carbon for simplicity)
    atom_types = [1] * len(cartesian_positions)
    masses = {1: 12.011}  # Carbon mass

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "test.data")
        create_lammps_data_file(
            data_file, cartesian_positions, cell, atom_types, masses
        )

        # Create LAMMPS instance and read the data file
        lmp = lammps.lammps(cmdargs=["-screen", "none", "-log", "none"])
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p p p")
        lmp.command(f"read_data {data_file}")

        # Extract box parameters from LAMMPS
        boxlo, boxhi, xy_lmp, yz_lmp, xz_lmp, periodicity, box_change = (
            lmp.extract_box()
        )

        # Extract atom positions from LAMMPS
        nlocal = lmp.get_natoms()
        x_lammps = lmp.numpy.extract_atom("x")[:nlocal].copy()

        # Get ASE cell from LAMMPS box parameters
        cell_from_lammps = restricted_cell_from_lammps_box(
            boxlo, boxhi, xy_lmp, yz_lmp, xz_lmp
        )
        cell_np = cell_from_lammps.squeeze().numpy()

        # Create ASE Atoms object
        ase_atoms = Atoms(
            symbols=["C"] * nlocal, positions=x_lammps, cell=cell_np, pbc=True
        )

        # Get scaled positions from ASE (wrap=False to avoid [0,1) wrapping issues)
        ase_scaled_positions = ase_atoms.get_scaled_positions(wrap=False)

        # The key validation: scaled positions from ASE should match our original
        # fractional positions. We compare modulo 1 to handle periodic boundary effects.
        # Use a reasonable tolerance for floating point precision through LAMMPS I/O.
        def normalize_fractional(frac):
            """Normalize fractional coordinates to [0, 1) handling numerical precision."""
            normalized = frac % 1.0
            # Handle values very close to 1.0 that should wrap to 0.0
            normalized = np.where(np.abs(normalized - 1.0) < 1e-6, 0.0, normalized)
            return normalized

        wrapped_original = normalize_fractional(fractional_positions)
        wrapped_ase = normalize_fractional(ase_scaled_positions)

        assert np.allclose(
            wrapped_ase, wrapped_original, atol=1e-5
        ), f"Scaled positions don't match original fractional:\nASE: {wrapped_ase}\nOriginal: {wrapped_original}"

        lmp.close()


@pytest.mark.parametrize(
    "cell_name,cell",
    [
        (
            "cubic",
            np.array(
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64
            ),
        ),
        (
            "orthorhombic",
            np.array(
                [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64
            ),
        ),
        (
            "monoclinic",
            np.array(
                [[4.0, 0.0, 0.0], [-1.5, 4.5, 0.0], [0.0, 0.0, 6.0]], dtype=np.float64
            ),
        ),
        (
            "triclinic",
            np.array(
                [[5.0, 0.0, 0.0], [1.2, 4.8, 0.0], [0.8, 0.5, 5.5]], dtype=np.float64
            ),
        ),
    ],
)
def test_scaled_positions_various_triclinic_cells(cell_name, cell):
    """
    Test scaled positions for various triclinic cell configurations.

    Tests cubic, orthorhombic, monoclinic, and general triclinic cells
    to ensure the cell conversion works correctly in all cases.
    """
    lammps = pytest.importorskip("lammps")

    fractional_positions = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.7, 0.8, 0.1],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    atom_types = [1] * len(fractional_positions)
    masses = {1: 12.011}

    cartesian_positions = fractional_positions @ cell

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "test.data")
        create_lammps_data_file(
            data_file, cartesian_positions, cell, atom_types, masses
        )

        lmp = lammps.lammps(cmdargs=["-screen", "none", "-log", "none"])
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p p p")
        lmp.command(f"read_data {data_file}")

        boxlo, boxhi, xy_lmp, yz_lmp, xz_lmp, periodicity, box_change = (
            lmp.extract_box()
        )

        nlocal = lmp.get_natoms()
        x_lammps = lmp.numpy.extract_atom("x")[:nlocal].copy()

        cell_from_lammps = restricted_cell_from_lammps_box(
            boxlo, boxhi, xy_lmp, yz_lmp, xz_lmp
        )
        cell_np = cell_from_lammps.squeeze().numpy()

        ase_atoms = Atoms(
            symbols=["C"] * nlocal, positions=x_lammps, cell=cell_np, pbc=True
        )

        ase_scaled_positions = ase_atoms.get_scaled_positions(wrap=False)

        # Verify scaled positions match original fractional positions
        def normalize_fractional(frac):
            """Normalize fractional coordinates to [0, 1) handling numerical precision."""
            normalized = frac % 1.0
            normalized = np.where(np.abs(normalized - 1.0) < 1e-6, 0.0, normalized)
            return normalized

        wrapped_original = normalize_fractional(fractional_positions)
        wrapped_ase = normalize_fractional(ase_scaled_positions)

        assert np.allclose(
            wrapped_ase, wrapped_original, atol=1e-5
        ), f"Cell {cell_name}: Scaled positions don't match.\nASE: {wrapped_ase}\nOriginal: {wrapped_original}"

        lmp.close()


@pytest.mark.parametrize(
    "box_name,boxlo,boxhi,xy,yz,xz",
    [
        ("cubic", [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 0.0, 0.0, 0.0),
        ("orthorhombic", [0.0, 0.0, 0.0], [3.0, 4.0, 5.0], 0.0, 0.0, 0.0),
        ("xy_tilt", [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], 1.0, 0.0, 0.0),
        ("yz_tilt", [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], 0.0, 0.5, 0.0),
        ("xz_tilt", [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], 0.0, 0.0, 0.3),
        ("full_triclinic", [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], 1.0, 0.5, 0.3),
        ("nonzero_boxlo", [1.0, 2.0, 3.0], [6.0, 7.0, 8.0], 0.5, 0.3, 0.2),
    ],
)
def test_cell_conversion_preserves_volume(box_name, boxlo, boxhi, xy, yz, xz):
    """
    Test that restricted_cell_from_lammps_box preserves cell volume.
    """
    cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
    cell_np = cell.squeeze().numpy()

    # Expected volume: lx * ly * lz (for restricted triclinic)
    lx = boxhi[0] - boxlo[0]
    ly = boxhi[1] - boxlo[1]
    lz = boxhi[2] - boxlo[2]
    expected_volume = lx * ly * lz

    actual_volume = np.abs(np.linalg.det(cell_np))

    assert np.isclose(expected_volume, actual_volume, atol=1e-6), (
        f"Volume mismatch for {box_name}: boxlo={boxlo}, boxhi={boxhi}, xy={xy}, yz={yz}, xz={xz}.\n"
        f"Expected: {expected_volume}, Actual: {actual_volume}"
    )
