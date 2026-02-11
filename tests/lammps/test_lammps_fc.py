from __future__ import annotations

import numpy as np
import torch
from fairchem.lammps.lammps_fc import restricted_cell_from_lammps_box


def general_to_restricted_triclinic(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """
    Convert general triclinic lattice vectors A, B, C to LAMMPS restricted triclinic parameters.

    Based on equations from https://docs.lammps.org/Howto_triclinic.html:
    - lx = |A|
    - xy = B · A_hat where A_hat = A/|A|
    - ly = sqrt(|B|² - xy²)
    - xz = C · A_hat
    - yz = (C · B - xy*xz) / ly
    - lz = sqrt(|C|² - xz² - yz²)

    Args:
        A: First lattice vector (3,)
        B: Second lattice vector (3,)
        C: Third lattice vector (3,)

    Returns:
        tuple: (lx, ly, lz, xy, xz, yz) restricted triclinic parameters
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    # lx = |A|
    lx = np.linalg.norm(A)
    A_hat = A / lx

    # xy = B · A_hat
    xy = np.dot(B, A_hat)

    # ly = sqrt(|B|² - xy²)
    ly = np.sqrt(np.dot(B, B) - xy**2)

    # xz = C · A_hat
    xz = np.dot(C, A_hat)

    # yz = (C · B - xy*xz) / ly
    yz = (np.dot(C, B) - xy * xz) / ly

    # lz = sqrt(|C|² - xz² - yz²)
    lz = np.sqrt(np.dot(C, C) - xz**2 - yz**2)

    return lx, ly, lz, xy, xz, yz


def cell_lengths_and_angles(cell: np.ndarray):
    """
    Compute cell lengths (a, b, c) and angles (alpha, beta, gamma) from a 3x3 cell matrix.

    Args:
        cell: 3x3 array where rows are lattice vectors

    Returns:
        tuple: (a, b, c, alpha, beta, gamma) where angles are in radians
    """
    a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    # alpha = angle between b and c
    alpha = np.arccos(np.dot(b_vec, c_vec) / (b * c))
    # beta = angle between a and c
    beta = np.arccos(np.dot(a_vec, c_vec) / (a * c))
    # gamma = angle between a and b
    gamma = np.arccos(np.dot(a_vec, b_vec) / (a * b))

    return a, b, c, alpha, beta, gamma


class TestGeneralTriclinicCellFromLammpsBox:
    """Tests for restricted_cell_from_lammps_box function.

    The function converts LAMMPS restricted triclinic box parameters to ASE cell format.
    Tests verify that starting from general lattice vectors A, B, C:
    1. Convert to restricted parameters (lx, ly, lz, xy, xz, yz)
    2. Call restricted_cell_from_lammps_box
    3. The resulting cell preserves volume, cell lengths, and angles
    """

    def test_cubic_cell(self):
        """Test with a simple cubic cell - lattice vectors along axes."""
        # Cubic cell with lattice parameter a = 5.0
        A = np.array([5.0, 0.0, 0.0])
        B = np.array([0.0, 5.0, 0.0])
        C = np.array([0.0, 0.0, 5.0])
        original_cell = np.stack([A, B, C])

        lx, ly, lz, xy, xz, yz = general_to_restricted_triclinic(A, B, C)

        boxlo = [0.0, 0.0, 0.0]
        boxhi = [lx, ly, lz]

        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
        cell_np = cell.squeeze().numpy()

        # For cubic, the restricted form equals the original
        expected_cell = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        assert torch.allclose(cell, expected_cell, atol=1e-5)

        # Verify volume is preserved
        original_volume = np.abs(np.linalg.det(original_cell))
        result_volume = np.abs(np.linalg.det(cell_np))
        assert np.isclose(original_volume, result_volume, atol=1e-4)

    def test_orthorhombic_cell(self):
        """Test with an orthorhombic cell - different lengths, right angles."""
        A = np.array([3.0, 0.0, 0.0])
        B = np.array([0.0, 4.0, 0.0])
        C = np.array([0.0, 0.0, 5.0])
        original_cell = np.stack([A, B, C])

        lx, ly, lz, xy, xz, yz = general_to_restricted_triclinic(A, B, C)

        boxlo = [0.0, 0.0, 0.0]
        boxhi = [lx, ly, lz]

        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
        cell_np = cell.squeeze().numpy()

        # For orthorhombic, the restricted form equals the original
        expected_cell = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        assert torch.allclose(cell, expected_cell, atol=1e-5)

        # Verify volume is preserved
        original_volume = np.abs(np.linalg.det(original_cell))
        result_volume = np.abs(np.linalg.det(cell_np))
        assert np.isclose(original_volume, result_volume, atol=1e-4)

    def test_monoclinic_cell(self):
        """Test with a monoclinic cell - one tilted angle.

        Starts with general lattice vectors, converts to restricted,
        and verifies that the resulting cell has the same volume, lengths, and angles.
        """
        # Monoclinic cell with beta angle != 90 degrees
        a, b, c = 4.0, 5.0, 6.0
        beta = np.radians(110)  # angle between A and C

        # General lattice vectors (A along x, B along y, C tilted in xz-plane)
        A = np.array([a, 0.0, 0.0])
        B = np.array([0.0, b, 0.0])
        C = np.array([c * np.cos(beta), 0.0, c * np.sin(beta)])
        original_cell = np.stack([A, B, C])

        # Convert to restricted triclinic parameters
        lx, ly, lz, xy, xz, yz = general_to_restricted_triclinic(A, B, C)

        boxlo = [0.0, 0.0, 0.0]
        boxhi = [lx, ly, lz]

        # Get the cell from the function
        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
        cell_np = cell.squeeze().numpy()

        # Verify cell volumes match
        original_volume = np.abs(np.linalg.det(original_cell))
        result_volume = np.abs(np.linalg.det(cell_np))
        assert np.isclose(
            original_volume, result_volume, atol=1e-4
        ), f"Volume mismatch: original={original_volume}, result={result_volume}"

        # Verify cell lengths match
        orig_a, orig_b, orig_c, orig_alpha, orig_beta, orig_gamma = (
            cell_lengths_and_angles(original_cell)
        )
        res_a, res_b, res_c, res_alpha, res_beta, res_gamma = cell_lengths_and_angles(
            cell_np
        )

        assert np.isclose(
            orig_a, res_a, atol=1e-4
        ), f"Length a mismatch: {orig_a} vs {res_a}"
        assert np.isclose(
            orig_b, res_b, atol=1e-4
        ), f"Length b mismatch: {orig_b} vs {res_b}"
        assert np.isclose(
            orig_c, res_c, atol=1e-4
        ), f"Length c mismatch: {orig_c} vs {res_c}"

        # Verify angles match
        assert np.isclose(
            orig_alpha, res_alpha, atol=1e-4
        ), f"Alpha mismatch: {np.degrees(orig_alpha)} vs {np.degrees(res_alpha)}"
        assert np.isclose(
            orig_beta, res_beta, atol=1e-4
        ), f"Beta mismatch: {np.degrees(orig_beta)} vs {np.degrees(res_beta)}"
        assert np.isclose(
            orig_gamma, res_gamma, atol=1e-4
        ), f"Gamma mismatch: {np.degrees(orig_gamma)} vs {np.degrees(res_gamma)}"

    def test_triclinic_cell(self):
        """Test with a general triclinic cell - all angles different.

        Starts with general lattice vectors, converts to restricted,
        and verifies that the resulting cell has the same volume, lengths, and angles.
        """
        # General triclinic cell with alpha, beta, gamma all != 90
        a, b, c = 4.0, 5.0, 6.0
        alpha = np.radians(80)  # angle between B and C
        beta = np.radians(85)  # angle between A and C
        gamma = np.radians(75)  # angle between A and B

        # Construct general lattice vectors from cell parameters
        # A along x-axis
        A = np.array([a, 0.0, 0.0])

        # B in xy-plane
        B = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

        # C in general direction
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = np.sqrt(c**2 - cx**2 - cy**2)
        C = np.array([cx, cy, cz])
        original_cell = np.stack([A, B, C])

        # Convert to restricted triclinic parameters
        lx, ly, lz, xy, xz, yz = general_to_restricted_triclinic(A, B, C)

        boxlo = [0.0, 0.0, 0.0]
        boxhi = [lx, ly, lz]

        # Get the cell from the function
        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
        cell_np = cell.squeeze().numpy()

        # Verify cell volumes match
        original_volume = np.abs(np.linalg.det(original_cell))
        result_volume = np.abs(np.linalg.det(cell_np))
        assert np.isclose(
            original_volume, result_volume, atol=1e-4
        ), f"Volume mismatch: original={original_volume}, result={result_volume}"

        # Verify cell lengths match
        orig_a, orig_b, orig_c, orig_alpha, orig_beta, orig_gamma = (
            cell_lengths_and_angles(original_cell)
        )
        res_a, res_b, res_c, res_alpha, res_beta, res_gamma = cell_lengths_and_angles(
            cell_np
        )

        assert np.isclose(
            orig_a, res_a, atol=1e-4
        ), f"Length a mismatch: {orig_a} vs {res_a}"
        assert np.isclose(
            orig_b, res_b, atol=1e-4
        ), f"Length b mismatch: {orig_b} vs {res_b}"
        assert np.isclose(
            orig_c, res_c, atol=1e-4
        ), f"Length c mismatch: {orig_c} vs {res_c}"

        # Verify angles match
        assert np.isclose(
            orig_alpha, res_alpha, atol=1e-4
        ), f"Alpha mismatch: {np.degrees(orig_alpha)} vs {np.degrees(res_alpha)}"
        assert np.isclose(
            orig_beta, res_beta, atol=1e-4
        ), f"Beta mismatch: {np.degrees(orig_beta)} vs {np.degrees(res_beta)}"
        assert np.isclose(
            orig_gamma, res_gamma, atol=1e-4
        ), f"Gamma mismatch: {np.degrees(orig_gamma)} vs {np.degrees(res_gamma)}"

    def test_hexagonal_cell(self):
        """Test with a hexagonal cell - gamma = 120 degrees."""
        a = 3.0
        c = 5.0
        gamma = np.radians(120)

        A = np.array([a, 0.0, 0.0])
        B = np.array([a * np.cos(gamma), a * np.sin(gamma), 0.0])
        C = np.array([0.0, 0.0, c])
        original_cell = np.stack([A, B, C])

        lx, ly, lz, xy, xz, yz = general_to_restricted_triclinic(A, B, C)

        boxlo = [0.0, 0.0, 0.0]
        boxhi = [lx, ly, lz]

        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
        cell_np = cell.squeeze().numpy()

        # Verify volume is preserved
        original_volume = np.abs(np.linalg.det(original_cell))
        result_volume = np.abs(np.linalg.det(cell_np))
        assert np.isclose(original_volume, result_volume, atol=1e-4)

        # Verify cell lengths and angles match
        orig_a, orig_b, orig_c, orig_alpha, orig_beta, orig_gamma = (
            cell_lengths_and_angles(original_cell)
        )
        res_a, res_b, res_c, res_alpha, res_beta, res_gamma = cell_lengths_and_angles(
            cell_np
        )

        assert np.isclose(orig_a, res_a, atol=1e-4)
        assert np.isclose(orig_b, res_b, atol=1e-4)
        assert np.isclose(orig_c, res_c, atol=1e-4)
        assert np.isclose(orig_alpha, res_alpha, atol=1e-4)
        assert np.isclose(orig_beta, res_beta, atol=1e-4)
        assert np.isclose(orig_gamma, res_gamma, atol=1e-4)

        # Check specific restricted values for hexagonal
        assert np.isclose(lx, a, atol=1e-5)
        assert np.isclose(xy, a * np.cos(gamma), atol=1e-5)  # -a/2
        assert np.isclose(ly, a * np.sin(gamma), atol=1e-5)  # a*sqrt(3)/2
        assert np.isclose(lz, c, atol=1e-5)

    def test_nonzero_boxlo(self):
        """Test that boxlo offsets are properly handled."""
        # Simple cubic with non-zero boxlo
        boxlo = [1.0, 2.0, 3.0]
        boxhi = [6.0, 7.0, 8.0]  # lx=ly=lz=5
        xy, yz, xz = 0.0, 0.0, 0.0

        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)

        expected_cell = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        assert torch.allclose(cell, expected_cell, atol=1e-5)

    def test_roundtrip_preserves_cell_properties(self):
        """Test that general -> restricted -> cell preserves all cell properties."""
        # Random general triclinic parameters
        np.random.seed(42)
        a, b, c = 3.0 + np.random.rand(3) * 2.0
        alpha = np.radians(70 + np.random.rand() * 40)
        beta = np.radians(70 + np.random.rand() * 40)
        gamma = np.radians(70 + np.random.rand() * 40)

        # Construct lattice vectors
        A = np.array([a, 0.0, 0.0])
        B = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0))
        C = np.array([cx, cy, cz])
        original_cell = np.stack([A, B, C])

        lx, ly, lz, xy, xz, yz = general_to_restricted_triclinic(A, B, C)

        boxlo = [0.0, 0.0, 0.0]
        boxhi = [lx, ly, lz]

        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
        cell_np = cell.squeeze().numpy()

        # Verify volume is preserved
        original_volume = np.abs(np.linalg.det(original_cell))
        result_volume = np.abs(np.linalg.det(cell_np))
        assert np.isclose(original_volume, result_volume, atol=1e-4)

        # Verify all cell lengths and angles are preserved
        orig_a, orig_b, orig_c, orig_alpha, orig_beta, orig_gamma = (
            cell_lengths_and_angles(original_cell)
        )
        res_a, res_b, res_c, res_alpha, res_beta, res_gamma = cell_lengths_and_angles(
            cell_np
        )

        assert np.isclose(orig_a, res_a, atol=1e-4)
        assert np.isclose(orig_b, res_b, atol=1e-4)
        assert np.isclose(orig_c, res_c, atol=1e-4)
        assert np.isclose(orig_alpha, res_alpha, atol=1e-4)
        assert np.isclose(orig_beta, res_beta, atol=1e-4)
        assert np.isclose(orig_gamma, res_gamma, atol=1e-4)

    def test_output_shape(self):
        """Test that output has correct shape (1, 3, 3)."""
        boxlo = [0.0, 0.0, 0.0]
        boxhi = [5.0, 5.0, 5.0]
        xy, yz, xz = 1.0, 0.5, 0.3

        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)

        assert cell.shape == (1, 3, 3)
        assert cell.dtype == torch.float32

    def test_output_restricted_form(self):
        """Test that output follows the restricted triclinic form exactly.

        The restricted form has:
        - a = (lx, 0, 0)
        - b = (xy, ly, 0)
        - c = (xz, yz, lz)
        """
        boxlo = [0.0, 0.0, 0.0]
        boxhi = [4.0, 5.0, 6.0]
        xy, yz, xz = 1.5, 0.8, 1.2

        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
        cell_sq = cell.squeeze()

        lx, ly, lz = 4.0, 5.0, 6.0

        # Row 0: (lx, 0, 0)
        assert torch.isclose(cell_sq[0, 0], torch.tensor(lx, dtype=torch.float32))
        assert torch.isclose(cell_sq[0, 1], torch.tensor(0.0, dtype=torch.float32))
        assert torch.isclose(cell_sq[0, 2], torch.tensor(0.0, dtype=torch.float32))

        # Row 1: (xy, ly, 0)
        assert torch.isclose(cell_sq[1, 0], torch.tensor(xy, dtype=torch.float32))
        assert torch.isclose(cell_sq[1, 1], torch.tensor(ly, dtype=torch.float32))
        assert torch.isclose(cell_sq[1, 2], torch.tensor(0.0, dtype=torch.float32))

        # Row 2: (xz, yz, lz)
        assert torch.isclose(cell_sq[2, 0], torch.tensor(xz, dtype=torch.float32))
        assert torch.isclose(cell_sq[2, 1], torch.tensor(yz, dtype=torch.float32))
        assert torch.isclose(cell_sq[2, 2], torch.tensor(lz, dtype=torch.float32))
