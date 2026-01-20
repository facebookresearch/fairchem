"""
Tests for common structure generation helper functions.
"""

from __future__ import annotations

import numpy as np

from fairchem.core.datasets.common_structures import get_copper_fcc, get_fcc_carbon_xtal


class TestGetFCCCarbonXtal:
    """Test the get_fcc_carbon_xtal function"""

    def test_returns_correct_number_of_atoms(self):
        """Test that function returns approximately the requested number of atoms"""
        num_atoms = 100
        atoms = get_fcc_carbon_xtal(num_atoms)
        assert len(atoms) == num_atoms

    def test_has_periodic_boundary_conditions(self):
        """Test that PBC is set"""
        atoms = get_fcc_carbon_xtal(50)
        assert all(atoms.get_pbc())

    def test_has_valid_cell(self):
        """Test that cell is non-zero"""
        atoms = get_fcc_carbon_xtal(50)
        assert np.all(atoms.cell.lengths() > 0)

    def test_different_sizes(self):
        """Test generation with different atom counts"""
        for num_atoms in [10, 50, 100, 200]:
            atoms = get_fcc_carbon_xtal(num_atoms)
            assert len(atoms) == num_atoms
            assert all(symbol == "C" for symbol in atoms.get_chemical_symbols())

    def test_custom_lattice_constant(self):
        """Test with custom lattice constant"""
        atoms = get_fcc_carbon_xtal(50, lattice_constant=4.0)
        assert len(atoms) == 50
        # Cell should reflect the custom lattice constant
        assert atoms.cell.lengths()[0] > 0

    def test_small_system(self):
        """Test with very small system (1 atom)"""
        atoms = get_fcc_carbon_xtal(1)
        assert len(atoms) == 1
        assert atoms.get_chemical_symbols()[0] == "C"


class TestGetCopperFCC:
    """Test the get_copper_fcc function"""

    def test_returns_correct_structure(self):
        """Test that function returns FCC copper structure"""
        n_cells = 2
        atoms = get_copper_fcc(n_cells)
        # FCC has 4 atoms per unit cell
        expected_atoms = 4 * (n_cells**3)
        assert len(atoms) == expected_atoms

    def test_returns_copper_atoms(self):
        """Test that all atoms are copper"""
        atoms = get_copper_fcc(2)
        assert all(symbol == "Cu" for symbol in atoms.get_chemical_symbols())

    def test_has_periodic_boundary_conditions(self):
        """Test that PBC is set"""
        atoms = get_copper_fcc(2)
        assert all(atoms.get_pbc())

    def test_has_charge_and_spin_info(self):
        """Test that info dict contains charge and spin"""
        atoms = get_copper_fcc(2)
        assert "charge" in atoms.info
        assert "spin" in atoms.info
        assert atoms.info["charge"] == 0
        assert atoms.info["spin"] == 0

    def test_has_valid_cell(self):
        """Test that cell is non-zero"""
        atoms = get_copper_fcc(2)
        assert np.all(atoms.cell.lengths() > 0)

    def test_different_sizes(self):
        """Test generation with different cell counts"""
        for n_cells in [1, 2, 3, 4]:
            atoms = get_copper_fcc(n_cells)
            expected_atoms = 4 * (n_cells**3)
            assert len(atoms) == expected_atoms
            assert all(symbol == "Cu" for symbol in atoms.get_chemical_symbols())

    def test_single_cell(self):
        """Test with single unit cell"""
        atoms = get_copper_fcc(1)
        assert len(atoms) == 4  # FCC has 4 atoms per unit cell

    def test_large_system(self):
        """Test with larger system"""
        n_cells = 5
        atoms = get_copper_fcc(n_cells)
        expected_atoms = 4 * (n_cells**3)
        assert len(atoms) == expected_atoms

    def test_cell_symmetry(self):
        """Test that FCC structure has cubic symmetry"""
        atoms = get_copper_fcc(2)
        cell_lengths = atoms.cell.lengths()
        # All cell lengths should be equal for cubic system
        assert np.allclose(cell_lengths[0], cell_lengths[1])
        assert np.allclose(cell_lengths[1], cell_lengths[2])

    def test_directions_are_orthogonal(self):
        """Test that crystal directions are orthogonal"""
        atoms = get_copper_fcc(2)
        cell = atoms.cell.array
        # Check that cell vectors are orthogonal (cubic cell)
        assert np.allclose(cell, np.diag(np.diagonal(cell)), atol=1e-10)
