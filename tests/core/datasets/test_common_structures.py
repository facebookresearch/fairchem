"""
Simple tests for common structure generation helper functions.
"""

from __future__ import annotations

from fairchem.core.datasets.common_structures import (
    get_fcc_crystal_by_num_atoms,
    get_fcc_crystal_by_num_cells,
)


class TestGetFCCCrystalByNumAtoms:
    """Test the get_fcc_crystal_by_num_atoms function"""

    def test_correct_number_of_atoms(self):
        """Test returns requested number of atoms"""
        atoms = get_fcc_crystal_by_num_atoms(100)
        assert len(atoms) == 100

    def test_correct_atom_type(self):
        """Test returns correct atom type"""
        atoms = get_fcc_crystal_by_num_atoms(50, atom_type="C")
        assert all(symbol == "C" for symbol in atoms.get_chemical_symbols())

        atoms = get_fcc_crystal_by_num_atoms(50, atom_type="Cu")
        assert all(symbol == "Cu" for symbol in atoms.get_chemical_symbols())

    def test_has_metadata(self):
        """Test has charge and spin info"""
        atoms = get_fcc_crystal_by_num_atoms(50)
        assert atoms.info["charge"] == 0
        assert atoms.info["spin"] == 0


class TestGetFCCCrystalByNumCells:
    """Test the get_fcc_crystal_by_num_cells function"""

    def test_correct_number_of_atoms(self):
        """Test returns correct number of atoms (4 per unit cell)"""
        atoms = get_fcc_crystal_by_num_cells(2)
        assert len(atoms) == 4 * (2**3)  # 4 atoms per unit cell

    def test_correct_atom_type(self):
        """Test returns correct atom type"""
        atoms = get_fcc_crystal_by_num_cells(2, atom_type="Fe")
        assert all(symbol == "Fe" for symbol in atoms.get_chemical_symbols())

        atoms = get_fcc_crystal_by_num_cells(2, atom_type="Cu")
        assert all(symbol == "Cu" for symbol in atoms.get_chemical_symbols())

    def test_has_metadata(self):
        """Test has charge and spin info"""
        atoms = get_fcc_crystal_by_num_cells(2)
        assert atoms.info["charge"] == 0
        assert atoms.info["spin"] == 0
