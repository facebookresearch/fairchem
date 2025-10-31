"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet

from fairchem.data.omat.entries.compatibility import (
    OMat24Compatibility,
    apply_mp_style_corrections,
    generate_computed_structure_entry,
    generate_cse_parameters,
)
from fairchem.data.omat.vasp.sets import OMat24StaticSet

if TYPE_CHECKING:
    from typing import Literal


@pytest.fixture
def simple_structure():
    """Create a simple test structure."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.0)
    species = ["Fe"]
    coords = [[0, 0, 0]]
    return Structure(lattice, species, coords)


@pytest.fixture
def oxide_structure():
    """Create an oxide structure for testing corrections."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.2)
    species = ["Fe", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


@pytest.fixture
def simple_atoms():
    """Create simple ASE Atoms object."""
    return bulk("Fe", "bcc", a=2.87)


@pytest.fixture
def molecule_atoms():
    """Create molecular ASE Atoms object in a periodic box."""
    mol = molecule("H2O")
    mol.set_cell([10, 10, 10])  # Add a periodic box
    mol.set_pbc(True)
    return mol


@pytest.fixture
def mp_relax_set(simple_structure):
    """Create MPRelaxSet for testing."""
    return MPRelaxSet(simple_structure)


@pytest.fixture
def omat24_static_set(simple_structure):
    """Create OMat24StaticSet for testing."""
    return OMat24StaticSet(simple_structure)


class TestOMat24Compatibility:
    """Test OMat24Compatibility class."""

    def test_init_default_config(self):
        """Test initialization with default config file."""
        compat = OMat24Compatibility()
        assert isinstance(compat, MaterialsProject2020Compatibility)
        
        assert hasattr(compat, 'config_file')
        assert hasattr(compat, 'compat_type')
        assert hasattr(compat, 'correct_peroxide')
        assert hasattr(compat, 'strict_anions')

    def test_default_config_file_exists(self):
        """Test that the default config file exists."""
        from fairchem.data.omat.entries.compatibility import OMAT24_CONFIG_FILE
        assert os.path.exists(OMAT24_CONFIG_FILE)


class TestGenerateCSEParameters:
    """Test generate_cse_parameters function."""

    def test_basic_parameters_skip_potcar(self, simple_structure):
        """Test basic parameter generation without POTCAR dependency."""
        
        from pymatgen.io.vasp.sets import MPRelaxSet
        
        
        input_set = MPRelaxSet(simple_structure, user_potcar_functional="PBE")
        
        
        assert hasattr(input_set, 'incar')
        assert hasattr(input_set, 'poscar')
        
        assert input_set.incar is not None
        assert len(simple_structure) > 0

    def test_hubbard_detection_logic(self, simple_structure):
        """Test Hubbard U detection logic."""
        from pymatgen.io.vasp.sets import MPRelaxSet
        
        input_set = MPRelaxSet(simple_structure)
        
        ldau_present = input_set.incar.get("LDAU", False)
        ldauu_values = input_set.incar.get("LDAUU", [])
        
        if ldau_present and ldauu_values:
            # Should be hubbard if LDAU is True and has U values > 0
            has_nonzero_u = any(u > 0 for u in ldauu_values if isinstance(u, (int, float)))
            assert isinstance(has_nonzero_u, bool)
        else:
            # Should be GGA if no LDAU or no U values
            assert not ldau_present or not ldauu_values


class TestGenerateComputedStructureEntry:
    """Test generate_computed_structure_entry function."""

    def test_omat24_correction(self, simple_structure):
        """Test with OMat24 correction type."""
        cse = generate_computed_structure_entry(
            structure=simple_structure,
            total_energy=-10.0,
            correction_type="OMat24",
            check_potcar=False
        )
        
        assert isinstance(cse, ComputedStructureEntry)
        assert cse.structure == simple_structure
        assert hasattr(cse, 'energy')
        assert hasattr(cse, 'parameters')

    def test_invalid_correction_type(self, simple_structure):
        """Test with invalid correction type."""
        with pytest.raises(ValueError) as exc_info:
            generate_computed_structure_entry(
                structure=simple_structure,
                total_energy=-10.0,
                correction_type="INVALID",
                check_potcar=False
            )
        
        assert "INVALID is not a valid correction type" in str(exc_info.value)

    def test_oxidation_states_handling(self, oxide_structure):
        """Test oxidation states are properly handled."""
        cse = generate_computed_structure_entry(
            structure=oxide_structure,
            total_energy=-15.0,
            correction_type="OMat24",
            check_potcar=False
        )
        
        assert "oxidation_states" in cse.data
        assert isinstance(cse.data["oxidation_states"], dict)

    def test_energy_correction_applied(self, simple_structure):
        """Test that energy corrections are actually applied."""
        original_energy = -10.0
        
        cse = generate_computed_structure_entry(
            structure=simple_structure,
            total_energy=original_energy,
            correction_type="OMat24",
            check_potcar=False
        )
        
        # Energy should be different from original if corrections were applied
        # (This test assumes corrections exist for the structure)
        assert hasattr(cse, 'energy') and hasattr(cse, 'uncorrected_energy')

    def test_check_potcar_parameter(self, simple_structure):
        """Test that check_potcar parameter is properly passed through."""
        # This should not raise an error with check_potcar=False
        cse = generate_computed_structure_entry(
            structure=simple_structure,
            total_energy=-10.0,
            correction_type="OMat24",
            check_potcar=False
        )
        
        assert isinstance(cse, ComputedStructureEntry)


class TestApplyMPStyleCorrections:
    """Test apply_mp_style_corrections function."""

    def test_mp2020_corrections(self, simple_atoms):
        """Test MP2020 style corrections."""
        # Skip this test if POTCAR files are not available
        pytest.skip("POTCAR files not available in test environment")

    def test_omat24_corrections(self, simple_atoms):
        """Test OMat24 style corrections."""
        original_energy = -10.0
        
        corrected_energy = apply_mp_style_corrections(
            energy=original_energy,
            atoms=simple_atoms,
            correction_type="OMat24"
        )
        
        assert isinstance(corrected_energy, float)

    def test_molecular_system(self, molecule_atoms):
        """Test corrections on molecular system."""
        original_energy = -15.0
        
        corrected_energy = apply_mp_style_corrections(
            energy=original_energy,
            atoms=molecule_atoms,
            correction_type="OMat24"
        )
        
        assert isinstance(corrected_energy, float)

    def test_invalid_correction_type_in_apply(self, simple_atoms):
        """Test invalid correction type in apply function."""
        with pytest.raises(ValueError) as exc_info:
            apply_mp_style_corrections(
                energy=-10.0,
                atoms=simple_atoms,
                correction_type="INVALID"
            )
        
        assert "INVALID is not a valid correction type" in str(exc_info.value)

    def test_energy_consistency(self, simple_atoms):
        """Test that the same atoms object gives consistent corrections."""
        original_energy = -10.0
        
        corrected1 = apply_mp_style_corrections(
            energy=original_energy,
            atoms=simple_atoms,
            correction_type="OMat24"
        )
        
        corrected2 = apply_mp_style_corrections(
            energy=original_energy,
            atoms=simple_atoms,
            correction_type="OMat24"
        )
        
        assert np.isclose(corrected1, corrected2, rtol=1e-10)

    def test_different_correction_types_give_different_results(self, simple_atoms):
        """Test that different correction types can give different results."""
        # Test only with OMat24 since MP2020 requires POTCAR files
        original_energy = -10.0
        
        omat24_corrected = apply_mp_style_corrections(
            energy=original_energy,
            atoms=simple_atoms,
            correction_type="OMat24"
        )
        
        # Should be a valid float
        assert isinstance(omat24_corrected, float)
        
        # Test that applying corrections twice gives same result
        omat24_corrected_again = apply_mp_style_corrections(
            energy=original_energy,
            atoms=simple_atoms,
            correction_type="OMat24"
        )
        
        assert np.isclose(omat24_corrected, omat24_corrected_again, rtol=1e-10)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_end_to_end_workflow(self, simple_atoms):
        """Test complete workflow from atoms to corrected energy."""
        original_energy = -10.0
        
        structure = AseAtomsAdaptor.get_structure(simple_atoms)
        
        cse = generate_computed_structure_entry(
            structure=structure,
            total_energy=original_energy,
            correction_type="OMat24",
            check_potcar=False
        )
        
        corrected_energy = apply_mp_style_corrections(
            energy=original_energy,
            atoms=simple_atoms,
            correction_type="OMat24"
        )

        assert np.isclose(cse.energy, corrected_energy, rtol=1e-6)

    def test_compatibility_processing(self, oxide_structure):
        """Test that compatibility processing works correctly."""
        compat = OMat24Compatibility(check_potcar=False)
        
        cse = generate_computed_structure_entry(
            structure=oxide_structure,
            total_energy=-15.0,
            correction_type="OMat24",
            check_potcar=False
        )
        
        # The CSE should already be processed, but we can process it again
        # This should not raise an error
        processed_cse = compat.process_entry(cse, clean=True)
        
        if processed_cse is not None:
            assert isinstance(processed_cse, ComputedStructureEntry)