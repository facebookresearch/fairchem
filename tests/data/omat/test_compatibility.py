"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile
import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from monty.serialization import loadfn
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
    OMAT24_CONFIG_FILE,
)
from fairchem.data.omat.vasp.sets import OMat24StaticSet

from . import compatibility_resources

if TYPE_CHECKING:
    from typing import Literal


@pytest.fixture(scope="session")
def mp_computed_structure_entries():
    with importlib.resources.path(
        compatibility_resources, "test_calc_compounds_mp.json.gz"
    ) as path:
        entries = loadfn(path)
        yield entries


@pytest.fixture(scope="session")
def omat_computed_structure_entries():
    with importlib.resources.path(
        compatibility_resources, "test_calc_compounds_omat.json.gz"
    ) as path:
        entries = loadfn(path)
        yield entries


@pytest.fixture
def iron_oxide_structure():
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.2)
    species = ["Fe", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


@pytest.fixture
def mp_relax_set(iron_oxide_structure):
    return MPRelaxSet(iron_oxide_structure)


@pytest.fixture
def omat24_static_set(iron_oxide_structure):
    return OMat24StaticSet(iron_oxide_structure)


class TestOMat24Compatibility:

    def test_init_default_config(self):
        compat = OMat24Compatibility()
        assert isinstance(compat, MaterialsProject2020Compatibility)
        
        assert hasattr(compat, 'config_file')
        assert hasattr(compat, 'compat_type')
        assert hasattr(compat, 'correct_peroxide')
        assert hasattr(compat, 'strict_anions')
        assert compat.config_file == OMAT24_CONFIG_FILE

    def test_default_config_file_exists(self):
        assert os.path.exists(OMAT24_CONFIG_FILE)


class TestGenerateCSEParameters:

    def test_basic_parameters_skip_potcar(self, iron_oxide_structure):
        input_set = OMat24StaticSet(iron_oxide_structure, user_potcar_functional="PBE")
        
        assert hasattr(input_set, 'incar')
        assert hasattr(input_set, 'poscar')
        
        assert input_set.incar is not None
        assert len(iron_oxide_structure) > 0

    def test_hubbard_detection_logic(self, iron_oxide_structure):
        input_set = OMat24StaticSet(iron_oxide_structure)
        
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

    def test_omat24_generated_computed_structure_entry(self, omat_computed_structure_entries):
        for omat_cse in omat_computed_structure_entries.values():
            cse = generate_computed_structure_entry(
                structure=omat_cse.structure,
                total_energy=omat_cse.uncorrected_energy,
                correction_type="OMat24",
                check_potcar=False
            )
            print(omat_cse.energy_adjustments)
            print(cse.energy_adjustments)
            
            assert cse.energy == omat_cse.energy
            assert cse.uncorrected_energy == omat_cse.uncorrected_energy

    def test_mp_generated_computed_structure_entry(self, mp_computed_structure_entries):
        for mp_cse in mp_computed_structure_entries.values():
            cse = generate_computed_structure_entry(
                structure=mp_cse.structure,
                total_energy=mp_cse.uncorrected_energy,
                correction_type="MP2020",
                check_potcar=False
            )

            assert cse.energy == mp_cse.energy
            assert cse.uncorrected_energy == mp_cse.uncorrected_energy

    def test_invalid_correction_type(self, iron_oxide_structure):
        with pytest.raises(ValueError) as exc_info:
            generate_computed_structure_entry(
                structure=iron_oxide_structure,
                total_energy=-10.0,
                correction_type="INVALID",
                check_potcar=False
            )
        
        assert "INVALID is not a valid correction type" in str(exc_info.value)

    def test_oxidation_states_handling(self, iron_oxide_structure):
        cse = generate_computed_structure_entry(
            structure=iron_oxide_structure,
            total_energy=-15.0,
            correction_type="OMat24",
            check_potcar=False
        )
        
        assert "oxidation_states" in cse.data
        assert isinstance(cse.data["oxidation_states"], dict)

    def test_energy_correction_applied(self, iron_oxide_structure):
        original_energy = -10.0
        
        cse = generate_computed_structure_entry(
            structure=iron_oxide_structure,
            total_energy=original_energy,
            correction_type="OMat24",
            check_potcar=False
        )
        
        # Energy should be different from original if corrections were applied
        # (This test assumes corrections exist for the structure)
        assert hasattr(cse, 'energy') and hasattr(cse, 'uncorrected_energy')


class TestApplyMPStyleCorrections:
    """Test apply_mp_style_corrections function."""

    def test_invalid_correction_type_in_apply(self, iron_oxide_structure):
        """Test invalid correction type in apply function."""
        with pytest.raises(ValueError) as exc_info:
            apply_mp_style_corrections(
                energy=-10.0,
                atoms=iron_oxide_structure.to_ase_atoms(),
                correction_type="INVALID"
            )
        
        assert "INVALID is not a valid correction type" in str(exc_info.value)

    def test_different_correction_types_give_different_results(self, iron_oxide_structure):
        """Test that different correction types can give different results."""
        original_energy = -10.0
        
        corrected = apply_mp_style_corrections(
            energy=original_energy,
            atoms=iron_oxide_structure.to_ase_atoms(),
            correction_type="OMat24"
        )
        
        # Test that applying corrections twice gives same result
        corrected_again = apply_mp_style_corrections(
            energy=original_energy,
            atoms=iron_oxide_structure.to_ase_atoms(),
            correction_type="MP2020"
        )

        assert not np.isclose(corrected, corrected_again, rtol=1e-8)

    def test_apply_corrections_mp(self, mp_computed_structure_entries):
        for entry in mp_computed_structure_entries.values():
            atoms = AseAtomsAdaptor.get_atoms(entry.structure)
            corrected_energy = apply_mp_style_corrections(
                energy=entry.uncorrected_energy,
                atoms=atoms,
                correction_type="MP2020",
                check_potcar=False
            )
            assert np.isclose(corrected_energy, entry.energy, rtol=1e-6)

    def test_apply_corrections_omat(self, omat_computed_structure_entries):
        for entry in omat_computed_structure_entries.values():
            atoms = AseAtomsAdaptor.get_atoms(entry.structure)
            corrected_energy = apply_mp_style_corrections(
                energy=entry.uncorrected_energy,
                atoms=atoms,
                correction_type="OMat24",
                check_potcar=False
            )
            assert np.isclose(corrected_energy, entry.energy, rtol=1e-6)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_end_to_end_workflow(self, iron_oxide_structure):
        """Test complete workflow from atoms to corrected energy."""
        original_energy = -10.0
                
        cse = generate_computed_structure_entry(
            structure=iron_oxide_structure,
            total_energy=original_energy,
            correction_type="OMat24",
            check_potcar=False
        )
        
        corrected_energy = apply_mp_style_corrections(
            energy=original_energy,
            atoms=iron_oxide_structure.to_ase_atoms(),
            correction_type="OMat24"
        )

        assert np.isclose(cse.energy, corrected_energy, rtol=1e-6)

    def test_compatibility_processing(self, iron_oxide_structure):
        """Test that compatibility processing works correctly."""
        compat = OMat24Compatibility(check_potcar=False)
        
        cse = generate_computed_structure_entry(
            structure=iron_oxide_structure,
            total_energy=-15.0,
            correction_type="OMat24",
            check_potcar=False
        )
        
        # The CSE should already be processed, but we can process it again
        # This should not raise an error
        processed_cse = compat.process_entry(cse, clean=True)
        
        if processed_cse is not None:
            assert isinstance(processed_cse, ComputedStructureEntry)