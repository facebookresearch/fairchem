"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from fairchem.core.components.calculate.recipes import phonons


@pytest.mark.parametrize("quasiharmonic", [False, True])
def test_phonon_units_temperature_alignment_and_dos(monkeypatch, quasiharmonic):
    atoms = bulk("Cu", cubic=True)
    atoms.calc = EMT()
    initial_cell = atoms.cell.copy()
    temperatures = np.arange(0, 50, 10)
    raw_thermal = {
        "temperatures": temperatures,
        "free_energy": np.full(5, 2.0),
        "entropy": np.full(5, 3.0),
        "heat_capacity": np.full(5, 4.0),
        "frequency_points": np.array([1.0, 2.0]),
        "total_dos": np.array([0.5, 0.5]),
    }
    thermal = Mock(
        side_effect=lambda *a, **kw: {k: v.copy() for k, v in raw_thermal.items()}
    )
    monkeypatch.setattr(phonons, "get_phonopy_object", Mock(return_value="phonon"))
    force_constants = Mock()
    monkeypatch.setattr(phonons, "produce_force_constants", force_constants)
    monkeypatch.setattr(phonons, "calculate_thermal_properties", thermal)
    relax = Mock()
    monkeypatch.setattr(phonons, "relax_atoms", relax)
    qha = Mock(
        return_value=SimpleNamespace(
            **{
                name: np.ones(4)
                for name in (
                    "thermal_expansion",
                    "gibbs_temperature",
                    "bulk_modulus_temperature",
                    "heat_capacity_P_polyfit",
                    "gruneisen_temperature",
                    "volume_temperature",
                )
            }
        )
    )
    monkeypatch.setattr(phonons, "PhonopyQHA", qha)
    result = phonons.calculate_vibrational_thermo(
        atoms,
        quasiharmonic=quasiharmonic,
        scale_factors=[0.9, 1.0, 1.1],
        t_min=0,
        t_max=40,
        t_step=10,
        relax_initial=True,
        relax_scaled=True,
        compute_dos=True,
    )
    assert np.array_equal(atoms.cell, initial_cell)
    assert np.array_equal(result["frequency_points"], raw_thermal["frequency_points"])
    assert np.array_equal(result["total_dos"], raw_thermal["total_dos"])
    assert np.array_equal(raw_thermal["free_energy"], np.full(5, 2.0))
    assert force_constants.call_count == (3 if quasiharmonic else 1)
    assert sum(call.kwargs["return_dos"] for call in thermal.call_args_list) == 1
    matrix = phonons.get_phonopy_object.call_args.kwargs["supercell_matrix"]
    assert matrix.shape == (3, 3)
    assert np.all(matrix.diagonal() * atoms.cell.lengths() >= 15)
    if quasiharmonic:
        assert np.array_equal(result["temperatures"], temperatures[:4])
        assert result["free_energy"] == pytest.approx(
            np.full(4, 2 * phonons.kJmol2eV + atoms.get_potential_energy())
        )
        assert result["entropy"] == pytest.approx(
            np.full(4, 3 * phonons.kJmol2eV / 1000)
        )
        assert result["heat_capacity"].shape == (4,)
        kwargs = qha.call_args.kwargs
        assert kwargs["free_energy"].shape == (5, 3)
        assert np.all(np.diff(kwargs["volumes"]) > 0)
        assert relax.call_count == 4
    else:
        assert result["free_energy"] == pytest.approx(np.full(5, 2 * phonons.kJmol2eV))
        assert np.array_equal(result["entropy"], raw_thermal["entropy"])
        assert relax.call_count == 1
        qha.assert_not_called()


def test_quasiharmonic_requires_equilibrium_scale():
    atoms = bulk("Cu", cubic=True)
    atoms.calc = EMT()
    with pytest.raises(ValueError, match="scale_factors does not include 1"):
        phonons.calculate_vibrational_thermo(atoms, scale_factors=[0.8, 1.2])
