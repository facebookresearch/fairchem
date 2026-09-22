"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for the ASE DFT-D3(BJ) calculator wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import numpy.testing as npt
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

import fairchem.core.calculate.dftd3_calculator as dftd3
from fairchem.core import DFTD3Calculator


class _ConstantCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "stress",
    ]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 10.0,
            "free_energy": 10.0,
            "forces": np.full((len(self.atoms), 3), 2.0),
            "stress": np.full(6, 0.5),
        }


@pytest.fixture()
def fake_nvalchemi(monkeypatch):
    calls = SimpleNamespace(model_kwargs=None, neighbors=0)

    class FakeModel:
        def __init__(self, **kwargs):
            calls.model_kwargs = kwargs
            self.model_config = SimpleNamespace(
                neighbor_config=SimpleNamespace(cutoff=kwargs["cutoff"])
            )

        def to(self, device):
            self.device = device
            return self

        def set_config(self, key, value):
            assert key == "active_outputs"
            self.active_outputs = value

        def eval(self):
            return self

        def __call__(self, batch):
            natoms = len(batch.data.atoms)
            output = {
                "energy": torch.tensor([[-1.0]]),
                "forces": torch.full((natoms, 3), 0.25),
            }
            if "stress" in self.active_outputs:
                output["stress"] = torch.tensor(
                    [[[1.0, 0.1, 0.2], [0.3, 2.0, 0.4], [0.5, 0.6, 3.0]]]
                )
            return output

    class FakeAtomicData:
        @staticmethod
        def from_atoms(atoms, *, device, dtype):
            assert device == torch.device("cpu")
            assert dtype == torch.float32
            return SimpleNamespace(atoms=atoms)

    class FakeBatch:
        @staticmethod
        def from_data_list(data, *, device, skip_validation):
            assert device == torch.device("cpu")
            assert skip_validation
            return SimpleNamespace(data=data[0])

    def compute_neighbors(batch, *, config):
        calls.neighbors += 1
        batch._neighbor_list_cutoff = config.cutoff

    monkeypatch.setattr(
        dftd3,
        "_import_nvalchemi",
        lambda: (FakeModel, FakeAtomicData, FakeBatch, compute_neighbors),
    )
    return calls


@pytest.mark.parametrize(
    ("functional", "expected"),
    [
        (
            "r2scan",
            {"a1": 0.49484001, "a2": 5.73083694, "s6": 1.0, "s8": 0.78981345},
        ),
        ("pbe", {"a1": 0.4289, "a2": 4.4407, "s6": 1.0, "s8": 0.7875}),
    ],
)
def test_named_parameters_and_nvalchemi_configuration(
    fake_nvalchemi, functional, expected
):
    calc = DFTD3Calculator(
        _ConstantCalculator(),
        functional=functional,
        device="cpu",
        param_file="parameters.pt",
        auto_download=False,
    )

    assert calc.functional == functional
    assert dftd3.asdict(calc.damping_parameters) == expected
    assert fake_nvalchemi.model_kwargs == {
        **expected,
        "k1": 16.0,
        "k3": -4.0,
        "cutoff": 15.0,
        "smoothing_fraction": 0.2,
        "param_file": "parameters.pt",
        "auto_download": False,
    }


def test_adds_energy_forces_and_symmetric_voigt_stress(fake_nvalchemi):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
        cell=[30.0, 30.0, 30.0],
        pbc=True,
    )
    atoms.calc = DFTD3Calculator(_ConstantCalculator(), functional="pbe", device="cpu")

    assert atoms.get_potential_energy() == pytest.approx(9.0)
    assert atoms.calc.get_property("free_energy", atoms) == pytest.approx(9.0)
    npt.assert_allclose(atoms.get_forces(), 2.25)
    npt.assert_allclose(
        atoms.get_stress(),
        [1.5, 2.5, 3.5, 1.0, 0.85, 0.7],
    )
    assert fake_nvalchemi.neighbors == 1


def test_rebuilds_neighbor_list_after_position_and_cell_changes(fake_nvalchemi):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.calc = DFTD3Calculator(
        _ConstantCalculator(), functional="r2scan", device="cpu"
    )

    atoms.get_potential_energy()
    atoms.positions[1, 0] += 0.01
    atoms.get_potential_energy()
    atoms.cell[0, 0] += 0.01
    atoms.get_potential_energy()

    assert fake_nvalchemi.neighbors == 3


def test_nonperiodic_system_provides_energy_and_forces_without_stress(
    fake_nvalchemi,
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    atoms.calc = DFTD3Calculator(_ConstantCalculator(), functional="pbe", device="cpu")

    assert atoms.get_potential_energy() == pytest.approx(9.0)
    npt.assert_allclose(atoms.get_forces(), 2.25)
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()

    assert fake_nvalchemi.neighbors == 2


def test_rejects_unknown_functional_before_import(monkeypatch):
    monkeypatch.setattr(
        dftd3,
        "_import_nvalchemi",
        lambda: pytest.fail("nvalchemi import should not be attempted"),
    )
    with pytest.raises(ValueError, match="Unknown DFT-D3 functional"):
        DFTD3Calculator(_ConstantCalculator(), functional="b3lyp", device="cpu")
