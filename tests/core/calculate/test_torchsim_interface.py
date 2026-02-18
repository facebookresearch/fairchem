"""Tests for FairChem torchsim integration.

Tests the FairChemModel wrapper that provides a torch-sim compatible interface
for FairChem models to compute energies, forces, and stresses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from ase.build import bulk, fcc100, molecule
from huggingface_hub.utils._auth import get_token

from fairchem.core.calculate.pretrained_mlip import (
    pretrained_checkpoint_path_from_name,
)
from fairchem.core.calculate.torchsim_interface import FairChemModel

if TYPE_CHECKING:
    from collections.abc import Callable

pytest.importorskip(
    "torch_sim",
    reason="torch_sim not installed. Install torch-sim-atomistic separately if needed.",
)

import torch_sim as ts  # noqa: E402
from torch_sim.models.interface import validate_model_outputs  # noqa: E402

DTYPE = torch.float32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture()
def uma_model_omat() -> FairChemModel:
    """UMA model for materials (periodic boundary conditions)."""
    return FairChemModel(model="uma-s-1", task_name="omat", device=DEVICE)


@pytest.fixture()
def uma_model_omol() -> FairChemModel:
    """UMA model for molecules."""
    return FairChemModel(model="uma-s-1", task_name="omol", device=DEVICE)


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
@pytest.mark.parametrize("task_name", ["omat", "omol", "oc20"])
def test_task_initialization(task_name: str) -> None:
    """Test that different UMA task names initialize correctly."""
    model = FairChemModel(
        model="uma-s-1", task_name=task_name, device=torch.device("cpu")
    )
    assert model.task_name
    assert str(model.task_name.value) == task_name
    assert hasattr(model, "predictor")


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
@pytest.mark.parametrize(
    ("task_name", "systems_func"),
    [
        (
            "omat",
            lambda: [
                bulk("Si", "diamond", a=5.43),
                bulk("Al", "fcc", a=4.05),
                bulk("Fe", "bcc", a=2.87),
                bulk("Cu", "fcc", a=3.61),
            ],
        ),
        (
            "omol",
            lambda: [
                molecule("H2O"),
                molecule("CO2"),
                molecule("CH4"),
                molecule("NH3"),
            ],
        ),
    ],
)
def test_homogeneous_batching(task_name: str, systems_func: Callable) -> None:
    """Test batching multiple systems with the same task."""
    systems = systems_func()

    if task_name == "omol":
        for mol in systems:
            mol.info |= {"charge": 0, "spin": 1}

    model = FairChemModel(model="uma-s-1", task_name=task_name, device=DEVICE)
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
    results = model(state)

    assert results["energy"].shape == (4,)
    assert results["forces"].shape[0] == sum(len(s) for s in systems)
    assert results["forces"].shape[1] == 3

    energies = results["energy"]
    uniq_energies = torch.unique(energies, dim=0)
    assert len(uniq_energies) > 1, "Different systems should have different energies"


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
def test_heterogeneous_tasks() -> None:
    """Test different task types work with appropriate systems."""
    test_cases = [
        ("omol", [molecule("H2O")]),
        ("omat", [bulk("Pt", cubic=True)]),
        ("oc20", [fcc100("Cu", (2, 2, 3), vacuum=8, periodic=True)]),
    ]

    for task_name, systems in test_cases:
        if task_name == "omol":
            systems[0].info |= {"charge": 0, "spin": 1}

        model = FairChemModel(
            model="uma-s-1",
            task_name=task_name,
            device=DEVICE,
        )
        state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
        results = model(state)

        assert results["energy"].shape[0] == 1
        assert results["forces"].dim() == 2
        assert results["forces"].shape[1] == 3


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
@pytest.mark.parametrize(
    ("systems_func", "expected_count"),
    [
        (lambda: [bulk("Si", "diamond", a=5.43)], 1),
        (
            lambda: [
                bulk("H", "bcc", a=2.0),
                bulk("Li", "bcc", a=3.0),
                bulk("Si", "diamond", a=5.43),
                bulk("Al", "fcc", a=4.05).repeat((2, 1, 1)),
            ],
            4,
        ),
        (
            lambda: [
                bulk(element, "fcc", a=4.0)
                for element in ("Al", "Cu", "Ni", "Pd", "Pt") * 3
            ],
            15,
        ),
    ],
)
def test_batch_size_variations(systems_func: Callable, expected_count: int) -> None:
    """Test batching with different numbers and sizes of systems."""
    systems = systems_func()

    model = FairChemModel(model="uma-s-1", task_name="omat", device=DEVICE)
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
    results = model(state)

    assert results["energy"].shape == (expected_count,)
    assert results["forces"].shape[0] == sum(len(s) for s in systems)
    assert results["forces"].shape[1] == 3
    assert torch.isfinite(results["energy"]).all()
    assert torch.isfinite(results["forces"]).all()


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
@pytest.mark.parametrize("compute_stress", [True, False])
def test_stress_computation(*, compute_stress: bool) -> None:
    """Test stress tensor computation."""
    systems = [bulk("Si", "diamond", a=5.43), bulk("Al", "fcc", a=4.05)]

    model = FairChemModel(
        model="uma-s-1",
        task_name="omat",
        device=DEVICE,
        compute_stress=compute_stress,
    )
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
    results = model(state)

    if compute_stress:
        assert "stress" in results
        assert results["stress"].shape == (2, 3, 3)
        assert torch.isfinite(results["stress"]).all()
    else:
        assert "stress" not in results


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
def test_device_consistency() -> None:
    """Test device consistency between model and data."""
    model = FairChemModel(model="uma-s-1", task_name="omat", device=DEVICE)
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    results = model(state)
    assert results["energy"].device == DEVICE
    assert results["forces"].device == DEVICE


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
def test_empty_batch_error() -> None:
    """Test that empty batches raise appropriate errors."""
    model = FairChemModel(model="uma-s-1", task_name="omat", device=torch.device("cpu"))
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        model(ts.io.atoms_to_state([], device=torch.device("cpu"), dtype=torch.float32))


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
def test_load_from_checkpoint_path() -> None:
    """Test loading model from a saved checkpoint file path."""
    checkpoint_path = pretrained_checkpoint_path_from_name("uma-s-1")
    loaded_model = FairChemModel(
        model=str(checkpoint_path), task_name="omat", device=DEVICE
    )

    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)
    results = loaded_model(state)

    assert "energy" in results
    assert "forces" in results
    assert results["energy"].shape == (1,)
    assert torch.isfinite(results["energy"]).all()
    assert torch.isfinite(results["forces"]).all()


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
@pytest.mark.parametrize(
    ("charge", "spin"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 0.0),
        (0.0, 2.0),
    ],
)
def test_charge_spin_handling(charge: float, spin: float) -> None:
    """Test that FairChemModel correctly handles charge and spin from atoms.info."""
    mol = molecule("H2O")
    mol.info["charge"] = charge
    mol.info["spin"] = spin

    state = ts.io.atoms_to_state([mol], device=DEVICE, dtype=DTYPE)

    assert state.charge[0].item() == charge
    assert state.spin[0].item() == spin

    model = FairChemModel(
        model="uma-s-1",
        task_name="omol",
        device=DEVICE,
    )

    result = model(state)

    assert "energy" in result
    assert result["energy"].shape == (1,)
    assert "forces" in result
    assert result["forces"].shape == (len(mol), 3)
    assert torch.isfinite(result["energy"]).all()
    assert torch.isfinite(result["forces"]).all()


@pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)
def test_model_output_validation(uma_model_omat: FairChemModel) -> None:
    """Test that the model implementation follows the ModelInterface contract."""
    validate_model_outputs(uma_model_omat, DEVICE, DTYPE)
