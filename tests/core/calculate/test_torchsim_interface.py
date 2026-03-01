"""Tests for FairChem torchsim integration.

Tests the FairChemModel wrapper that provides a torch-sim compatible interface
for FairChem models to compute energies, forces, and stresses.

To run just these tests with uv(from the repo root):
    uv run --project packages/fairchem-core --extra torchsim --extra dev --extra extras -- pytest tests/core/calculate/test_torchsim_interface.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from ase.build import bulk, molecule

from fairchem.core.calculate.torchsim_interface import FairChemModel

if TYPE_CHECKING:
    from collections.abc import Callable

pytest.importorskip(
    "torch_sim",
    reason="torch_sim not installed. Install with: pip install fairchem-core[torchsim]",
)

import torch_sim as ts  # noqa: E402
from torch_sim.models.interface import validate_model_outputs  # noqa: E402

DTYPE = torch.float32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture()
def torchsim_model_oc20(direct_checkpoint) -> FairChemModel:
    """Model for materials (periodic boundary conditions) using locally-trained checkpoint.

    Note: The checkpoint is trained on oc20_omol tasks, so it supports both:
    - oc20 task (PBC - surfaces/catalysis)
    - omol task (non-PBC - molecules)
    """
    checkpoint_path, _ = direct_checkpoint
    return FairChemModel(model=checkpoint_path, task_name="oc20", device=DEVICE)


@pytest.fixture()
def torchsim_model_omol(direct_checkpoint) -> FairChemModel:
    """Model for molecules (non-PBC) using locally-trained checkpoint.

    Note: The checkpoint is trained on oc20_omol tasks, so it supports both:
    - oc20 task (PBC - surfaces/catalysis)
    - omol task (non-PBC - molecules)
    """
    checkpoint_path, _ = direct_checkpoint
    return FairChemModel(model=checkpoint_path, task_name="omol", device=DEVICE)


@pytest.mark.parametrize("task_name", ["oc20", "omol"])
def test_task_initialization(direct_checkpoint, task_name: str) -> None:
    """Test that different task names initialize correctly."""
    checkpoint_path, _ = direct_checkpoint
    model = FairChemModel(
        model=checkpoint_path, task_name=task_name, device=torch.device("cpu")
    )
    assert model.task_name
    assert str(model.task_name.value) == task_name
    assert hasattr(model, "predictor")


@pytest.mark.parametrize(
    ("task_name", "systems_func"),
    [
        (
            "oc20",
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
def test_homogeneous_batching(
    direct_checkpoint, task_name: str, systems_func: Callable
) -> None:
    """Test batching multiple systems with the same task."""
    systems = systems_func()
    checkpoint_path, _ = direct_checkpoint

    if task_name == "omol":
        for mol in systems:
            mol.info |= {"charge": 0, "spin": 1}

    model = FairChemModel(model=checkpoint_path, task_name=task_name, device=DEVICE)
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
    results = model(state)

    assert results["energy"].shape == (4,)
    assert results["forces"].shape[0] == sum(len(s) for s in systems)
    assert results["forces"].shape[1] == 3

    energies = results["energy"]
    uniq_energies = torch.unique(energies, dim=0)
    assert len(uniq_energies) > 1, "Different systems should have different energies"


def test_heterogeneous_tasks(direct_checkpoint) -> None:
    """Test different task types work with appropriate systems."""
    checkpoint_path, _ = direct_checkpoint
    test_cases = [
        ("omol", [molecule("H2O")]),
        ("oc20", [bulk("Pt", cubic=True)]),
    ]

    for task_name, systems in test_cases:
        if task_name == "omol":
            systems[0].info |= {"charge": 0, "spin": 1}

        model = FairChemModel(
            model=checkpoint_path,
            task_name=task_name,
            device=DEVICE,
        )
        state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
        results = model(state)

        assert results["energy"].shape[0] == 1
        assert results["forces"].dim() == 2
        assert results["forces"].shape[1] == 3


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
def test_batch_size_variations(
    direct_checkpoint, systems_func: Callable, expected_count: int
) -> None:
    """Test batching with different numbers and sizes of systems."""
    systems = systems_func()
    checkpoint_path, _ = direct_checkpoint

    model = FairChemModel(model=checkpoint_path, task_name="oc20", device=DEVICE)
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
    results = model(state)

    assert results["energy"].shape == (expected_count,)
    assert results["forces"].shape[0] == sum(len(s) for s in systems)
    assert results["forces"].shape[1] == 3
    assert torch.isfinite(results["energy"]).all()
    assert torch.isfinite(results["forces"]).all()


@pytest.mark.parametrize("compute_stress", [True, False])
def test_stress_computation(
    conserving_mole_checkpoint, *, compute_stress: bool
) -> None:
    """Test stress tensor computation using a conservative (non-direct-force) model."""
    systems = [bulk("Si", "diamond", a=5.43), bulk("Al", "fcc", a=4.05)]
    checkpoint_path, _ = conserving_mole_checkpoint

    model = FairChemModel(
        model=checkpoint_path,
        task_name="oc20",
        device=DEVICE,
        compute_stress=compute_stress,
    )
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)
    results = model(state)

    assert "energy" in results
    assert "forces" in results
    if compute_stress:
        assert "stress" in results
        assert results["stress"].shape == (2, 3, 3)
        assert torch.isfinite(results["stress"]).all()
    else:
        assert "stress" not in results


def test_device_consistency(direct_checkpoint) -> None:
    """Test device consistency between model and data."""
    checkpoint_path, _ = direct_checkpoint
    model = FairChemModel(model=checkpoint_path, task_name="oc20", device=DEVICE)
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    results = model(state)
    assert results["energy"].device == DEVICE
    assert results["forces"].device == DEVICE


def test_empty_batch_error(direct_checkpoint) -> None:
    """Test that empty batches raise appropriate errors."""
    checkpoint_path, _ = direct_checkpoint
    model = FairChemModel(
        model=checkpoint_path, task_name="oc20", device=torch.device("cpu")
    )
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        model(ts.io.atoms_to_state([], device=torch.device("cpu"), dtype=torch.float32))


def test_load_from_checkpoint_path(direct_checkpoint) -> None:
    """Test loading model from a saved checkpoint file path."""
    checkpoint_path, _ = direct_checkpoint
    loaded_model = FairChemModel(model=checkpoint_path, task_name="oc20", device=DEVICE)

    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)
    results = loaded_model(state)

    assert "energy" in results
    assert "forces" in results
    assert results["energy"].shape == (1,)
    assert torch.isfinite(results["energy"]).all()
    assert torch.isfinite(results["forces"]).all()


@pytest.mark.parametrize(
    ("charge", "spin"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 0.0),
        (0.0, 2.0),
    ],
)
def test_charge_spin_handling(direct_checkpoint, charge: float, spin: float) -> None:
    """Test that FairChemModel correctly handles charge and spin from atoms.info."""
    mol = molecule("H2O")
    mol.info["charge"] = charge
    mol.info["spin"] = spin

    state = ts.io.atoms_to_state([mol], device=DEVICE, dtype=DTYPE)

    assert state.charge[0].item() == charge
    assert state.spin[0].item() == spin

    checkpoint_path, _ = direct_checkpoint
    model = FairChemModel(
        model=checkpoint_path,
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


def test_model_output_validation(torchsim_model_oc20: FairChemModel) -> None:
    """Test that the model implementation follows the ModelInterface contract."""
    validate_model_outputs(torchsim_model_oc20, DEVICE, DTYPE)


def test_model_output_validation_with_stress(conserving_mole_checkpoint) -> None:
    """Test ModelInterface contract for a conservative model that predicts stresses."""
    checkpoint_path, _ = conserving_mole_checkpoint
    model = FairChemModel(
        model=checkpoint_path, task_name="oc20", device=DEVICE, compute_stress=True
    )
    validate_model_outputs(model, DEVICE, DTYPE)


def test_missing_torchsim_raises_import_error(monkeypatch) -> None:
    """Test that FairChemModel raises ImportError when torch-sim is not installed."""
    # Mock the module-level variables to simulate torch-sim not being installed
    import fairchem.core.calculate.torchsim_interface as torchsim_module

    # Save original values
    original_ts = torchsim_module.ts
    original_model_interface = torchsim_module.ModelInterface

    # Set to None to simulate missing torch-sim
    monkeypatch.setattr(torchsim_module, "ts", None)
    monkeypatch.setattr(torchsim_module, "ModelInterface", None)

    # Now try to instantiate - should raise ImportError
    with pytest.raises(
        ImportError, match="torch-sim is required to use FairChemModel.*Install it with"
    ):
        FairChemModel(model="dummy", task_name="oc20")

    # Restore original values (monkeypatch will do this automatically, but being explicit)
    monkeypatch.setattr(torchsim_module, "ts", original_ts)
    monkeypatch.setattr(torchsim_module, "ModelInterface", original_model_interface)


def test_invalid_model_path_raises_error() -> None:
    """Test that FairChemModel raises ValueError for invalid model path."""
    with pytest.raises(ValueError, match="Invalid model name or checkpoint path"):
        FairChemModel(model="/nonexistent/path/to/checkpoint.pt", task_name="oc20")


def test_invalid_task_name_raises_error(direct_checkpoint) -> None:
    """Test that FairChemModel raises error for invalid task name."""
    checkpoint_path, _ = direct_checkpoint
    with pytest.raises((ValueError, KeyError)):
        FairChemModel(model=checkpoint_path, task_name="invalid_task")


def test_custom_neighbor_list_raises_error(direct_checkpoint) -> None:
    """Test that FairChemModel raises NotImplementedError for custom neighbor list."""
    checkpoint_path, _ = direct_checkpoint
    with pytest.raises(
        NotImplementedError, match="Custom neighbor list is not supported"
    ):
        FairChemModel(
            model=checkpoint_path,
            task_name="oc20",
            neighbor_list_fn=lambda x: x,
        )
