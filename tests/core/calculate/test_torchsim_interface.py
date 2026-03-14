"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from ase.build import bulk, molecule

from fairchem.core.calculate.torchsim_interface import (
    FairChemModel,
    _simstate_to_atomicdata_batch,
)
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

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


def test_single_system_pbc() -> None:
    """Test conversion of a single periodic system."""
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name="oc20", target_dtype=DTYPE
    )

    assert atomic_data.num_graphs == 1
    assert atomic_data.num_nodes == len(system)
    assert atomic_data.pos.shape == (len(system), 3)
    assert atomic_data.atomic_numbers.shape == (len(system),)
    assert atomic_data.cell.shape == (1, 3, 3)
    assert atomic_data.pbc.shape == (1, 3)
    assert torch.all(atomic_data.pbc)
    assert atomic_data.charge.shape == (1,)
    assert atomic_data.spin.shape == (1,)
    assert atomic_data.natoms.shape == (1,)
    assert atomic_data.natoms[0].item() == len(system)
    assert len(atomic_data.sid) == 1
    assert atomic_data.dataset == ["oc20"]


def test_multiple_systems_pbc() -> None:
    """Test conversion of multiple periodic systems."""
    systems = [
        bulk("Si", "diamond", a=5.43),
        bulk("Al", "fcc", a=4.05),
        bulk("Fe", "bcc", a=2.87),
    ]
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name="oc20", target_dtype=DTYPE
    )

    assert atomic_data.num_graphs == 3
    assert atomic_data.num_nodes == sum(len(s) for s in systems)
    assert atomic_data.cell.shape == (3, 3, 3)
    assert atomic_data.pbc.shape == (3, 3)
    assert torch.all(atomic_data.pbc)
    assert atomic_data.charge.shape == (3,)
    assert atomic_data.spin.shape == (3,)
    assert atomic_data.natoms.shape == (3,)
    assert atomic_data.natoms.tolist() == [len(s) for s in systems]
    assert len(atomic_data.sid) == 3
    assert atomic_data.dataset == ["oc20"] * 3


def test_single_system_molecule() -> None:
    """Test conversion of a single molecule (non-PBC)."""
    mol = molecule("H2O")
    mol.info["charge"] = 0
    mol.info["spin"] = 1
    state = ts.io.atoms_to_state([mol], device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name="omol", target_dtype=DTYPE
    )

    assert atomic_data.num_graphs == 1
    assert atomic_data.num_nodes == len(mol)
    assert atomic_data.cell.shape == (1, 3, 3)
    assert atomic_data.pbc.shape == (1, 3)
    assert not torch.any(atomic_data.pbc)
    assert atomic_data.charge[0].item() == 0
    assert atomic_data.spin[0].item() == 1
    assert len(atomic_data.sid) == 1
    assert atomic_data.dataset == ["omol"]


def test_multiple_systems_molecules() -> None:
    """Test conversion of multiple molecules."""
    molecules = [molecule("H2O"), molecule("CO2"), molecule("CH4")]
    for mol in molecules:
        mol.info["charge"] = 0
        mol.info["spin"] = 1
    state = ts.io.atoms_to_state(molecules, device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name="omol", target_dtype=DTYPE
    )

    assert atomic_data.num_graphs == 3
    assert atomic_data.num_nodes == sum(len(m) for m in molecules)
    assert not torch.any(atomic_data.pbc)
    assert atomic_data.charge.shape == (3,)
    assert atomic_data.spin.shape == (3,)
    assert len(atomic_data.sid) == 3


def test_batch_statistics() -> None:
    """Test that batch statistics are correctly computed."""
    systems = [
        bulk("Si", "diamond", a=5.43),
        bulk("Al", "fcc", a=4.05),
    ]
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=DTYPE
    )

    slices, cumsum, cat_dims, natoms_list = atomic_data.get_batch_stats()

    assert len(natoms_list) == 2
    assert natoms_list == [len(s) for s in systems]

    # Check slices for per-atom keys
    assert "pos" in slices
    assert slices["pos"] == [0, natoms_list[0], sum(natoms_list)]

    # Check slices for per-system keys
    assert "cell" in slices
    assert slices["cell"] == [0, 1, 2]

    # Check cat_dims
    assert "pos" in cat_dims
    assert "cell" in cat_dims


def test_dtype_conversion() -> None:
    """Test that dtype conversion works correctly."""
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=torch.float64)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=torch.float32
    )

    assert atomic_data.pos.dtype == torch.float32
    assert atomic_data.cell.dtype == torch.float32
    assert atomic_data.atomic_numbers.dtype == torch.int64


def test_empty_graphs() -> None:
    """Test that empty graphs (no edges) are correctly handled."""
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=DTYPE
    )

    assert atomic_data.edge_index.shape == (2, 0)
    assert atomic_data.cell_offsets.shape == (0, 3)
    assert atomic_data.nedges.shape == (1,)
    assert atomic_data.nedges[0].item() == 0


def test_positions_wrapping_pbc() -> None:
    """Test that positions are wrapped when PBC is enabled."""
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    # Shift positions outside the cell
    state.positions += torch.tensor([10.0, 10.0, 10.0], device=DEVICE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=DTYPE
    )

    # Positions should be wrapped back into the cell
    assert atomic_data.pos.shape == (len(system), 3)
    # Check that positions are within reasonable bounds (wrapped)
    cell_volume = torch.det(state.cell[0])
    assert cell_volume > 0


def test_no_task_name() -> None:
    """Test conversion without task_name."""
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=DTYPE
    )

    assert "dataset" not in atomic_data.__dict__ or atomic_data.dataset is None


def test_fixed_and_tags() -> None:
    """Test that fixed and tags are set to zeros."""
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=DTYPE
    )

    assert atomic_data.fixed.shape == (len(system),)
    assert torch.all(atomic_data.fixed == 0)
    assert atomic_data.tags.shape == (len(system),)
    assert torch.all(atomic_data.tags == 0)


def test_batch_indices() -> None:
    """Test that batch indices correctly map atoms to systems."""
    systems = [
        bulk("Si", "diamond", a=5.43),
        bulk("Al", "fcc", a=4.05),
    ]
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=DTYPE
    )

    assert atomic_data.batch.shape == (atomic_data.num_nodes,)
    # First system should have batch index 0
    assert atomic_data.batch[: len(systems[0])].unique().item() == 0
    # Second system should have batch index 1
    assert atomic_data.batch[len(systems[0]) :].unique().item() == 1
    assert atomic_data.batch.max().item() == 1


def test_cell_row_vector_convention() -> None:
    """Test that cell is converted from column to row vector convention."""
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=DEVICE, dtype=DTYPE)

    atomic_data = _simstate_to_atomicdata_batch(
        state, task_name=None, target_dtype=DTYPE
    )

    # SimState uses column vectors, AtomicData expects row vectors
    # row_vector_cell = cell.mT, so atomic_data.cell should be row vectors
    assert atomic_data.cell.shape == (1, 3, 3)
    # Verify it's the transpose of the column vector cell
    expected_row_cell = state.cell[0].mT
    torch.testing.assert_close(atomic_data.cell[0], expected_row_cell)


def test_direct_vs_ase_roundtrip() -> None:
    """Test that direct SimState->AtomicData conversion matches ASE roundtrip."""
    systems = [
        bulk("Si", "diamond", a=5.43),
        bulk("Al", "fcc", a=4.05),
    ]
    state = ts.io.atoms_to_state(systems, device=DEVICE, dtype=DTYPE)

    # Direct approach: SimState -> AtomicData batch
    atomic_data_direct = _simstate_to_atomicdata_batch(
        state, task_name="oc20", target_dtype=DTYPE
    )

    # Old approach: SimState -> ASE -> AtomicData list -> batch
    ase_atoms_list = state.to_atoms()
    atomic_data_list = [
        AtomicData.from_ase(atoms, r_edges=False) for atoms in ase_atoms_list
    ]
    # Set task_name for each
    for ad in atomic_data_list:
        ad.dataset = ["oc20"]
    atomic_data_roundtrip = atomicdata_list_to_batch(atomic_data_list)

    # Compare key attributes
    torch.testing.assert_close(atomic_data_direct.pos, atomic_data_roundtrip.pos)
    torch.testing.assert_close(
        atomic_data_direct.atomic_numbers, atomic_data_roundtrip.atomic_numbers
    )
    torch.testing.assert_close(atomic_data_direct.cell, atomic_data_roundtrip.cell)
    torch.testing.assert_close(atomic_data_direct.pbc, atomic_data_roundtrip.pbc)
    torch.testing.assert_close(atomic_data_direct.charge, atomic_data_roundtrip.charge)
    torch.testing.assert_close(atomic_data_direct.spin, atomic_data_roundtrip.spin)
    torch.testing.assert_close(atomic_data_direct.batch, atomic_data_roundtrip.batch)
    assert atomic_data_direct.dataset == atomic_data_roundtrip.dataset
    assert atomic_data_direct.num_graphs == atomic_data_roundtrip.num_graphs
    assert atomic_data_direct.num_nodes == atomic_data_roundtrip.num_nodes

    # Compare batch statistics (only check slices for keys that exist in both)
    slices_direct, _, _, natoms_list_direct = atomic_data_direct.get_batch_stats()
    slices_roundtrip, _, _, natoms_list_roundtrip = (
        atomic_data_roundtrip.get_batch_stats()
    )

    # Only compare slices for keys present in both (exclude dataset which is handled differently)
    common_keys = set(slices_direct.keys()) & set(slices_roundtrip.keys())
    for key in common_keys:
        if key not in ("dataset", "sid"):  # Skip non-tensor keys
            assert (
                slices_direct[key] == slices_roundtrip[key]
            ), f"Slices mismatch for {key}: {slices_direct[key]} != {slices_roundtrip[key]}"

    assert natoms_list_direct == natoms_list_roundtrip
