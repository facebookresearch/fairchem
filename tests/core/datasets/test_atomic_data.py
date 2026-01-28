"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import ase
import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)


@pytest.fixture()
def ase_atoms():
    return molecule("H2O")


@pytest.fixture()
def ase_bulk_atoms():
    """Create an ASE Atoms object with a unit cell and PBC."""
    atoms = bulk("Cu", "fcc", a=3.6)
    return atoms


@pytest.fixture()
def ase_atoms_with_calc():
    """Create an ASE Atoms object with a SinglePointCalculator."""
    atoms = molecule("H2O")
    atoms.center(vacuum=5.0)
    atoms.pbc = True
    calc = SinglePointCalculator(
        atoms,
        energy=-10.0,
        forces=np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [0.0, 0.0, 0.0]]),
        stress=np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03]),  # Voigt 6 format
    )
    atoms.calc = calc
    return atoms


@pytest.fixture()
def ase_atoms_with_constraints():
    """Create an ASE Atoms object with fixed atoms."""
    atoms = molecule("H2O")
    atoms.center(vacuum=5.0)
    atoms.pbc = True
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


@pytest.fixture()
def atomic_data_single(ase_bulk_atoms):
    """Create a single AtomicData object."""
    return AtomicData.from_ase(ase_bulk_atoms)


@pytest.fixture()
def atomic_data_batch(ase_bulk_atoms):
    """Create a batched AtomicData object with 3 graphs."""
    data_list = [AtomicData.from_ase(ase_bulk_atoms) for _ in range(3)]
    return atomicdata_list_to_batch(data_list)


def test_to_ase_single(ase_atoms):
    atoms = AtomicData.from_ase(ase_atoms).to_ase_single()
    assert atoms.get_chemical_formula() == "H2O"


@pytest.mark.gpu()
def test_to_ase_single_cuda(ase_atoms):
    atomic_data = AtomicData.from_ase(ase_atoms).cuda()
    atoms = atomic_data.to_ase_single()
    assert atoms.get_chemical_formula() == "H2O"


@pytest.fixture()
def batch_edgeless():
    # Create AtomicData batch of two ase.Atoms molecules without edges
    ase_atoms = ase.Atoms(positions=[[0.5, 0, 0], [1, 0, 0]], cell=(2, 2, 2), pbc=True)
    atomicdata_list_edgeless = [AtomicData.from_ase(ase_atoms) for _ in range(2)]
    batch_edgeless = atomicdata_list_to_batch(atomicdata_list_edgeless)
    return batch_edgeless


def test_to_ase_batch(batch_edgeless):
    # Define edge targets
    edge_index = torch.tensor([[1, 0, 3, 2], [0, 1, 2, 3]])
    cell_offsets = torch.zeros((4, 3))
    neighbors = torch.tensor([2, 2])
    # or equivalently:
    # edge_index, cell_offsets, neighbors = radius_graph_pbc_v2(
    #     batch_edgeless,
    #     radius=1,
    #     max_num_neighbors_threshold=100,
    #     pbc=batch_edgeless["pbc"][0],  # use the PBC from molecule 0
    # )

    # Add edge information to batch and check it is correct
    batch = batch_edgeless.clone()
    batch.update_batch_edges(edge_index, cell_offsets, neighbors)
    # or equivalently:
    # batch = batch_edgeless.update_batch_edges(edge_index, cell_offsets, neighbors)
    assert (batch.edge_index == edge_index).all()

    # Note: if we simply do `batch.edge_index = edge_index`, there will be no edges
    # after unbatching because `batch.__slices__` would contain only zeros.

    # Unbatch and check that edges have been added correctly
    atomicdata_list = batch.batch_to_atomicdata_list()
    assert (atomicdata_list[0].edge_index == edge_index[:, :2]).all()
    assert (atomicdata_list[1].edge_index == edge_index[:, :2]).all()


# =============================================================================
# Tests for AtomicData.from_ase
# =============================================================================


class TestAtomicDataFromAse:
    def test_basic_creation(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms)
        assert data.num_nodes == len(ase_bulk_atoms)
        assert data.num_graphs == 1

    def test_with_calculator(self, ase_atoms_with_calc):
        data = AtomicData.from_ase(ase_atoms_with_calc)
        assert hasattr(data, "energy")
        assert hasattr(data, "forces")
        assert hasattr(data, "stress")
        assert data.energy.shape == (1,)
        assert data.forces.shape == (3, 3)
        assert data.stress.shape == (1, 3, 3)

    def test_with_constraints(self, ase_atoms_with_constraints):
        data = AtomicData.from_ase(ase_atoms_with_constraints)
        assert data.fixed[0] == 1
        assert data.fixed[1] == 0
        assert data.fixed[2] == 0

    def test_with_molecule_cell_size(self, ase_atoms):
        data = AtomicData.from_ase(ase_atoms, molecule_cell_size=10.0)
        assert data.cell.shape == (1, 3, 3)
        assert torch.all(data.pbc)

    def test_with_sid(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, sid="test_id")
        assert data.sid == ["test_id"]

    def test_with_task_name(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, task_name="my_task")
        assert data.dataset == ["my_task"]

    def test_with_edges(self, ase_bulk_atoms):
        data = AtomicData.from_ase(
            ase_bulk_atoms, r_edges=True, radius=6.0, max_neigh=50
        )
        assert data.edge_index.shape[0] == 2
        assert data.nedges.item() == data.edge_index.shape[1]

    def test_without_edges(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, r_edges=False)
        assert data.edge_index.shape[1] == 0
        assert data.nedges.item() == 0

    def test_target_dtype_float64(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, target_dtype=torch.float64)
        assert data.pos.dtype == torch.float64
        assert data.cell.dtype == torch.float64

    def test_no_cell_no_pbc_allowed(self, ase_atoms):
        # Molecule without cell and without PBC should work
        data = AtomicData.from_ase(ase_atoms)
        assert data.num_nodes == 3

    def test_invalid_cell_raises(self):
        atoms = molecule("H2O")
        atoms.pbc = [True, False, False]  # Mixed PBC without cell
        with pytest.raises(ValueError, match="atoms must either have a cell"):
            AtomicData.from_ase(atoms)

    def test_info_overrides_calc(self, ase_atoms_with_calc):
        # Test that atoms.info overrides calculator results
        ase_atoms_with_calc.info["energy"] = -20.0
        data = AtomicData.from_ase(ase_atoms_with_calc)
        assert data.energy.item() == -20.0

    def test_stress_3x3_format(self, ase_bulk_atoms):
        atoms = ase_bulk_atoms.copy()
        atoms.center(vacuum=5.0)
        stress_3x3 = np.eye(3) * 0.1
        calc = SinglePointCalculator(atoms, stress=stress_3x3)
        atoms.calc = calc
        data = AtomicData.from_ase(atoms)
        assert data.stress.shape == (1, 3, 3)


# =============================================================================
# Tests for AtomicData properties
# =============================================================================


class TestAtomicDataProperties:
    def test_num_nodes(self, atomic_data_single):
        assert atomic_data_single.num_nodes == atomic_data_single.pos.shape[0]

    def test_num_edges(self, atomic_data_single):
        assert atomic_data_single.num_edges == atomic_data_single.edge_index.shape[1]

    def test_num_graphs_single(self, atomic_data_single):
        assert atomic_data_single.num_graphs == 1

    def test_num_graphs_batch(self, atomic_data_batch):
        assert atomic_data_batch.num_graphs == 3

    def test_len(self, atomic_data_batch):
        assert len(atomic_data_batch) == 3

    def test_task_name_property(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, task_name="test_task")
        assert data.task_name == ["test_task"]
        data.task_name = ["new_task"]
        assert data.dataset == ["new_task"]


# =============================================================================
# Tests for AtomicData dict-like interface
# =============================================================================


class TestAtomicDataDictInterface:
    def test_getitem_string(self, atomic_data_single):
        pos = atomic_data_single["pos"]
        assert torch.equal(pos, atomic_data_single.pos)

    def test_getitem_int(self, atomic_data_batch):
        example = atomic_data_batch[0]
        assert example.num_graphs == 1

    def test_getitem_negative_int(self, atomic_data_batch):
        example = atomic_data_batch[-1]
        assert example.num_graphs == 1

    def test_getitem_slice(self, atomic_data_batch):
        examples = atomic_data_batch[0:2]
        assert len(examples) == 2
        assert all(ex.num_graphs == 1 for ex in examples)

    def test_setitem(self, atomic_data_single):
        new_tensor = torch.ones(atomic_data_single.num_nodes)
        atomic_data_single["custom_field"] = new_tensor
        assert "custom_field" in atomic_data_single
        assert torch.equal(atomic_data_single["custom_field"], new_tensor)

    def test_delitem(self, atomic_data_single):
        atomic_data_single["temp_field"] = torch.zeros(1)
        assert "temp_field" in atomic_data_single
        del atomic_data_single["temp_field"]
        assert "temp_field" not in atomic_data_single

    def test_delitem_nonexistent_raises(self, atomic_data_single):
        with pytest.raises(AssertionError):
            del atomic_data_single["nonexistent_key"]

    def test_contains(self, atomic_data_single):
        assert "pos" in atomic_data_single
        assert "nonexistent" not in atomic_data_single

    def test_keys(self, atomic_data_single):
        keys = atomic_data_single.keys()
        assert "pos" in keys
        assert "atomic_numbers" in keys
        assert "cell" in keys

    def test_get_existing(self, atomic_data_single):
        result = atomic_data_single.get("pos", None)
        assert result is not None
        assert torch.equal(result, atomic_data_single.pos)

    def test_get_nonexistent(self, atomic_data_single):
        result = atomic_data_single.get("nonexistent", "default")
        assert result == "default"

    def test_iter(self, atomic_data_single):
        items = list(atomic_data_single)
        keys = [k for k, v in items]
        assert "pos" in keys

    def test_call(self, atomic_data_single):
        items = list(atomic_data_single("pos", "atomic_numbers"))
        keys = [k for k, v in items]
        assert keys == ["pos", "atomic_numbers"]


# =============================================================================
# Tests for AtomicData.from_dict and to_dict
# =============================================================================


class TestAtomicDataFromDict:
    def test_from_dict_basic(self, atomic_data_single):
        data_dict = atomic_data_single.to_dict()
        data_dict["sid"] = atomic_data_single.sid
        data_dict["batch"] = atomic_data_single.batch
        reconstructed = AtomicData.from_dict(data_dict)
        assert reconstructed.num_nodes == atomic_data_single.num_nodes

    def test_from_dict_missing_required_raises(self):
        incomplete_dict = {"pos": torch.zeros(3, 3)}
        with pytest.raises(AssertionError, match="Missing required keys"):
            AtomicData.from_dict(incomplete_dict)

    def test_from_dict_with_extra_keys(self, atomic_data_single):
        data_dict = atomic_data_single.to_dict()
        data_dict["sid"] = atomic_data_single.sid
        data_dict["batch"] = atomic_data_single.batch
        data_dict["custom_extra"] = torch.ones(5)
        reconstructed = AtomicData.from_dict(data_dict)
        assert "custom_extra" in reconstructed

    def test_to_dict(self, atomic_data_single):
        data_dict = atomic_data_single.to_dict()
        assert isinstance(data_dict, dict)
        assert "pos" in data_dict
        assert "atomic_numbers" in data_dict

    def test_values(self, atomic_data_single):
        values = atomic_data_single.values()
        assert isinstance(values, list)
        assert len(values) > 0


# =============================================================================
# Tests for AtomicData tensor operations
# =============================================================================


class TestAtomicDataTensorOps:
    def test_to_device(self, atomic_data_single):
        data = atomic_data_single.to("cpu")
        assert data.pos.device.type == "cpu"

    def test_cpu(self, atomic_data_single):
        data = atomic_data_single.cpu()
        assert data.pos.device.type == "cpu"

    @pytest.mark.gpu()
    def test_cuda(self, atomic_data_single):
        data = atomic_data_single.cuda()
        assert data.pos.device.type == "cuda"

    def test_contiguous(self, atomic_data_single):
        data = atomic_data_single.contiguous()
        assert data.pos.is_contiguous()

    def test_apply(self, atomic_data_single):
        original_dtype = atomic_data_single.pos.dtype
        data = atomic_data_single.apply(
            lambda x: x.double() if x.is_floating_point() else x
        )
        # Check that floating point tensors were converted
        if original_dtype == torch.float32:
            assert data.pos.dtype == torch.float64

    def test_clone(self, atomic_data_single):
        cloned = atomic_data_single.clone()
        assert cloned is not atomic_data_single
        assert torch.equal(cloned.pos, atomic_data_single.pos)
        # Verify it's a deep copy
        cloned.pos[0, 0] = 999.0
        assert atomic_data_single.pos[0, 0] != 999.0


# =============================================================================
# Tests for AtomicData.to_ase
# =============================================================================


class TestAtomicDataToAse:
    def test_to_ase_single_basic(self, atomic_data_single):
        atoms = atomic_data_single.to_ase_single()
        assert isinstance(atoms, ase.Atoms)
        assert len(atoms) == atomic_data_single.num_nodes

    def test_to_ase_single_with_targets(self, ase_atoms_with_calc):
        data = AtomicData.from_ase(ase_atoms_with_calc)
        atoms = data.to_ase_single()
        assert atoms.calc is not None
        assert "energy" in atoms.calc.results

    def test_to_ase_single_preserves_info(self, atomic_data_single):
        atoms = atomic_data_single.to_ase_single()
        assert "charge" in atoms.info
        assert "spin" in atoms.info

    def test_to_ase_single_preserves_sid(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, sid="my_sid")
        atoms = data.to_ase_single()
        assert atoms.info["sid"] == ["my_sid"]

    def test_to_ase_batch(self, atomic_data_batch):
        atoms_list = atomic_data_batch.to_ase()
        assert len(atoms_list) == 3
        assert all(isinstance(a, ase.Atoms) for a in atoms_list)

    def test_to_ase_single_batch_raises(self, atomic_data_batch):
        with pytest.raises(AssertionError, match="single graph"):
            atomic_data_batch.to_ase_single()


# =============================================================================
# Tests for AtomicData batch operations
# =============================================================================


class TestAtomicDataBatchOps:
    def test_get_example(self, atomic_data_batch):
        example = atomic_data_batch.get_example(0)
        assert example.num_graphs == 1

    def test_get_example_negative_index(self, atomic_data_batch):
        example = atomic_data_batch.get_example(-1)
        assert example.num_graphs == 1

    def test_get_example_single_graph_returns_self(self, atomic_data_single):
        example = atomic_data_single.get_example(0)
        assert example is atomic_data_single

    def test_index_select_slice(self, atomic_data_batch):
        selected = atomic_data_batch.index_select(slice(0, 2))
        assert len(selected) == 2

    def test_index_select_tensor(self, atomic_data_batch):
        idx = torch.tensor([0, 2])
        selected = atomic_data_batch.index_select(idx)
        assert len(selected) == 2

    def test_index_select_bool_tensor(self, atomic_data_batch):
        idx = torch.tensor([True, False, True])
        selected = atomic_data_batch.index_select(idx)
        assert len(selected) == 2

    def test_index_select_list(self, atomic_data_batch):
        selected = atomic_data_batch.index_select([0, 1])
        assert len(selected) == 2

    def test_index_select_invalid_type_raises(self, atomic_data_batch):
        with pytest.raises(IndexError):
            atomic_data_batch.index_select("invalid")

    def test_batch_to_atomicdata_list(self, atomic_data_batch):
        data_list = atomic_data_batch.batch_to_atomicdata_list()
        assert len(data_list) == 3
        assert all(d.num_graphs == 1 for d in data_list)

    def test_assign_and_get_batch_stats(self, atomic_data_single):
        slices = {"pos": [0, 1]}
        cumsum = {"pos": [0, 0]}
        cat_dims = {"pos": 0}
        natoms_list = [1]
        atomic_data_single.assign_batch_stats(slices, cumsum, cat_dims, natoms_list)
        retrieved = atomic_data_single.get_batch_stats()
        assert retrieved[0] == slices
        assert retrieved[1] == cumsum
        assert retrieved[2] == cat_dims
        assert retrieved[3] == natoms_list


# =============================================================================
# Tests for atomicdata_list_to_batch
# =============================================================================


class TestAtomicDataListToBatch:
    def test_basic_batching(self, ase_bulk_atoms):
        data_list = [AtomicData.from_ase(ase_bulk_atoms) for _ in range(3)]
        batch = atomicdata_list_to_batch(data_list)
        assert batch.num_graphs == 3
        assert batch.num_nodes == 3 * len(ase_bulk_atoms)

    def test_batch_preserves_sids(self, ase_bulk_atoms):
        data_list = [
            AtomicData.from_ase(ase_bulk_atoms, sid=f"sid_{i}") for i in range(3)
        ]
        batch = atomicdata_list_to_batch(data_list)
        assert batch.sid == ["sid_0", "sid_1", "sid_2"]

    def test_batch_from_already_batched_data(self, atomic_data_batch):
        # This tests the re-batching scenario
        data_list = atomic_data_batch.batch_to_atomicdata_list()
        rebatched = atomicdata_list_to_batch(data_list)
        assert rebatched.num_graphs == 3


# =============================================================================
# Tests for AtomicData __cat_dim__ and __inc__
# =============================================================================


class TestAtomicDataCatDimAndInc:
    def test_cat_dim_index_key(self, atomic_data_single):
        result = atomic_data_single.__cat_dim__("edge_index", None)
        assert result == -1

    def test_cat_dim_regular_key(self, atomic_data_single):
        result = atomic_data_single.__cat_dim__("pos", None)
        assert result == 0

    def test_inc_index_key(self, atomic_data_single):
        result = atomic_data_single.__inc__("edge_index", None)
        assert result == atomic_data_single.natoms.item()

    def test_inc_regular_key(self, atomic_data_single):
        result = atomic_data_single.__inc__("pos", None)
        assert result == 0


# =============================================================================
# Tests for AtomicData validation
# =============================================================================


class TestAtomicDataValidation:
    def test_validate_called_on_init(self, ase_bulk_atoms):
        # Should not raise - validation happens in __init__
        data = AtomicData.from_ase(ase_bulk_atoms)
        assert data is not None

    def test_invalid_dataset_type_raises(self, ase_bulk_atoms):
        with pytest.raises(ValueError, match="dataset must be a string or list"):
            atoms = ase_bulk_atoms.copy()
            atoms.center(vacuum=5.0)
            data = AtomicData.from_ase(atoms)
            # Directly construct with invalid dataset type
            AtomicData(
                pos=data.pos,
                atomic_numbers=data.atomic_numbers,
                cell=data.cell,
                pbc=data.pbc,
                natoms=data.natoms,
                edge_index=data.edge_index,
                cell_offsets=data.cell_offsets,
                nedges=data.nedges,
                charge=data.charge,
                spin=data.spin,
                fixed=data.fixed,
                tags=data.tags,
                dataset=123,  # Invalid type
            )


# =============================================================================
# Tests for update_batch_edges
# =============================================================================


class TestUpdateBatchEdges:
    def test_update_batch_edges(self, atomic_data_batch):
        new_edge_index = torch.tensor([[0, 1], [1, 0]])
        new_cell_offsets = torch.zeros(2, 3)
        new_nedges = torch.tensor([1, 1, 0])

        result = atomic_data_batch.update_batch_edges(
            new_edge_index, new_cell_offsets, new_nedges
        )

        assert torch.equal(result.edge_index, new_edge_index)
        assert torch.equal(result.cell_offsets, new_cell_offsets)
        assert torch.equal(result.nedges, new_nedges)


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestAtomicDataEdgeCases:
    def test_sid_as_string(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, sid="single_string")
        assert data.sid == ["single_string"]

    def test_sid_as_list(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, sid=["list_sid"])
        assert data.sid == ["list_sid"]

    def test_sid_none(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, sid=None)
        assert data.sid == [""]

    def test_dataset_as_string_normalized_to_list(self, ase_bulk_atoms):
        data = AtomicData.from_ase(ase_bulk_atoms, task_name="single_task")
        assert isinstance(data.dataset, list)
        assert data.dataset == ["single_task"]

    def test_clone_preserves_batch_stats(self, atomic_data_batch):
        cloned = atomic_data_batch.clone()
        original_stats = atomic_data_batch.get_batch_stats()
        cloned_stats = cloned.get_batch_stats()
        assert original_stats[0] == cloned_stats[0]  # slices
        assert original_stats[3] == cloned_stats[3]  # natoms_list
