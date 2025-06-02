"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

modified from troch_geometric Data class
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from typing import List, Optional, Union

import ase
import ase.db.sqlite
import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator
from ase.constraints import FixAtoms
from ase.geometry import wrap_positions
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from pymatgen.io.ase import AseAtomsAdaptor

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]

# these are all currently certainly output by the current a2g
# except for tags, all fields are required for network inference.
_REQUIRED_KEYS = [
    "pos",
    "atomic_numbers",
    "cell",
    "pbc",
    "natoms",
    "edge_index",
    "cell_offsets",
    "nedges",
    "charge",
    "spin",
    "fixed",
    "tags",
]

_OPTIONAL_KEYS = ["energy", "forces", "stress", "dataset"]

# TODO: potential future keys
# ["virials", "atom_attr", "edge_attr"]


def size_repr(key: str, item: torch.Tensor, indent=0) -> str:
    indent_str = " " * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = item.item()
    elif torch.is_tensor(item):
        out = str(list(item.size()))
    elif isinstance(item, (List, tuple)):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + indent_str + "}"
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)

    return f"{indent_str}{key}={out}"


def get_neighbors_pymatgen(atoms: ase.Atoms, cutoff, max_neigh):
    """Preforms nearest neighbor search and returns edge index, distances,
    and cell offsets"""
    if AseAtomsAdaptor is None:
        raise RuntimeError(
            "Unable to import pymatgen.io.ase.AseAtomsAdaptor. Make sure pymatgen is properly installed."
        )

    struct = AseAtomsAdaptor.get_structure(atoms)

    # tol of 1e-8 should remove all self loops
    _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
        r=cutoff, numerical_tol=1e-8, exclude_self=True
    )
    _nonmax_idx = []
    for i in range(len(atoms)):
        idx_i = (_c_index == i).nonzero()[0]
        # sort neighbors by distance, remove edges larger than max_neighbors
        idx_sorted = np.argsort(n_distance[idx_i])[:max_neigh]
        _nonmax_idx.append(idx_i[idx_sorted])
    _nonmax_idx = np.concatenate(_nonmax_idx)

    _c_index = _c_index[_nonmax_idx]
    _n_index = _n_index[_nonmax_idx]
    n_distance = n_distance[_nonmax_idx]
    _offsets = _offsets[_nonmax_idx]

    return _c_index, _n_index, n_distance, _offsets


def reshape_features(
    c_index: np.ndarray,
    n_index: np.ndarray,
    n_distance: np.ndarray,
    offsets: np.ndarray,
):
    """Stack center and neighbor index and reshapes distances,
    takes in np.arrays and returns torch tensors"""
    edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
    edge_distances = torch.FloatTensor(n_distance)
    cell_offsets = torch.FloatTensor(offsets)

    # remove distances smaller than a tolerance ~ 0. The small tolerance is
    # needed to correct for pymatgen's neighbor_list returning self atoms
    # in a few edge cases.
    nonzero = torch.where(edge_distances >= 1e-8)[0]
    edge_index = edge_index[:, nonzero]
    cell_offsets = cell_offsets[nonzero]

    return edge_index, cell_offsets


class AtomicData:
    def __init__(
        self,
        pos: torch.Tensor,  # (num_node, 3)
        atomic_numbers: torch.Tensor,  # (num_node,)
        cell: torch.Tensor,  # (num_graph, 3, 3)
        pbc: torch.Tensor,  # (num_graph, 3)
        natoms: torch.Tensor,  # (1,)
        edge_index: torch.Tensor,  # (2, num_edge)
        cell_offsets: torch.Tensor,  # (num_edge, 3)
        nedges: torch.Tensor,  # (1,)
        charge: torch.Tensor,  # (num_graph,)
        spin: torch.Tensor,  # (num_graph,)
        fixed: torch.Tensor,  # (num_node,)
        tags: torch.Tensor,  # (num_node,)
        energy: torch.Tensor | None = None,  # (num_graph,)
        forces: torch.Tensor | None = None,  # (num_node, 3)
        stress: torch.Tensor | None = None,  # (num_graph, 3, 3)
        batch: torch.Tensor | None = None,  # (num_node,)
        sid: list[str] | None = None,
        dataset: list[str] | str | None = None,
    ):
        self.__keys__ = set(_REQUIRED_KEYS)

        # this conversion must have been done somewhere in
        # pytorch geoemtric data
        self.pos = pos.to(torch.float32)
        self.atomic_numbers = atomic_numbers
        self.cell = cell.to(self.pos.dtype)
        self.pbc = pbc
        self.natoms = natoms
        self.edge_index = edge_index
        self.cell_offsets = cell_offsets
        self.nedges = nedges
        self.charge = charge
        self.spin = spin
        self.fixed = fixed
        self.tags = tags
        self.sid = sid if sid is not None else [""]

        if dataset is not None:
            self.dataset = dataset

        # tagets
        if energy is not None:
            self.energy = energy
        if forces is not None:
            self.forces = forces
        if stress is not None:
            self.stress = stress

        # batch related
        if batch is not None:
            self.batch = batch
        else:
            self.batch = torch.zeros_like(self.atomic_numbers)

        # id
        if isinstance(sid, str):
            self.sid = [sid]
        elif isinstance(sid, list):
            self.sid = sid
        else:
            self.sid = [""]

        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__natoms_list__ = None

        # self.custom_fields = {}

        self.validate()

    @property
    def task_name(self):
        return getattr(self, "dataset", None)

    @task_name.setter
    def task_name(self, value):
        self.dataset = value

    def assign_batch_stats(self, slices, cumsum, cat_dims, natoms_list):
        self.__slices__ = slices
        self.__cumsum__ = cumsum
        self.__cat_dims__ = cat_dims
        self.__natoms_list__ = natoms_list

    def get_batch_stats(self):
        return self.__slices__, self.__cumsum__, self.__cat_dims__, self.__natoms_list__

    def validate(self):
        # shape checks
        assert (
            self.pos.shape[0]
            == self.atomic_numbers.shape[0]
            == self.num_nodes
            == self.natoms.sum().item()
        )
        assert self.pos.dim() == 2
        assert self.pos.shape[1] == 3
        assert self.atomic_numbers.dim() == 1
        assert self.cell.shape[0] == self.pbc.shape[0] == self.num_graphs
        assert self.cell.dim() == 3
        assert self.cell.shape[1:] == (3, 3)
        assert self.pbc.dim() == 2
        assert self.pbc.shape[1] == 3
        assert self.edge_index.shape[0] == 2
        assert self.cell_offsets.shape[0] == self.edge_index.shape[1]
        assert self.nedges.sum().item() == self.edge_index.shape[1]
        assert self.fixed.shape[0] == self.pos.shape[0]
        assert self.tags.shape[0] == self.pos.shape[0]
        assert self.batch.shape == self.atomic_numbers.shape
        assert int(self.batch.max()) + 1 == self.num_graphs
        assert len(self.sid) == self.num_graphs

        if "dataset" in self.__keys__:
            if isinstance(self.dataset, list):
                assert len(self.dataset) == self.num_graphs
            else:
                assert isinstance(self.dataset, str)
                assert self.num_graphs == 1

        # dtype checks
        assert (
            self.pos.dtype == self.cell.dtype == self.cell_offsets.dtype == torch.float
        ), "Positions, cell, cell_offsets are all expected to be float32. Check data going into AtomicData is correct dtype"
        assert self.atomic_numbers.dtype == torch.long
        assert self.edge_index.dtype == torch.long
        assert self.pbc.dtype == torch.bool
        assert self.fixed.dtype == self.tags.dtype == torch.long
        assert self.batch.dtype == torch.long
        assert isinstance(self.num_nodes, int)
        assert isinstance(self.num_graphs, int)

        # NOTE maybe do re matching instead of exact key matching for energy/forces/stress
        if hasattr(self, "energy"):
            assert self.energy.dim() == 1
            assert self.energy.shape[0] == self.num_graphs
            assert self.energy.dtype == torch.float
        if hasattr(self, "forces"):
            assert self.forces.shape[0] == self.pos.shape[0]
            assert self.forces.shape[1] == 3
            assert self.forces.dtype == torch.float
        if hasattr(self, "stress"):
            # NOTE: usually decomposed. for EFS prediction right now we reshape to (9,). need to discuss, perhaps use (1,3,3)
            assert (
                self.stress.dim() == 3
                and self.stress.shape[1:] == (3, 3)
                or (self.stress.dim() == 2 and self.stress.shape[1:] == (9,))
            )
            assert self.stress.shape[0] == self.num_graphs
            assert self.stress.dtype == torch.float

        if self.sid is not None:
            assert isinstance(self.sid, list)
            assert all(isinstance(s, str) for s in self.sid)
            assert len(self.sid) == self.num_graphs

        # for k, v in self.custom_fields.items():
        #     assert self[k].shape == v['shape']
        #     assert self[k].dtype == v['dtype']

    @classmethod
    def from_ase(
        cls,
        input_atoms: ase.Atoms,
        r_edges: bool = False,
        radius: float = 6.0,
        max_neigh: int | None = None,
        sid: str | None = None,
        molecule_cell_size: float | None = None,
        r_energy: bool = True,  # deprecated
        r_forces: bool = True,
        r_stress: bool = True,
        r_data_keys: list[str] | None = None,  # NOT USED, compat for now
        task_name: str | None = None,
    ) -> AtomicData:
        atoms = input_atoms.copy()
        calc = input_atoms.calc
        # TODO: maybe compute a safe cell size if not provided.
        if molecule_cell_size is not None:
            assert (
                atoms.cell.volume == 0.0
            ), "atoms must not have a unit cell to begin with to create a molecule cell"
            # create a molecule box with the molecule centered on it if specified
            atoms.center(vacuum=(molecule_cell_size))
            atoms.pbc = np.array([True, True, True])
        elif np.all(~atoms.pbc):
            pass  # This is fine
        elif not np.all(atoms.pbc) and atoms.cell.volume < 0.1:
            raise ValueError(
                "atoms must either have a cell or have a cell created by setting <molecule_cell_size>."
            )

        atomic_numbers = np.array(atoms.get_atomic_numbers(), copy=True)
        pos = np.array(atoms.get_positions(), copy=True)
        pbc = np.array(atoms.pbc, copy=True)
        cell = np.array(atoms.get_cell(complete=True), copy=True)
        pos = wrap_positions(pos, cell, pbc=pbc, eps=0)

        # wrap positions for CPU graph Generation
        atoms.set_positions(pos)

        atomic_numbers = torch.from_numpy(atomic_numbers).long()
        pos = torch.from_numpy(pos).float()
        pbc = torch.from_numpy(pbc).bool().view(1, 3)
        cell = torch.from_numpy(cell).float().view(1, 3, 3)
        natoms = torch.tensor([pos.shape[0]], dtype=torch.long)

        # graph construction
        if r_edges:
            assert (
                radius is not None
            ), "cutoff must be specified for cpu graph construction."
            assert (
                max_neigh is not None
            ), "max_neigh must be specified for cpu graph construction."
            split_idx_dist = get_neighbors_pymatgen(atoms, radius, max_neigh)
            edge_index, cell_offsets = reshape_features(*split_idx_dist)
            nedges = torch.tensor([edge_index.shape[1]], dtype=torch.long)
        else:
            # empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            cell_offsets = torch.empty((0, 3), dtype=torch.float)
            nedges = torch.tensor([0], dtype=torch.long)

        # initialized to torch.zeros(natoms) if tags missing.
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_tags
        tags = torch.from_numpy(atoms.get_tags()).long()
        fixed = torch.zeros(natoms, dtype=torch.long)
        if hasattr(atoms, "constraints"):
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    fixed[constraint.index] = 1

        if isinstance(calc, (SinglePointCalculator, SinglePointDFTCalculator)):
            results = calc.results
            energy = (
                torch.FloatTensor([results["energy"]]).view(1)
                if "energy" in results
                else None
            )
            forces = (
                torch.FloatTensor(results["forces"]).view(-1, 3)
                if "forces" in results
                else None
            )
            stress = results.get("stress", None)
            if stress is not None and r_stress:
                if stress.shape == (6,):
                    stress = torch.FloatTensor(voigt_6_to_full_3x3_stress(stress)).view(
                        1, 3, 3
                    )
                elif stress.shape in ((3, 3), (9,)):
                    stress = torch.FloatTensor(stress).view(1, 3, 3)
                else:
                    raise ValueError(f"Unknown stress shape, {stress.shape}")
            else:
                stress = None
        else:
            energy = None
            forces = None
            stress = None

        energy = (
            torch.FloatTensor([atoms.info["energy"]])
            if "energy" in atoms.info
            else energy
        )
        forces = (
            torch.FloatTensor(atoms.info["forces"])
            if "forces" in atoms.info
            else forces
        )
        stress = (
            torch.FloatTensor(atoms.info["stress"]).view(1, 3, 3)
            if "stress" in atoms.info
            else stress
        )

        # TODO another way to specify this is to spcify a key. maybe total_charge
        charge = torch.LongTensor(
            [
                atoms.info.get("charge", 0)
                if r_data_keys is not None and "charge" in r_data_keys
                else 0
            ]
        )
        spin = torch.LongTensor(
            [
                atoms.info.get("spin", 0)
                if r_data_keys is not None and "spin" in r_data_keys
                else 0
            ]
        )

        # NOTE: code assumes these are ints.. not tensors
        # charge = atoms.info.get("charge", 0)
        # spin = atoms.info.get("spin", 0)
        data = cls(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=tensor_or_int_to_tensor(charge, torch.long),
            spin=tensor_or_int_to_tensor(spin, torch.long),
            fixed=fixed,
            tags=tags,
            energy=energy,
            forces=forces,
            stress=stress,
            sid=[sid] if isinstance(sid, str) else sid,
            dataset=task_name,
        )

        return data

    def to_ase_single(self) -> ase.Atoms:
        assert self.num_graphs == 1, "Data object must contain a single graph."

        atoms = ase.Atoms(
            numbers=self.atomic_numbers.numpy(),
            positions=self.pos.numpy(),
            cell=self.cell.squeeze().numpy(),
            pbc=self.pbc.squeeze().tolist(),
            constraint=FixAtoms(mask=self.fixed.tolist()),
            tags=self.tags.numpy(),
        )

        if self.__keys__.intersection(["energy", "forces", "stress"]):
            fields = {}
            if self.energy is not None:
                fields["energy"] = self.energy.numpy()
            if self.forces is not None:
                fields["forces"] = self.forces.numpy()
            if self.stress is not None:
                if self.stress.shape == (3, 3):
                    fields["stress"] = full_3x3_to_voigt_6_stress(
                        self.stress.squeeze().numpy()
                    )
                elif self.stress.shape == (6,):
                    fields["stress"] = self.stress.squeeze().numpy()
            atoms.calc = SinglePointCalculator(atoms=atoms, **fields)

        atoms.info = {
            "charge": self.charge.item(),
            "spin": self.spin.item(),
        }

        if self.sid is not None:
            atoms.info["sid"] = self.sid

        return atoms

    def to_ase(self) -> list[ase.Atoms]:
        return [self.get_example(i).to_ase_single() for i in range(self.num_graphs)]

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        assert set(_REQUIRED_KEYS).issubset(
            dictionary.keys()
        ), f"Missing required keys: {set(_REQUIRED_KEYS) - set(dictionary.keys())}"

        data = cls(
            pos=dictionary["pos"],
            atomic_numbers=dictionary["atomic_numbers"],
            cell=dictionary["cell"],
            pbc=dictionary["pbc"],
            natoms=dictionary["natoms"],
            edge_index=dictionary["edge_index"],
            cell_offsets=dictionary["cell_offsets"],
            nedges=dictionary["nedges"],
            charge=dictionary["charge"],
            spin=dictionary["spin"],
            fixed=dictionary.get("fixed", None),
            tags=dictionary.get("tags", None),
            energy=dictionary.get("energy", None),
            forces=dictionary.get("forces", None),
            stress=dictionary.get("stress", None),
            batch=dictionary.get("batch", None),
            sid=dictionary.get("sid", None),
            dataset=dictionary.get("dataset", None),
        )

        # TODO: may require validation for them in the future
        # assign extra keys in dict...
        for key in set(dictionary.keys()) - set(_REQUIRED_KEYS + _OPTIONAL_KEYS):
            data[key] = dictionary[key]

        return data

    # TODO clean
    def to_dict(self):
        return dict(self)

    # TODO clean
    def values(self):
        return [item for _, item in self]

    ###############################
    #       basic operations      #
    ###############################

    @property
    def num_nodes(self) -> int:
        r"""Returns or sets the number of nodes in the graph."""
        return self.pos.size(0)

    @property
    def num_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return self["edge_index"].size(1)

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        return int(self.batch.max()) + 1

    def __len__(self):
        return self.num_graphs

    def get(self, key, default):
        if key in self:
            return self[key]
        return default

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return getattr(self, idx)
        elif isinstance(idx, (int, np.integer)):
            return self.get_example(idx)
        else:
            return self.index_select(idx)

    # TODO: maybe should not allow this. this is currently used a lot though.
    def __setitem__(self, key: str, value: torch.Tensor):
        # TODO call add custom field? if validation checks are given?
        """Sets the attribute :obj:`key` to :obj:`value`."""
        # renaming keys is not supported if this function validate keys.
        # assert key in self.__keys__ or key in set(_REQUIRED_KEYS + _OPTIONAL_KEYS + ['batch']), f"Invalid key: {key}"
        setattr(self, key, value)
        if key not in self.__keys__:
            self.__keys__.add(key)
        # self.validate()

    # TODO: maybe should not allow this. this is currently used a lot though.
    def __setattr__(self, key: str, value: torch.Tensor):
        super().__setattr__(key, value)
        if not key.startswith("__") and key not in self.__keys__:
            self.__keys__.add(key)

    # TODO: maybe should not allow this. this is currently used a lot though.
    def __delitem__(self, key: str):
        """Deletes the attribute :obj:`key`."""
        assert key in self.__keys__, f"Key: <{key}> not found."
        delattr(self, key)
        self.__keys__.remove(key)

    def keys(self):
        return self.__keys__

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.__keys__

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.__keys__):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in keys if keys else sorted(self.__keys__):
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key, value) -> int:
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if bool(re.search("index", key)):
            return -1
        return 0

    def __inc__(self, key, value) -> int:
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only the `*index*` attribute should be cumulatively summed
        # up when creating batches.
        assert self.num_graphs == 1
        return self.natoms.item() if bool(re.search("index", key)) else 0

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func):
        r"""Applies the function :obj:`func` to all tensor attributes"""
        for key in self.__keys__:
            self[key] = self.__apply__(self[key], func)

        self["batch"] = self.__apply__(self["batch"], func)

        return self

    def contiguous(self):
        r"""Ensures a contiguous memory layout for all tensor attributes"""
        return self.apply(lambda x: x.contiguous())

    def to(self, device, **kwargs):
        r"""Performs tensor dtype and/or device conversion for all tensor attributes"""
        return self.apply(lambda x: x.to(device, **kwargs))

    def cpu(self):
        r"""Copies all tensor attributes to CPU memory."""
        return self.apply(lambda x: x.cpu())

    def cuda(self, device=None, non_blocking=False):
        r"""Copies all tensor attributes to GPU memory."""
        return self.apply(lambda x: x.cuda(device=device, non_blocking=non_blocking))

    def clone(self):
        r"""Performs a deep-copy of the data object."""
        data_dict = {}
        for key in self.__keys__:
            if torch.is_tensor(self[key]):
                data_dict[key] = self[key].clone()
            else:
                # TODO: with this we should stop making sid special.
                data_dict[key] = copy.deepcopy(self[key])
                # ""
                # print(key)
                # raise ValueError("keys must correspond to torch tensors.")
        data_dict["sid"] = copy.deepcopy(self.sid)
        data_dict["batch"] = self.batch.clone()
        batch_stats = copy.deepcopy(self.get_batch_stats())
        cloned = AtomicData.from_dict(data_dict)
        cloned.assign_batch_stats(*batch_stats)

        return cloned

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any(isinstance(item, dict) for _, item in self)

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))

    ###############################
    # operations related to batch #
    ###############################

    def get_example(self, idx: int) -> AtomicData:
        r"""
        Reconstructs the :class:`AtomicData` object at index
        :obj:`idx` from a batched AtomicData object.
        """
        if self.num_graphs == 1 and (idx == 0 or idx == -1):
            return self

        data_dict = {}
        idx = self.num_graphs + idx if idx < 0 else idx
        for key in self.__slices__:
            item = self[key]
            if self.__cat_dims__[key] is None:
                # The item was concatenated along a new batch dimension,
                # so just index in that dimension:
                item = item[idx]
            else:
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, torch.Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item.narrow(dim, start, end - start)
                else:
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item[start:end]
                    item = item[0] if len(item) == 1 else item

            # Decrease its value by `cumsum` value:
            cum = self.__cumsum__[key][idx]
            if isinstance(item, torch.Tensor):
                if not isinstance(cum, int) or cum != 0:
                    item = item - cum
            elif isinstance(item, (int, float)):
                item = item - cum

            data_dict[key] = item

        data_dict["batch"] = torch.zeros_like(data_dict["atomic_numbers"])
        data_dict["sid"] = [self.sid[idx]]

        if hasattr(self, "dataset"):
            data_dict["dataset"] = self.dataset[idx]

        return AtomicData.from_dict(data_dict)

    def index_select(self, idx: IndexType) -> list[AtomicData]:
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
            idx = idx.flatten().tolist()

        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass

        else:
            raise IndexError(
                f"Only integers, slices (':'), list, tuples, torch.tensor"
                f"are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        return [self.get_example(i) for i in idx]

    def batch_to_atomicdata_list(self) -> list[AtomicData]:
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""
        return [self.get_example(i) for i in range(self.num_graphs)]


def atomicdata_list_to_batch(
    data_list: list[AtomicData], exclude_keys: Optional[list] = None
) -> AtomicData:
    """
    all data points must be single graphs and have the same set of keys.
    TODO: exclude keys?
    """
    # this does not include the sid/fids
    # arbitrary set of keys handled at set_item?
    if exclude_keys is None:
        exclude_keys = []
    keys = list(set(data_list[0].keys()))

    batched_data_dict = {k: [] for k in keys}
    batch = []

    device = None
    slices = {key: [0] for key in keys}
    cumsum = {key: [0] for key in keys}
    cat_dims = {}
    natoms_list, sid_list = [], []

    for i, data in enumerate(data_list):
        assert (
            data.num_graphs == 1
        ), "data list must only contain single-graph AtomicData objects."

        for key in keys:
            item = data[key]

            # Increase values by `cumsum` value.
            cum = cumsum[key][-1]
            if isinstance(item, torch.Tensor) and item.dtype != torch.bool:
                if not isinstance(cum, int) or cum != 0:
                    item = item + cum
            elif isinstance(item, (int, float)):
                item = item + cum

            # Gather the size of the `cat` dimension.
            size = 1
            cat_dim = data.__cat_dim__(key, data[key])
            # 0-dimensional torch.Tensors have no dimension along which to
            # concatenate, so we set `cat_dim` to `None`.
            if isinstance(item, torch.Tensor) and item.dim() == 0:
                cat_dim = None
            cat_dims[key] = cat_dim

            # Add a batch dimension to items whose `cat_dim` is `None`:
            if isinstance(item, torch.Tensor) and cat_dim is None:
                cat_dim = 0  # Concatenate along this new batch dimension.
                item = item.unsqueeze(0)
                device = item.device
            elif isinstance(item, torch.Tensor):
                size = item.size(cat_dim)
                device = item.device

            batched_data_dict[key].append(item)  # Append item to the attribute list.

            slices[key].append(size + slices[key][-1])
            inc = data.__inc__(key, item)
            if isinstance(inc, (tuple, list)):
                inc = torch.tensor(inc)
            cumsum[key].append(inc + cumsum[key][-1])

        natoms_list.append(data.natoms.item())
        sid_list.extend(data.sid)
        item = torch.full((data.natoms,), i, dtype=torch.long, device=device)
        batch.append(item)

    ref_data = data_list[0]
    for key in keys:
        items = batched_data_dict[key]
        item = items[0]
        cat_dim = ref_data.__cat_dim__(key, item)
        cat_dim = 0 if cat_dim is None else cat_dim
        if torch.is_tensor(item):
            batched_data_dict[key] = torch.cat(items, cat_dim)

        # TODO: this allows non-tensor fields to be batched.
        # we might want to remove support for that.
        else:
            batched_data_dict[key] = items

    batched_data_dict["batch"] = torch.cat(batch, dim=-1)
    batched_data_dict["sid"] = sid_list
    atomic_data_batch = AtomicData.from_dict(batched_data_dict)
    atomic_data_batch.assign_batch_stats(slices, cumsum, cat_dims, natoms_list)

    return atomic_data_batch.contiguous()


def tensor_or_int_to_tensor(x, dtype=torch.int):
    if isinstance(x, int):
        return torch.Tensor(x, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        assert x.dtype == dtype, "Tensor is not of right dtype"
        return x
    raise ValueError(f"type({x}) is not an int or tensor")
