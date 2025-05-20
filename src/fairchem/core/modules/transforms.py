"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from ase import Atoms

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.utils.irreps import cg_change_mat, irreps_sum


def _get_molecule_cell(data_object: AtomicData):
    # create an Atoms object and center molecule in cell
    mol = Atoms(
        numbers=data_object.atomic_numbers,
        positions=data_object.pos,
    )
    # largest radius cutoff is ~12A include that and a safety factor of 10
    mol.center(vacuum=(12.0 * 10.0))
    mol.pbc = [True, True, True]

    positions = np.array(mol.get_positions(), copy=True)
    # pbc = np.array(mol.pbc, copy=True)
    cell = np.array(mol.get_cell(complete=True), copy=True)

    atomic_numbers = torch.Tensor(mol.get_atomic_numbers())
    positions = torch.from_numpy(positions).float()
    cell = torch.from_numpy(cell).view(1, 3, 3).float()
    natoms = positions.shape[0]
    assert data_object.natoms == natoms

    return atomic_numbers, positions, cell


def common_transform(data_object: AtomicData, config) -> AtomicData:
    data_object.dataset = config["dataset_name"]

    if not hasattr(data_object, "charge"):
        data_object.charge = 0
    if not hasattr(data_object, "spin"):
        data_object.spin = 0
    ensure_tensor(data_object, "energy")
    return data_object


def ensure_tensor(data_object, keys):
    # ensure dataset_energy is a tensor
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if hasattr(data_object, key):
            if not torch.is_tensor(getattr(data_object, key)):
                setattr(
                    data_object,
                    key,
                    torch.tensor(getattr(data_object, key), dtype=torch.float),
                )
            setattr(data_object, key, getattr(data_object, key).view(-1).float())
    return data_object


def ani1x_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    # ensure ani1x_energy is a tensor
    return data_object


def trans1x_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed and cell
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    # ensure trans1x_energy is a tensor

    return data_object


def spice_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)
    # this is necessary for SPICE maceoff split to work with GemNet-OC
    data_object.tags = torch.full(data_object.tags.shape, 2, dtype=torch.long)

    # ensure spice_energy is a tensor

    return data_object


def qmof_transform(data_object: AtomicData, config) -> AtomicData:
    # add fixed and cell
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    return data_object


def qm9_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    return data_object


def omol_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell
    assert hasattr(
        data_object, "charge"
    ), "no charge in omol dataset set a2g_args: {r_energy: True, r_forces: True, r_data_keys: ['spin', 'charge']}"
    assert hasattr(
        data_object, "spin"
    ), "no spin in omol dataset set a2g_args: {r_energy: True, r_forces: True, r_data_keys: ['spin', 'charge']}"

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    return common_transform(data_object, config)


def stress_reshape_transform(data_object: AtomicData, config) -> AtomicData:
    for k in data_object.keys():  # noqa: SIM118
        if "stress" in k and ("iso" not in k and "aniso" not in k):
            data_object[k] = data_object[k].reshape(1, 9)
    return data_object


def asedb_transform(data_object: AtomicData, config) -> AtomicData:
    data_object.dataset = config["dataset_name"]
    data_object.sid = str(
        data_object.sid.item() if torch.is_tensor(data_object) else data_object.sid
    )
    return data_object


class DataTransforms:
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, data_object):
        if not self.config:
            return data_object

        for transform_fn in self.config:
            # TODO: Normalization information used in the trainers. Ignore here for now
            # TODO: if we dont use them here, these should not be defined as "transforms" in the config
            # TODO: add them as another entry under dataset, maybe "standardize"?
            if transform_fn in ("normalizer", "element_references"):
                continue

            data_object = eval(transform_fn)(data_object, self.config[transform_fn])

        return data_object


def decompose_tensor(data_object, config) -> AtomicData:
    tensor_key = config["tensor"]
    rank = config["rank"]

    if tensor_key not in data_object:
        return data_object

    if rank != 2:
        raise NotImplementedError

    tensor_decomposition = torch.einsum(
        "ab, cb->ca",
        cg_change_mat(rank),
        data_object[tensor_key].reshape(1, irreps_sum(rank)),
    )

    for decomposition_key in config["decomposition"]:
        irrep_dim = config["decomposition"][decomposition_key]["irrep_dim"]
        data_object[decomposition_key] = tensor_decomposition[
            :,
            max(0, irreps_sum(irrep_dim - 1)) : irreps_sum(irrep_dim),
        ]

    return data_object
