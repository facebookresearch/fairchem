"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import pytest
import torch
from ase import Atoms
from ase.build import molecule as get_molecule
from e3nn.math import direct_sum
from e3nn.o3 import matrix_to_angles, spherical_harmonics, wigner_D

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_md import (
    MLP_EFS_Head,
    eSCNMDBackbone,
    resolve_dataset_mapping,
)


@pytest.mark.parametrize(
    "atoms, orthogonal_direction",
    [
        (get_molecule("Be2"), torch.eye(3, dtype=torch.float32)[:2, :]),
        (
            Atoms(symbols="C3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
            torch.eye(3, dtype=torch.float32)[:2, :],
        ),
    ],
)
def test_escnmd_backbone_impossible_vectors(
    atoms: Atoms, orthogonal_direction: torch.Tensor
) -> None:
    # TODO test could be improved by randomly rotating input and orthogonal directions
    torch.manual_seed(42)
    lmax = 2
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=lmax,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )
    g = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=25,
        radius=12,
        task_name="diatomic_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    out = backbone(g)

    # L=1 -> orthogonality as in R^3
    sh1 = spherical_harmonics(1, orthogonal_direction, normalize=True)
    l1 = out["node_embedding"][:, 1:4]
    orthogonal = torch.einsum(
        "...ijk,...j->...ijk", l1, sh1
    )  # n_ortho_directions, edges, 3, channels
    orthogonal = orthogonal.norm(dim=2)  # n_ortho_directions, edges, channels
    assert torch.allclose(
        orthogonal, torch.zeros_like(orthogonal), atol=1e-6
    ), "Orthogonal directions should be zero"


@pytest.mark.parametrize(
    "atoms, symmetry_matrix",
    [
        (
            Atoms(
                symbols="C4",
                positions=[
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.0],
                ],
            ),
            torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32),
        ),
    ],
)
def test_escnmd_backbone_symmetries(
    atoms: Atoms, symmetry_matrix: torch.Tensor
) -> None:
    torch.manual_seed(42)
    lmax = 2
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=lmax,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )
    g0 = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=25,
        radius=12,
        task_name="diatomic_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    out0 = backbone(g0)

    atoms1 = atoms.copy()
    atoms1.positions = atoms1.positions @ torch.linalg.inv(symmetry_matrix).T.numpy()
    g1 = AtomicData.from_ase(
        input_atoms=atoms1,
        max_neigh=25,
        radius=12,
        task_name="diatomic_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    out1 = backbone(g1)

    wigner_d = direct_sum(
        *[wigner_D(l, *matrix_to_angles(symmetry_matrix)) for l in range(lmax + 1)]
    )
    out1["node_embedding"] = torch.einsum(
        "aj,ijk->iak", wigner_d, out1["node_embedding"]
    )

    assert (
        (out0["node_embedding"] - out1["node_embedding"]).abs().max() < 5e-4
    ), f"For this molecule {atoms.positions=}, node embeddings should be invariant under this symmetry transformation {symmetry_matrix=}."  # high tolerance due to low precision


def test_resolve_dataset_mapping_valid_mapping():
    mapping = {"oc20": "oc20", "oc20_subset": "oc20"}
    result = resolve_dataset_mapping(deprecated_list=None, dataset_mapping=mapping)
    assert result == {"oc20": "oc20", "oc20_subset": "oc20"}


def test_resolve_dataset_mapping_deprecated_list(caplog):
    with caplog.at_level(logging.WARNING):
        result = resolve_dataset_mapping(
            deprecated_list=["omol", "omat"], dataset_mapping=None
        )
    assert result == {"omol": "omol", "omat": "omat"}
    assert "deprecated" in caplog.text.lower()


def test_resolve_dataset_mapping_both_raises():
    with pytest.raises(ValueError, match="Both"):
        resolve_dataset_mapping(
            deprecated_list=["oc20"], dataset_mapping={"oc20": "oc20"}
        )


def test_resolve_dataset_mapping_neither_raises():
    with pytest.raises(ValueError, match="dataset_mapping"):
        resolve_dataset_mapping(deprecated_list=None, dataset_mapping=None)


def test_resolve_dataset_mapping_empty_dict_raises():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_dataset_mapping(deprecated_list=None, dataset_mapping={})


def test_resolve_dataset_mapping_values_not_subset_raises():
    with pytest.raises(ValueError, match="subset"):
        resolve_dataset_mapping(
            deprecated_list=None,
            dataset_mapping={"oc20_subset1": "oc20", "oc20subset2": "oc20"},
        )


def test_resolve_dataset_mapping_custom_param_name(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_dataset_mapping(
            deprecated_list=["omol"],
            dataset_mapping=None,
            deprecated_param_name="dataset_names",
        )
    assert "dataset_names" in caplog.text


@pytest.mark.parametrize("hessian_vmap", [True, False])
def test_force_only_hessian_inference(hessian_vmap):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        backbone = (
            eSCNMDBackbone(
                max_num_elements=10,
                sphere_channels=4,
                lmax=1,
                mmax=1,
                hidden_channels=8,
                num_layers=1,
                edge_channels=4,
                num_distance_basis=6,
                use_dataset_embedding=False,
                always_use_pbc=False,
                otf_graph=True,
                direct_forces=False,
                regress_forces=True,
                regress_hessian=True,
                regress_stress=False,
            )
            .double()
            .eval()
        )
        head = MLP_EFS_Head(backbone, wrap_property=False).double().eval()
    backbone.regress_config.hessian_vmap = hessian_vmap
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.2, 0.1]], dtype=torch.float64)

    def predict(coordinates, hessian):
        backbone.regress_config.hessian = hessian
        atoms = Atoms("H2", positions=coordinates.numpy(), cell=[10.0] * 3, pbc=False)
        atoms.info.update(charge=0, spin=1)
        data = atomicdata_list_to_batch(
            [
                AtomicData.from_ase(
                    atoms,
                    task_name="omol",
                    r_data_keys=["charge", "spin"],
                    r_edges=False,
                )
            ]
        )
        data["pos"] = data["pos"].double()
        data["cell"] = data["cell"].double()
        with torch.no_grad():
            return head(data, backbone(data))

    prediction = predict(positions, hessian=True)
    hessian = prediction["hessian"].reshape(6, 6)
    force_only = predict(positions, hessian=False)
    torch.testing.assert_close(prediction["forces"], force_only["forces"])
    assert not force_only["forces"].requires_grad

    epsilon = 3e-3
    columns = []
    for index in range(positions.numel()):
        displacement = torch.zeros_like(positions)
        displacement.view(-1)[index] = epsilon
        positive = predict(positions + displacement, hessian=False)["forces"]
        negative = predict(positions - displacement, hessian=False)["forces"]
        columns.append(-(positive - negative).flatten() / (2 * epsilon))
    finite_difference = torch.stack(columns, dim=1)

    torch.testing.assert_close(hessian, finite_difference, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(hessian, hessian.T, rtol=1e-8, atol=1e-10)
    translations = torch.eye(3, dtype=torch.float64).repeat(2, 1)
    torch.testing.assert_close(
        hessian @ translations,
        torch.zeros(6, 3, dtype=torch.float64),
        rtol=0,
        atol=1e-9,
    )
