from __future__ import annotations

import pytest
import torch
from ase import Atoms
from ase.build import molecule as get_molecule
from e3nn.math import direct_sum
from e3nn.o3 import matrix_to_angles, spherical_harmonics, wigner_D

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md import eSCNMDBackbone


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
        (out0["node_embedding"] - out1["node_embedding"]).abs().max()
        < 5e-4  # high tolerance due to low precision
    ), f"For this molecule {atoms.positions=}, node embeddings should be invariant under this symmetry transformation {symmetry_matrix=}."
