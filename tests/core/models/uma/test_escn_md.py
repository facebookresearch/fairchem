from __future__ import annotations

import pytest
import torch
from ase import Atoms
from ase.build import molecule as get_molecule
from e3nn.o3 import spherical_harmonics

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md import eSCNMDBackbone


@pytest.mark.parametrize(
    "atoms, orthogonal_direction",
    [
        (get_molecule("Be2"), torch.eye(3, dtype=torch.float32)[:2, :]),
        (
            Atoms(symbols="C4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
            torch.eye(3, dtype=torch.float32)[2:, :],
        ),
    ],
)
def test_escnmd_backbone_impossible_vectors(
    atoms: Atoms, orthogonal_direction: torch.Tensor
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
        "bijk,bj->bijk", l1[None, ...], sh1
    )  # n_ortho_directions, edges, 3, channels
    orthogonal = orthogonal.norm(dim=2)  # n_ortho_directions, edges, channels
    assert torch.allclose(
        orthogonal, torch.zeros_like(orthogonal), atol=1e-4
    ), "Orthogonal directions should be zero"
