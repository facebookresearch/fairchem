"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.geometry.geometry import general_find_mic

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.graph.geometry import (
    compute_minkowski_op,
    minimum_image_displacement,
)

_ORTHO_CELL = torch.diag(
    torch.tensor([10.0, 12.0, 8.0], dtype=torch.float64)
).unsqueeze(0)
_TRICLINIC_CELL = torch.tensor(
    [[[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [0.0, 0.0, 8.0]]], dtype=torch.float64
)
_SKEWED_CELL = torch.tensor(
    [[[6.0, 0.0, 0.0], [5.5, 3.0, 0.0], [2.0, 1.0, 7.0]]], dtype=torch.float64
)


def _assert_matches_general_find_mic(
    dr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    minkowski_op: torch.Tensor | None = None,
    atol: float = 1e-10,
) -> None:
    result = minimum_image_displacement(dr, cell, pbc, batch, minkowski_op=minkowski_op)
    pbc_per_atom = pbc[batch]
    cell_per_atom = cell[batch]

    for i in range(dr.shape[0]):
        expected, _ = general_find_mic(
            dr[i].numpy()[None],
            cell_per_atom[i].numpy(),
            pbc=pbc_per_atom[i].numpy(),
        )
        np.testing.assert_allclose(
            result[i].numpy(),
            expected[0],
            atol=atol,
            err_msg=f"atom {i}: got {result[i].numpy()}, expected {expected[0]}",
        )


# ── Correctness against ASE general_find_mic ──


@pytest.mark.parametrize(
    "cell, pbc, dr",
    [
        pytest.param(
            _ORTHO_CELL,
            torch.tensor([[True, False, True]]),
            torch.tensor([[6.0, 7.0, 5.0], [-7.0, 3.0, 0.0]], dtype=torch.float64),
            id="orthorhombic-partial-pbc",
        ),
        pytest.param(
            _SKEWED_CELL,
            torch.tensor([[True, True, True]]),
            torch.tensor(
                (torch.manual_seed(42), torch.randn(5, 3, dtype=torch.float64) * 5)[1]
            ),
            id="highly-skewed-triclinic",
        ),
    ],
)
def test_minimum_image_matches_general_find_mic(cell, pbc, dr):
    batch = torch.zeros(dr.shape[0], dtype=torch.long)
    _assert_matches_general_find_mic(dr, cell, pbc, batch)


# ── Precomputed minkowski_op ──


def test_precomputed_op_matches_on_the_fly():
    batch = torch.zeros(4, dtype=torch.long)
    pbc = torch.tensor([[True, True, True]])
    torch.manual_seed(7)
    dr = torch.randn(4, 3, dtype=torch.float64) * 3

    result_fly = minimum_image_displacement(dr, _TRICLINIC_CELL, pbc, batch)
    op = compute_minkowski_op(_TRICLINIC_CELL, pbc)
    result_pre = minimum_image_displacement(
        dr, _TRICLINIC_CELL, pbc, batch, minkowski_op=op
    )

    torch.testing.assert_close(result_pre, result_fly)


def test_precomputed_op_matches_general_find_mic():
    batch = torch.zeros(3, dtype=torch.long)
    pbc = torch.tensor([[True, True, True]])
    torch.manual_seed(99)
    dr = torch.randn(3, 3, dtype=torch.float64) * 4
    op = compute_minkowski_op(_SKEWED_CELL, pbc)

    _assert_matches_general_find_mic(dr, _SKEWED_CELL, pbc, batch, minkowski_op=op)


# ── Multi-system batch ──


def test_mixed_cells_in_batch():
    cell = torch.cat([_ORTHO_CELL, _TRICLINIC_CELL], dim=0)
    pbc = torch.tensor([[True, True, True], [True, True, True]])
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    dr = torch.tensor(
        [[6.0, 0.0, 0.0], [0.5, 0.5, 0.5], [3.0, 2.0, 1.0], [1.0, -1.0, 0.0]],
        dtype=torch.float64,
    )
    _assert_matches_general_find_mic(dr, cell, pbc, batch)


# ── Collation ──


def _make_sample(n_atoms, cell, pbc):
    return AtomicData.from_dict(
        {
            "pos": torch.randn(n_atoms, 3),
            "atomic_numbers": torch.ones(n_atoms, dtype=torch.long),
            "cell": cell.view(1, 3, 3),
            "pbc": pbc.view(1, 3),
            "natoms": torch.tensor([n_atoms]),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "cell_offsets": torch.empty((0, 3)),
            "nedges": torch.tensor([0]),
            "charge": torch.tensor([0]),
            "spin": torch.tensor([0]),
            "fixed": torch.zeros(n_atoms, dtype=torch.long),
            "tags": torch.zeros(n_atoms, dtype=torch.long),
            "sid": ["s"],
            "dataset": "test",
            "minkowski_op": compute_minkowski_op(cell.view(1, 3, 3), pbc.view(1, 3)),
        }
    )


def test_collated_batch_matches_general_find_mic():
    pbc = torch.tensor([True, True, True])
    d1 = _make_sample(3, _TRICLINIC_CELL.squeeze(0), pbc)
    d2 = _make_sample(
        2, torch.diag(torch.tensor([12.0, 12.0, 12.0], dtype=torch.float64)), pbc
    )
    batch = data_list_collater([d1, d2], otf_graph=True)

    dr = torch.tensor(
        [
            [8.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            [-5.0, 6.0, 4.5],
            [7.0, 0.0, 0.0],
            [0.0, -3.0, 3.0],
        ],
        dtype=torch.float64,
    )
    _assert_matches_general_find_mic(
        dr,
        batch.cell.double(),
        batch.pbc,
        batch.batch,
        minkowski_op=batch.minkowski_op.double(),
    )
