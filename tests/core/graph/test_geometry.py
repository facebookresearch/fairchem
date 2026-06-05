"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.geometry import wrap_positions as ase_wrap_positions

from fairchem.core.graph.geometry import (
    minimum_image_displacement,
    wrap_positions,
)


def _random_cell(rng, min_length=3.0, max_length=10.0):
    raw = rng.standard_normal((3, 3))
    raw += np.eye(3) * 3.0
    lengths = rng.uniform(min_length, max_length, size=3)
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True) * lengths[:, None]
    if np.linalg.det(raw) < 0:
        raw[0] *= -1
    return raw


class TestWrapPositionsAgainstASE:
    """Both numpy and torch paths must agree with ASE on a random non-orthogonal cell."""

    @pytest.fixture()
    def random_system(self):
        rng = np.random.default_rng(42)
        cell = _random_cell(rng)
        frac = rng.uniform(-0.5, 1.5, size=(50, 3))
        pos = frac @ cell
        return pos, cell

    @pytest.mark.parametrize(
        "pbc",
        [
            [True, True, True],
            [True, False, True],
            [False, False, False],
        ],
        ids=["full-pbc", "mixed-pbc", "no-pbc"],
    )
    def test_numpy_and_torch_match_ase(self, random_system, pbc):
        pos, cell = random_system
        pbc_arr = np.array(pbc)

        expected = ase_wrap_positions(pos, cell, pbc=pbc_arr, eps=0)
        result_np = wrap_positions(pos, cell, pbc_arr)
        result_torch = wrap_positions(
            torch.from_numpy(pos),
            torch.from_numpy(cell),
            torch.from_numpy(pbc_arr),
        ).numpy()

        np.testing.assert_allclose(result_np, expected, atol=1e-10)
        np.testing.assert_allclose(result_torch, expected, atol=1e-10)


class TestMinimumImageDisplacement:
    """Minimum-image convention folds displacements to the shortest periodic image."""

    def test_known_displacements_with_mixed_pbc(self):
        L = 10.0
        cell = (torch.eye(3, dtype=torch.float64) * L).unsqueeze(0).expand(3, -1, -1)
        pbc = torch.tensor([
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ])

        dr = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [9.5, -9.8, 0.0],
                [9.5, -9.8, 25.0],
            ],
            dtype=torch.float64,
        )

        result = minimum_image_displacement(dr, cell, pbc)

        expected = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [-0.5, 0.2, 0.0],
                [-0.5, 0.2, 25.0],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)
