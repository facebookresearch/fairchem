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


class TestMinimumImageDisplacementAgainstASE:
    """Minimum-image displacements must agree with ASE's mic distances."""

    @pytest.fixture()
    def random_system(self):
        rng = np.random.default_rng(42)
        cell = _random_cell(rng)
        n_atoms = 20
        frac = rng.uniform(0.0, 1.0, size=(n_atoms, 3))
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
    def test_matches_ase_get_all_distances(self, random_system, pbc):
        from ase import Atoms

        pos, cell = random_system
        pbc_arr = np.array(pbc)

        atoms = Atoms(
            symbols=["H"] * len(pos),
            positions=pos,
            cell=cell,
            pbc=pbc_arr,
        )

        # ASE reference: all pairwise mic displacement vectors, shape (n, n, 3)
        ase_vecs = atoms.get_all_distances(mic=True, vector=True)

        n = len(pos)
        for i in range(n):
            # Displacement vectors from atom i to every other atom
            dr_np = pos - pos[i]  # (n, 3) raw displacements

            dr = torch.from_numpy(dr_np)
            cell_t = torch.from_numpy(cell).unsqueeze(0).expand(n, -1, -1)
            pbc_t = torch.tensor(pbc).unsqueeze(0).expand(n, -1)

            result = minimum_image_displacement(dr, cell_t, pbc_t).numpy()

            np.testing.assert_allclose(
                result,
                ase_vecs[i],
                atol=1e-10,
                err_msg=f"MIC displacement mismatch for reference atom {i}",
            )

    def test_gradients_flow_through_cell(self):
        """Verify autograd gradients survive through cell."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        cell.requires_grad_(True)
        cell_b = cell.unsqueeze(0).expand(2, -1, -1)
        pbc = torch.tensor([[True, True, True], [True, True, True]])
        dr = torch.tensor([[9.5, 0.5, 0.5], [0.5, 9.8, 0.0]], dtype=torch.float64)

        result = minimum_image_displacement(dr, cell_b, pbc)
        loss = result.sum()
        loss.backward()

        assert cell.grad is not None
        assert cell.grad.abs().sum().item() > 0
