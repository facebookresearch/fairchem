"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid
from fairchem.core.models.uma.nn.activation import (
    GateActivation,
    SeparableS2Activation_M,
)


def _coefficient_degrees(lmax, mmax, m_prime):
    """
    Degree l of each spherical harmonic coefficient, in L-major order or, with
    m_prime, in M-major order (l0m0, l1m0, ..., l1m1, l2m1, ..., l1m-1, ...).
    """
    if m_prime:
        degrees = list(range(lmax + 1))
        for mval in range(1, mmax + 1):
            degrees += list(range(mval, lmax + 1)) * 2
    else:
        degrees = []
        for lval in range(lmax + 1):
            degrees += [lval] * min(2 * lval + 1, 2 * mmax + 1)
    return torch.tensor(degrees)


class TestGateActivation:
    """
    Test the GateActivation class.
    """

    num_nodes = 5
    num_channels = 4

    @pytest.mark.parametrize(
        ("lmax", "mmax", "m_prime"),
        [(2, 2, False), (2, 2, True), (3, 1, False), (3, 1, True)],
    )
    def test_matches_per_degree_reference(self, lmax, mmax, m_prime):
        """
        The l=0 coefficient goes through SiLU, and every coefficient of degree
        l > 0 is scaled by the sigmoid of that degree's gate, in both
        coefficient orderings.
        """
        torch.manual_seed(0)
        activation = GateActivation(lmax, mmax, self.num_channels, m_prime=m_prime)
        degrees = _coefficient_degrees(lmax, mmax, m_prime)
        x = torch.randn(self.num_nodes, len(degrees), self.num_channels)
        gating_scalars = torch.randn(self.num_nodes, lmax * self.num_channels)

        gates = torch.sigmoid(gating_scalars).view(
            self.num_nodes, lmax, self.num_channels
        )
        expected = x * gates[:, (degrees - 1).clamp(min=0)]
        expected[:, 0] = F.silu(x[:, 0])

        torch.testing.assert_close(activation(gating_scalars, x), expected)


class TestSeparableS2ActivationM:
    """
    Test the SeparableS2Activation_M class.
    """

    @pytest.mark.parametrize(("lmax", "mmax"), [(2, 2), (3, 2)])
    def test_matches_l_ordered_grid_activation(self, lmax, mmax):
        """
        Acting on M-ordered coefficients gives the same result as converting to
        L order, applying SiLU on the sphere grid and converting back, with the
        l=0 coefficient replaced by SiLU of the separate scalar input.
        """
        torch.manual_seed(0)
        num_channels = 4
        so3_grid = torch.nn.ModuleDict({"lmax_mmax": SO3_Grid(lmax, mmax)})
        to_m = CoefficientMapping(lmax, mmax).to_m
        activation = SeparableS2Activation_M(lmax, mmax, so3_grid, to_m)
        x_m = torch.randn(3, to_m.shape[0], num_channels)
        scalars = torch.randn(3, num_channels)

        # to_m maps L-ordered to M-ordered coefficients and is a permutation
        x_l = torch.einsum("ml,nmc->nlc", to_m, x_m)
        to_grid = so3_grid["lmax_mmax"].get_to_grid_mat()
        from_grid = so3_grid["lmax_mmax"].get_from_grid_mat()
        on_grid = F.silu(torch.einsum("bai,nic->nbac", to_grid, x_l))
        out_l = torch.einsum("bai,nbac->nic", from_grid, on_grid)
        expected = torch.einsum("ml,nlc->nmc", to_m, out_l)
        expected[:, 0] = F.silu(scalars)

        torch.testing.assert_close(activation(scalars, x_m), expected)
