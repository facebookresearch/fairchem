"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.nn.mole import (
    MOLE,
    MOLEGlobals,
    set_padded_segments,
)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("sizes", [[7, 3, 12], [5, 5], [1, 9, 2, 4]])
def test_batched_matches_loop(ndim, sizes):
    """
    The padded bmm path must match the per-system loop, values and gradients.
    """
    torch.manual_seed(0)
    num_experts, in_f, out_f = 4, 6, 5
    coeffs = torch.softmax(torch.randn(len(sizes), num_experts), dim=1)
    shape = (sum(sizes), in_f) if ndim == 2 else (sum(sizes), 2, in_f)

    def run(batched):
        torch.manual_seed(1)
        g = MOLEGlobals(
            expert_mixing_coefficients=coeffs,
            mole_sizes=torch.tensor(sizes),
        )
        layer = MOLE(num_experts, in_f, out_f, g, bias=True)
        if batched:
            set_padded_segments(g, sizes, torch.device("cpu"))
        x = torch.randn(shape, requires_grad=True)
        out = layer(x)
        (out**2).sum().backward()
        return out, x.grad, layer.weights.grad, layer.bias.grad

    ref = run(batched=False)
    got = run(batched=True)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_single_system_keeps_loop():
    g = MOLEGlobals(expert_mixing_coefficients=None, mole_sizes=torch.tensor([9]))
    set_padded_segments(g, [9], torch.device("cpu"))
    assert g.pad_index is None


def test_chunked_input_falls_back_to_loop():
    """
    Activation-checkpoint chunks (ac_start_idx != 0) keep the loop path.
    """
    torch.manual_seed(0)
    sizes = [6, 4]
    coeffs = torch.softmax(torch.randn(2, 3), dim=1)
    g = MOLEGlobals(expert_mixing_coefficients=coeffs, mole_sizes=torch.tensor(sizes))
    layer = MOLE(3, 5, 4, g, bias=False)
    x = torch.randn(10, 5)
    full = layer(x)
    set_padded_segments(g, sizes, torch.device("cpu"))
    g.ac_start_idx = 6
    chunk = layer(x[6:])
    assert torch.allclose(chunk, full[6:], atol=1e-6)
