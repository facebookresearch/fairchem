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
@pytest.mark.parametrize("sizes", [[7, 6, 8], [5, 5], [3, 4, 4, 4]])
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


def test_uneven_sizes_skip_padding():
    g = MOLEGlobals(
        expert_mixing_coefficients=None, mole_sizes=torch.tensor([1, 9, 2, 4])
    )
    set_padded_segments(g, [1, 9, 2, 4], torch.device("cpu"))
    assert g.pad_index is None
    set_padded_segments(g, [1, 9, 2, 4], torch.device("cpu"), max_pad_ratio=10.0)
    assert g.pad_index is not None


@pytest.mark.parametrize("sizes", [[7, 3, 12], [1, 9, 2, 4], [6]])
def test_split_loop_matches_slice_loop(sizes):
    """
    The split-view loop (no index maps) must match the original slice loop,
    exercised here through the activation-checkpoint chunk path.
    """
    torch.manual_seed(0)
    coeffs = torch.softmax(torch.randn(len(sizes), 3), dim=1)
    g = MOLEGlobals(expert_mixing_coefficients=coeffs, mole_sizes=torch.tensor(sizes))
    layer = MOLE(3, 5, 4, g, bias=True)
    x = torch.randn(sum(sizes), 5, requires_grad=True)
    out = layer(x)
    (out**2).sum().backward()
    x_grad = x.grad.clone()
    x.grad = None
    layer.weights.grad = None
    # chunk path == original slice loop, two chunks
    k = sizes[0]
    g.ac_start_idx = 0
    part0 = layer(x[:k])
    g.ac_start_idx = k
    part1 = layer(x[k:])
    g.ac_start_idx = 0
    ref = torch.cat([part0, part1], dim=0)
    (ref**2).sum().backward()
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(x_grad, x.grad, atol=1e-6)


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
