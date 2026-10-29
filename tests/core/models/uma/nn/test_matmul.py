"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from fairchem.core.models.uma.nn.matmul import linear_with_folded_batch

EDGES = 16
CHANNELS = 8
IN_FEATURES = 2 * CHANNELS
OUT_FEATURES = 12
MODES = ("no_grad", "grad", "input_grad")


def _make_leaf(kind: str, requires_grad: bool) -> torch.Tensor:
    if kind == "contiguous":
        shape = (EDGES, 2, IN_FEATURES)
    elif kind == "rank4":
        shape = (3, EDGES, 2, IN_FEATURES)
    else:
        shape = (EDGES, 9, CHANNELS)
    return torch.randn(shape, requires_grad=requires_grad)


def _derive_input(leaf: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "contiguous":
        return leaf
    if kind == "rank4":
        return leaf.transpose(0, 1)
    # The m=1 slice of lmax=2 edge features, viewed as SO2_Convolution passes it.
    x = leaf.split((3, 4, 2), dim=1)[1].view(EDGES, 2, -1)
    # One system's edges, as an unmerged MOLE layer passes them.
    return x[3:11] if kind == "sliced" else x


def _make_parameters(
    trainable: bool, use_bias: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = torch.randn(OUT_FEATURES, IN_FEATURES, requires_grad=trainable)
    bias = torch.randn(OUT_FEATURES, requires_grad=trainable) if use_bias else None
    return weight, bias


def _clone_leaf(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().clone().requires_grad_(tensor.requires_grad)


def _call(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, mode: str
) -> torch.Tensor:
    with torch.no_grad() if mode == "no_grad" else nullcontext():
        return linear_with_folded_batch(x, weight, bias)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("kind", ["contiguous", "strided", "sliced", "rank4"])
def test_matches_contiguous_linear(kind, trainable, use_bias, mode):
    torch.manual_seed(0)
    leaf = _make_leaf(kind, requires_grad=mode == "input_grad")
    weight, bias = _make_parameters(trainable, use_bias)
    leaf_ref, weight_ref, bias_ref = map(_clone_leaf, (leaf, weight, bias))

    output = _call(_derive_input(leaf, kind), weight, bias, mode)
    x_ref = _derive_input(leaf_ref, kind)
    with torch.no_grad() if mode == "no_grad" else nullcontext():
        reference = F.linear(x_ref.contiguous().flatten(0, -2), weight_ref, bias_ref)
    reference = reference.reshape(*x_ref.shape[:-1], OUT_FEATURES)

    assert output.shape == reference.shape
    assert output.requires_grad == reference.requires_grad
    torch.testing.assert_close(output, reference)

    if not output.requires_grad:
        return
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    reference.backward(grad_output)
    for tensor, tensor_ref in ((leaf, leaf_ref), (weight, weight_ref)):
        if tensor.requires_grad:
            torch.testing.assert_close(tensor.grad, tensor_ref.grad)
    if bias is not None and bias.requires_grad:
        torch.testing.assert_close(bias.grad, bias_ref.grad)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["strided", "sliced", "rank4"])
def test_frozen_weight_folds_strided_input_into_one_gemm(kind, mode):
    """
    A frozen weight must not fall back to one broadcast-BMM GEMM per leading index.

    PyTorch only folds such inputs itself when the weight requires gradients, so
    this checks the dispatched kernels for every autograd mode, including
    inference without input gradients.
    """
    x = _derive_input(_make_leaf(kind, requires_grad=mode == "input_grad"), kind)
    with pytest.raises(RuntimeError):
        x.view(-1, x.shape[-1])  # the leading dimensions are not viewable as one
    weight, _ = _make_parameters(trainable=False, use_bias=False)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _call(x, weight, None, mode)
    dispatched = {event.key for event in prof.key_averages()}

    assert "aten::bmm" not in dispatched
    assert "aten::mm" in dispatched
