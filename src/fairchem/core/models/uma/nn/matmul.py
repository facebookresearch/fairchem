"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional as F

if TYPE_CHECKING:
    import torch


def linear_with_folded_batch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a linear operation with the leading input dimensions folded into one.

    PyTorch folds the leading dimensions of a rank-three or higher input into one
    GEMM only when the fold is a free view or the weight requires gradients.
    Otherwise, as for an order-m slice of the SO(2) edge features multiplied by a
    frozen weight, it runs a broadcast BMM with one small GEMM per leading index,
    whether or not autograd records the operation. Folding copies such an input
    once and runs a single two-dimensional GEMM; for a viewable input the fold is a
    free view. Frozen weights therefore always fold, while weights that require
    gradients keep the native path, which already folds them.

    Args:
        x: Input to the linear operation.
        weight: Linear weight used to determine whether gradients are required.
        bias: Optional linear bias.

    Returns:
        The linear result with the original leading dimensions restored.
    """
    should_fold = x.ndim > 2 and not weight.requires_grad
    x_shape = x.shape
    linear_input = x.flatten(0, -2) if should_fold else x
    output = F.linear(linear_input, weight, bias)
    return output.reshape(*x_shape[:-1], output.shape[-1]) if should_fold else output
