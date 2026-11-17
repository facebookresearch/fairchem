"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def linear_with_folded_batch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a linear operation after folding batch dimensions for input gradients.

    PyTorch can route a non-contiguous, rank-three activation multiplied by a frozen
    weight through broadcast BMM. UMA force inference differentiates with respect to
    positions while model parameters are frozen, so this path can retain a
    batch-expanded weight for backward. Folding the leading dimensions produces one
    two-dimensional GEMM instead, reducing backward memory and improving latency for
    representative UMA edge counts. When weights require gradients or input
    gradients are not being recorded, the original shape is preserved because the
    native linear path can be faster and has no frozen-weight expansion to avoid.

    Args:
        x: Input to the linear operation.
        weight: Linear weight used to determine whether gradients are required.
        bias: Optional linear bias.

    Returns:
        The linear result with the original leading dimensions restored.
    """
    should_fold = (
        x.ndim > 2
        and torch.is_grad_enabled()
        and x.requires_grad
        and not weight.requires_grad
    )
    x_shape = x.shape
    linear_input = x.flatten(0, -2) if should_fold else x
    output = F.linear(linear_input, weight, bias)
    return output.reshape(*x_shape[:-1], output.shape[-1]) if should_fold else output
