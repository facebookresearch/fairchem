"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Test-only Triton kernel launchers.

These launchers directly invoke Triton kernels (bypassing @triton_op wrappers)
and use num_edges as GRID_E_STRIDE for simpler testing. Production code uses
the autograd Functions in the triton/ package which use fixed GRID_E_STRIDE=2048
for torch.compile compatibility.

These should NOT be used in production - only for unit testing kernel correctness.
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.triton.constants import BLOCK_C
from fairchem.core.models.uma.triton.kernels import (
    node_to_edge_wigner_permute_kernel,
    permute_wigner_inv_edge_to_node_kernel,
)


def node_to_edge_wigner_permute_launcher(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    """
    Test-only launcher: gather + Wigner + L→M permute.

    Uses num_edges as GRID_E_STRIDE (not the production constant 2048).
    For testing kernel correctness, not for production use.

    Args:
        x: Node features [N, 9, C] in L-major order
        edge_index: Edge indices [2, E]
        wigner: Wigner matrices [E, 9, 9] (block-diagonal structure)

    Returns:
        out: Rotated edge features [E, 9, 2C] in M-major order (src||tgt)
    """
    assert x.ndim == 3, "x must be 3D [N, 9, C]"
    assert x.shape[1] == 9, "x must have 9 coefficients (lmax=2)"
    assert wigner.ndim == 3, "wigner must be 3D [E, 9, 9]"
    assert wigner.shape[1] == 9, "wigner must have shape [E, 9, 9]"
    assert wigner.shape[2] == 9, "wigner must have shape [E, 9, 9]"
    assert wigner.is_contiguous(), "wigner must be contiguous"

    num_edges = edge_index.shape[1]
    sphere_channels = x.shape[2]

    wigner_flat = wigner.reshape(num_edges, -1)

    out = torch.empty(
        (num_edges, 9, sphere_channels * 2),
        dtype=x.dtype,
        device=x.device,
    )

    num_c_blocks = (sphere_channels + BLOCK_C - 1) // BLOCK_C
    grid = (num_edges, num_c_blocks)

    node_to_edge_wigner_permute_kernel[grid](
        x,
        edge_index,
        wigner_flat,
        out,
        num_edges,
        sphere_channels,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        edge_index.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_C=BLOCK_C,
        GRID_E_STRIDE=num_edges,
    )

    return out


def permute_wigner_inv_edge_to_node_launcher(
    x: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    """
    Test-only launcher: M→L permute + Wigner inverse.

    Uses E as GRID_E_STRIDE (not the production constant 2048).
    For testing kernel correctness, not for production use.

    Args:
        x: Edge features [E, 9, C] in M-major order
        wigner: Wigner inverse matrices [E, 9, 9]

    Returns:
        out: Rotated features [E, 9, C] in L-major order
    """
    assert x.ndim == 3, "x must be 3D [E, 9, C]"
    assert x.shape[1] == 9, "x must have 9 coefficients (lmax=2)"
    assert wigner.ndim == 3, "wigner must be 3D [E, 9, 9]"
    assert wigner.shape[1] == 9, "wigner must have shape [E, 9, 9]"
    assert wigner.shape[2] == 9, "wigner must have shape [E, 9, 9]"
    assert x.is_contiguous(), "x must be contiguous"
    assert wigner.is_contiguous(), "wigner must be contiguous"

    E, num_coeffs, C = x.shape
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    out = torch.empty_like(x)

    permute_wigner_inv_edge_to_node_kernel[(E, num_c_blocks)](
        x,
        wigner,
        out,
        E,
        C,
        BLOCK_C=BLOCK_C,
        GRID_E_STRIDE=E,
    )
    return out
