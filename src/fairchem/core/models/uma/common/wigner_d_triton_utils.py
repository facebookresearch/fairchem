"""
Shared utilities for Triton Wigner D kernels.

This module provides the coefficient loading and caching utilities
needed by the generated Triton kernels.

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_COEFFICIENTS_FILE = Path(__file__).parent / "wigner_d_coefficients.pt"
_KERNEL_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


@functools.lru_cache(maxsize=1)
def _load_coefficients() -> dict:
    """Load precomputed coefficients from file."""
    raw = torch.load(_COEFFICIENTS_FILE, map_location="cpu", weights_only=True)
    result = {}
    for ell in [3, 4]:
        key = f"C_l{ell}"
        palette = raw[f"{key}_palette"]
        indices = raw[f"{key}_indices"]
        shape = tuple(raw[f"{key}_shape"].tolist())
        result[key] = palette[indices.long()].reshape(shape)
    return result


def _get_kernel_data(
    ell: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Get transposed coefficient matrix C.T for the Triton kernel.

    The coefficient matrix C has shape (n_outputs, n_monomials).
    We return C.T with shape (n_monomials, n_outputs) for efficient
    memory access in the kernel.

    Args:
        ell: Angular momentum (3 or 4)
        dtype: Data type for the coefficients
        device: Device for the tensors

    Returns:
        Transposed coefficient matrix of shape (n_monomials, n_outputs)
    """
    key = (ell, dtype, device)
    if key not in _KERNEL_CACHE:
        coeffs = _load_coefficients()
        C = coeffs[f"C_l{ell}"].to(dtype=dtype, device=device)
        # Store transposed for efficient kernel access
        C_T = C.T.contiguous()
        _KERNEL_CACHE[key] = C_T
    return _KERNEL_CACHE[key]


def preload_triton_caches(
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> None:
    """
    Preload coefficient caches for Triton kernels.

    Call this before torch.compile to avoid tracing through torch.load.

    Args:
        dtype: Data type for coefficients
        device: Device for tensors (default: CUDA if available)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ell in (3, 4):
        _get_kernel_data(ell, dtype, device)


def clear_triton_caches() -> None:
    """Clear all cached coefficient tensors."""
    _KERNEL_CACHE.clear()
    _load_coefficients.cache_clear()
