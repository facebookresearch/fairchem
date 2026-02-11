"""
Wigner D matrices via hybrid approach (fastest method per l).

This module provides Wigner D computation using the optimal method for each l:
- l=0: Trivial (identity)
- l=1: Direct quaternion to rotation matrix (fastest for 3x3)
- l=2: Quaternion einsum tensor contraction (~20x faster on GPU)
- l=3: Triton kernel (if available) or quaternion matmul
- l=4: Triton kernel (if available) or quaternion matmul (with l4_kernel=True)
- l>=4 or 5: Ra/Rb polynomial (faster than matrix_exp on GPU)

Entry point:
- axis_angle_wigner_hybrid: Main function with configurable arithmetic mode

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    compute_euler_matching_gamma,
    get_ra_rb_coefficients,
    get_ra_rb_coefficients_real,
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_to_ra_rb,
    quaternion_to_ra_rb_real,
    quaternion_y_rotation,
    wigner_d_complex_to_real,
    wigner_d_matrix_complex,
    wigner_d_matrix_real,
    wigner_d_pair_to_real,
)
from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    quaternion_to_rotation_matrix,
    quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_matmul,
)

# =============================================================================
# Triton Kernel Support
# =============================================================================

# Try to import Triton kernels
_TRITON_L3_AVAILABLE = False
_TRITON_L4_AVAILABLE = False

try:
    from fairchem.core.models.uma.common.wigner_d_l3_triton_generated import (
        quaternion_to_wigner_d_l3_triton,
    )
    _TRITON_L3_AVAILABLE = True
except ImportError:
    quaternion_to_wigner_d_l3_triton = None

try:
    from fairchem.core.models.uma.common.wigner_d_l4_triton_generated import (
        quaternion_to_wigner_d_l4_triton,
    )
    _TRITON_L4_AVAILABLE = True
except ImportError:
    quaternion_to_wigner_d_l4_triton = None

# Global flag to enable/disable Triton (can be set by user)
_USE_TRITON = True

def set_use_triton(enabled: bool) -> None:
    """Enable or disable Triton kernels globally."""
    global _USE_TRITON
    _USE_TRITON = enabled
    logging.info(f"Triton kernels {'enabled' if enabled else 'disabled'}")


def get_triton_status() -> dict:
    """Get status of Triton kernel availability."""
    return {
        "enabled": _USE_TRITON,
        "l3_available": _TRITON_L3_AVAILABLE,
        "l4_available": _TRITON_L4_AVAILABLE,
    }


def _should_use_triton(q: torch.Tensor) -> bool:
    """Check if we should use Triton for this computation."""
    return (
        _USE_TRITON
        and q.is_cuda
        and (_TRITON_L3_AVAILABLE or _TRITON_L4_AVAILABLE)
    )


# =============================================================================
# Hybrid Wigner D Computation
# =============================================================================


def wigner_d_from_quaternion_hybrid(
    q: torch.Tensor,
    lmax: int,
    l4_kernel: bool = False,
    use_real_arithmetic: bool = False,
    use_triton: Optional[bool] = None,
    coeffs: Optional[object] = None,
    U_blocks: Optional[list] = None,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from quaternion using hybrid approach.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomial einsum
    - l=3: Triton kernel (GPU) or quaternion matmul (CPU/fallback)
    - l=4: Triton kernel (GPU) or quaternion matmul if l4_kernel=True
    - l>=lmin: Ra/Rb polynomial (lmin=5 if l4_kernel else 4)

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum
        l4_kernel: If True, use custom kernel for l=4 (matmul or Triton)
        use_real_arithmetic: If True, use real-pair arithmetic for Ra/Rb
                            (torch.compile compatible, avoids complex tensors)
        use_triton: If True, prefer Triton kernels for l=3,4 when available.
                   If None, uses global setting. If False, always use matmul.
        coeffs: Optional pre-computed WignerCoefficients. If provided with U_blocks,
                skips the cache lookup for better performance in hot paths.
        U_blocks: Optional pre-computed U transformation blocks.

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2
    """
    N = q.shape[0]
    device = q.device
    dtype = q.dtype
    size = (lmax + 1) ** 2

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    lmin = 5 if l4_kernel else 4

    # Determine whether to use Triton
    if use_triton is None:
        use_triton_kernels = _should_use_triton(q)
    else:
        use_triton_kernels = use_triton and q.is_cuda

    # Compute l=0,1,2,3 (and optionally 4) using direct quaternion methods
    for ell in range(min(lmax + 1, lmin)):
        if ell == 0:
            D[:, 0, 0] = 1.0
        elif ell == 1:
            D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)
        elif ell == 2:
            D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q)
        elif ell == 3:
            # Use Triton if available and enabled, else matmul
            if use_triton_kernels and _TRITON_L3_AVAILABLE:
                D[:, 9:16, 9:16] = quaternion_to_wigner_d_l3_triton(q)
            else:
                D[:, 9:16, 9:16] = quaternion_to_wigner_d_matmul(q, 3)
        elif l4_kernel and ell == 4:
            # Use Triton if available and enabled, else matmul
            if use_triton_kernels and _TRITON_L4_AVAILABLE:
                D[:, 16:25, 16:25] = quaternion_to_wigner_d_l4_triton(q)
            else:
                D[:, 16:25, 16:25] = quaternion_to_wigner_d_matmul(q, 4)

    # Compute l>=lmin using Ra/Rb polynomial
    if lmax >= lmin:
        if use_real_arithmetic:
            # Real-pair arithmetic (torch.compile compatible)
            if coeffs is None or U_blocks is None:
                coeffs, U_blocks = get_ra_rb_coefficients_real(
                    lmax, dtype, device, lmin=lmin
                )
            ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
            D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
            D_range = wigner_d_pair_to_real(D_re, D_im, U_blocks, lmin=lmin, lmax=lmax)
        else:
            # Complex arithmetic
            if coeffs is None or U_blocks is None:
                coeffs, U_blocks = get_ra_rb_coefficients(
                    lmax, dtype, device, lmin=lmin
                )
            Ra, Rb = quaternion_to_ra_rb(q)
            D_complex = wigner_d_matrix_complex(Ra, Rb, coeffs)
            D_range = wigner_d_complex_to_real(
                D_complex, U_blocks, lmin=lmin, lmax=lmax
            )

        block_offset = lmin * lmin  # 9 for lmin=3, 25 for lmin=5
        D[:, block_offset:, block_offset:] = D_range

    return D


# =============================================================================
# Main Entry Point
# =============================================================================


def axis_angle_wigner_hybrid(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
    l4_kernel: bool = False,
    use_real_arithmetic: bool = False,
    use_triton: Optional[bool] = None,
    coeffs: Optional[object] = None,
    U_blocks: Optional[list] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using hybrid approach (optimal method per l).

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion einsum tensor contraction
    - l=3: Triton kernel (GPU) or quaternion matmul
    - l=4: Triton kernel (GPU) or quaternion matmul if l4_kernel=True
    - l>=lmin: Ra/Rb polynomial (lmin=5 if l4_kernel else 4)

    Combines the edge->Y and gamma rotations into a single quaternion before
    computing the Wigner D, avoiding the overhead of computing two separate
    Wigner D matrices and multiplying them.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.
        l4_kernel: If True, use custom kernel for l=4
        use_real_arithmetic: If True, use real-pair arithmetic for Ra/Rb
               (torch.compile compatible, avoids complex tensors)
        use_triton: If True, prefer Triton kernels for l=3,4 when available.
                   If None, uses global setting. If False, always use matmul.
        coeffs: Optional pre-computed WignerCoefficients. If provided with U_blocks,
               skips the cache lookup for better performance in hot paths.
        U_blocks: Optional pre-computed U transformation blocks.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
    """
    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    N = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype

    # Step 1: Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Step 2: Compute gamma if not provided
    if gamma is None:
        if use_euler_gamma:
            gamma = compute_euler_matching_gamma(edge_normalized)
        else:
            gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 3: Compute quaternion (edge -> +Y)
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge->Y
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Compute Wigner D using hybrid approach
    D = wigner_d_from_quaternion_hybrid(
        q_combined,
        lmax,
        l4_kernel=l4_kernel,
        use_real_arithmetic=use_real_arithmetic,
        use_triton=use_triton,
        coeffs=coeffs,
        U_blocks=U_blocks,
    )

    # Step 6: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv


# =============================================================================
# Preloading for torch.compile compatibility
# =============================================================================


def preload_wigner_caches(
    lmax: int = 6,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> None:
    """
    Preload all caches needed for Wigner D computation.

    This should be called before torch.compile to avoid tracing through
    torch.load and cache initialization.

    Args:
        lmax: Maximum angular momentum to preload for
        dtype: Data type for coefficients
        device: Device for tensors (default: CUDA if available, else CPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preload custom kernel caches (l=2,3,4)
    from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
        preload_kernel_caches,
    )
    preload_kernel_caches(dtype, device)

    # Preload Ra/Rb coefficients for higher l
    if lmax >= 4:
        get_ra_rb_coefficients_real(lmax, dtype, device, lmin=4)

    # Preload Triton kernel caches if available
    if _TRITON_L3_AVAILABLE or _TRITON_L4_AVAILABLE:
        try:
            from fairchem.core.models.uma.common.wigner_d_triton_utils import (
                _get_kernel_data,
            )
            if _TRITON_L3_AVAILABLE:
                _get_kernel_data(3, dtype, device)
            if _TRITON_L4_AVAILABLE:
                _get_kernel_data(4, dtype, device)
        except ImportError:
            pass  # Triton cache loading is optional

    logging.info(
        f"Preloaded Wigner caches: lmax={lmax}, dtype={dtype}, device={device}, "
        f"triton_l3={_TRITON_L3_AVAILABLE}, triton_l4={_TRITON_L4_AVAILABLE}"
    )
