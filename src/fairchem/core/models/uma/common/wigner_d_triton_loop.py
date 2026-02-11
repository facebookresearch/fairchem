"""
Loop-based Triton kernels for Wigner D matrix computation.

These kernels use loops instead of full unrolling, trading a small amount
of runtime performance for dramatically faster compilation (~5s vs ~5min).

Use these when:
- You need fast compilation (development, first-time runs)
- The fully unrolled kernels take too long to compile

The fully unrolled kernels in wigner_d_l3/l4_triton_generated.py are
faster at runtime but take minutes to compile for l=4.

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


_COEFFICIENTS_FILE = Path(__file__).parent / "wigner_d_coefficients.pt"
_KERNEL_CACHE: dict[tuple[str, torch.dtype, torch.device], tuple] = {}


def _generate_monomials(n_vars: int, total_degree: int) -> list[tuple[int, ...]]:
    """Generate all monomials of given degree in n_vars variables."""
    monomials = []
    def generate(remaining_vars, remaining_deg, current):
        if remaining_vars == 1:
            monomials.append(tuple(current + [remaining_deg]))
            return
        for i in range(remaining_deg + 1):
            generate(remaining_vars - 1, remaining_deg - i, current + [i])
    generate(n_vars, total_degree, [])
    return monomials


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


def _get_loop_kernel_data(
    ell: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get data for loop-based kernel.

    Returns:
        C_T: Transposed coefficient matrix (n_monomials, n_outputs)
        exponents: Monomial exponents (n_monomials, 4) as int32
    """
    key = (f"loop_{ell}", dtype, device)
    if key not in _KERNEL_CACHE:
        coeffs = _load_coefficients()
        C = coeffs[f"C_l{ell}"].to(dtype=dtype, device=device)
        C_T = C.T.contiguous()

        # Generate exponent tensor
        monomials = _generate_monomials(4, 2 * ell)
        exponents = torch.tensor(monomials, dtype=torch.int32, device=device)

        _KERNEL_CACHE[key] = (C_T, exponents)
    return _KERNEL_CACHE[key]


if TRITON_AVAILABLE:
    @triton.jit(do_not_specialize=['N', 'n_outputs', 'n_monomials', 'degree'])
    def _wigner_d_loop_kernel(
        q_ptr,              # Input: (N, 4)
        D_ptr,              # Output: (N, n_outputs)
        C_T_ptr,            # Coefficients: (n_monomials, n_outputs)
        exp_ptr,            # Exponents: (n_monomials, 4)
        N,                  # Number of quaternions
        n_outputs,          # Number of output elements (49 for l=3, 81 for l=4)
        n_monomials,        # Number of monomials (84 for l=3, 165 for l=4)
        degree,             # Polynomial degree (6 for l=3, 8 for l=4)
        stride_q: tl.constexpr,
        stride_d: tl.constexpr,
        stride_c_row: tl.constexpr,
        stride_c_col: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        """
        Loop-based Wigner D kernel.

        Uses loops over outputs and monomials instead of full unrolling.
        Compiles much faster (~5s vs ~5min) with modest runtime overhead.
        """
        pid = tl.program_id(0)

        q_start = pid * BLOCK_Q
        q_offsets = q_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < N

        # Load quaternion components
        w = tl.load(q_ptr + q_offsets * stride_q + 0, mask=q_mask, other=0.0)
        x = tl.load(q_ptr + q_offsets * stride_q + 1, mask=q_mask, other=0.0)
        y = tl.load(q_ptr + q_offsets * stride_q + 2, mask=q_mask, other=0.0)
        z = tl.load(q_ptr + q_offsets * stride_q + 3, mask=q_mask, other=0.0)

        # Precompute powers (up to degree 8 to handle l=4)
        # We'll select the right ones based on exponents
        w1, w2, w3, w4 = w, w*w, w*w*w, w*w*w*w
        w5, w6, w7, w8 = w4*w, w4*w2, w4*w3, w4*w4

        x1, x2, x3, x4 = x, x*x, x*x*x, x*x*x*x
        x5, x6, x7, x8 = x4*x, x4*x2, x4*x3, x4*x4

        y1, y2, y3, y4 = y, y*y, y*y*y, y*y*y*y
        y5, y6, y7, y8 = y4*y, y4*y2, y4*y3, y4*y4

        z1, z2, z3, z4 = z, z*z, z*z*z, z*z*z*z
        z5, z6, z7, z8 = z4*z, z4*z2, z4*z3, z4*z4

        # Process each output element
        for out_idx in range(n_outputs):
            acc = tl.zeros((BLOCK_Q,), dtype=w.dtype)

            # Sum over all monomials
            for mono_idx in range(n_monomials):
                # Load coefficient
                c = tl.load(C_T_ptr + mono_idx * stride_c_row + out_idx * stride_c_col)

                # Load exponents
                a = tl.load(exp_ptr + mono_idx * 4 + 0)
                b = tl.load(exp_ptr + mono_idx * 4 + 1)
                c_exp = tl.load(exp_ptr + mono_idx * 4 + 2)
                d = tl.load(exp_ptr + mono_idx * 4 + 3)

                # Compute w^a using nested ternary
                w_pow = tl.where(a == 0, 1.0,
                        tl.where(a == 1, w1,
                        tl.where(a == 2, w2,
                        tl.where(a == 3, w3,
                        tl.where(a == 4, w4,
                        tl.where(a == 5, w5,
                        tl.where(a == 6, w6,
                        tl.where(a == 7, w7, w8))))))))

                x_pow = tl.where(b == 0, 1.0,
                        tl.where(b == 1, x1,
                        tl.where(b == 2, x2,
                        tl.where(b == 3, x3,
                        tl.where(b == 4, x4,
                        tl.where(b == 5, x5,
                        tl.where(b == 6, x6,
                        tl.where(b == 7, x7, x8))))))))

                y_pow = tl.where(c_exp == 0, 1.0,
                        tl.where(c_exp == 1, y1,
                        tl.where(c_exp == 2, y2,
                        tl.where(c_exp == 3, y3,
                        tl.where(c_exp == 4, y4,
                        tl.where(c_exp == 5, y5,
                        tl.where(c_exp == 6, y6,
                        tl.where(c_exp == 7, y7, y8))))))))

                z_pow = tl.where(d == 0, 1.0,
                        tl.where(d == 1, z1,
                        tl.where(d == 2, z2,
                        tl.where(d == 3, z3,
                        tl.where(d == 4, z4,
                        tl.where(d == 5, z5,
                        tl.where(d == 6, z6,
                        tl.where(d == 7, z7, z8))))))))

                acc += c * w_pow * x_pow * y_pow * z_pow

            # Store result
            tl.store(D_ptr + q_offsets * stride_d + out_idx, acc, mask=q_mask)


def quaternion_to_wigner_d_l3_triton_loop(q: torch.Tensor) -> torch.Tensor:
    """
    Compute l=3 Wigner D using loop-based Triton kernel.

    This version compiles in ~5 seconds vs ~30 seconds for the unrolled version.
    Runtime is slightly slower but compilation is much faster.

    Args:
        q: Unit quaternions of shape (N, 4) in (w, x, y, z) convention.

    Returns:
        Wigner D matrices of shape (N, 7, 7).
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    N = q.shape[0]
    n_outputs = 49
    n_monomials = 84
    degree = 6

    D_flat = torch.empty(N, n_outputs, dtype=q.dtype, device=q.device)
    C_T, exponents = _get_loop_kernel_data(3, q.dtype, q.device)

    BLOCK_Q = 32
    grid = ((N + BLOCK_Q - 1) // BLOCK_Q,)

    _wigner_d_loop_kernel[grid](
        q, D_flat, C_T, exponents, N,
        n_outputs, n_monomials, degree,
        stride_q=q.stride(0),
        stride_d=D_flat.stride(0),
        stride_c_row=C_T.stride(0),
        stride_c_col=C_T.stride(1),
        BLOCK_Q=BLOCK_Q,
    )

    return D_flat.view(N, 7, 7)


def quaternion_to_wigner_d_l4_triton_loop(q: torch.Tensor) -> torch.Tensor:
    """
    Compute l=4 Wigner D using loop-based Triton kernel.

    This version compiles in ~5-10 seconds vs ~5 minutes for the unrolled version.
    Runtime is slightly slower but compilation is much faster.

    Args:
        q: Unit quaternions of shape (N, 4) in (w, x, y, z) convention.

    Returns:
        Wigner D matrices of shape (N, 9, 9).
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    N = q.shape[0]
    n_outputs = 81
    n_monomials = 165
    degree = 8

    D_flat = torch.empty(N, n_outputs, dtype=q.dtype, device=q.device)
    C_T, exponents = _get_loop_kernel_data(4, q.dtype, q.device)

    BLOCK_Q = 32
    grid = ((N + BLOCK_Q - 1) // BLOCK_Q,)

    _wigner_d_loop_kernel[grid](
        q, D_flat, C_T, exponents, N,
        n_outputs, n_monomials, degree,
        stride_q=q.stride(0),
        stride_d=D_flat.stride(0),
        stride_c_row=C_T.stride(0),
        stride_c_col=C_T.stride(1),
        BLOCK_Q=BLOCK_Q,
    )

    return D_flat.view(N, 9, 9)


# =============================================================================
# Test
# =============================================================================


def test_loop_kernels():
    """Test loop-based kernels against reference."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("SKIP: Triton or CUDA not available")
        return

    from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
        quaternion_to_wigner_d_matmul,
    )

    device = torch.device("cuda")
    dtype = torch.float64

    torch.manual_seed(42)
    q = torch.randn(100, 4, dtype=dtype, device=device)
    q = q / q.norm(dim=1, keepdim=True)

    print("Testing loop-based kernels...")

    # l=3
    D_ref = quaternion_to_wigner_d_matmul(q, 3)
    D_loop = quaternion_to_wigner_d_l3_triton_loop(q)
    err_l3 = (D_ref - D_loop).abs().max().item()
    print(f"  l=3: max_err={err_l3:.2e} {'PASS' if err_l3 < 1e-10 else 'FAIL'}")

    # l=4
    D_ref = quaternion_to_wigner_d_matmul(q, 4)
    D_loop = quaternion_to_wigner_d_l4_triton_loop(q)
    err_l4 = (D_ref - D_loop).abs().max().item()
    print(f"  l=4: max_err={err_l4:.2e} {'PASS' if err_l4 < 1e-10 else 'FAIL'}")


if __name__ == "__main__":
    test_loop_kernels()
