"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fused LayerNorm + SiLU used by the radial MLPs.
"""

from __future__ import annotations

import triton
import triton.language as tl


def _ln_configs():
    # Single-row-per-program kernel, so warps are the only useful axis.
    # ``num_stages`` is pinned to 1 on purpose. It controls software pipelining of
    # a loop, and this kernel has none -- the grid covers the whole problem. Extra
    # stages would only multiply the autotuning cold start for identical code.
    configs = []
    for w in [2, 4, 8]:
        configs.append(triton.Config({}, num_warps=w, num_stages=1))
    return configs


# =========================================================================
# FORWARD KERNEL: LayerNorm + SiLU
# =========================================================================
@triton.autotune(cache_results=True, configs=_ln_configs(), key=["D"])
@triton.jit
def _ln_silu_fwd_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    M,
    D,
    eps,
    stride_xm,
    stride_xd,
    stride_om,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # Load row
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)

    # Compute Mean and Variance
    sum_x = tl.sum(tl.where(d_mask, x, 0.0), axis=0)
    mean = sum_x / D
    xc = tl.where(d_mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    # Save Mean and Rstd for the backward pass
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Apply Normalization
    xhat = xc * rstd
    gamma = tl.load(gamma_ptr + d_offs, mask=d_mask, other=0.0)
    beta = tl.load(beta_ptr + d_offs, mask=d_mask, other=0.0)

    # Linear Affine Transform
    y = xhat * gamma + beta

    # SiLU Activation (y * sigmoid(y))
    sig = tl.sigmoid(y)
    out = y * sig

    # Store Output
    tl.store(out_ptr + row * stride_om + d_offs * stride_od, out, mask=d_mask)


# =========================================================================
# BACKWARD KERNEL: Optimized for Inference (No d_gamma / d_beta)
# =========================================================================
@triton.autotune(cache_results=True, configs=_ln_configs(), key=["D"])
@triton.jit
def _ln_silu_bwd_dx_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    g_out_ptr,
    g_x_ptr,
    M,
    D,
    stride_xm,
    stride_xd,
    stride_gom,
    stride_god,
    stride_gxm,
    stride_gxd,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # 1. Recompute forward values
    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    gamma = tl.load(gamma_ptr + d_offs, mask=d_mask, other=0.0)
    beta = tl.load(beta_ptr + d_offs, mask=d_mask, other=0.0)

    xhat = (x - mean) * rstd
    y = xhat * gamma + beta
    sig = tl.sigmoid(y)

    # 2. SiLU Derivative: d(silu(y))/dy = sig + y * sig * (1 - sig)
    dsilu_dy = sig + y * sig * (1.0 - sig)

    # 3. Chain Rule: g_out -> g_y -> g_xhat
    g_out = tl.load(
        g_out_ptr + row * stride_gom + d_offs * stride_god, mask=d_mask, other=0.0
    )
    g_y = g_out * dsilu_dy
    g_xhat = g_y * gamma

    # 4. Standard LayerNorm backward formula for g_x
    # g_x = rstd * (g_xhat - (mean(g_xhat) + xhat * mean(g_xhat * xhat)))
    sum_gxhat = tl.sum(tl.where(d_mask, g_xhat, 0.0), axis=0)
    sum_gxhat_xhat = tl.sum(tl.where(d_mask, g_xhat * xhat, 0.0), axis=0)

    g_x = rstd * (g_xhat - (sum_gxhat + xhat * sum_gxhat_xhat) / D)

    # 5. Store gradient w.r.t inputs
    tl.store(g_x_ptr + row * stride_gxm + d_offs * stride_gxd, g_x, mask=d_mask)
