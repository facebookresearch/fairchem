"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

First stage of the radial MLP, fused end to end.

The eager spelling is four lines that look like two adds and a norm::

    h = F.linear(gauss, W_gauss, bias)
    h = h + table_src[z_i] + table_tgt[z_j]
    h = layer_norm_silu(h, gamma, beta, eps)

and it moves seven [E, H] tensors through HBM: the GEMM result, a gathered
copy per table, a sum per table, and the norm's own input and output. The
arithmetic behind it is trivial -- the GEMM contracts over the Gaussian basis,
which is 32 wide for uma-s -- so the stage is bandwidth bound by a wide margin,
and on an RTX 4070 the two table lookups alone cost two to six times the GEMM
they follow.

This kernel does the whole stage in one pass: read the 32 Gaussian
coefficients, contract them against the weight tile in registers, add the bias
and both element rows, normalise, apply SiLU, write once. The element tables
are (max_num_elements, H) and stay resident in cache.

``tl.dot`` is pinned to ``ieee``. It is the only matrix multiply in any flash
kernel, and letting it default to tf32 would silently drop mantissa bits
regardless of the tf32 inference setting, which is decided per predict() and
is not visible from here. The contraction is 32 deep, so tensor cores would
buy nothing anyway.
"""

from __future__ import annotations

import triton
import triton.language as tl


def _generate_configs():
    # Each program holds BLOCK_E rows of H floats in registers, so the useful
    # range of BLOCK_E is narrow: too small wastes the dot, too large spills.
    # No loop to pipeline, hence num_stages=1 (see the note in scatter.py).
    configs = []
    for e in [16, 32, 64]:
        for w in [2, 4, 8]:
            configs.append(triton.Config({"BLOCK_E": e}, num_warps=w, num_stages=1))
    return configs


@triton.autotune(cache_results=True, configs=_generate_configs(), key=["B", "H"])
@triton.jit
def _radial_stage1_fwd_kernel(
    gauss_ptr,
    W_ptr,
    bias_ptr,
    ts_ptr,
    tt_ptr,
    zi_ptr,
    zj_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    E,
    B,
    H,
    eps,
    stride_gauss_e,
    stride_out_e,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    b_offs = tl.arange(0, BLOCK_B)
    h_offs = tl.arange(0, BLOCK_H)
    b_mask = b_offs < B
    h_mask = h_offs < H

    # Zeros outside the real extents keep the padded dot exact.
    g = tl.load(
        gauss_ptr + e_offs[:, None] * stride_gauss_e + b_offs[None, :],
        mask=e_mask[:, None] & b_mask[None, :],
        other=0.0,
    )
    w = tl.load(
        W_ptr + h_offs[:, None] * B + b_offs[None, :],
        mask=h_mask[:, None] & b_mask[None, :],
        other=0.0,
    )
    acc = tl.dot(g, tl.trans(w), input_precision="ieee")
    acc += tl.load(bias_ptr + h_offs, mask=h_mask, other=0.0)[None, :]

    zi = tl.load(zi_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    zj = tl.load(zj_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    tbl_mask = e_mask[:, None] & h_mask[None, :]
    acc += tl.load(ts_ptr + zi[:, None] * H + h_offs[None, :], mask=tbl_mask, other=0.0)
    acc += tl.load(tt_ptr + zj[:, None] * H + h_offs[None, :], mask=tbl_mask, other=0.0)
    acc = tl.where(h_mask[None, :], acc, 0.0)

    mean = tl.sum(acc, axis=1) / H
    centred = tl.where(h_mask[None, :], acc - mean[:, None], 0.0)
    var = tl.sum(centred * centred, axis=1) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    x_hat = centred * rstd[:, None]
    y = (
        x_hat * tl.load(gamma_ptr + h_offs, mask=h_mask, other=0.0)[None, :]
        + tl.load(beta_ptr + h_offs, mask=h_mask, other=0.0)[None, :]
    )

    tl.store(mean_ptr + e_offs, mean, mask=e_mask)
    tl.store(rstd_ptr + e_offs, rstd, mask=e_mask)
    tl.store(
        out_ptr + e_offs[:, None] * stride_out_e + h_offs[None, :],
        y * tl.sigmoid(y),
        mask=tbl_mask,
    )


@triton.autotune(cache_results=True, configs=_generate_configs(), key=["B", "H"])
@triton.jit
def _radial_stage1_bwd_kernel(
    g_out_ptr,
    gauss_ptr,
    W_ptr,
    bias_ptr,
    ts_ptr,
    tt_ptr,
    zi_ptr,
    zj_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    g_gauss_ptr,
    E,
    B,
    H,
    stride_gauss_e,
    stride_gout_e,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    b_offs = tl.arange(0, BLOCK_B)
    h_offs = tl.arange(0, BLOCK_H)
    b_mask = b_offs < B
    h_mask = h_offs < H
    tbl_mask = e_mask[:, None] & h_mask[None, :]

    # Recompute the pre-norm activations rather than staging them in the
    # forward: they are the [E, H] tensor this kernel exists to avoid writing,
    # and the inputs are all either tiny or already being read.
    g = tl.load(
        gauss_ptr + e_offs[:, None] * stride_gauss_e + b_offs[None, :],
        mask=e_mask[:, None] & b_mask[None, :],
        other=0.0,
    )
    w = tl.load(
        W_ptr + h_offs[:, None] * B + b_offs[None, :],
        mask=h_mask[:, None] & b_mask[None, :],
        other=0.0,
    )
    acc = tl.dot(g, tl.trans(w), input_precision="ieee")
    acc += tl.load(bias_ptr + h_offs, mask=h_mask, other=0.0)[None, :]
    zi = tl.load(zi_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    zj = tl.load(zj_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    acc += tl.load(ts_ptr + zi[:, None] * H + h_offs[None, :], mask=tbl_mask, other=0.0)
    acc += tl.load(tt_ptr + zj[:, None] * H + h_offs[None, :], mask=tbl_mask, other=0.0)
    acc = tl.where(h_mask[None, :], acc, 0.0)

    mean = tl.load(mean_ptr + e_offs, mask=e_mask, other=0.0)
    rstd = tl.load(rstd_ptr + e_offs, mask=e_mask, other=0.0)
    gamma = tl.load(gamma_ptr + h_offs, mask=h_mask, other=0.0)[None, :]
    beta = tl.load(beta_ptr + h_offs, mask=h_mask, other=0.0)[None, :]

    x_hat = tl.where(h_mask[None, :], (acc - mean[:, None]) * rstd[:, None], 0.0)
    y = x_hat * gamma + beta

    # SiLU'(y) = s * (1 + y * (1 - s))
    s = tl.sigmoid(y)
    g_out = tl.load(
        g_out_ptr + e_offs[:, None] * stride_gout_e + h_offs[None, :],
        mask=tbl_mask,
        other=0.0,
    )
    g_y = g_out * s * (1.0 + y * (1.0 - s))
    g_hat = tl.where(h_mask[None, :], g_y * gamma, 0.0)

    mean_g = tl.sum(g_hat, axis=1) / H
    mean_gx = tl.sum(g_hat * x_hat, axis=1) / H
    g_acc = (g_hat - mean_g[:, None] - x_hat * mean_gx[:, None]) * rstd[:, None]
    g_acc = tl.where(h_mask[None, :], g_acc, 0.0)

    # Only the Gaussian input carries a gradient: the weights and the element
    # tables are inference buffers, matching flash_ln_silu_bwd.
    g_gauss = tl.dot(g_acc, w, input_precision="ieee")
    tl.store(
        g_gauss_ptr + e_offs[:, None] * stride_gauss_e + b_offs[None, :],
        g_gauss,
        mask=e_mask[:, None] & b_mask[None, :],
    )
