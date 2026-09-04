"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Node-to-edge gather fused with the L->M Wigner rotation.

Produces the three dense m-order blocks consumed by SO(2) convolution 1,
scaling by the per-edge radial output on the way out, so the rotated
messages never round-trip through global memory.
"""

from __future__ import annotations

import triton
import triton.language as tl

from fairchem.core.models.uma.flash.constants import (
    W11,
    W12,
    W13,
    W21,
    W22,
    W23,
    W31,
    W32,
    W33,
    W44,
    W45,
    W46,
    W47,
    W48,
    W54,
    W55,
    W56,
    W57,
    W58,
    W64,
    W65,
    W66,
    W67,
    W68,
    W74,
    W75,
    W76,
    W77,
    W78,
    W84,
    W85,
    W86,
    W87,
    W88,
    X_M0_MULT,
    X_M1_MULT,
    X_M2_MULT,
)


# ``num_stages`` is pinned to 1 for the forward kernels and swept for the
# backward ones. Stages control software pipelining of a loop: the forward
# grid covers (E, C) directly and has no loop to pipeline, while the backward
# kernels walk the channel dimension in a loop, which is what Triton
# pipelines. Sweeping stages on the forward pass would only multiply the
# autotuning cold start over identical code.
def _generate_fwd_configs():
    configs = []
    for e in [16, 32, 64]:
        for c in [16, 32, 64, 128]:
            for w in [4, 8]:
                configs.append(
                    triton.Config(
                        {"BLOCK_E": e, "BLOCK_C": c}, num_warps=w, num_stages=1
                    )
                )
    return configs


def _generate_bwd_configs():
    configs = []
    for e in [16, 32, 64]:
        for c in [16, 32, 64, 128]:
            for w in [4, 8]:
                for s in [1, 2, 3, 4]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_E": e, "BLOCK_C": c}, num_warps=w, num_stages=s
                        )
                    )
    return configs


# =========================================================================
# FORWARD KERNEL
# =========================================================================
# Keyed on H as well as C now that the radial epilogue lives here: the two
# are independent model knobs, and the winning tile shape depends on both.
@triton.autotune(cache_results=True, configs=_generate_fwd_configs(), key=["C"])
@triton.jit
def _gather_split_fwd_kernel(
    x_ptr,
    idx_i_ptr,
    idx_j_ptr,
    wig_ptr,
    rad_ptr,
    X_m0_ptr,
    X_m1_ptr,
    X_m2_ptr,
    E,
    N,
    C,
    stride_xn,
    stride_xl,
    stride_xc,
    stride_wig_e,
    stride_wig_k,
    stride_rad_e,
    stride_rad_c,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)

    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    e_mask = e_offs < E
    c_mask = c_offs < C
    mask = e_mask[:, None] & c_mask[None, :]

    i = tl.load(idx_i_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)

    wb = e_offs * stride_wig_e
    rb = e_offs[:, None] * stride_rad_e + c_offs[None, :] * stride_rad_c

    # Load L=0 and L=1 Node Features
    # (2.1) `csd_emb` is already folded into the l=0 channel by the caller.
    vi0 = tl.load(
        x_ptr + i[:, None] * stride_xn + 0 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vi1 = tl.load(
        x_ptr + i[:, None] * stride_xn + 1 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vi2 = tl.load(
        x_ptr + i[:, None] * stride_xn + 2 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vi3 = tl.load(
        x_ptr + i[:, None] * stride_xn + 3 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )

    vj0 = tl.load(
        x_ptr + j[:, None] * stride_xn + 0 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vj1 = tl.load(
        x_ptr + j[:, None] * stride_xn + 1 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vj2 = tl.load(
        x_ptr + j[:, None] * stride_xn + 2 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vj3 = tl.load(
        x_ptr + j[:, None] * stride_xn + 3 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )

    w11 = tl.load(wig_ptr + wb + W11 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w12 = tl.load(wig_ptr + wb + W12 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w13 = tl.load(wig_ptr + wb + W13 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w21 = tl.load(wig_ptr + wb + W21 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w22 = tl.load(wig_ptr + wb + W22 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w23 = tl.load(wig_ptr + wb + W23 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w31 = tl.load(wig_ptr + wb + W31 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w32 = tl.load(wig_ptr + wb + W32 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w33 = tl.load(wig_ptr + wb + W33 * stride_wig_k, mask=e_mask, other=0.0)[:, None]

    # L=1 -> M=1
    cs1 = w21 * vi1 + w22 * vi2 + w23 * vi3
    ts1 = w21 * vj1 + w22 * vj2 + w23 * vj3
    cs3 = w31 * vi1 + w32 * vi2 + w33 * vi3
    ts3 = w31 * vj1 + w32 * vj2 + w33 * vj3
    cs5 = w11 * vi1 + w12 * vi2 + w13 * vi3
    ts5 = w11 * vj1 + w12 * vj2 + w13 * vj3

    # Load L=2 Node Features
    vi4 = tl.load(
        x_ptr + i[:, None] * stride_xn + 4 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vi5 = tl.load(
        x_ptr + i[:, None] * stride_xn + 5 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vi6 = tl.load(
        x_ptr + i[:, None] * stride_xn + 6 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vi7 = tl.load(
        x_ptr + i[:, None] * stride_xn + 7 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vi8 = tl.load(
        x_ptr + i[:, None] * stride_xn + 8 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )

    vj4 = tl.load(
        x_ptr + j[:, None] * stride_xn + 4 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vj5 = tl.load(
        x_ptr + j[:, None] * stride_xn + 5 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vj6 = tl.load(
        x_ptr + j[:, None] * stride_xn + 6 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vj7 = tl.load(
        x_ptr + j[:, None] * stride_xn + 7 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )
    vj8 = tl.load(
        x_ptr + j[:, None] * stride_xn + 8 * stride_xl + c_offs[None, :] * stride_xc,
        mask=mask,
        other=0.0,
    )

    w44 = tl.load(wig_ptr + wb + W44 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w45 = tl.load(wig_ptr + wb + W45 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w46 = tl.load(wig_ptr + wb + W46 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w47 = tl.load(wig_ptr + wb + W47 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w48 = tl.load(wig_ptr + wb + W48 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w54 = tl.load(wig_ptr + wb + W54 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w55 = tl.load(wig_ptr + wb + W55 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w56 = tl.load(wig_ptr + wb + W56 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w57 = tl.load(wig_ptr + wb + W57 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w58 = tl.load(wig_ptr + wb + W58 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w64 = tl.load(wig_ptr + wb + W64 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w65 = tl.load(wig_ptr + wb + W65 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w66 = tl.load(wig_ptr + wb + W66 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w67 = tl.load(wig_ptr + wb + W67 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w68 = tl.load(wig_ptr + wb + W68 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w74 = tl.load(wig_ptr + wb + W74 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w75 = tl.load(wig_ptr + wb + W75 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w76 = tl.load(wig_ptr + wb + W76 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w77 = tl.load(wig_ptr + wb + W77 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w78 = tl.load(wig_ptr + wb + W78 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w84 = tl.load(wig_ptr + wb + W84 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w85 = tl.load(wig_ptr + wb + W85 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w86 = tl.load(wig_ptr + wb + W86 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w87 = tl.load(wig_ptr + wb + W87 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w88 = tl.load(wig_ptr + wb + W88 * stride_wig_k, mask=e_mask, other=0.0)[:, None]

    # L=2 -> M=2
    cs2 = w64 * vi4 + w65 * vi5 + w66 * vi6 + w67 * vi7 + w68 * vi8
    ts2 = w64 * vj4 + w65 * vj5 + w66 * vj6 + w67 * vj7 + w68 * vj8
    cs4 = w74 * vi4 + w75 * vi5 + w76 * vi6 + w77 * vi7 + w78 * vi8
    ts4 = w74 * vj4 + w75 * vj5 + w76 * vj6 + w77 * vj7 + w78 * vj8
    cs6 = w54 * vi4 + w55 * vi5 + w56 * vi6 + w57 * vi7 + w58 * vi8
    ts6 = w54 * vj4 + w55 * vj5 + w56 * vj6 + w57 * vj7 + w58 * vj8
    cs7 = w84 * vi4 + w85 * vi5 + w86 * vi6 + w87 * vi7 + w88 * vi8
    ts7 = w84 * vj4 + w85 * vj5 + w86 * vj6 + w87 * vj7 + w88 * vj8
    cs8 = w44 * vi4 + w45 * vi5 + w46 * vi6 + w47 * vi7 + w48 * vi8
    ts8 = w44 * vj4 + w45 * vj5 + w46 * vj6 + w47 * vj7 + w48 * vj8

    # Load Radial Outputs
    r0 = tl.load(rad_ptr + rb + 0 * C * stride_rad_c, mask=mask, other=0.0)
    r1 = tl.load(rad_ptr + rb + 1 * C * stride_rad_c, mask=mask, other=0.0)
    r2 = tl.load(rad_ptr + rb + 2 * C * stride_rad_c, mask=mask, other=0.0)
    r3 = tl.load(rad_ptr + rb + 3 * C * stride_rad_c, mask=mask, other=0.0)
    r4 = tl.load(rad_ptr + rb + 4 * C * stride_rad_c, mask=mask, other=0.0)
    r5 = tl.load(rad_ptr + rb + 5 * C * stride_rad_c, mask=mask, other=0.0)
    r6 = tl.load(rad_ptr + rb + 6 * C * stride_rad_c, mask=mask, other=0.0)
    r7 = tl.load(rad_ptr + rb + 7 * C * stride_rad_c, mask=mask, other=0.0)
    r8 = tl.load(rad_ptr + rb + 8 * C * stride_rad_c, mask=mask, other=0.0)
    r9 = tl.load(rad_ptr + rb + 9 * C * stride_rad_c, mask=mask, other=0.0)
    r10 = tl.load(rad_ptr + rb + 10 * C * stride_rad_c, mask=mask, other=0.0)
    r11 = tl.load(rad_ptr + rb + 11 * C * stride_rad_c, mask=mask, other=0.0)

    # Store directly into the 3 dense matrices
    base_m0 = e_offs[:, None] * X_M0_MULT * C + c_offs[None, :]
    base_m1 = e_offs[:, None] * X_M1_MULT * C + c_offs[None, :]
    base_m2 = e_offs[:, None] * X_M2_MULT * C + c_offs[None, :]

    # M=0
    tl.store(X_m0_ptr + base_m0 + 0 * C, vi0 * r0, mask=mask)
    tl.store(X_m0_ptr + base_m0 + 1 * C, vj0 * r1, mask=mask)
    tl.store(X_m0_ptr + base_m0 + 2 * C, cs1 * r2, mask=mask)
    tl.store(X_m0_ptr + base_m0 + 3 * C, ts1 * r3, mask=mask)
    tl.store(X_m0_ptr + base_m0 + 4 * C, cs2 * r4, mask=mask)
    tl.store(X_m0_ptr + base_m0 + 5 * C, ts2 * r5, mask=mask)

    # M=1 (Real & Imag)
    tl.store(X_m1_ptr + base_m1 + 0 * C, cs3 * r6, mask=mask)
    tl.store(X_m1_ptr + base_m1 + 1 * C, ts3 * r7, mask=mask)
    tl.store(X_m1_ptr + base_m1 + 2 * C, cs4 * r8, mask=mask)
    tl.store(X_m1_ptr + base_m1 + 3 * C, ts4 * r9, mask=mask)
    tl.store(X_m1_ptr + base_m1 + 4 * C, cs5 * r6, mask=mask)
    tl.store(X_m1_ptr + base_m1 + 5 * C, ts5 * r7, mask=mask)
    tl.store(X_m1_ptr + base_m1 + 6 * C, cs6 * r8, mask=mask)
    tl.store(X_m1_ptr + base_m1 + 7 * C, ts6 * r9, mask=mask)

    # M=2 (Real & Imag)
    tl.store(X_m2_ptr + base_m2 + 0 * C, cs7 * r10, mask=mask)
    tl.store(X_m2_ptr + base_m2 + 1 * C, ts7 * r11, mask=mask)
    tl.store(X_m2_ptr + base_m2 + 2 * C, cs8 * r10, mask=mask)
    tl.store(X_m2_ptr + base_m2 + 3 * C, ts8 * r11, mask=mask)


# =========================================================================
# BACKWARD KERNEL 1: L=1 Split
# =========================================================================
@triton.autotune(
    cache_results=True,
    configs=_generate_bwd_configs(),
    key=["C"],
    reset_to_zero=["g_x_ptr"],
)
@triton.jit
def _gather_split_bwd_l1_kernel(
    g_X_m0_ptr,
    g_X_m1_ptr,
    g_X_m2_ptr,
    x_ptr,
    idx_i_ptr,
    idx_j_ptr,
    wig_ptr,
    rad_ptr,
    g_x_ptr,
    g_wig_ptr,
    g_rad_ptr,
    E,
    N,
    C,
    stride_xn,
    stride_xl,
    stride_xc,
    stride_wig_e,
    stride_wig_k,
    stride_rad_e,
    stride_rad_c,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    i = tl.load(idx_i_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    wb = e_offs * stride_wig_e

    w11 = tl.load(wig_ptr + wb + W11 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w12 = tl.load(wig_ptr + wb + W12 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w13 = tl.load(wig_ptr + wb + W13 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w21 = tl.load(wig_ptr + wb + W21 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w22 = tl.load(wig_ptr + wb + W22 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w23 = tl.load(wig_ptr + wb + W23 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w31 = tl.load(wig_ptr + wb + W31 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w32 = tl.load(wig_ptr + wb + W32 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w33 = tl.load(wig_ptr + wb + W33 * stride_wig_k, mask=e_mask, other=0.0)[:, None]

    gw11_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw12_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw13_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw21_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw22_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw23_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw31_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw32_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw33_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        mask = e_mask[:, None] & c_mask[None, :]

        base_m0 = e_offs[:, None] * X_M0_MULT * C + c_offs[None, :]
        base_m1 = e_offs[:, None] * X_M1_MULT * C + c_offs[None, :]

        gx_0 = tl.load(g_X_m0_ptr + base_m0 + 0 * C, mask=mask, other=0.0)
        gx_1 = tl.load(g_X_m0_ptr + base_m0 + 1 * C, mask=mask, other=0.0)
        gx_2 = tl.load(g_X_m0_ptr + base_m0 + 2 * C, mask=mask, other=0.0)
        gx_3 = tl.load(g_X_m0_ptr + base_m0 + 3 * C, mask=mask, other=0.0)

        gx_6 = tl.load(g_X_m1_ptr + base_m1 + 0 * C, mask=mask, other=0.0)
        gx_7 = tl.load(g_X_m1_ptr + base_m1 + 1 * C, mask=mask, other=0.0)
        gx_10 = tl.load(g_X_m1_ptr + base_m1 + 4 * C, mask=mask, other=0.0)
        gx_11 = tl.load(g_X_m1_ptr + base_m1 + 5 * C, mask=mask, other=0.0)

        vi0 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 0 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vi1 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 1 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vi2 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 2 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vi3 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 3 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj0 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 0 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj1 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 1 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj2 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 2 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj3 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 3 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )

        rb = e_offs[:, None] * stride_rad_e + c_offs[None, :] * stride_rad_c
        r0 = tl.load(rad_ptr + rb + 0 * C * stride_rad_c, mask=mask, other=0.0)
        r1 = tl.load(rad_ptr + rb + 1 * C * stride_rad_c, mask=mask, other=0.0)
        r2 = tl.load(rad_ptr + rb + 2 * C * stride_rad_c, mask=mask, other=0.0)
        r3 = tl.load(rad_ptr + rb + 3 * C * stride_rad_c, mask=mask, other=0.0)
        r6 = tl.load(rad_ptr + rb + 6 * C * stride_rad_c, mask=mask, other=0.0)
        r7 = tl.load(rad_ptr + rb + 7 * C * stride_rad_c, mask=mask, other=0.0)

        cs1 = w21 * vi1 + w22 * vi2 + w23 * vi3
        ts1 = w21 * vj1 + w22 * vj2 + w23 * vj3
        cs3 = w31 * vi1 + w32 * vi2 + w33 * vi3
        ts3 = w31 * vj1 + w32 * vj2 + w33 * vj3
        cs5 = w11 * vi1 + w12 * vi2 + w13 * vi3
        ts5 = w11 * vj1 + w12 * vj2 + w13 * vj3

        tl.store(g_rad_ptr + rb + 0 * C * stride_rad_c, gx_0 * vi0, mask=mask)
        tl.store(g_rad_ptr + rb + 1 * C * stride_rad_c, gx_1 * vj0, mask=mask)
        tl.store(g_rad_ptr + rb + 2 * C * stride_rad_c, gx_2 * cs1, mask=mask)
        tl.store(g_rad_ptr + rb + 3 * C * stride_rad_c, gx_3 * ts1, mask=mask)
        tl.store(
            g_rad_ptr + rb + 6 * C * stride_rad_c, gx_6 * cs3 + gx_10 * cs5, mask=mask
        )
        tl.store(
            g_rad_ptr + rb + 7 * C * stride_rad_c, gx_7 * ts3 + gx_11 * ts5, mask=mask
        )

        gcs1 = gx_2 * r2
        gts1 = gx_3 * r3
        gcs3 = gx_6 * r6
        gts3 = gx_7 * r7
        gcs5 = gx_10 * r6
        gts5 = gx_11 * r7

        gvi0 = gx_0 * r0
        gvj0 = gx_1 * r1
        gvi1 = gcs1 * w21 + gcs3 * w31 + gcs5 * w11
        gvj1 = gts1 * w21 + gts3 * w31 + gts5 * w11
        gvi2 = gcs1 * w22 + gcs3 * w32 + gcs5 * w12
        gvj2 = gts1 * w22 + gts3 * w32 + gts5 * w12
        gvi3 = gcs1 * w23 + gcs3 * w33 + gcs5 * w13
        gvj3 = gts1 * w23 + gts3 * w33 + gts5 * w13

        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 0 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi0,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 1 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi1,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 2 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi2,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 3 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi3,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 0 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj0,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 1 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj1,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 2 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj2,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 3 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj3,
            mask=mask,
        )

        gw11_acc += tl.sum(gcs5 * vi1 + gts5 * vj1, axis=1)
        gw12_acc += tl.sum(gcs5 * vi2 + gts5 * vj2, axis=1)
        gw13_acc += tl.sum(gcs5 * vi3 + gts5 * vj3, axis=1)
        gw21_acc += tl.sum(gcs1 * vi1 + gts1 * vj1, axis=1)
        gw22_acc += tl.sum(gcs1 * vi2 + gts1 * vj2, axis=1)
        gw23_acc += tl.sum(gcs1 * vi3 + gts1 * vj3, axis=1)
        gw31_acc += tl.sum(gcs3 * vi1 + gts3 * vj1, axis=1)
        gw32_acc += tl.sum(gcs3 * vi2 + gts3 * vj2, axis=1)
        gw33_acc += tl.sum(gcs3 * vi3 + gts3 * vj3, axis=1)

    tl.store(g_wig_ptr + wb + W11 * stride_wig_k, gw11_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W12 * stride_wig_k, gw12_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W13 * stride_wig_k, gw13_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W21 * stride_wig_k, gw21_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W22 * stride_wig_k, gw22_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W23 * stride_wig_k, gw23_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W31 * stride_wig_k, gw31_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W32 * stride_wig_k, gw32_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W33 * stride_wig_k, gw33_acc, mask=e_mask)


# =========================================================================
# BACKWARD KERNEL 2: L=2 Split
# =========================================================================
@triton.autotune(
    cache_results=True,
    configs=_generate_bwd_configs(),
    key=["C"],
    reset_to_zero=["g_x_ptr"],
)
@triton.jit
def _gather_split_bwd_l2_kernel(
    g_X_m0_ptr,
    g_X_m1_ptr,
    g_X_m2_ptr,
    x_ptr,
    idx_i_ptr,
    idx_j_ptr,
    wig_ptr,
    rad_ptr,
    g_x_ptr,
    g_wig_ptr,
    g_rad_ptr,
    E,
    N,
    C,
    stride_xn,
    stride_xl,
    stride_xc,
    stride_wig_e,
    stride_wig_k,
    stride_rad_e,
    stride_rad_c,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    i = tl.load(idx_i_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    wb = e_offs * stride_wig_e

    w44 = tl.load(wig_ptr + wb + W44 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w45 = tl.load(wig_ptr + wb + W45 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w46 = tl.load(wig_ptr + wb + W46 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w47 = tl.load(wig_ptr + wb + W47 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w48 = tl.load(wig_ptr + wb + W48 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w54 = tl.load(wig_ptr + wb + W54 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w55 = tl.load(wig_ptr + wb + W55 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w56 = tl.load(wig_ptr + wb + W56 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w57 = tl.load(wig_ptr + wb + W57 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w58 = tl.load(wig_ptr + wb + W58 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w64 = tl.load(wig_ptr + wb + W64 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w65 = tl.load(wig_ptr + wb + W65 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w66 = tl.load(wig_ptr + wb + W66 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w67 = tl.load(wig_ptr + wb + W67 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w68 = tl.load(wig_ptr + wb + W68 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w74 = tl.load(wig_ptr + wb + W74 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w75 = tl.load(wig_ptr + wb + W75 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w76 = tl.load(wig_ptr + wb + W76 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w77 = tl.load(wig_ptr + wb + W77 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w78 = tl.load(wig_ptr + wb + W78 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w84 = tl.load(wig_ptr + wb + W84 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w85 = tl.load(wig_ptr + wb + W85 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w86 = tl.load(wig_ptr + wb + W86 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w87 = tl.load(wig_ptr + wb + W87 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w88 = tl.load(wig_ptr + wb + W88 * stride_wig_k, mask=e_mask, other=0.0)[:, None]

    gw44_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw45_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw46_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw47_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw48_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw54_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw55_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw56_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw57_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw58_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw64_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw65_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw66_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw67_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw68_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw74_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw75_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw76_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw77_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw78_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw84_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw85_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw86_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw87_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw88_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        mask = e_mask[:, None] & c_mask[None, :]

        base_m0 = e_offs[:, None] * X_M0_MULT * C + c_offs[None, :]
        base_m1 = e_offs[:, None] * X_M1_MULT * C + c_offs[None, :]
        base_m2 = e_offs[:, None] * X_M2_MULT * C + c_offs[None, :]

        gx_4 = tl.load(g_X_m0_ptr + base_m0 + 4 * C, mask=mask, other=0.0)
        gx_5 = tl.load(g_X_m0_ptr + base_m0 + 5 * C, mask=mask, other=0.0)

        gx_8 = tl.load(g_X_m1_ptr + base_m1 + 2 * C, mask=mask, other=0.0)
        gx_9 = tl.load(g_X_m1_ptr + base_m1 + 3 * C, mask=mask, other=0.0)
        gx_12 = tl.load(g_X_m1_ptr + base_m1 + 6 * C, mask=mask, other=0.0)
        gx_13 = tl.load(g_X_m1_ptr + base_m1 + 7 * C, mask=mask, other=0.0)

        gx_14 = tl.load(g_X_m2_ptr + base_m2 + 0 * C, mask=mask, other=0.0)
        gx_15 = tl.load(g_X_m2_ptr + base_m2 + 1 * C, mask=mask, other=0.0)
        gx_16 = tl.load(g_X_m2_ptr + base_m2 + 2 * C, mask=mask, other=0.0)
        gx_17 = tl.load(g_X_m2_ptr + base_m2 + 3 * C, mask=mask, other=0.0)

        vi4 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 4 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vi5 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 5 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vi6 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 6 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vi7 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 7 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vi8 = tl.load(
            x_ptr
            + i[:, None] * stride_xn
            + 8 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj4 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 4 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj5 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 5 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj6 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 6 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj7 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 7 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )
        vj8 = tl.load(
            x_ptr
            + j[:, None] * stride_xn
            + 8 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=mask,
            other=0.0,
        )

        rb = e_offs[:, None] * stride_rad_e + c_offs[None, :] * stride_rad_c
        r4 = tl.load(rad_ptr + rb + 4 * C * stride_rad_c, mask=mask, other=0.0)
        r5 = tl.load(rad_ptr + rb + 5 * C * stride_rad_c, mask=mask, other=0.0)
        r8 = tl.load(rad_ptr + rb + 8 * C * stride_rad_c, mask=mask, other=0.0)
        r9 = tl.load(rad_ptr + rb + 9 * C * stride_rad_c, mask=mask, other=0.0)
        r10 = tl.load(rad_ptr + rb + 10 * C * stride_rad_c, mask=mask, other=0.0)
        r11 = tl.load(rad_ptr + rb + 11 * C * stride_rad_c, mask=mask, other=0.0)

        cs2 = w64 * vi4 + w65 * vi5 + w66 * vi6 + w67 * vi7 + w68 * vi8
        ts2 = w64 * vj4 + w65 * vj5 + w66 * vj6 + w67 * vj7 + w68 * vj8
        cs4 = w74 * vi4 + w75 * vi5 + w76 * vi6 + w77 * vi7 + w78 * vi8
        ts4 = w74 * vj4 + w75 * vj5 + w76 * vj6 + w77 * vj7 + w78 * vj8
        cs6 = w54 * vi4 + w55 * vi5 + w56 * vi6 + w57 * vi7 + w58 * vi8
        ts6 = w54 * vj4 + w55 * vj5 + w56 * vj6 + w57 * vj7 + w58 * vj8
        cs7 = w84 * vi4 + w85 * vi5 + w86 * vi6 + w87 * vi7 + w88 * vi8
        ts7 = w84 * vj4 + w85 * vj5 + w86 * vj6 + w87 * vj7 + w88 * vj8
        cs8 = w44 * vi4 + w45 * vi5 + w46 * vi6 + w47 * vi7 + w48 * vi8
        ts8 = w44 * vj4 + w45 * vj5 + w46 * vj6 + w47 * vj7 + w48 * vj8

        tl.store(g_rad_ptr + rb + 4 * C * stride_rad_c, gx_4 * cs2, mask=mask)
        tl.store(g_rad_ptr + rb + 5 * C * stride_rad_c, gx_5 * ts2, mask=mask)
        tl.store(
            g_rad_ptr + rb + 8 * C * stride_rad_c, gx_8 * cs4 + gx_12 * cs6, mask=mask
        )
        tl.store(
            g_rad_ptr + rb + 9 * C * stride_rad_c, gx_9 * ts4 + gx_13 * ts6, mask=mask
        )
        tl.store(
            g_rad_ptr + rb + 10 * C * stride_rad_c, gx_14 * cs7 + gx_16 * cs8, mask=mask
        )
        tl.store(
            g_rad_ptr + rb + 11 * C * stride_rad_c, gx_15 * ts7 + gx_17 * ts8, mask=mask
        )

        gcs2 = gx_4 * r4
        gts2 = gx_5 * r5
        gcs4 = gx_8 * r8
        gts4 = gx_9 * r9
        gcs6 = gx_12 * r8
        gts6 = gx_13 * r9
        gcs7 = gx_14 * r10
        gts7 = gx_15 * r11
        gcs8 = gx_16 * r10
        gts8 = gx_17 * r11

        gvi4 = gcs2 * w64 + gcs4 * w74 + gcs6 * w54 + gcs7 * w84 + gcs8 * w44
        gvj4 = gts2 * w64 + gts4 * w74 + gts6 * w54 + gts7 * w84 + gts8 * w44
        gvi5 = gcs2 * w65 + gcs4 * w75 + gcs6 * w55 + gcs7 * w85 + gcs8 * w45
        gvj5 = gts2 * w65 + gts4 * w75 + gts6 * w55 + gts7 * w85 + gts8 * w45
        gvi6 = gcs2 * w66 + gcs4 * w76 + gcs6 * w56 + gcs7 * w86 + gcs8 * w46
        gvj6 = gts2 * w66 + gts4 * w76 + gts6 * w56 + gts7 * w86 + gts8 * w46
        gvi7 = gcs2 * w67 + gcs4 * w77 + gcs6 * w57 + gcs7 * w87 + gcs8 * w47
        gvj7 = gts2 * w67 + gts4 * w77 + gts6 * w57 + gts7 * w87 + gts8 * w47
        gvi8 = gcs2 * w68 + gcs4 * w78 + gcs6 * w58 + gcs7 * w88 + gcs8 * w48
        gvj8 = gts2 * w68 + gts4 * w78 + gts6 * w58 + gts7 * w88 + gts8 * w48

        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 4 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi4,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 5 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi5,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 6 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi6,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 7 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi7,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + i[:, None] * stride_xn
            + 8 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvi8,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 4 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj4,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 5 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj5,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 6 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj6,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 7 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj7,
            mask=mask,
        )
        tl.atomic_add(
            g_x_ptr
            + j[:, None] * stride_xn
            + 8 * stride_xl
            + c_offs[None, :] * stride_xc,
            gvj8,
            mask=mask,
        )

        gw44_acc += tl.sum(gcs8 * vi4 + gts8 * vj4, axis=1)
        gw45_acc += tl.sum(gcs8 * vi5 + gts8 * vj5, axis=1)
        gw46_acc += tl.sum(gcs8 * vi6 + gts8 * vj6, axis=1)
        gw47_acc += tl.sum(gcs8 * vi7 + gts8 * vj7, axis=1)
        gw48_acc += tl.sum(gcs8 * vi8 + gts8 * vj8, axis=1)
        gw54_acc += tl.sum(gcs6 * vi4 + gts6 * vj4, axis=1)
        gw55_acc += tl.sum(gcs6 * vi5 + gts6 * vj5, axis=1)
        gw56_acc += tl.sum(gcs6 * vi6 + gts6 * vj6, axis=1)
        gw57_acc += tl.sum(gcs6 * vi7 + gts6 * vj7, axis=1)
        gw58_acc += tl.sum(gcs6 * vi8 + gts6 * vj8, axis=1)
        gw64_acc += tl.sum(gcs2 * vi4 + gts2 * vj4, axis=1)
        gw65_acc += tl.sum(gcs2 * vi5 + gts2 * vj5, axis=1)
        gw66_acc += tl.sum(gcs2 * vi6 + gts2 * vj6, axis=1)
        gw67_acc += tl.sum(gcs2 * vi7 + gts2 * vj7, axis=1)
        gw68_acc += tl.sum(gcs2 * vi8 + gts2 * vj8, axis=1)
        gw74_acc += tl.sum(gcs4 * vi4 + gts4 * vj4, axis=1)
        gw75_acc += tl.sum(gcs4 * vi5 + gts4 * vj5, axis=1)
        gw76_acc += tl.sum(gcs4 * vi6 + gts4 * vj6, axis=1)
        gw77_acc += tl.sum(gcs4 * vi7 + gts4 * vj7, axis=1)
        gw78_acc += tl.sum(gcs4 * vi8 + gts4 * vj8, axis=1)
        gw84_acc += tl.sum(gcs7 * vi4 + gts7 * vj4, axis=1)
        gw85_acc += tl.sum(gcs7 * vi5 + gts7 * vj5, axis=1)
        gw86_acc += tl.sum(gcs7 * vi6 + gts7 * vj6, axis=1)
        gw87_acc += tl.sum(gcs7 * vi7 + gts7 * vj7, axis=1)
        gw88_acc += tl.sum(gcs7 * vi8 + gts7 * vj8, axis=1)

    tl.store(g_wig_ptr + wb + W44 * stride_wig_k, gw44_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W45 * stride_wig_k, gw45_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W46 * stride_wig_k, gw46_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W47 * stride_wig_k, gw47_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W48 * stride_wig_k, gw48_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W54 * stride_wig_k, gw54_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W55 * stride_wig_k, gw55_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W56 * stride_wig_k, gw56_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W57 * stride_wig_k, gw57_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W58 * stride_wig_k, gw58_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W64 * stride_wig_k, gw64_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W65 * stride_wig_k, gw65_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W66 * stride_wig_k, gw66_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W67 * stride_wig_k, gw67_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W68 * stride_wig_k, gw68_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W74 * stride_wig_k, gw74_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W75 * stride_wig_k, gw75_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W76 * stride_wig_k, gw76_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W77 * stride_wig_k, gw77_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W78 * stride_wig_k, gw78_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W84 * stride_wig_k, gw84_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W85 * stride_wig_k, gw85_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W86 * stride_wig_k, gw86_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W87 * stride_wig_k, gw87_acc, mask=e_mask)
    tl.store(g_wig_ptr + wb + W88 * stride_wig_k, gw88_acc, mask=e_mask)
