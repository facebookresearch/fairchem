"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

M->L Wigner rotation fused with the edge-to-node scatter.

Covers both the per-layer message scatter and the edge-degree
initialisation. Both take ``scatter_target``: the destination row for each
edge, already remapped to this rank's local numbering by upstream's
``_generate_graph``. That indirection is what lets a rank holding a slice of
the nodes under graph parallelism scatter into its own rows, and it handles
non-contiguous partitions that a scalar offset could not express.
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
    Z_M0_MULT,
    Z_M1_MULT,
    Z_M2_MULT,
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


def _generate_configs():
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
# PHASE 2 FORWARD: COO Edge-Parallel Scatter (FP32 Atomics)
# =========================================================================
# ``restore_value`` rather than ``reset_to_zero``: the caller seeds this buffer
# with the block's residual and the kernel accumulates on top, which saves an
# [N, 9, C] memset and a full-tensor add per layer. Zeroing between autotune
# trials would destroy the residual; restoring it does not. The snapshot costs
# one clone per trial, during tuning only -- the cached path never calls the
# hook (see Autotuner.run).
@triton.autotune(
    cache_results=True,
    configs=_generate_fwd_configs(),
    key=["C"],
    restore_value=["x_out_ptr"],
)
@triton.jit
def _scatter_split_fwd_kernel(
    scatter_target_ptr,
    Z_m0_ptr,
    Z_m1_ptr,
    Z_m2_ptr,
    wig_ptr,
    env_ptr,
    x_out_ptr,
    E,
    C,
    stride_xn,
    stride_xl,
    stride_xc,
    stride_wig_e,
    stride_wig_k,
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

    # Already this rank's local row for each edge, so no remapping here.
    j = tl.load(scatter_target_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)[:, None]
    wb = e_offs * stride_wig_e

    # Load Wigner Scalars
    w11 = tl.load(wig_ptr + wb + W11 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w12 = tl.load(wig_ptr + wb + W12 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w13 = tl.load(wig_ptr + wb + W13 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w21 = tl.load(wig_ptr + wb + W21 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w22 = tl.load(wig_ptr + wb + W22 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w23 = tl.load(wig_ptr + wb + W23 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w31 = tl.load(wig_ptr + wb + W31 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w32 = tl.load(wig_ptr + wb + W32 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w33 = tl.load(wig_ptr + wb + W33 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
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

    # Base Offsets
    z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
    z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]
    z_base_m2 = e_offs[:, None] * Z_M2_MULT * C + c_offs[None, :]

    # Load Z components
    Z0_0 = tl.load(Z_m0_ptr + z_base_m0 + 0 * C, mask=mask, other=0.0)
    Z0_1 = tl.load(Z_m0_ptr + z_base_m0 + 1 * C, mask=mask, other=0.0)
    Z0_2 = tl.load(Z_m0_ptr + z_base_m0 + 2 * C, mask=mask, other=0.0)

    Z1R_0 = tl.load(Z_m1_ptr + z_base_m1 + 0 * C, mask=mask, other=0.0)
    Z1R_1 = tl.load(Z_m1_ptr + z_base_m1 + 1 * C, mask=mask, other=0.0)
    Z1I_0 = tl.load(Z_m1_ptr + z_base_m1 + 2 * C, mask=mask, other=0.0)
    Z1I_1 = tl.load(Z_m1_ptr + z_base_m1 + 3 * C, mask=mask, other=0.0)

    Z2R_0 = tl.load(Z_m2_ptr + z_base_m2 + 0 * C, mask=mask, other=0.0)
    Z2I_0 = tl.load(Z_m2_ptr + z_base_m2 + 1 * C, mask=mask, other=0.0)

    # M -> L Inverse Rotation (FP32)
    v0 = Z0_0 * env
    v1_1 = (w21 * Z0_1 + w11 * Z1I_0 + w31 * Z1R_0) * env
    v1_2 = (w22 * Z0_1 + w12 * Z1I_0 + w32 * Z1R_0) * env
    v1_3 = (w23 * Z0_1 + w13 * Z1I_0 + w33 * Z1R_0) * env

    v2_4 = (w64 * Z0_2 + w54 * Z1I_1 + w74 * Z1R_1 + w44 * Z2I_0 + w84 * Z2R_0) * env
    v2_5 = (w65 * Z0_2 + w55 * Z1I_1 + w75 * Z1R_1 + w45 * Z2I_0 + w85 * Z2R_0) * env
    v2_6 = (w66 * Z0_2 + w56 * Z1I_1 + w76 * Z1R_1 + w46 * Z2I_0 + w86 * Z2R_0) * env
    v2_7 = (w67 * Z0_2 + w57 * Z1I_1 + w77 * Z1R_1 + w47 * Z2I_0 + w87 * Z2R_0) * env
    v2_8 = (w68 * Z0_2 + w58 * Z1I_1 + w78 * Z1R_1 + w48 * Z2I_0 + w88 * Z2R_0) * env

    # Atomic Add to target node
    out_base = j[:, None] * stride_xn + c_offs[None, :] * stride_xc
    tl.atomic_add(x_out_ptr + out_base + 0 * stride_xl, v0, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 1 * stride_xl, v1_1, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 2 * stride_xl, v1_2, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 3 * stride_xl, v1_3, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 4 * stride_xl, v2_4, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 5 * stride_xl, v2_5, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 6 * stride_xl, v2_6, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 7 * stride_xl, v2_7, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 8 * stride_xl, v2_8, mask=mask)


# =========================================================================
# PHASE 2 BACKWARD: Split L1 and L2 (Unaffected logically, just config)
# =========================================================================
@triton.autotune(cache_results=True, configs=_generate_configs(), key=["C"])
@triton.jit
def _scatter_split_bwd_l1_kernel(
    g_out_ptr,
    scatter_target_ptr,
    Z_m0_ptr,
    Z_m1_ptr,
    Z_m2_ptr,
    wig_ptr,
    env_ptr,
    g_z_m0_ptr,
    g_z_m1_ptr,
    g_z_m2_ptr,
    g_wig_ptr,
    g_env_l1_ptr,
    E,
    C,
    stride_xn,
    stride_xl,
    stride_xc,
    stride_wig_e,
    stride_wig_k,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # Already this rank's local row for each edge, so no remapping here.
    j = tl.load(scatter_target_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)
    env_e = env[:, None]
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
    g_env_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        full_mask = e_mask[:, None] & c_mask[None, :]

        g0 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 0 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )
        g1 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 1 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )
        g2 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 2 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )
        g3 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 3 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]

        Z0_0 = tl.load(Z_m0_ptr + z_base_m0 + 0 * C, mask=full_mask, other=0.0)
        Z0_1 = tl.load(Z_m0_ptr + z_base_m0 + 1 * C, mask=full_mask, other=0.0)
        Z1R_0 = tl.load(Z_m1_ptr + z_base_m1 + 0 * C, mask=full_mask, other=0.0)
        Z1I_0 = tl.load(Z_m1_ptr + z_base_m1 + 2 * C, mask=full_mask, other=0.0)

        aL1_1 = g1 * env_e
        aL1_2 = g2 * env_e
        aL1_3 = g3 * env_e

        # Store L1 gradients to M0 and M1 blocks
        tl.store(g_z_m0_ptr + z_base_m0 + 0 * C, g0 * env_e, mask=full_mask)
        tl.store(
            g_z_m0_ptr + z_base_m0 + 1 * C,
            aL1_1 * w21 + aL1_2 * w22 + aL1_3 * w23,
            mask=full_mask,
        )
        tl.store(
            g_z_m1_ptr + z_base_m1 + 0 * C,
            aL1_1 * w31 + aL1_2 * w32 + aL1_3 * w33,
            mask=full_mask,
        )
        tl.store(
            g_z_m1_ptr + z_base_m1 + 2 * C,
            aL1_1 * w11 + aL1_2 * w12 + aL1_3 * w13,
            mask=full_mask,
        )

        gw11_acc += tl.sum(g1 * Z1I_0, axis=1)
        gw12_acc += tl.sum(g2 * Z1I_0, axis=1)
        gw13_acc += tl.sum(g3 * Z1I_0, axis=1)
        gw21_acc += tl.sum(g1 * Z0_1, axis=1)
        gw22_acc += tl.sum(g2 * Z0_1, axis=1)
        gw23_acc += tl.sum(g3 * Z0_1, axis=1)
        gw31_acc += tl.sum(g1 * Z1R_0, axis=1)
        gw32_acc += tl.sum(g2 * Z1R_0, axis=1)
        gw33_acc += tl.sum(g3 * Z1R_0, axis=1)

        g_env_acc += tl.sum(g0 * Z0_0, axis=1) + tl.sum(
            g1 * (w21 * Z0_1 + w11 * Z1I_0 + w31 * Z1R_0)
            + g2 * (w22 * Z0_1 + w12 * Z1I_0 + w32 * Z1R_0)
            + g3 * (w23 * Z0_1 + w13 * Z1I_0 + w33 * Z1R_0),
            axis=1,
        )

    tl.store(g_wig_ptr + wb + W11 * stride_wig_k, gw11_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W12 * stride_wig_k, gw12_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W13 * stride_wig_k, gw13_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W21 * stride_wig_k, gw21_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W22 * stride_wig_k, gw22_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W23 * stride_wig_k, gw23_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W31 * stride_wig_k, gw31_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W32 * stride_wig_k, gw32_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W33 * stride_wig_k, gw33_acc * env, mask=e_mask)
    tl.store(g_env_l1_ptr + e_offs, g_env_acc, mask=e_mask)


@triton.autotune(cache_results=True, configs=_generate_configs(), key=["C"])
@triton.jit
def _scatter_split_bwd_l2_kernel(
    g_out_ptr,
    scatter_target_ptr,
    Z_m0_ptr,
    Z_m1_ptr,
    Z_m2_ptr,
    wig_ptr,
    env_ptr,
    g_z_m0_ptr,
    g_z_m1_ptr,
    g_z_m2_ptr,
    g_wig_ptr,
    g_env_l2_ptr,
    E,
    C,
    stride_xn,
    stride_xl,
    stride_xc,
    stride_wig_e,
    stride_wig_k,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # Already this rank's local row for each edge, so no remapping here.
    j = tl.load(scatter_target_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)
    env_e = env[:, None]
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
    g_env_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        full_mask = e_mask[:, None] & c_mask[None, :]

        g4 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 4 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )
        g5 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 5 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )
        g6 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 6 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )
        g7 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 7 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )
        g8 = tl.load(
            g_out_ptr
            + j[:, None] * stride_xn
            + 8 * stride_xl
            + c_offs[None, :] * stride_xc,
            mask=full_mask,
            other=0.0,
        )

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]
        z_base_m2 = e_offs[:, None] * Z_M2_MULT * C + c_offs[None, :]

        Z0_2 = tl.load(Z_m0_ptr + z_base_m0 + 2 * C, mask=full_mask, other=0.0)
        Z1R_1 = tl.load(Z_m1_ptr + z_base_m1 + 1 * C, mask=full_mask, other=0.0)
        Z1I_1 = tl.load(Z_m1_ptr + z_base_m1 + 3 * C, mask=full_mask, other=0.0)
        Z2R_0 = tl.load(Z_m2_ptr + z_base_m2 + 0 * C, mask=full_mask, other=0.0)
        Z2I_0 = tl.load(Z_m2_ptr + z_base_m2 + 1 * C, mask=full_mask, other=0.0)

        aL2_4 = g4 * env_e
        aL2_5 = g5 * env_e
        aL2_6 = g6 * env_e
        aL2_7 = g7 * env_e
        aL2_8 = g8 * env_e

        tl.store(
            g_z_m0_ptr + z_base_m0 + 2 * C,
            aL2_4 * w64 + aL2_5 * w65 + aL2_6 * w66 + aL2_7 * w67 + aL2_8 * w68,
            mask=full_mask,
        )
        tl.store(
            g_z_m1_ptr + z_base_m1 + 1 * C,
            aL2_4 * w74 + aL2_5 * w75 + aL2_6 * w76 + aL2_7 * w77 + aL2_8 * w78,
            mask=full_mask,
        )
        tl.store(
            g_z_m1_ptr + z_base_m1 + 3 * C,
            aL2_4 * w54 + aL2_5 * w55 + aL2_6 * w56 + aL2_7 * w57 + aL2_8 * w58,
            mask=full_mask,
        )
        tl.store(
            g_z_m2_ptr + z_base_m2 + 0 * C,
            aL2_4 * w84 + aL2_5 * w85 + aL2_6 * w86 + aL2_7 * w87 + aL2_8 * w88,
            mask=full_mask,
        )
        tl.store(
            g_z_m2_ptr + z_base_m2 + 1 * C,
            aL2_4 * w44 + aL2_5 * w45 + aL2_6 * w46 + aL2_7 * w47 + aL2_8 * w48,
            mask=full_mask,
        )

        gw44_acc += tl.sum(g4 * Z2I_0, axis=1)
        gw45_acc += tl.sum(g5 * Z2I_0, axis=1)
        gw46_acc += tl.sum(g6 * Z2I_0, axis=1)
        gw47_acc += tl.sum(g7 * Z2I_0, axis=1)
        gw48_acc += tl.sum(g8 * Z2I_0, axis=1)
        gw54_acc += tl.sum(g4 * Z1I_1, axis=1)
        gw55_acc += tl.sum(g5 * Z1I_1, axis=1)
        gw56_acc += tl.sum(g6 * Z1I_1, axis=1)
        gw57_acc += tl.sum(g7 * Z1I_1, axis=1)
        gw58_acc += tl.sum(g8 * Z1I_1, axis=1)
        gw64_acc += tl.sum(g4 * Z0_2, axis=1)
        gw65_acc += tl.sum(g5 * Z0_2, axis=1)
        gw66_acc += tl.sum(g6 * Z0_2, axis=1)
        gw67_acc += tl.sum(g7 * Z0_2, axis=1)
        gw68_acc += tl.sum(g8 * Z0_2, axis=1)
        gw74_acc += tl.sum(g4 * Z1R_1, axis=1)
        gw75_acc += tl.sum(g5 * Z1R_1, axis=1)
        gw76_acc += tl.sum(g6 * Z1R_1, axis=1)
        gw77_acc += tl.sum(g7 * Z1R_1, axis=1)
        gw78_acc += tl.sum(g8 * Z1R_1, axis=1)
        gw84_acc += tl.sum(g4 * Z2R_0, axis=1)
        gw85_acc += tl.sum(g5 * Z2R_0, axis=1)
        gw86_acc += tl.sum(g6 * Z2R_0, axis=1)
        gw87_acc += tl.sum(g7 * Z2R_0, axis=1)
        gw88_acc += tl.sum(g8 * Z2R_0, axis=1)

        g_env_acc += tl.sum(
            g4 * (w64 * Z0_2 + w54 * Z1I_1 + w74 * Z1R_1 + w44 * Z2I_0 + w84 * Z2R_0)
            + g5 * (w65 * Z0_2 + w55 * Z1I_1 + w75 * Z1R_1 + w45 * Z2I_0 + w85 * Z2R_0)
            + g6 * (w66 * Z0_2 + w56 * Z1I_1 + w76 * Z1R_1 + w46 * Z2I_0 + w86 * Z2R_0)
            + g7 * (w67 * Z0_2 + w57 * Z1I_1 + w77 * Z1R_1 + w47 * Z2I_0 + w87 * Z2R_0)
            + g8 * (w68 * Z0_2 + w58 * Z1I_1 + w78 * Z1R_1 + w48 * Z2I_0 + w88 * Z2R_0),
            axis=1,
        )

    tl.store(g_wig_ptr + wb + W44 * stride_wig_k, gw44_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W45 * stride_wig_k, gw45_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W46 * stride_wig_k, gw46_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W47 * stride_wig_k, gw47_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W48 * stride_wig_k, gw48_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W54 * stride_wig_k, gw54_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W55 * stride_wig_k, gw55_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W56 * stride_wig_k, gw56_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W57 * stride_wig_k, gw57_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W58 * stride_wig_k, gw58_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W64 * stride_wig_k, gw64_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W65 * stride_wig_k, gw65_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W66 * stride_wig_k, gw66_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W67 * stride_wig_k, gw67_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W68 * stride_wig_k, gw68_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W74 * stride_wig_k, gw74_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W75 * stride_wig_k, gw75_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W76 * stride_wig_k, gw76_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W77 * stride_wig_k, gw77_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W78 * stride_wig_k, gw78_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W84 * stride_wig_k, gw84_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W85 * stride_wig_k, gw85_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W86 * stride_wig_k, gw86_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W87 * stride_wig_k, gw87_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W88 * stride_wig_k, gw88_acc * env, mask=e_mask)
    tl.store(g_env_l2_ptr + e_offs, g_env_acc, mask=e_mask)


# =========================================================================
# PHASE 1: Init Node Scatter (Edge-Parallel FP32 Atomics)
# =========================================================================
# (2.3) Node-parallel epilogue: adds the l=0 self-embedding, replacing the
# eager `x_out[:, 0, :] = W_sphere[Z] + csd_emb[batch]` (two gathers, an add and
# a strided assign) with one kernel. It runs *after* the atomic scatter because
# that scatter's target is declared `reset_to_zero`.
#
# Deliberately NOT autotuned. It is a read-modify-write over an N-sized tensor,
# so every autotune trial would re-add the embedding - the same defect as the
# unshielded forward scatters, and it is not idempotent so `reset_to_zero`
# cannot express it either. There is nothing worth tuning in an N*C pointwise
# pass, so a fixed config is both correct and fast.
@triton.jit
def _init_node_l0_add_kernel(
    Z_ptr,
    batch_ptr,
    Wsph_ptr,
    ce_ptr,
    x_out_ptr,
    N,
    C,
    stride_ws_z,
    stride_ws_c,
    stride_ce_b,
    stride_ce_c,
    stride_xn,
    stride_xl,
    stride_xc,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offs = tl.arange(0, BLOCK_C)
    n_mask = n_offs < N
    c_mask = c_offs < C
    mask = n_mask[:, None] & c_mask[None, :]

    z = tl.load(Z_ptr + n_offs, mask=n_mask, other=0).to(tl.int64)
    b = tl.load(batch_ptr + n_offs, mask=n_mask, other=0).to(tl.int64)

    sph = tl.load(
        Wsph_ptr + z[:, None] * stride_ws_z + c_offs[None, :] * stride_ws_c,
        mask=mask,
        other=0.0,
    )
    ce = tl.load(
        ce_ptr + b[:, None] * stride_ce_b + c_offs[None, :] * stride_ce_c,
        mask=mask,
        other=0.0,
    )

    p = (
        x_out_ptr
        + n_offs[:, None] * stride_xn
        + 0 * stride_xl
        + c_offs[None, :] * stride_xc
    )
    cur = tl.load(p, mask=mask, other=0.0)
    tl.store(p, cur + sph + ce, mask=mask)


@triton.autotune(
    cache_results=True,
    configs=_generate_fwd_configs(),
    key=["C"],
    reset_to_zero=["x_out_ptr"],
)
@triton.jit
def _init_scatter_fwd_kernel(
    rad_out_ptr,
    wig_ptr,
    env_ptr,
    scatter_target_ptr,
    x_out_ptr,
    E,
    C,
    inv_rescale,
    stride_rad_e,
    stride_rad_c,
    stride_wm_e,
    stride_wig_k,
    stride_xn,
    stride_xl,
    stride_xc,
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

    # Already this rank's local row for each edge, so no remapping here.
    j = tl.load(scatter_target_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)[:, None]
    wb = e_offs * stride_wm_e

    w21 = tl.load(wig_ptr + wb + W21 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w22 = tl.load(wig_ptr + wb + W22 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w23 = tl.load(wig_ptr + wb + W23 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w64 = tl.load(wig_ptr + wb + W64 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w65 = tl.load(wig_ptr + wb + W65 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w66 = tl.load(wig_ptr + wb + W66 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w67 = tl.load(wig_ptr + wb + W67 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w68 = tl.load(wig_ptr + wb + W68 * stride_wig_k, mask=e_mask, other=0.0)[:, None]

    rb = e_offs[:, None] * stride_rad_e + c_offs[None, :] * stride_rad_c
    r0 = tl.load(rad_out_ptr + rb + 0 * C, mask=mask, other=0.0)
    r1 = tl.load(rad_out_ptr + rb + 1 * C, mask=mask, other=0.0)
    r2 = tl.load(rad_out_ptr + rb + 2 * C, mask=mask, other=0.0)

    v0 = (r0 * env) * inv_rescale
    v1_1 = (r1 * w21 * env) * inv_rescale
    v1_2 = (r1 * w22 * env) * inv_rescale
    v1_3 = (r1 * w23 * env) * inv_rescale

    v2_4 = (r2 * w64 * env) * inv_rescale
    v2_5 = (r2 * w65 * env) * inv_rescale
    v2_6 = (r2 * w66 * env) * inv_rescale
    v2_7 = (r2 * w67 * env) * inv_rescale
    v2_8 = (r2 * w68 * env) * inv_rescale

    # Atomic Add to Target Node
    out_base = j[:, None] * stride_xn + c_offs[None, :] * stride_xc
    tl.atomic_add(x_out_ptr + out_base + 0 * stride_xl, v0, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 1 * stride_xl, v1_1, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 2 * stride_xl, v1_2, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 3 * stride_xl, v1_3, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 4 * stride_xl, v2_4, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 5 * stride_xl, v2_5, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 6 * stride_xl, v2_6, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 7 * stride_xl, v2_7, mask=mask)
    tl.atomic_add(x_out_ptr + out_base + 8 * stride_xl, v2_8, mask=mask)


@triton.autotune(cache_results=True, configs=_generate_configs(), key=["C"])
@triton.jit
def _init_scatter_bwd_kernel(
    g_out_ptr,
    scatter_target_ptr,
    rad_out_ptr,
    wig_ptr,
    env_ptr,
    g_rad_ptr,
    g_wig_ptr,
    g_env_ptr,
    E,
    C,
    inv_rescale,
    stride_xn,
    stride_xl,
    stride_xc,
    stride_rad_e,
    stride_rad_c,
    stride_wm_e,
    stride_wig_k,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # Already this rank's local row for each edge, so no remapping here.
    j = tl.load(scatter_target_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env_1d = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)
    env = env_1d[:, None]
    wb = e_offs * stride_wm_e

    w21 = tl.load(wig_ptr + wb + W21 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w22 = tl.load(wig_ptr + wb + W22 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w23 = tl.load(wig_ptr + wb + W23 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w64 = tl.load(wig_ptr + wb + W64 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w65 = tl.load(wig_ptr + wb + W65 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w66 = tl.load(wig_ptr + wb + W66 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w67 = tl.load(wig_ptr + wb + W67 * stride_wig_k, mask=e_mask, other=0.0)[:, None]
    w68 = tl.load(wig_ptr + wb + W68 * stride_wig_k, mask=e_mask, other=0.0)[:, None]

    gw21_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw22_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw23_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw64_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw65_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw66_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw67_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw68_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    g_env_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        full_mask = e_mask[:, None] & c_mask[None, :]
        rb = e_offs[:, None] * stride_rad_e + c_offs[None, :] * stride_rad_c

        g0 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 0 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g1 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 1 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g2 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 2 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g3 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 3 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g4 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 4 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g5 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 5 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g6 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 6 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g7 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 7 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )
        g8 = (
            tl.load(
                g_out_ptr
                + j[:, None] * stride_xn
                + 8 * stride_xl
                + c_offs[None, :] * stride_xc,
                mask=full_mask,
                other=0.0,
            )
            * inv_rescale
        )

        r1 = tl.load(rad_out_ptr + rb + 1 * C, mask=full_mask, other=0.0)
        r2 = tl.load(rad_out_ptr + rb + 2 * C, mask=full_mask, other=0.0)

        tl.store(g_rad_ptr + rb + 0 * C, g0 * env, mask=full_mask)
        tl.store(
            g_rad_ptr + rb + 1 * C,
            (g1 * w21 + g2 * w22 + g3 * w23) * env,
            mask=full_mask,
        )
        tl.store(
            g_rad_ptr + rb + 2 * C,
            (g4 * w64 + g5 * w65 + g6 * w66 + g7 * w67 + g8 * w68) * env,
            mask=full_mask,
        )

        gw21_acc += tl.sum(g1 * r1, axis=1)
        gw22_acc += tl.sum(g2 * r1, axis=1)
        gw23_acc += tl.sum(g3 * r1, axis=1)
        gw64_acc += tl.sum(g4 * r2, axis=1)
        gw65_acc += tl.sum(g5 * r2, axis=1)
        gw66_acc += tl.sum(g6 * r2, axis=1)
        gw67_acc += tl.sum(g7 * r2, axis=1)
        gw68_acc += tl.sum(g8 * r2, axis=1)

        r0 = tl.load(rad_out_ptr + rb + 0 * C, mask=full_mask, other=0.0)
        g_env_acc += tl.sum(
            g0 * r0
            + (g1 * w21 + g2 * w22 + g3 * w23) * r1
            + (g4 * w64 + g5 * w65 + g6 * w66 + g7 * w67 + g8 * w68) * r2,
            axis=1,
        )

    tl.store(g_wig_ptr + wb + W21 * stride_wig_k, gw21_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W22 * stride_wig_k, gw22_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W23 * stride_wig_k, gw23_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W64 * stride_wig_k, gw64_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W65 * stride_wig_k, gw65_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W66 * stride_wig_k, gw66_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W67 * stride_wig_k, gw67_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W68 * stride_wig_k, gw68_acc * env_1d, mask=e_mask)
    tl.store(g_env_ptr + e_offs, g_env_acc, mask=e_mask)
