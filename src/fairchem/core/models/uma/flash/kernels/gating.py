"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fused SO(2) gating: splits the m=0 output into gate and value
halves and applies the activation without materialising either half.
"""

from __future__ import annotations

import triton
import triton.language as tl

from fairchem.core.models.uma.flash.constants import (
    Y_M0_MULT,
    Y_M1_MULT,
    Y_M2_MULT,
    Z_M0_MULT,
    Z_M1_MULT,
    Z_M2_MULT,
)


# Both the forward and the backward walk the channel dimension in a loop, so
# ``num_stages`` has something to pipeline in either direction and is swept up
# to 4 rather than 2.
def _generate_configs():
    configs = []
    for e in [16, 32, 64, 128]:
        for c in [16, 32, 64, 128]:
            for w in [4, 8]:
                for ns in [1, 2, 3, 4]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_E": e, "BLOCK_C": c}, num_warps=w, num_stages=ns
                        )
                    )
    return configs


# =========================================================================
# FORWARD KERNEL: Split Gating
# =========================================================================
@triton.autotune(cache_results=True, configs=_generate_configs(), key=["C"])
@triton.jit
def _gating_split_fwd_kernel(
    Y_m0_ptr,
    Y_m1_ptr,
    Y_m2_ptr,
    Z_m0_ptr,
    Z_m1_ptr,
    Z_m2_ptr,
    E,
    C,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # 1D Grid over Edges
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # Loop over Channels to reuse registers
    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        mask = e_mask[:, None] & c_mask[None, :]

        # Calculate base offsets for each block
        y_base_m0 = e_offs[:, None] * Y_M0_MULT * C + c_offs[None, :]
        y_base_m1 = e_offs[:, None] * Y_M1_MULT * C + c_offs[None, :]
        y_base_m2 = e_offs[:, None] * Y_M2_MULT * C + c_offs[None, :]

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]
        z_base_m2 = e_offs[:, None] * Z_M2_MULT * C + c_offs[None, :]

        # ----------------------------------------------------
        # 1. Load Inputs from 3 Split Pointers
        # ----------------------------------------------------
        # M=0
        x0_e0 = tl.load(Y_m0_ptr + y_base_m0 + 0 * C, mask=mask, other=0.0)
        x0_e1 = tl.load(Y_m0_ptr + y_base_m0 + 1 * C, mask=mask, other=0.0)
        y2 = tl.load(Y_m0_ptr + y_base_m0 + 2 * C, mask=mask, other=0.0)
        y3 = tl.load(Y_m0_ptr + y_base_m0 + 3 * C, mask=mask, other=0.0)
        y4 = tl.load(Y_m0_ptr + y_base_m0 + 4 * C, mask=mask, other=0.0)

        # M=1
        y1r_0 = tl.load(Y_m1_ptr + y_base_m1 + 0 * C, mask=mask, other=0.0)
        y1r_1 = tl.load(Y_m1_ptr + y_base_m1 + 1 * C, mask=mask, other=0.0)
        y1i_0 = tl.load(Y_m1_ptr + y_base_m1 + 2 * C, mask=mask, other=0.0)
        y1i_1 = tl.load(Y_m1_ptr + y_base_m1 + 3 * C, mask=mask, other=0.0)

        # M=2
        y2r = tl.load(Y_m2_ptr + y_base_m2 + 0 * C, mask=mask, other=0.0)
        y2i = tl.load(Y_m2_ptr + y_base_m2 + 1 * C, mask=mask, other=0.0)

        # ----------------------------------------------------
        # 2. Compute Activation Gates
        # ----------------------------------------------------
        g0 = tl.sigmoid(x0_e0)
        g1 = tl.sigmoid(x0_e1)
        sy2 = tl.sigmoid(y2)

        # ----------------------------------------------------
        # 3. Write Gated Output to 3 Split Pointers
        # ----------------------------------------------------
        # M=0 (3C Output)
        tl.store(Z_m0_ptr + z_base_m0 + 0 * C, y2 * sy2, mask=mask)
        tl.store(Z_m0_ptr + z_base_m0 + 1 * C, y3 * g0, mask=mask)
        tl.store(Z_m0_ptr + z_base_m0 + 2 * C, y4 * g1, mask=mask)

        # M=1 (4C Output)
        tl.store(Z_m1_ptr + z_base_m1 + 0 * C, y1r_0 * g0, mask=mask)
        tl.store(Z_m1_ptr + z_base_m1 + 1 * C, y1r_1 * g1, mask=mask)
        tl.store(Z_m1_ptr + z_base_m1 + 2 * C, y1i_0 * g0, mask=mask)
        tl.store(Z_m1_ptr + z_base_m1 + 3 * C, y1i_1 * g1, mask=mask)

        # M=2 (2C Output)
        tl.store(Z_m2_ptr + z_base_m2 + 0 * C, y2r * g1, mask=mask)
        tl.store(Z_m2_ptr + z_base_m2 + 1 * C, y2i * g1, mask=mask)


# =========================================================================
# BACKWARD KERNEL: Split Gating Adjoints
# =========================================================================
@triton.autotune(cache_results=True, configs=_generate_configs(), key=["C"])
@triton.jit
def _gating_split_bwd_kernel(
    g_z_m0_ptr,
    g_z_m1_ptr,
    g_z_m2_ptr,
    Y_m0_ptr,
    Y_m1_ptr,
    Y_m2_ptr,
    g_y_m0_ptr,
    g_y_m1_ptr,
    g_y_m2_ptr,
    E,
    C,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        mask = e_mask[:, None] & c_mask[None, :]

        y_base_m0 = e_offs[:, None] * Y_M0_MULT * C + c_offs[None, :]
        y_base_m1 = e_offs[:, None] * Y_M1_MULT * C + c_offs[None, :]
        y_base_m2 = e_offs[:, None] * Y_M2_MULT * C + c_offs[None, :]

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]
        z_base_m2 = e_offs[:, None] * Z_M2_MULT * C + c_offs[None, :]

        # 1. Recompute forward values (faster than saving them to HBM)
        x0_e0 = tl.load(Y_m0_ptr + y_base_m0 + 0 * C, mask=mask, other=0.0)
        x0_e1 = tl.load(Y_m0_ptr + y_base_m0 + 1 * C, mask=mask, other=0.0)
        y2 = tl.load(Y_m0_ptr + y_base_m0 + 2 * C, mask=mask, other=0.0)
        y3 = tl.load(Y_m0_ptr + y_base_m0 + 3 * C, mask=mask, other=0.0)
        y4 = tl.load(Y_m0_ptr + y_base_m0 + 4 * C, mask=mask, other=0.0)

        y1r_0 = tl.load(Y_m1_ptr + y_base_m1 + 0 * C, mask=mask, other=0.0)
        y1r_1 = tl.load(Y_m1_ptr + y_base_m1 + 1 * C, mask=mask, other=0.0)
        y1i_0 = tl.load(Y_m1_ptr + y_base_m1 + 2 * C, mask=mask, other=0.0)
        y1i_1 = tl.load(Y_m1_ptr + y_base_m1 + 3 * C, mask=mask, other=0.0)

        y2r = tl.load(Y_m2_ptr + y_base_m2 + 0 * C, mask=mask, other=0.0)
        y2i = tl.load(Y_m2_ptr + y_base_m2 + 1 * C, mask=mask, other=0.0)

        g0 = tl.sigmoid(x0_e0)
        dg0 = g0 * (1.0 - g0)
        g1 = tl.sigmoid(x0_e1)
        dg1 = g1 * (1.0 - g1)
        sy2 = tl.sigmoid(y2)
        dsy2 = sy2 * (1.0 - sy2)

        # 2. Load incoming gradients from Conv2
        gy2 = tl.load(g_z_m0_ptr + z_base_m0 + 0 * C, mask=mask, other=0.0)
        gy3 = tl.load(g_z_m0_ptr + z_base_m0 + 1 * C, mask=mask, other=0.0)
        gy4 = tl.load(g_z_m0_ptr + z_base_m0 + 2 * C, mask=mask, other=0.0)

        gy1r_0 = tl.load(g_z_m1_ptr + z_base_m1 + 0 * C, mask=mask, other=0.0)
        gy1r_1 = tl.load(g_z_m1_ptr + z_base_m1 + 1 * C, mask=mask, other=0.0)
        gy1i_0 = tl.load(g_z_m1_ptr + z_base_m1 + 2 * C, mask=mask, other=0.0)
        gy1i_1 = tl.load(g_z_m1_ptr + z_base_m1 + 3 * C, mask=mask, other=0.0)

        gy2r = tl.load(g_z_m2_ptr + z_base_m2 + 0 * C, mask=mask, other=0.0)
        gy2i = tl.load(g_z_m2_ptr + z_base_m2 + 1 * C, mask=mask, other=0.0)

        # 3. Chain Rule: Derivative of Output w.r.t Y_in
        g_x0_e0 = dg0 * (gy3 * y3 + gy1r_0 * y1r_0 + gy1i_0 * y1i_0)
        g_x0_e1 = dg1 * (
            gy4 * y4 + gy1r_1 * y1r_1 + gy1i_1 * y1i_1 + gy2r * y2r + gy2i * y2i
        )
        g_y2 = gy2 * (sy2 + y2 * dsy2)

        # 4. Store Gradients to 3 Split Pointers
        tl.store(g_y_m0_ptr + y_base_m0 + 0 * C, g_x0_e0, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 1 * C, g_x0_e1, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 2 * C, g_y2, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 3 * C, gy3 * g0, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 4 * C, gy4 * g1, mask=mask)

        tl.store(g_y_m1_ptr + y_base_m1 + 0 * C, gy1r_0 * g0, mask=mask)
        tl.store(g_y_m1_ptr + y_base_m1 + 1 * C, gy1r_1 * g1, mask=mask)
        tl.store(g_y_m1_ptr + y_base_m1 + 2 * C, gy1i_0 * g0, mask=mask)
        tl.store(g_y_m1_ptr + y_base_m1 + 3 * C, gy1i_1 * g1, mask=mask)

        tl.store(g_y_m2_ptr + y_base_m2 + 0 * C, gy2r * g1, mask=mask)
        tl.store(g_y_m2_ptr + y_base_m2 + 1 * C, gy2i * g1, mask=mask)
