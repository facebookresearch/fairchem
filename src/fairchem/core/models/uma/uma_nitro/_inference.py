"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

UMA Nitro: a standalone UMA-small model for energy, force, and stress inference.

``UMANitroModel`` is built directly from an inference checkpoint
(``model_config``, ``ema_state_dict`` and ``tasks_config``) without constructing
the composable ``HydraModel``. The architecture is fixed to uma-s-1p2p1's, and
``verify_checkpoint_file`` accepts only the checkpoints on its MD5 allowlist
(uma-s-1p2p1 and fine-tunes of it). It owns everything between the checkpoint
and the kernels: the mixture-of-linear-experts (MOLE) merge for the system being
simulated, the SO(2) block-diagonal weight layout, and the unified radial MLP
layout. It implements the subset of the ``HydraModel`` interface that
``MLIPPredictUnit`` uses, so the predict unit's task handling and its fast-path
fallback (``MergeMoleConsistencyError``) apply unchanged.

The forward combines a fixed-capacity padded radius graph, fused Triton/CuTe
kernels, architecture-specific kernel selection, and optional CUDA graph
capture. Model parameters remain frozen while input derivatives are computed
for forces and stress. Capacity and finite-output guards prevent an incomplete
graph or invalid numerical result from being returned.

Third-party source included in this file
----------------------------------------

This file contains one copied and adapted third-party source block:

* Project: NVIDIA CUTLASS, https://github.com/NVIDIA/cutlass
* Upstream release and commit: ``v4.7.0`` at
  ``dcf215af68a2d08d305076c152a06f201728cd53``
* Original path:
  ``examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py``
* Copied/adapted symbol: the complete dense-GEMM kernel class, named
  ``HopperWgmmaGemmKernel`` here, including all of its methods
* Associated integration runtime: ``nvidia-cutlass-dsl==4.7.1``, pinned by the
  ``fairchem-core`` dependencies
* License: ``BSD-3-Clause``
* Upstream notice: ``Copyright (c) 2025 - 2026 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved.``

FairChem renamed the class for the local namespace, removed the upstream CLI,
benchmark driver, and ``__main__`` entry point, and added optional bias,
per-row rescaling, and accumulator-epilogue integration. The copied block is
delimited below by explicit ``BEGIN`` and ``END`` provenance markers and keeps
the complete upstream copyright notice, license conditions, and disclaimer.

This file also copies or adapts MIT-licensed, first-party FairChem UMA source:
``M_TO_L_GATHER_IDX`` from ``triton/constants.py``; ``_wig_rot9``,
``_wig_rotT9``, ``_wig_dw_store1``, ``_wig_dw_store2``; and fused Wigner/conv1
kernels from ``triton/kernels.py``. No QuACK or FlashAttention source is
incorporated. See ``UMA_NITRO_LICENSE_PROVENANCE.md`` for the expanded source
inventory and distribution requirements.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.profiler import record_function
from torch.library import triton_op, wrap_triton

from fairchem.core.common.checksums import cached_file_md5
from fairchem.core.models.uma.escn_md import GradRegressConfig
from fairchem.core.units.mlip_unit.api.inference import (
    NITRO_CHECKPOINT,
    UMANitroNotSupportedError,
)


_CUDA_CAPABILITY = (
    torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
)


@dataclass(frozen=True)
class _KernelPolicy:
    """
    Architecture-specific UMA Nitro primitive selection.
    """

    name: str
    use_cute_kernels: bool
    use_hopper_gemm: bool
    use_tf32: bool
    supports_tf32_round: bool
    supports_vector_atomics: bool


def _kernel_policy(capability: tuple[int, int]) -> _KernelPolicy:
    """
    Select architecture-compatible UMA Nitro primitives.

    Args:
        capability: CUDA compute capability as ``(major, minor)``.

    Returns:
        The primitive policy for the requested architecture.
    """
    major, _ = capability
    use_hopper_gemm = major == 9
    use_blackwell_hybrid = major == 10
    use_cute_kernels = use_hopper_gemm or use_blackwell_hybrid
    use_tf32 = major in (9, 10)
    if use_hopper_gemm:
        name = "native_sm90_plus"
    elif use_blackwell_hybrid:
        name = "blackwell_cute_cublas"
    elif use_tf32:
        name = "portable_pytorch_tf32"
    else:
        name = "portable_pytorch_ieee"
    return _KernelPolicy(
        name=name,
        use_cute_kernels=use_cute_kernels,
        use_hopper_gemm=use_hopper_gemm,
        use_tf32=use_tf32,
        supports_tf32_round=use_cute_kernels,
        supports_vector_atomics=use_cute_kernels,
    )


_KERNEL_POLICY = _kernel_policy(_CUDA_CAPABILITY)

if _KERNEL_POLICY.use_cute_kernels:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

    if _KERNEL_POLICY.use_hopper_gemm:
        import cutlass.utils.hopper_helpers as sm90_utils
    else:
        sm90_utils = None
else:
    cuda = None
    cutlass = None
    pipeline = None
    utils = None
    sm90_utils = None
    from_dlpack = None
    pipeline_init_arrive = None
    pipeline_init_wait = None

    class _UnavailableCuTe:
        @staticmethod
        def jit(function):
            return function

        kernel = jit

    cute = _UnavailableCuTe()


_SUPPORTS_VECTOR_ATOMICS = _KERNEL_POLICY.supports_vector_atomics
_USE_TF32 = _KERNEL_POLICY.use_tf32
_USE_CUTEDSL = _KERNEL_POLICY.use_cute_kernels
_USE_HOPPER_GEMM = _KERNEL_POLICY.use_hopper_gemm
_SUPPORTS_TF32_ROUND = _KERNEL_POLICY.supports_tf32_round


def kernel_backend_summary() -> str:
    """
    Describe the effective UMA Nitro primitive backend for logs and diagnostics.
    """
    major, minor = _CUDA_CAPABILITY
    return f"{_KERNEL_POLICY.name} (CUDA capability {major}.{minor})"


# ============================================================================
# Allocator configuration
# ============================================================================
# Expandable segments reduce fragmentation during CUDA graph warmup and capture.
# Configure them before the first CUDA allocation when possible; the runtime
# setter covers callers that initialized CUDA before importing this module.
EXPANDABLE_SEGMENTS = os.environ.get("UMA_NITRO_EXPANDABLE_SEGMENTS", "1") == "1"


def _configure_allocator():
    """
    Put ``expandable_segments:True`` in force, whether or not CUDA is up.

    Returns:
        ``True`` if the setting is believed to be active.
    """
    if not EXPANDABLE_SEGMENTS:
        return False
    key = "PYTORCH_CUDA_ALLOC_CONF"
    cur = os.environ.get(key, "")
    if "expandable_segments" not in cur:
        os.environ[key] = f"{cur},expandable_segments:True".lstrip(",")
    if torch.cuda.is_initialized():
        # The environment variable is read once, at allocator init, so a caller
        # that allocated before importing this module needs the runtime setter.
        # It applies to segments created from here on, which is every segment
        # that matters: the warmup and the graph pool are both still ahead.
        try:
            torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        except Exception:  # pragma: no cover - older torch, or no accelerator
            return False
    return True


_ALLOC_CONFIGURED = _configure_allocator()

# ============================================================================
# FairChem UMA Triton helpers
# ============================================================================
# These helpers are copied or adapted from the MIT-licensed UMA Triton kernels.
# Keep the index arithmetic synchronized with the canonical implementation. See
# ``UMA_NITRO_LICENSE_PROVENANCE.md`` for the exact source inventory.
M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]


@triton.jit
def _wig_rot9(w_ptr, wb, x0, x1, x2, x3, x4, x5, x6, x7, x8):
    """
    Block-diagonal Wigner rotate y = W @ x for the 9 lmax=2 coefficients.

    Loops over the three diagonal blocks (rows/cols only over nonzeros).
    Returns the 9 rotated coefficients y0..y8 as a tuple.
    """
    xs = (x0, x1, x2, x3, x4, x5, x6, x7, x8)
    y0 = tl.load(w_ptr + wb + 0) * xs[0]
    y1 = tl.load(w_ptr + wb + 1 * 9 + 1) * xs[1]
    y2 = tl.load(w_ptr + wb + 2 * 9 + 1) * xs[1]
    y3 = tl.load(w_ptr + wb + 3 * 9 + 1) * xs[1]
    for j in tl.static_range(2, 4):
        y1 += tl.load(w_ptr + wb + 1 * 9 + j) * xs[j]
        y2 += tl.load(w_ptr + wb + 2 * 9 + j) * xs[j]
        y3 += tl.load(w_ptr + wb + 3 * 9 + j) * xs[j]
    y4 = tl.load(w_ptr + wb + 4 * 9 + 4) * xs[4]
    y5 = tl.load(w_ptr + wb + 5 * 9 + 4) * xs[4]
    y6 = tl.load(w_ptr + wb + 6 * 9 + 4) * xs[4]
    y7 = tl.load(w_ptr + wb + 7 * 9 + 4) * xs[4]
    y8 = tl.load(w_ptr + wb + 8 * 9 + 4) * xs[4]
    for j in tl.static_range(5, 9):
        y4 += tl.load(w_ptr + wb + 4 * 9 + j) * xs[j]
        y5 += tl.load(w_ptr + wb + 5 * 9 + j) * xs[j]
        y6 += tl.load(w_ptr + wb + 6 * 9 + j) * xs[j]
        y7 += tl.load(w_ptr + wb + 7 * 9 + j) * xs[j]
        y8 += tl.load(w_ptr + wb + 8 * 9 + j) * xs[j]
    return y0, y1, y2, y3, y4, y5, y6, y7, y8


@triton.jit
def _wig_rotT9(w_ptr, wb, g0, g1, g2, g3, g4, g5, g6, g7, g8):
    """
    Block-diagonal transpose rotate dx = W^T @ g for the 9 lmax=2 coefficients.

    dx[j] = sum_i W[i, j] * g[i] over each diagonal block. Returns dx0..dx8.
    """
    gs = (g0, g1, g2, g3, g4, g5, g6, g7, g8)
    dx0 = tl.load(w_ptr + wb + 0) * gs[0]
    dx1 = tl.load(w_ptr + wb + 1 * 9 + 1) * gs[1]
    dx2 = tl.load(w_ptr + wb + 1 * 9 + 2) * gs[1]
    dx3 = tl.load(w_ptr + wb + 1 * 9 + 3) * gs[1]
    for i in tl.static_range(2, 4):
        dx1 += tl.load(w_ptr + wb + i * 9 + 1) * gs[i]
        dx2 += tl.load(w_ptr + wb + i * 9 + 2) * gs[i]
        dx3 += tl.load(w_ptr + wb + i * 9 + 3) * gs[i]
    dx4 = tl.load(w_ptr + wb + 4 * 9 + 4) * gs[4]
    dx5 = tl.load(w_ptr + wb + 4 * 9 + 5) * gs[4]
    dx6 = tl.load(w_ptr + wb + 4 * 9 + 6) * gs[4]
    dx7 = tl.load(w_ptr + wb + 4 * 9 + 7) * gs[4]
    dx8 = tl.load(w_ptr + wb + 4 * 9 + 8) * gs[4]
    for i in tl.static_range(5, 9):
        dx4 += tl.load(w_ptr + wb + i * 9 + 4) * gs[i]
        dx5 += tl.load(w_ptr + wb + i * 9 + 5) * gs[i]
        dx6 += tl.load(w_ptr + wb + i * 9 + 6) * gs[i]
        dx7 += tl.load(w_ptr + wb + i * 9 + 7) * gs[i]
        dx8 += tl.load(w_ptr + wb + i * 9 + 8) * gs[i]
    return dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8


@triton.jit
def _wig_dw_store1(dw_ptr, wb, a, b):
    """
    Store block-diagonal dW[i,j] = sum_c a[i]*b[j] (single-term outer product).

    a, b are 9-tuples of per-coefficient tensors; only the three diagonal
    blocks are written, looping over their contiguous row/col ranges.
    """
    tl.store(dw_ptr + wb + 0, tl.sum(a[0] * b[0]))
    for i in tl.static_range(1, 4):
        for j in tl.static_range(1, 4):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a[i] * b[j]))
    for i in tl.static_range(4, 9):
        for j in tl.static_range(4, 9):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a[i] * b[j]))


@triton.jit
def _wig_dw_store2(dw_ptr, wb, a1, b1, a2, b2):
    """
    Store block-diagonal dW[i,j] = sum_c (a1[i]*b1[j] + a2[i]*b2[j]).

    Two-term variant summing src and tgt contributions in one pass.
    """
    tl.store(dw_ptr + wb + 0, tl.sum(a1[0] * b1[0] + a2[0] * b2[0]))
    for i in tl.static_range(1, 4):
        for j in tl.static_range(1, 4):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a1[i] * b1[j] + a2[i] * b2[j]))
    for i in tl.static_range(4, 9):
        for j in tl.static_range(4, 9):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a1[i] * b1[j] + a2[i] * b2[j]))


@triton.jit
def wigner_conv1_fused_fwd_kernel(
    x_ptr,  # [N, 9, C] node feats (L-major)
    edge_index_ptr,  # [2, E]
    wigner_ptr,  # [E, 81] flattened 9x9
    radial_ptr,  # [E, RTOT] conv1 radial (RTOT = 6*C for lmax2: 768+512+256)
    m0_ptr,  # [E, 3*2C]
    m1_ptr,  # [E, 4*2C]
    m2_ptr,  # [E, 2*2C]
    num_edges,
    C: tl.constexpr,  # sphere_channels (128)
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    BLOCK_C: tl.constexpr,  # == C
    GRID_E_STRIDE: tl.constexpr,
    X_ABL: tl.constexpr = 0,
    EV_S: tl.constexpr = "",
    EV_R: tl.constexpr = "",
):
    edge_id = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C

    C2 = 2 * C  # width of one packed M-row (src|tgt)
    m0_row = 3 * C2
    m1_row = 4 * C2
    m2_row = 2 * C2
    # radial block offsets (in units of C, radial row is [E, 6C])
    RAD = 6 * C2  # 1536 for C=128 (radial row width = 768+512+256)

    while edge_id < num_edges:
        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)
        w_base = edge_id * 81

        # ---- load all 9 L-major coeffs, src & tgt ----
        # X_ABL == 2: see the conv1 backward.
        if X_ABL == 2:
            idx0 = edge_id % 64
            idx1 = (edge_id + 1) % 64
        s_base = idx0 * x_stride_n + c * x_stride_c
        t_base = idx1 * x_stride_n + c * x_stride_c
        xs = (
            tl.load(
                x_ptr + s_base + 0 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 1 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 2 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 3 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 4 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 5 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 6 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 7 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 8 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
        )
        xt = (
            tl.load(
                x_ptr + t_base + 0 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 1 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 2 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 3 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 4 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 5 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 6 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 7 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 8 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
        )

        # ---- block-diagonal Wigner rotate (M->L y indices below) ----
        y0s, y1s, y2s, y3s, y4s, y5s, y6s, y7s, y8s = _wig_rot9(wigner_ptr, w_base, *xs)
        y0t, y1t, y2t, y3t, y4t, y5t, y6t, y7t, y8t = _wig_rot9(wigner_ptr, w_base, *xt)

        # ---- M-major rows (M<-L permute) then per-m radial scale + pack ----
        # M0<-y0 M1<-y2 M2<-y6 | M3<-y3 M4<-y7 M5<-y1 M6<-y5 | M7<-y8 M8<-y4
        # radial src at [base + c], tgt at [base + C + c]
        rb = edge_id * RAD
        rs = (
            tl.load(
                radial_ptr + rb + 0 * C2 + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 1 * C2 + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 2 * C2 + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 3 * C2 + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 4 * C2 + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 5 * C2 + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
        )
        rt = (
            tl.load(
                radial_ptr + rb + 0 * C2 + C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 1 * C2 + C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 2 * C2 + C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 3 * C2 + C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 4 * C2 + C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
            tl.load(
                radial_ptr + rb + 5 * C2 + C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_S,
            ),
        )
        # per-M-row operands, pre-permuted to M-order (radial block per row:
        # 0,1,2 | 3,4,3,4 | 5,5) so only the direct loop var indexes each tuple
        ys_perm = (y0s, y2s, y6s, y3s, y7s, y1s, y5s, y8s, y4s)
        yt_perm = (y0t, y2t, y6t, y3t, y7t, y1t, y5t, y8t, y4t)
        rs_m = (rs[0], rs[1], rs[2], rs[3], rs[4], rs[3], rs[4], rs[5], rs[5])
        rt_m = (rt[0], rt[1], rt[2], rt[3], rt[4], rt[3], rt[4], rt[5], rt[5])
        mb = edge_id * m0_row
        m1b = edge_id * m1_row
        m2b = edge_id * m2_row
        for m in tl.static_range(3):  # m0 buffer: rows M0,M1,M2
            tl.store(m0_ptr + mb + m * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m0_ptr + mb + m * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)
        for m in tl.static_range(3, 7):  # m1 buffer: rows M3,M4,M5,M6
            row = m - 3
            tl.store(m1_ptr + m1b + row * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m1_ptr + m1b + row * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)
        for m in tl.static_range(7, 9):  # m2 buffer: rows M7,M8
            row = m - 7
            tl.store(m2_ptr + m2b + row * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m2_ptr + m2b + row * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)

        edge_id += GRID_E_STRIDE


# Optional kernel-development ablations. These deliberately change numerical
# results and must remain disabled in normal inference.
_ABLATIONS = {t for t in os.environ.get("UMA_NITRO_ABLATIONS", "").split(",") if t}
if _ABLATIONS:
    print(
        "[uma_nitro] kernel ablations are active; outputs are not valid: "
        f"{sorted(_ABLATIONS)}"
    )


# The coefficient group size must divide the nine spherical coefficients. This
# guard prevents a partial group from leaving output rows unwritten.
_NUM_SPH = 9


def _coeff_groups(cpg: int, knob: str) -> int:
    """
    Programs needed to cover the nine coefficients at ``cpg`` each.

    Args:
        cpg: Coefficients per program.
        knob: Name of the environment knob, for the message.

    Returns:
        ``9 // cpg``.

    Raises:
        RuntimeError: If ``cpg`` does not divide nine, which would drop the
            remaining coefficients without a trace.
    """
    if cpg < 1 or _NUM_SPH % cpg:
        raise RuntimeError(
            f"UMA Nitro: {knob}={cpg} does not divide {_NUM_SPH}, so "
            f"coefficients {(_NUM_SPH // cpg) * cpg}..{_NUM_SPH - 1} would "
            f"never be written; use 1, 3 or 9"
        )
    return _NUM_SPH // cpg


def _abl_row0(a):
    """
    Return a stride-zero view used by diagnostic kernel ablations.
    """
    return a[:1].expand(a.shape[0], a.shape[1]) if a.shape[0] else a


# ============================================================================
# Edge CSR representation
# ============================================================================
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# A checked CSR bundle for the edge list. Its required invariants are:
#
# 1. ``edge_index[1]`` is sorted ascending over the live prefix, so each target
#    node's edges are one contiguous range
# 2. ``offsets`` is the matching row-pointer array, ``offsets[-1] == live``
# 3. only ``[0, live)`` is real; the tail up to ``E_cap`` is padding whose
#    distances are set to the cutoff so the envelope is zero
#
# ``EdgeCSR.validate()`` checks these invariants:
#
#     bincount(target[:live]) == diff(offsets)   and   offsets[-1] == live
#
# It costs a bincount, so it is opt-in via ``UMA_NITRO_CHECK_CSR=1`` rather than run
# every step.
#
# Validation is opt-in because it requires an additional bincount. The container
# implements tensor flattening so tracing can treat it as a pytree-like value.
CHECK = os.environ.get("UMA_NITRO_CHECK_CSR") == "1"


class EdgeCSR:
    """
    An edge list plus the CSR row pointers over its target nodes.
    """

    __slots__ = ("edge_index", "offsets", "live", "cell_offsets", "n_edges")

    def __init__(self, edge_index, offsets, live, cell_offsets=None, n_edges=None):
        """
        Bundle an edge list with its target-side CSR.

        Args:
            edge_index: ``(2, E_cap)``; row 0 source, row 1 target.
            offsets: CSR row pointers over targets, ``(N + 1,)``.
            live: Number of real edges; the rest of ``edge_index`` is padding.
            cell_offsets: Optional per-edge PBC image offset, ``(E_cap, 3)``.
            n_edges: Optional per-system edge count, ``(B,)``.
        """
        self.edge_index = edge_index
        self.offsets = offsets
        self.live = live
        self.cell_offsets = cell_offsets
        self.n_edges = n_edges
        if CHECK:
            self.validate()

    @property
    def source(self):
        """
        Source node per edge, ``(E_cap,)``.
        """
        return self.edge_index[0]

    @property
    def target(self):
        """
        Target node per edge, ``(E_cap,)``.
        """
        return self.edge_index[1]

    @property
    def num_nodes(self):
        """
        Node count implied by the row pointers.
        """
        return self.offsets.shape[0] - 1

    def validate(self):
        """
        Assert the three invariants. Costs a bincount; opt-in via env.

        Raises:
            AssertionError: If the CSR and the edge list disagree.
        """
        live = int(self.live)
        nodes = self.num_nodes
        if int(self.offsets[-1]) != live:
            raise AssertionError(f"offsets[-1]={int(self.offsets[-1])} != live={live}")
        tgt = self.target[:live]
        if live > 1 and not bool((tgt[1:] >= tgt[:-1]).all()):
            raise AssertionError(
                "edge_index[1] is not sorted ascending over the live prefix, so "
                "the CSR ranges are not contiguous"
            )
        degrees = torch.bincount(tgt, minlength=nodes)
        expected = self.offsets[1:] - self.offsets[:-1]
        if not bool(torch.equal(degrees, expected)):
            bad = int((degrees != expected).sum())
            raise AssertionError(
                f"CSR row pointers disagree with the live edge targets at "
                f"{bad} of {nodes} nodes"
            )
        return self

    def __tensor_flatten__(self):
        """
        Traceable-container protocol: inner tensor names plus static metadata.
        """
        names = ["edge_index", "offsets"]
        if self.cell_offsets is not None:
            names.append("cell_offsets")
        if self.n_edges is not None:
            names.append("n_edges")
        return names, {"live": self.live}

    @staticmethod
    def __tensor_unflatten__(inner, meta, outer_size=None, outer_stride=None):
        """
        Rebuild from flattened inner tensors.
        """
        del outer_size, outer_stride
        return EdgeCSR(
            inner["edge_index"],
            inner["offsets"],
            meta["live"],
            inner.get("cell_offsets"),
            inner.get("n_edges"),
        )

    def __repr__(self):
        return (
            f"EdgeCSR(E_cap={self.edge_index.shape[1]}, live={int(self.live)}, "
            f"nodes={self.num_nodes})"
        )


EQN_GRID = int(os.environ.get("UMA_NITRO_EQN_GRID", "2048"))
EQN_WARPS = int(os.environ.get("UMA_NITRO_EQN_WARPS", "4"))


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    affine_ptr,
    bias_ptr,
    expand_ptr,
    out_ptr,
    inv_ptr,
    row0_ptr,
    num_nodes,
    eps,
    NUM_COEFFS: tl.constexpr,
    CHANNELS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    HAS_ROW0: tl.constexpr,
):
    """
    One node per program: reduce over ``(9, C)``, then rescale in place.
    """
    chan = tl.arange(0, CHANNELS)
    node = tl.program_id(0)
    while node < num_nodes:
        base = node * NUM_COEFFS * CHANNELS + chan

        # l=0 has its channel mean removed before everything else
        x0 = tl.load(x_ptr + base, mask=chan < CHANNELS, other=0.0)
        f0 = x0 - tl.sum(x0) / CHANNELS

        acc = tl.load(w_ptr) * f0 * f0
        x1 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x2 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x3 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x4 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x5 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x6 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x7 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x8 = tl.zeros((CHANNELS,), dtype=tl.float32)
        if NUM_COEFFS > 1:
            x1 = tl.load(x_ptr + base + 1 * CHANNELS)
            acc += tl.load(w_ptr + 1) * x1 * x1
        if NUM_COEFFS > 2:
            x2 = tl.load(x_ptr + base + 2 * CHANNELS)
            acc += tl.load(w_ptr + 2) * x2 * x2
        if NUM_COEFFS > 3:
            x3 = tl.load(x_ptr + base + 3 * CHANNELS)
            acc += tl.load(w_ptr + 3) * x3 * x3
        if NUM_COEFFS > 4:
            x4 = tl.load(x_ptr + base + 4 * CHANNELS)
            acc += tl.load(w_ptr + 4) * x4 * x4
        if NUM_COEFFS > 5:
            x5 = tl.load(x_ptr + base + 5 * CHANNELS)
            acc += tl.load(w_ptr + 5) * x5 * x5
        if NUM_COEFFS > 6:
            x6 = tl.load(x_ptr + base + 6 * CHANNELS)
            acc += tl.load(w_ptr + 6) * x6 * x6
        if NUM_COEFFS > 7:
            x7 = tl.load(x_ptr + base + 7 * CHANNELS)
            acc += tl.load(w_ptr + 7) * x7 * x7
        if NUM_COEFFS > 8:
            x8 = tl.load(x_ptr + base + 8 * CHANNELS)
            acc += tl.load(w_ptr + 8) * x8 * x8
        inv = 1.0 / tl.sqrt(tl.sum(acc) / CHANNELS + eps)
        tl.store(inv_ptr + node, inv)

        # affine is per (l, channel); expand maps each coefficient to its l
        aw0 = tl.load(affine_ptr + tl.load(expand_ptr) * CHANNELS + chan)
        y0 = f0 * inv * aw0 + tl.load(bias_ptr + chan)
        # Optional per-node addend on the l=0 row. It exists so the caller does not
        # have to write ``out[:, 0, :] += emb`` afterwards: that is an in-place write
        # into a slice of a grad-requiring tensor, which autograd services with
        # CopySlices -- a full clone of the (N, 9, C) base plus a zeros and two copies
        # per call, four calls per iteration. The addend is a frozen embedding, so it
        # carries no gradient and folding it here leaves the backward untouched.
        if HAS_ROW0:
            y0 += tl.load(row0_ptr + node * CHANNELS + chan)
        tl.store(out_ptr + base, y0)
        if NUM_COEFFS > 1:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 1) * CHANNELS + chan)
            tl.store(out_ptr + base + 1 * CHANNELS, x1 * inv * aw)
        if NUM_COEFFS > 2:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 2) * CHANNELS + chan)
            tl.store(out_ptr + base + 2 * CHANNELS, x2 * inv * aw)
        if NUM_COEFFS > 3:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 3) * CHANNELS + chan)
            tl.store(out_ptr + base + 3 * CHANNELS, x3 * inv * aw)
        if NUM_COEFFS > 4:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 4) * CHANNELS + chan)
            tl.store(out_ptr + base + 4 * CHANNELS, x4 * inv * aw)
        if NUM_COEFFS > 5:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 5) * CHANNELS + chan)
            tl.store(out_ptr + base + 5 * CHANNELS, x5 * inv * aw)
        if NUM_COEFFS > 6:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 6) * CHANNELS + chan)
            tl.store(out_ptr + base + 6 * CHANNELS, x6 * inv * aw)
        if NUM_COEFFS > 7:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 7) * CHANNELS + chan)
            tl.store(out_ptr + base + 7 * CHANNELS, x7 * inv * aw)
        if NUM_COEFFS > 8:
            aw = tl.load(affine_ptr + tl.load(expand_ptr + 8) * CHANNELS + chan)
            tl.store(out_ptr + base + 8 * CHANNELS, x8 * inv * aw)
        node += GRID_STRIDE


@triton_op("uma_nitro::equivariant_rms_norm", mutates_args={"out", "inv"})
def _eqn_op(
    x: torch.Tensor,
    w: torch.Tensor,
    affine: torch.Tensor,
    bias: torch.Tensor,
    expand: torch.Tensor,
    out: torch.Tensor,
    inv: torch.Tensor,
    eps: float,
    row0: torch.Tensor | None = None,
) -> None:
    """
    Launch the fused equivariant RMS norm.
    """
    nodes, num_coeffs, channels = x.shape
    _eg = min(x.shape[0], EQN_GRID)
    if "norm" not in _ABLATIONS:
        wrap_triton(_rms_norm_kernel)[(_eg,)](
            x,
            w,
            affine,
            bias,
            expand,
            out,
            inv,
            x if row0 is None else row0,
            nodes,
            eps,
            NUM_COEFFS=num_coeffs,
            CHANNELS=channels,
            GRID_STRIDE=_eg,
            HAS_ROW0=row0 is not None,
            num_warps=4,
        )


def equivariant_rms_norm(module, node_input, row0=None):
    """
    Fused replacement for the eager equivariant RMS norm.

    Args:
        module: The norm module, supplying ``balance_degree_weight``,
            ``affine_weight``, ``affine_bias``, ``expand_index`` and ``eps``.
        node_input: Node features, ``(N, 9, C)``.
        row0: Optional ``(N, C)`` addend for the ``l=0`` output row, folded into the
            kernel's epilogue. Must not require grad.

    Returns:
        Normalized features, ``(N, 9, C)``.
    """
    channels = module.num_channels
    out, _inv = torch.ops.uma_nitro.eq_norm(
        node_input.contiguous(),
        module.balance_degree_weight.reshape(-1).contiguous(),
        module.affine_weight.reshape(-1, channels).contiguous(),
        module.affine_bias.reshape(-1).contiguous(),
        module.expand_index,
        float(module.eps),
        None if row0 is None else row0.contiguous(),
    )
    return out


@triton.jit
def _rms_norm_bwd_kernel(
    x_ptr,
    g_ptr,
    inv_ptr,
    w_ptr,
    affine_ptr,
    expand_ptr,
    dx_ptr,
    num_nodes,
    g_stride_n,
    g_stride_p,
    NUM_COEFFS: tl.constexpr,
    CHANNELS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
):
    """
    Compute the input gradient of the fused equivariant RMS norm.

    The incoming gradient is read with explicit strides. With f the mean-centered
    input, the kernel differentiates the normalization and then applies the linear
    centering adjoint to coefficient zero.
    """
    chan = tl.arange(0, CHANNELS)
    node = tl.program_id(0)
    while node < num_nodes:
        base = node * NUM_COEFFS * CHANNELS + chan
        inv = tl.load(inv_ptr + node)

        x0 = tl.load(x_ptr + base)
        f0 = x0 - tl.sum(x0) / CHANNELS

        # T = sum over every coefficient and channel of g * f * a
        aw0 = tl.load(affine_ptr + tl.load(expand_ptr) * CHANNELS + chan)
        gbase = node * g_stride_n + chan
        g0 = tl.load(g_ptr + gbase)
        # Same register-hoist as the forward: `g`, `x` and the affine row for each
        # coefficient were read once for the T reduction and again for the store.
        acc = g0 * f0 * aw0
        g1 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x1 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a1 = tl.zeros((CHANNELS,), dtype=tl.float32)
        g2 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x2 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a2 = tl.zeros((CHANNELS,), dtype=tl.float32)
        g3 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x3 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a3 = tl.zeros((CHANNELS,), dtype=tl.float32)
        g4 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x4 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a4 = tl.zeros((CHANNELS,), dtype=tl.float32)
        g5 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x5 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a5 = tl.zeros((CHANNELS,), dtype=tl.float32)
        g6 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x6 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a6 = tl.zeros((CHANNELS,), dtype=tl.float32)
        g7 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x7 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a7 = tl.zeros((CHANNELS,), dtype=tl.float32)
        g8 = tl.zeros((CHANNELS,), dtype=tl.float32)
        x8 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a8 = tl.zeros((CHANNELS,), dtype=tl.float32)
        if NUM_COEFFS > 1:
            g1 = tl.load(g_ptr + gbase + 1 * g_stride_p)
            x1 = tl.load(x_ptr + base + 1 * CHANNELS)
            a1 = tl.load(affine_ptr + tl.load(expand_ptr + 1) * CHANNELS + chan)
            acc += g1 * x1 * a1
        if NUM_COEFFS > 2:
            g2 = tl.load(g_ptr + gbase + 2 * g_stride_p)
            x2 = tl.load(x_ptr + base + 2 * CHANNELS)
            a2 = tl.load(affine_ptr + tl.load(expand_ptr + 2) * CHANNELS + chan)
            acc += g2 * x2 * a2
        if NUM_COEFFS > 3:
            g3 = tl.load(g_ptr + gbase + 3 * g_stride_p)
            x3 = tl.load(x_ptr + base + 3 * CHANNELS)
            a3 = tl.load(affine_ptr + tl.load(expand_ptr + 3) * CHANNELS + chan)
            acc += g3 * x3 * a3
        if NUM_COEFFS > 4:
            g4 = tl.load(g_ptr + gbase + 4 * g_stride_p)
            x4 = tl.load(x_ptr + base + 4 * CHANNELS)
            a4 = tl.load(affine_ptr + tl.load(expand_ptr + 4) * CHANNELS + chan)
            acc += g4 * x4 * a4
        if NUM_COEFFS > 5:
            g5 = tl.load(g_ptr + gbase + 5 * g_stride_p)
            x5 = tl.load(x_ptr + base + 5 * CHANNELS)
            a5 = tl.load(affine_ptr + tl.load(expand_ptr + 5) * CHANNELS + chan)
            acc += g5 * x5 * a5
        if NUM_COEFFS > 6:
            g6 = tl.load(g_ptr + gbase + 6 * g_stride_p)
            x6 = tl.load(x_ptr + base + 6 * CHANNELS)
            a6 = tl.load(affine_ptr + tl.load(expand_ptr + 6) * CHANNELS + chan)
            acc += g6 * x6 * a6
        if NUM_COEFFS > 7:
            g7 = tl.load(g_ptr + gbase + 7 * g_stride_p)
            x7 = tl.load(x_ptr + base + 7 * CHANNELS)
            a7 = tl.load(affine_ptr + tl.load(expand_ptr + 7) * CHANNELS + chan)
            acc += g7 * x7 * a7
        if NUM_COEFFS > 8:
            g8 = tl.load(g_ptr + gbase + 8 * g_stride_p)
            x8 = tl.load(x_ptr + base + 8 * CHANNELS)
            a8 = tl.load(affine_ptr + tl.load(expand_ptr + 8) * CHANNELS + chan)
            acc += g8 * x8 * a8
        t_scalar = tl.sum(acc)
        coef = inv * inv * inv * t_scalar / CHANNELS

        df0 = inv * aw0 * g0 - coef * tl.load(w_ptr) * f0
        # the l=0 centering is linear, so its adjoint is the same centering
        tl.store(dx_ptr + base, df0 - tl.sum(df0) / CHANNELS)
        if NUM_COEFFS > 1:
            tl.store(
                dx_ptr + base + 1 * CHANNELS,
                inv * a1 * g1 - coef * tl.load(w_ptr + 1) * x1,
            )
        if NUM_COEFFS > 2:
            tl.store(
                dx_ptr + base + 2 * CHANNELS,
                inv * a2 * g2 - coef * tl.load(w_ptr + 2) * x2,
            )
        if NUM_COEFFS > 3:
            tl.store(
                dx_ptr + base + 3 * CHANNELS,
                inv * a3 * g3 - coef * tl.load(w_ptr + 3) * x3,
            )
        if NUM_COEFFS > 4:
            tl.store(
                dx_ptr + base + 4 * CHANNELS,
                inv * a4 * g4 - coef * tl.load(w_ptr + 4) * x4,
            )
        if NUM_COEFFS > 5:
            tl.store(
                dx_ptr + base + 5 * CHANNELS,
                inv * a5 * g5 - coef * tl.load(w_ptr + 5) * x5,
            )
        if NUM_COEFFS > 6:
            tl.store(
                dx_ptr + base + 6 * CHANNELS,
                inv * a6 * g6 - coef * tl.load(w_ptr + 6) * x6,
            )
        if NUM_COEFFS > 7:
            tl.store(
                dx_ptr + base + 7 * CHANNELS,
                inv * a7 * g7 - coef * tl.load(w_ptr + 7) * x7,
            )
        if NUM_COEFFS > 8:
            tl.store(
                dx_ptr + base + 8 * CHANNELS,
                inv * a8 * g8 - coef * tl.load(w_ptr + 8) * x8,
            )
        node += GRID_STRIDE


@triton_op("uma_nitro::equivariant_rms_norm_bwd", mutates_args={"dx"})
def _bwd_op(
    x: torch.Tensor,
    g: torch.Tensor,
    inv: torch.Tensor,
    w: torch.Tensor,
    affine: torch.Tensor,
    expand: torch.Tensor,
    dx: torch.Tensor,
) -> None:
    """
    Launch the fused equivariant RMS norm backward.
    """
    nodes, num_coeffs, channels = x.shape
    _eg = min(x.shape[0], EQN_GRID)
    if "norm" not in _ABLATIONS:
        wrap_triton(_rms_norm_bwd_kernel)[(_eg,)](
            x,
            g,
            inv,
            w,
            affine,
            expand,
            dx,
            nodes,
            g.stride(0),
            g.stride(1),
            NUM_COEFFS=num_coeffs,
            CHANNELS=channels,
            GRID_STRIDE=_eg,
            num_warps=EQN_WARPS,
        )


@torch.library.custom_op("uma_nitro::eq_norm", mutates_args=(), device_types="cuda")
def _eq_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    affine: torch.Tensor,
    bias: torch.Tensor,
    expand: torch.Tensor,
    eps: float,
    row0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused equivariant RMS norm, returning the output and the per-node ``inv``.
    """
    out = torch.empty_like(x)
    inv = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    torch.ops.uma_nitro.equivariant_rms_norm(
        x, w, affine, bias, expand, out, inv, eps, row0
    )
    return out, inv


@_eq_norm.register_fake
def _(x, w, affine, bias, expand, eps, row0=None):
    """
    Shape function.
    """
    return torch.empty_like(x), x.new_empty(x.shape[0])


def _setup(ctx, inputs, output):
    """
    Save the input, the per-node inv, and the parameters the backward needs.
    """
    x, w, affine, bias, expand, _eps = inputs[:6]
    del bias
    # ``inv`` is an implementation detail the backward never differentiates, but
    # autograd still materializes a zeroed ``(N,)`` gradient for it before calling
    # the backward -- one aten::fill_ per call, 9 calls per iteration. Both flags are
    # needed: mark_non_differentiable alone does not suppress materialization.
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, output[1], w, affine, expand)


def _eqn_backward(ctx, grad_out, grad_inv):
    """
    Gradients for ``x``; the norm's parameters are frozen at inference.
    """
    del grad_inv
    x, inv, w, affine, expand = ctx.saved_tensors
    if grad_out is None:
        return (None,) * 7
    dx = torch.empty_like(x)
    torch.ops.uma_nitro.equivariant_rms_norm_bwd(
        x, grad_out, inv, w, affine, expand, dx
    )
    return dx, None, None, None, None, None, None


torch.library.register_autograd(
    "uma_nitro::eq_norm", _eqn_backward, setup_context=_setup
)


@triton.jit
def _invrot_segreduce_kernel(
    g0_ptr,
    g1_ptr,
    g2_ptr,
    w_ptr,
    offsets_ptr,
    residual_ptr,
    out_ptr,
    num_nodes,
    C: tl.constexpr,
    GRID: tl.constexpr,
    NSPLIT: tl.constexpr,
    CBLK: tl.constexpr,
    ESPLIT: tl.constexpr,
    G_ABL: tl.constexpr = 0,
    RED: tl.constexpr = 0,
    EV_S2: tl.constexpr = "",
    EV_R2: tl.constexpr = "",
):
    """
    Unpack M-to-L, inverse-rotate, and accumulate incoming edges per node.

    NSPLIT partitions channels and ESPLIT partitions edges while retaining a
    stable accumulation order within each partition.
    """
    pid = tl.program_id(0)
    node = pid // (NSPLIT * ESPLIT)
    lane = pid % (NSPLIT * ESPLIT)
    chan = (lane // ESPLIT) * CBLK + tl.arange(0, CBLK)
    part = lane % ESPLIT
    while node < num_nodes:
        base = node * 9 * C
        _r = 1.0 if ESPLIT == 1 else (part == 0).to(tl.float32)
        a0 = tl.load(residual_ptr + base + 0 * C + chan, eviction_policy=EV_R2) * _r
        a1 = tl.load(residual_ptr + base + 1 * C + chan, eviction_policy=EV_R2) * _r
        a2 = tl.load(residual_ptr + base + 2 * C + chan, eviction_policy=EV_R2) * _r
        a3 = tl.load(residual_ptr + base + 3 * C + chan, eviction_policy=EV_R2) * _r
        a4 = tl.load(residual_ptr + base + 4 * C + chan, eviction_policy=EV_R2) * _r
        a5 = tl.load(residual_ptr + base + 5 * C + chan, eviction_policy=EV_R2) * _r
        a6 = tl.load(residual_ptr + base + 6 * C + chan, eviction_policy=EV_R2) * _r
        a7 = tl.load(residual_ptr + base + 7 * C + chan, eviction_policy=EV_R2) * _r
        a8 = tl.load(residual_ptr + base + 8 * C + chan, eviction_policy=EV_R2) * _r

        start = tl.load(offsets_ptr + node)
        end = tl.load(offsets_ptr + node + 1)
        # ESPLIT programs share a node, each striding every ESPLIT-th edge. This is a
        # clean test of the occupancy hypothesis: it multiplies the program count while
        # keeping the loads a full C wide, where the channel split (NSPLIT) raised
        # occupancy but halved the transaction width, confounding the two.
        for edge in range(start + part, end, ESPLIT):
            # G_ABL: every edge reads g row 0, so the g stream leaves DRAM.
            _ga = (edge % 64) if G_ABL == 2 else (edge * 0 if G_ABL else edge)
            g0b = _ga * 3 * C
            g1b = _ga * 4 * C
            g2b = _ga * 2 * C
            # M->L permute absorbed into the offsets, per kernels.py:1994-2002.
            x0 = tl.load(g0_ptr + g0b + 0 * C + chan, eviction_policy=EV_S2)  # L0 <- M0
            x1 = tl.load(g1_ptr + g1b + 2 * C + chan, eviction_policy=EV_S2)  # L1 <- M5
            x2 = tl.load(g0_ptr + g0b + 1 * C + chan, eviction_policy=EV_S2)  # L2 <- M1
            x3 = tl.load(g1_ptr + g1b + 0 * C + chan, eviction_policy=EV_S2)  # L3 <- M3
            x4 = tl.load(g2_ptr + g2b + 1 * C + chan, eviction_policy=EV_S2)  # L4 <- M8
            x5 = tl.load(g1_ptr + g1b + 3 * C + chan, eviction_policy=EV_S2)  # L5 <- M6
            x6 = tl.load(g0_ptr + g0b + 2 * C + chan, eviction_policy=EV_S2)  # L6 <- M2
            x7 = tl.load(g1_ptr + g1b + 1 * C + chan, eviction_policy=EV_S2)  # L7 <- M4
            x8 = tl.load(g2_ptr + g2b + 0 * C + chan, eviction_policy=EV_S2)  # L8 <- M7
            ys = _wig_rot9(w_ptr, edge * 81, x0, x1, x2, x3, x4, x5, x6, x7, x8)
            a0 += ys[0]
            a1 += ys[1]
            a2 += ys[2]
            a3 += ys[3]
            a4 += ys[4]
            a5 += ys[5]
            a6 += ys[6]
            a7 += ys[7]
            a8 += ys[8]

        if ESPLIT == 1:
            tl.store(out_ptr + base + 0 * C + chan, a0)
            tl.store(out_ptr + base + 1 * C + chan, a1)
            tl.store(out_ptr + base + 2 * C + chan, a2)
            tl.store(out_ptr + base + 3 * C + chan, a3)
            tl.store(out_ptr + base + 4 * C + chan, a4)
            tl.store(out_ptr + base + 5 * C + chan, a5)
            tl.store(out_ptr + base + 6 * C + chan, a6)
            tl.store(out_ptr + base + 7 * C + chan, a7)
            tl.store(out_ptr + base + 8 * C + chan, a8)
        elif RED:
            # The ESPLIT accumulation is accumulate-only -- nothing reads the
            # returned old value -- so the reduction-only ``red.global.add.v4.f32``
            # is available. ``tl.atomic_add`` on this CBLK-wide fp32 tile is already
            # vector-4, but the *fetching* ``atom`` form; see _red_add_v4.
            _red_add_v4(out_ptr + base + 0 * C + chan, a0)
            _red_add_v4(out_ptr + base + 1 * C + chan, a1)
            _red_add_v4(out_ptr + base + 2 * C + chan, a2)
            _red_add_v4(out_ptr + base + 3 * C + chan, a3)
            _red_add_v4(out_ptr + base + 4 * C + chan, a4)
            _red_add_v4(out_ptr + base + 5 * C + chan, a5)
            _red_add_v4(out_ptr + base + 6 * C + chan, a6)
            _red_add_v4(out_ptr + base + 7 * C + chan, a7)
            _red_add_v4(out_ptr + base + 8 * C + chan, a8)
        else:
            tl.atomic_add(out_ptr + base + 0 * C + chan, a0)
            tl.atomic_add(out_ptr + base + 1 * C + chan, a1)
            tl.atomic_add(out_ptr + base + 2 * C + chan, a2)
            tl.atomic_add(out_ptr + base + 3 * C + chan, a3)
            tl.atomic_add(out_ptr + base + 4 * C + chan, a4)
            tl.atomic_add(out_ptr + base + 5 * C + chan, a5)
            tl.atomic_add(out_ptr + base + 6 * C + chan, a6)
            tl.atomic_add(out_ptr + base + 7 * C + chan, a7)
            tl.atomic_add(out_ptr + base + 8 * C + chan, a8)
        node += GRID // (NSPLIT * ESPLIT)


FED_GRID_STRIDE = 1024
FED_GRID_BWD = int(os.environ.get("UMA_NITRO_FED_GRID_BWD", "1048576"))
FED_WARPS_BWD = int(os.environ.get("UMA_NITRO_FED_WARPS_BWD", "1"))
FED_GRID_FWD = int(os.environ.get("UMA_NITRO_FED_GRID_FWD", "1048576"))
FED_ESPLIT = int(os.environ.get("UMA_NITRO_FED_ESPLIT", "1"))
# ``red.global.add.v4.f32`` for this kernel's ESPLIT accumulation, as SEG_RED is for
# _invrot_segreduce's. Only reachable when FED_ESPLIT > 1, which is off by default:
# the ESPLIT path also needs ``out = residual.clone()``, and re-priced with the cheap
# atomic it is still a loss here (see FED_ESPLIT's own sweep). Kept so the two
# reductions stay symmetric and the re-price is one flag away.
FED_RED = int(os.environ.get("UMA_NITRO_FED_RED", "1"))
FED_STRIDE_BUG = int(os.environ.get("UMA_NITRO_FED_STRIDE_BUG", "0"))
FED_WARPS_FWD = int(os.environ.get("UMA_NITRO_FED_WARPS_FWD", "0"))
FED_WARPS = 4
FED_SPH_PAD = 16


@triton.jit
def _fused_m0_segreduce_kernel(
    wig_ptr,
    rad_ptr,
    offsets_ptr,
    residual_ptr,
    out_ptr,
    num_nodes,
    rescale,
    NUM_SPH: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPH_BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    ESPLIT: tl.constexpr,
    CPG: tl.constexpr,
    NGRP: tl.constexpr,
    RED: tl.constexpr = 0,
    STRIDE_BUG: tl.constexpr = 0,
):
    """
    Build each incoming edge's m=0 embedding and reduce it onto its node.

    Explicit coefficient accumulators avoid padding a register tile. ESPLIT
    programs share a node and stride over its incoming edges.
    """
    chan = tl.arange(0, CHANNELS)
    _lane = tl.program_id(0) // ESPLIT
    # Coefficient split: the nine accumulators are independent chains, so a program
    # can own CPG of them and the program count multiplies without reassociating any
    # sum. Unlike the target reduction this does duplicate work -- the three radial
    # vectors are re-read once per group -- so it only pays where occupancy, not
    # bandwidth, is the constraint. Regime-gated on exactly that.
    grp = _lane % NGRP
    node = _lane // NGRP
    cb = grp * CPG
    part = tl.program_id(0) % ESPLIT
    while node < num_nodes:
        base = node * NUM_SPH * CHANNELS + chan
        a0 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a1 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a2 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a3 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a4 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a5 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a6 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a7 = tl.zeros((CHANNELS,), dtype=tl.float32)
        a8 = tl.zeros((CHANNELS,), dtype=tl.float32)
        start = tl.load(offsets_ptr + node)
        end = tl.load(offsets_ptr + node + 1)
        for edge in range(start + part, end, ESPLIT):
            wb = edge * NUM_SPH * NUM_SPH
            rb = edge * 3 * CHANNELS
            r0 = tl.load(rad_ptr + rb + 0 * CHANNELS + chan)
            r1 = tl.load(rad_ptr + rb + 1 * CHANNELS + chan)
            r2 = tl.load(rad_ptr + rb + 2 * CHANNELS + chan)
            # acc[j] += W[j,0]*rad0 + W[j,2]*rad1 + W[j,6]*rad2
            a0 += tl.load(wig_ptr + wb + (cb + 0) * NUM_SPH + 0) * r0
            a0 += tl.load(wig_ptr + wb + (cb + 0) * NUM_SPH + 2) * r1
            a0 += tl.load(wig_ptr + wb + (cb + 0) * NUM_SPH + 6) * r2
            if CPG > 1:
                a1 += tl.load(wig_ptr + wb + (cb + 1) * NUM_SPH + 0) * r0
                a1 += tl.load(wig_ptr + wb + (cb + 1) * NUM_SPH + 2) * r1
                a1 += tl.load(wig_ptr + wb + (cb + 1) * NUM_SPH + 6) * r2
            if CPG > 2:
                a2 += tl.load(wig_ptr + wb + (cb + 2) * NUM_SPH + 0) * r0
                a2 += tl.load(wig_ptr + wb + (cb + 2) * NUM_SPH + 2) * r1
                a2 += tl.load(wig_ptr + wb + (cb + 2) * NUM_SPH + 6) * r2
            if CPG > 3:
                a3 += tl.load(wig_ptr + wb + (cb + 3) * NUM_SPH + 0) * r0
                a3 += tl.load(wig_ptr + wb + (cb + 3) * NUM_SPH + 2) * r1
                a3 += tl.load(wig_ptr + wb + (cb + 3) * NUM_SPH + 6) * r2
            if CPG > 4:
                a4 += tl.load(wig_ptr + wb + (cb + 4) * NUM_SPH + 0) * r0
                a4 += tl.load(wig_ptr + wb + (cb + 4) * NUM_SPH + 2) * r1
                a4 += tl.load(wig_ptr + wb + (cb + 4) * NUM_SPH + 6) * r2
            if CPG > 5:
                a5 += tl.load(wig_ptr + wb + (cb + 5) * NUM_SPH + 0) * r0
                a5 += tl.load(wig_ptr + wb + (cb + 5) * NUM_SPH + 2) * r1
                a5 += tl.load(wig_ptr + wb + (cb + 5) * NUM_SPH + 6) * r2
            if CPG > 6:
                a6 += tl.load(wig_ptr + wb + (cb + 6) * NUM_SPH + 0) * r0
                a6 += tl.load(wig_ptr + wb + (cb + 6) * NUM_SPH + 2) * r1
                a6 += tl.load(wig_ptr + wb + (cb + 6) * NUM_SPH + 6) * r2
            if CPG > 7:
                a7 += tl.load(wig_ptr + wb + (cb + 7) * NUM_SPH + 0) * r0
                a7 += tl.load(wig_ptr + wb + (cb + 7) * NUM_SPH + 2) * r1
                a7 += tl.load(wig_ptr + wb + (cb + 7) * NUM_SPH + 6) * r2
            if CPG > 8:
                a8 += tl.load(wig_ptr + wb + (cb + 8) * NUM_SPH + 0) * r0
                a8 += tl.load(wig_ptr + wb + (cb + 8) * NUM_SPH + 2) * r1
                a8 += tl.load(wig_ptr + wb + (cb + 8) * NUM_SPH + 6) * r2
        if ESPLIT == 1:
            tl.store(
                out_ptr + base + (cb + 0) * CHANNELS,
                a0 / rescale + tl.load(residual_ptr + base + (cb + 0) * CHANNELS),
            )
            if CPG > 1:
                tl.store(
                    out_ptr + base + (cb + 1) * CHANNELS,
                    a1 / rescale + tl.load(residual_ptr + base + (cb + 1) * CHANNELS),
                )
            if CPG > 2:
                tl.store(
                    out_ptr + base + (cb + 2) * CHANNELS,
                    a2 / rescale + tl.load(residual_ptr + base + (cb + 2) * CHANNELS),
                )
            if CPG > 3:
                tl.store(
                    out_ptr + base + (cb + 3) * CHANNELS,
                    a3 / rescale + tl.load(residual_ptr + base + (cb + 3) * CHANNELS),
                )
            if CPG > 4:
                tl.store(
                    out_ptr + base + (cb + 4) * CHANNELS,
                    a4 / rescale + tl.load(residual_ptr + base + (cb + 4) * CHANNELS),
                )
            if CPG > 5:
                tl.store(
                    out_ptr + base + (cb + 5) * CHANNELS,
                    a5 / rescale + tl.load(residual_ptr + base + (cb + 5) * CHANNELS),
                )
            if CPG > 6:
                tl.store(
                    out_ptr + base + (cb + 6) * CHANNELS,
                    a6 / rescale + tl.load(residual_ptr + base + (cb + 6) * CHANNELS),
                )
            if CPG > 7:
                tl.store(
                    out_ptr + base + (cb + 7) * CHANNELS,
                    a7 / rescale + tl.load(residual_ptr + base + (cb + 7) * CHANNELS),
                )
            if CPG > 8:
                tl.store(
                    out_ptr + base + (cb + 8) * CHANNELS,
                    a8 / rescale + tl.load(residual_ptr + base + (cb + 8) * CHANNELS),
                )
        else:
            if RED:
                _red_add_v4(out_ptr + base + (cb + 0) * CHANNELS, a0 / rescale)
                if CPG > 1:
                    _red_add_v4(out_ptr + base + (cb + 1) * CHANNELS, a1 / rescale)
                if CPG > 2:
                    _red_add_v4(out_ptr + base + (cb + 2) * CHANNELS, a2 / rescale)
                if CPG > 3:
                    _red_add_v4(out_ptr + base + (cb + 3) * CHANNELS, a3 / rescale)
                if CPG > 4:
                    _red_add_v4(out_ptr + base + (cb + 4) * CHANNELS, a4 / rescale)
                if CPG > 5:
                    _red_add_v4(out_ptr + base + (cb + 5) * CHANNELS, a5 / rescale)
                if CPG > 6:
                    _red_add_v4(out_ptr + base + (cb + 6) * CHANNELS, a6 / rescale)
                if CPG > 7:
                    _red_add_v4(out_ptr + base + (cb + 7) * CHANNELS, a7 / rescale)
                if CPG > 8:
                    _red_add_v4(out_ptr + base + (cb + 8) * CHANNELS, a8 / rescale)
            else:
                tl.atomic_add(out_ptr + base + (cb + 0) * CHANNELS, a0 / rescale)
                if CPG > 1:
                    tl.atomic_add(out_ptr + base + (cb + 1) * CHANNELS, a1 / rescale)
                if CPG > 2:
                    tl.atomic_add(out_ptr + base + (cb + 2) * CHANNELS, a2 / rescale)
                if CPG > 3:
                    tl.atomic_add(out_ptr + base + (cb + 3) * CHANNELS, a3 / rescale)
                if CPG > 4:
                    tl.atomic_add(out_ptr + base + (cb + 4) * CHANNELS, a4 / rescale)
                if CPG > 5:
                    tl.atomic_add(out_ptr + base + (cb + 5) * CHANNELS, a5 / rescale)
                if CPG > 6:
                    tl.atomic_add(out_ptr + base + (cb + 6) * CHANNELS, a6 / rescale)
                if CPG > 7:
                    tl.atomic_add(out_ptr + base + (cb + 7) * CHANNELS, a7 / rescale)
                if CPG > 8:
                    tl.atomic_add(out_ptr + base + (cb + 8) * CHANNELS, a8 / rescale)
        if STRIDE_BUG:
            node += GRID // ESPLIT
        else:
            node += GRID


def _fed_fwd_warps():
    """
    Return the configured warp count for the fused m=0 reduction.
    """
    if FED_WARPS_FWD:
        return FED_WARPS_FWD
    return 1 if FED_CPG == 1 else FED_WARPS


@triton_op("uma_nitro::fused_m0_segreduce", mutates_args={"out"})
def _fused_m0_segreduce(
    wig: torch.Tensor,
    radial: torch.Tensor,
    offsets: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    rescale: float,
) -> None:
    """
    Launch the fused m=0 embedding plus node reduction.

    Args:
        wig: Full inverse Wigner with the envelope, ``(E, 9, 9)``; only the
            m=0 columns (0, 2, 6) are read.
        radial: Radial features, ``(E, 3, C)``.
        offsets: Target-side CSR row pointers, ``(N + 1,)``.
        residual: Node residual, ``(N, 9, C)``; added unscaled.
        out: Output, ``(N, 9, C)``; mutated.
        rescale: ``edge_degree.rescale_factor``; the edge sum is divided by it.
    """
    num_nodes, num_sph, channels = residual.shape
    _g = min(num_nodes, FED_GRID_FWD)
    if "m0seg" not in _ABLATIONS:
        _ngrp = _coeff_groups(FED_CPG, "FED_CPG")
        wrap_triton(_fused_m0_segreduce_kernel)[(_g * FED_ESPLIT * _ngrp,)](
            wig,
            radial,
            offsets,
            residual,
            out,
            num_nodes,
            rescale,
            NUM_SPH=num_sph,
            CHANNELS=channels,
            SPH_BLOCK=FED_SPH_PAD,
            GRID=_g,
            ESPLIT=FED_ESPLIT,
            CPG=FED_CPG,
            NGRP=_ngrp,
            RED=FED_RED and _red_v4_ok(channels, _fed_fwd_warps()),
            STRIDE_BUG=FED_STRIDE_BUG,
            num_warps=_fed_fwd_warps(),
            num_stages=FED_STAGES,
        )


class _FusedM0EdgeToNode(torch.autograd.Function):
    """
    Autograd wrapper: fused forward, torch backward.

    The forward deletes the ``(E, 9, C)`` intermediate. The backward rebuilds it
    (``grad_out[target[e]]``) and uses the same two bmms as before, so the backward
    cost is unchanged and this is a forward-only optimization.
    """

    @staticmethod
    def forward(ctx, wig, radial, offsets, residual, target, rescale):
        """
        Args:
            wig: ``(E, 9, 9)``.
            radial: ``(E, 3, C)``.
            offsets: CSR row pointers.
            residual: Node residual.
            target: Target node per edge, for the backward gather.
            rescale: Rescale factor.

        Returns:
            ``(N, 9, C)``.
        """
        out = residual.clone() if FED_ESPLIT > 1 else torch.empty_like(residual)
        torch.ops.uma_nitro.fused_m0_segreduce(
            wig.contiguous(),
            radial.contiguous(),
            offsets,
            residual,
            out,
            rescale,
        )
        ctx.save_for_backward(wig, radial, target)
        ctx.rescale = rescale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: ``(N, 9, C)``.

        Returns:
            Gradients for ``(wig, radial, offsets, residual, target, rescale)``.
        """
        wig, radial, target = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_wig = torch.zeros_like(wig)
        grad_rad = torch.empty_like(radial)
        torch.ops.uma_nitro.fused_m0_bwd(
            grad_out, target, wig, radial, grad_wig, grad_rad, ctx.rescale
        )
        return grad_wig, grad_rad, None, grad_out, None, None


def fused_m0_edge_to_node(wig, radial, offsets, residual, target, rescale_factor):
    """
    Compute the radial m=0 contribution and add the node residual.

    The Wigner input must have shape (E, 9, 9); the kernel reads columns 0, 2,
    and 6. The output has shape (N, 9, C).
    """
    return _FusedM0EdgeToNode.apply(
        wig, radial, offsets, residual, target, rescale_factor
    )


@triton.jit
def _fused_m0_bwd_kernel(
    gout_ptr,
    target_ptr,
    wig_ptr,
    rad_ptr,
    grad_wig_ptr,
    grad_rad_ptr,
    num_edges,
    rescale,
    NUM_SPH: tl.constexpr,
    CHANNELS: tl.constexpr,
    GRID: tl.constexpr,
):
    """
    Compute both m=0 gradients directly from the node gradient.

    Target-sorted edges reuse node-gradient rows without materializing a per-edge
    gradient buffer.
    """
    edge = tl.program_id(0)
    c = tl.arange(0, CHANNELS)
    while edge < num_edges:
        node = tl.load(target_ptr + edge)
        gb = node * NUM_SPH * CHANNELS
        dy = (
            tl.load(gout_ptr + gb + 0 * CHANNELS + c),
            tl.load(gout_ptr + gb + 1 * CHANNELS + c),
            tl.load(gout_ptr + gb + 2 * CHANNELS + c),
            tl.load(gout_ptr + gb + 3 * CHANNELS + c),
            tl.load(gout_ptr + gb + 4 * CHANNELS + c),
            tl.load(gout_ptr + gb + 5 * CHANNELS + c),
            tl.load(gout_ptr + gb + 6 * CHANNELS + c),
            tl.load(gout_ptr + gb + 7 * CHANNELS + c),
            tl.load(gout_ptr + gb + 8 * CHANNELS + c),
        )
        wb = edge * NUM_SPH * NUM_SPH
        rb = edge * 3 * CHANNELS
        # COLS is (0, 2, 6): the m=0 columns of the 9x9 inverse Wigner.
        for i in tl.static_range(3):
            col = 0 if i == 0 else (2 if i == 1 else 6)
            acc = tl.zeros((CHANNELS,), dtype=tl.float32)
            for j in tl.static_range(9):
                acc += tl.load(wig_ptr + wb + j * NUM_SPH + col) * dy[j]
            tl.store(
                grad_rad_ptr + rb + i * CHANNELS + c,
                (acc / rescale).to(grad_rad_ptr.dtype.element_ty),
            )
        for i in tl.static_range(3):
            col = 0 if i == 0 else (2 if i == 1 else 6)
            r = tl.load(rad_ptr + rb + i * CHANNELS + c)
            for j in tl.static_range(9):
                tl.store(
                    grad_wig_ptr + wb + j * NUM_SPH + col,
                    tl.sum(dy[j] * r) / rescale,
                )
        edge += GRID


@triton_op("uma_nitro::fused_m0_bwd", mutates_args={"grad_wig", "grad_rad"})
def _fused_m0_bwd(
    grad_out: torch.Tensor,
    target: torch.Tensor,
    wig: torch.Tensor,
    radial: torch.Tensor,
    grad_wig: torch.Tensor,
    grad_rad: torch.Tensor,
    rescale: float,
) -> None:
    """
    Launch the fused m=0 backward.

    Args:
        grad_out: Node gradient, ``(N, 9, C)``.
        target: Target node per edge, ``(E,)``.
        wig: Forward's ``(E, 9, 9)``.
        radial: Forward's ``(E, 3, C)``.
        grad_wig: Output ``(E, 9, 9)``; mutated, must start zeroed since only
            the m=0 columns are written.
        grad_rad: Output ``(E, 3, C)``; mutated.
        rescale: Rescale factor.
    """
    num_sph, channels = grad_out.shape[1], grad_out.shape[2]
    _gb = min(target.shape[0], FED_GRID_BWD)
    wrap_triton(_fused_m0_bwd_kernel)[(_gb,)](
        grad_out,
        target,
        wig,
        radial,
        grad_wig,
        grad_rad,
        target.shape[0],
        rescale,
        NUM_SPH=num_sph,
        CHANNELS=channels,
        GRID=_gb,
        num_warps=FED_WARPS_BWD,
    )


LNS_GRID = int(os.environ.get("UMA_NITRO_LNS_GRID", "8192"))

# The exponent every per-row / per-tensor fp16 power-of-two scale aims its maximum
# at, and the magnitude that follows from it. 14 is the top of fp16's normal range
# on an exact power of two, so a scaled row max lands just under 65504 with the
# whole subnormal range still below it.
#
# One source of truth, because the two host-side *bounds* that certify those scales
# (`_dbeta_bound`, `_gradblk_bound`) have to agree with the kernels that produce
# them, and until now both sides wrote the literal `16384.0`. Editing one and not
# the other silently invalidates a bound whose failure mode is `inf` in a stored
# buffer and NaN forces with a finite energy -- the exact shape of the `exp2`
# overflow and the `D_BETA` near-miss. Triton cannot read a plain module global, so
# the kernels take the `tl.constexpr` alias; it is derived from the same float
# rather than written out again.
_FP16_SCALE_EXP = 14.0
_FP16_SCALE_TARGET = 2.0**_FP16_SCALE_EXP
FP16_SCALE_TARGET = tl.constexpr(_FP16_SCALE_TARGET)
LNS_WARPS = int(os.environ.get("UMA_NITRO_LNS_WARPS", "8"))
BLOCK_R = int(os.environ.get("UMA_NITRO_LNS_BLOCK_R", "8"))


@triton.jit
def _ln_silu_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    num_rows,
    eps,
    x_stride,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    Apply LayerNorm and SiLU to a row tile, saving its mean and rstd.

    An explicit row stride accepts column slices without a contiguous copy.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    w = tl.load(w_ptr + c)[None, :]
    b = tl.load(b_ptr + c)[None, :]
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        off = rows[:, None] * x_stride + c[None, :]
        out_off = rows[:, None] * C + c[None, :]
        x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        y = d * rstd[:, None] * w + b
        tl.store(out_ptr + out_off, y * tl.sigmoid(y), mask=m[:, None])
        tl.store(mean_ptr + rows, mean, mask=m)
        tl.store(rstd_ptr + rows, rstd, mask=m)
        blk += GRID_S


@triton.jit
def _silu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    out_ptr,
    num_rows,
    x_stride,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    Multiply the upstream gradient by the SiLU derivative.

    The pre-SiLU value is reconstructed from the saved LayerNorm statistics.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    w = tl.load(w_ptr + c)[None, :]
    bb = tl.load(b_ptr + c)[None, :]
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        off = rows[:, None] * x_stride + c[None, :]
        flat = rows[:, None] * C + c[None, :]
        x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=m, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=m, other=0.0)
        y = (x - mean[:, None]) * rstd[:, None] * w + bb
        sg = tl.sigmoid(y)
        g = tl.load(grad_out_ptr + flat, mask=m[:, None], other=0.0).to(tl.float32)
        # d(y*sigmoid(y))/dy = s * (1 + y * (1 - s))
        tl.store(out_ptr + flat, g * sg * (1.0 + y * (1.0 - sg)), mask=m[:, None])
        blk += GRID_S


@triton.jit
def _ln_silu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    num_rows,
    x_stride,
    dx_stride,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    Compute the SiLU and LayerNorm input gradient in one pass.

    Inference parameters are frozen, so only within-row reductions are needed.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    w = tl.load(w_ptr + c)[None, :]
    bb = tl.load(b_ptr + c)[None, :]
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        off = rows[:, None] * x_stride + c[None, :]
        flat = rows[:, None] * C + c[None, :]
        dxo = rows[:, None] * dx_stride + c[None, :]
        x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=m, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=m, other=0.0)
        xhat = (x - mean[:, None]) * rstd[:, None]
        y = xhat * w + bb
        sg = tl.sigmoid(y)
        g = tl.load(grad_out_ptr + flat, mask=m[:, None], other=0.0).to(tl.float32)
        # d(y*sigmoid(y))/dy = s * (1 + y * (1 - s)), then through the affine.
        dxhat = g * sg * (1.0 + y * (1.0 - sg)) * w
        # Standard LayerNorm dx. Both means are over the channel axis only.
        c1 = tl.sum(dxhat, axis=1) / C
        c2 = tl.sum(dxhat * xhat, axis=1) / C
        dx = rstd[:, None] * (dxhat - c1[:, None] - xhat * c2[:, None])
        tl.store(dx_ptr + dxo, dx, mask=m[:, None])
        blk += GRID_S


@triton_op("uma_nitro::ln_silu_bwd_dx", mutates_args={"dx"})
def _ln_silu_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
) -> None:
    """
    Launch the fused SiLU'+LayerNorm backward.

    Args:
        grad_out: Upstream gradient, ``(R, C)``, contiguous.
        x: Forward input, ``(R, C)``, possibly a strided column slice.
        weight: LayerNorm weight, ``(C,)``.
        bias: LayerNorm bias, ``(C,)``.
        mean: Saved mean, ``(R,)``.
        rstd: Saved reciprocal std, ``(R,)``.
        dx: Result, ``(R, C)``; mutated.
    """
    rows, channels = x.shape
    _g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_ln_silu_bwd_kernel)[(_g,)](
        grad_out,
        x,
        weight,
        bias,
        mean,
        rstd,
        dx,
        rows,
        x.stride(0),
        dx.stride(0),
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=_g,
        num_warps=LNS_WARPS,
    )


@triton_op("uma_nitro::ln_silu_fwd", mutates_args={"out", "mean", "rstd"})
def _ln_silu_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
) -> None:
    """
    Launch the fused LayerNorm+SiLU forward.

    Args:
        x: Input, ``(R, C)``.
        weight: LayerNorm weight, ``(C,)``.
        bias: LayerNorm bias, ``(C,)``.
        out: Output, ``(R, C)``; mutated.
        mean: Saved mean, ``(R,)``; mutated.
        rstd: Saved reciprocal std, ``(R,)``; mutated.
        eps: LayerNorm epsilon.
    """
    rows, channels = x.shape
    _g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_ln_silu_fwd_kernel)[(_g,)](
        x,
        weight,
        bias,
        out,
        mean,
        rstd,
        rows,
        eps,
        x.stride(0),
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=_g,
        num_warps=LNS_WARPS,
    )


@triton_op("uma_nitro::ln_silu_silubwd", mutates_args={"out"})
def _silu_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Launch the SiLU-derivative pass that precedes the weight-gradient reduction.

    Args:
        grad_out: Upstream gradient, ``(R, C)``.
        x: Forward input, ``(R, C)``.
        weight: LayerNorm weight, ``(C,)``.
        bias: LayerNorm bias, ``(C,)``.
        mean: Saved mean, ``(R,)``.
        rstd: Saved reciprocal std, ``(R,)``.
        out: Result, ``(R, C)``; mutated.
    """
    rows, channels = x.shape
    _g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_silu_bwd_kernel)[(_g,)](
        grad_out,
        x,
        weight,
        bias,
        mean,
        rstd,
        out,
        rows,
        x.stride(0),
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=_g,
        num_warps=LNS_WARPS,
    )


@triton.jit
def _ln_dw_kernel(
    x_ptr,
    g_ptr,
    mean_ptr,
    rstd_ptr,
    part_ptr,
    rows,
    x_row_stride,
    g_row_stride,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID: tl.constexpr,
):
    """
    Compute deterministic fp32 partials of the LayerNorm weight gradient.
    """
    pid = tl.program_id(0)
    c = tl.arange(0, C)
    acc = tl.zeros((C,), dtype=tl.float32)
    row = pid.to(tl.int64) * BLOCK_R
    while row < rows:
        for i in tl.static_range(BLOCK_R):
            r = row + i
            m = r < rows
            xv = tl.load(x_ptr + r * x_row_stride + c, mask=m, other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + r * g_row_stride + c, mask=m, other=0.0).to(tl.float32)
            mu = tl.load(mean_ptr + r, mask=m, other=0.0).to(tl.float32)
            rs = tl.load(rstd_ptr + r, mask=m, other=0.0).to(tl.float32)
            acc += gv * (xv - mu) * rs
        row += GRID * BLOCK_R
    tl.store(part_ptr + pid * C + c, acc)


def _ln_weight_grad(x, g, mean, rstd):
    """
    Compute the LayerNorm weight gradient from row-wise partials.
    """
    rows, channels = x.shape
    # One source of truth for the row block: the grid is sized from it, and a
    # grid larger than the row count would leave programs writing partials for
    # rows that do not exist.
    block_r = 8
    grid = min(LNS_GRID, max(1, triton.cdiv(rows, block_r)))
    part = torch.empty(grid, channels, device=x.device, dtype=torch.float32)
    wrap_triton(_ln_dw_kernel)[(grid,)](
        x,
        g,
        mean,
        rstd,
        part,
        rows,
        x.stride(0),
        g.stride(0),
        C=channels,
        BLOCK_R=block_r,
        GRID=grid,
        num_warps=4,
    )
    return part.sum(0).to(x.dtype)


class _LNSiLU(torch.autograd.Function):
    """
    Autograd wrapper for the fused LayerNorm+SiLU.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        """
        Args:
            x: Input, ``(R, C)``.
            weight: LayerNorm weight.
            bias: LayerNorm bias.
            eps: LayerNorm epsilon.

        Returns:
            ``silu(layer_norm(x) * weight + bias)``.
        """
        rows = x.shape[0]
        # empty_like would inherit x's stride when x is a column slice.
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        torch.ops.uma_nitro.ln_silu_fwd(x, weight, bias, out, mean, rstd, eps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: Upstream gradient, ``(R, C)``.

        Returns:
            Gradients for ``(x, weight, bias, eps)``.
        """
        x, weight, bias, mean, rstd = ctx.saved_tensors
        if not ctx.needs_input_grad[1]:
            # The common inference path: parameters are frozen, so dw is dead and the
            # whole backward is one kernel. See _ln_silu_bwd_kernel.
            dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            torch.ops.uma_nitro.ln_silu_bwd_dx(
                grad_out.contiguous(), x, weight, bias, mean, rstd, dx
            )
            return dx, None, None, None

        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        torch.ops.uma_nitro.ln_silu_bwd_dx(
            grad_out.contiguous(), x, weight, bias, mean, rstd, dx
        )
        g = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        torch.ops.uma_nitro.ln_silu_silubwd(
            grad_out.contiguous(), x, weight, bias, mean, rstd, g
        )
        # has_bias=False in the original: the affine parameters are frozen in
        # inference, so db is a value autograd discards. dx does not depend on the
        # bias either way, so dropping db cannot change a result.
        return dx, _ln_weight_grad(x, g, mean, rstd), None, None


def ln_silu(x, weight, bias, eps):
    """
    Fused ``silu(layer_norm(x, weight, bias))`` over the last dimension.

    Args:
        x: Input, ``(R, C)``.
        weight: LayerNorm weight, ``(C,)``.
        bias: LayerNorm bias, ``(C,)``.
        eps: LayerNorm epsilon.

    Returns:
        ``(R, C)``.
    """
    return _LNSiLU.apply(x, weight, bias, eps)


CO_BLOCK = 256


@triton.jit
def _offsets_fwd_kernel(
    off_ptr,
    batch_ptr,
    row_ptr,
    cell_ptr,
    out_ptr,
    n_edges,
    BLOCK_N: tl.constexpr,
):
    """
    ``out[e, j] = sum_k cell_offsets[e, k] * cell[system[e], k, j]``.
    """
    e = (tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    m = e < n_edges
    # batch[row[e]] read here instead of materialized by an `index_select`: the
    # kernel is already gathering per edge, so the extra indirection is free and
    # the (E,) intermediate and its launch both disappear.
    s = tl.load(batch_ptr + tl.load(row_ptr + e, mask=m, other=0), mask=m, other=0)
    o0 = tl.load(off_ptr + e * 3 + 0, mask=m, other=0).to(tl.float32)
    o1 = tl.load(off_ptr + e * 3 + 1, mask=m, other=0).to(tl.float32)
    o2 = tl.load(off_ptr + e * 3 + 2, mask=m, other=0).to(tl.float32)
    base = s * 9
    for j in tl.static_range(3):
        v = (
            o0 * tl.load(cell_ptr + base + 0 * 3 + j, mask=m, other=0.0)
            + o1 * tl.load(cell_ptr + base + 1 * 3 + j, mask=m, other=0.0)
            + o2 * tl.load(cell_ptr + base + 2 * 3 + j, mask=m, other=0.0)
        )
        tl.store(out_ptr + e * 3 + j, v, mask=m)


@triton.jit
def _offsets_bwd_kernel(
    gout_ptr,
    off_ptr,
    batch_ptr,
    row_ptr,
    gcell_ptr,
    n_edges,
    n_sys,
    BLOCK_N: tl.constexpr,
):
    """
    ``grad_cell[s, k, j] = sum_{e : system[e] = s} cell_offsets[e, k] * g[e, j]``.

    Reduced within the block first, so each program contributes at most
    ``n_sys * 9`` atomics rather than one per edge. That is what removes the
    contention the eager ``index_add_`` pays.
    """
    e = (tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    m = e < n_edges
    s = tl.where(
        m,
        tl.load(batch_ptr + tl.load(row_ptr + e, mask=m, other=0), mask=m, other=0),
        -1,
    )
    o0 = tl.load(off_ptr + e * 3 + 0, mask=m, other=0).to(tl.float32)
    o1 = tl.load(off_ptr + e * 3 + 1, mask=m, other=0).to(tl.float32)
    o2 = tl.load(off_ptr + e * 3 + 2, mask=m, other=0).to(tl.float32)
    g0 = tl.load(gout_ptr + e * 3 + 0, mask=m, other=0.0)
    g1 = tl.load(gout_ptr + e * 3 + 1, mask=m, other=0.0)
    g2 = tl.load(gout_ptr + e * 3 + 2, mask=m, other=0.0)
    for sid in range(n_sys):
        keep = m & (s == sid)
        for k in tl.static_range(3):
            ok = tl.where(k == 0, o0, tl.where(k == 1, o1, o2))
            for j in tl.static_range(3):
                gj = tl.where(j == 0, g0, tl.where(j == 1, g1, g2))
                part = tl.sum(tl.where(keep, ok * gj, 0.0), axis=0)
                tl.atomic_add(gcell_ptr + sid * 9 + k * 3 + j, part)


@triton_op("uma_nitro::cell_offsets_fwd", mutates_args={"out"})
def _fwd(
    cell_offsets: torch.Tensor,
    batch: torch.Tensor,
    row: torch.Tensor,
    cell: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Launch the per-edge cell-offset product.

    Args:
        cell_offsets: Integer image offsets, ``(E, 3)``.
        batch: System index per atom, ``(N,)``.
        row: Source atom per edge, ``(E,)``.
        cell: Lattice matrices, ``(S, 3, 3)``.
        out: Output ``(E, 3)``; mutated.
    """
    n = cell_offsets.shape[0]
    wrap_triton(_offsets_fwd_kernel)[(triton.cdiv(n, CO_BLOCK),)](
        cell_offsets, batch, row, cell, out, n, BLOCK_N=CO_BLOCK, num_warps=4
    )


@triton_op("uma_nitro::cell_offsets_bwd", mutates_args={"gcell"})
def _bwd(
    grad_out: torch.Tensor,
    cell_offsets: torch.Tensor,
    batch: torch.Tensor,
    row: torch.Tensor,
    gcell: torch.Tensor,
) -> None:
    """
    Launch the cell-gradient reduction.

    Args:
        grad_out: Gradient of ``out``, ``(E, 3)``.
        cell_offsets: Integer image offsets, ``(E, 3)``.
        system: System index per edge, ``(E,)``.
        gcell: Output ``(S, 3, 3)``, pre-zeroed; mutated.
    """
    n = cell_offsets.shape[0]
    wrap_triton(_offsets_bwd_kernel)[(triton.cdiv(n, CO_BLOCK),)](
        grad_out,
        cell_offsets,
        batch,
        row,
        gcell,
        n,
        gcell.shape[0],
        BLOCK_N=CO_BLOCK,
        num_warps=4,
    )


class _CellOffsets(torch.autograd.Function):
    """
    Autograd wrapper for the per-edge cell-offset product.
    """

    @staticmethod
    def forward(ctx, cell_offsets, batch, row, cell):
        """
        Args:
            cell_offsets: Integer image offsets, ``(E, 3)``.
            system: System index per edge, ``(E,)``.
            cell: Lattice matrices, ``(S, 3, 3)``.

        Returns:
            ``(E, 3)`` offsets in the cell's dtype.
        """
        out = torch.empty(
            cell_offsets.shape[0], 3, device=cell.device, dtype=cell.dtype
        )
        torch.ops.uma_nitro.cell_offsets_fwd(cell_offsets, batch, row, cell, out)
        ctx.save_for_backward(cell_offsets, batch, row)
        ctx.cell_shape = cell.shape
        ctx.cell_dtype = cell.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: Gradient of the offsets, ``(E, 3)``.

        Returns:
            A gradient for ``cell`` only; the offsets and system ids are integral.
        """
        cell_offsets, batch, row = ctx.saved_tensors
        gcell = torch.zeros(
            ctx.cell_shape, device=grad_out.device, dtype=ctx.cell_dtype
        )
        torch.ops.uma_nitro.cell_offsets_bwd(
            grad_out.contiguous(), cell_offsets, batch, row, gcell
        )
        return None, None, None, gcell


def cell_offset_product(cell_offsets, batch, row, cell):
    """
    Per-edge ``cell_offsets @ cell[batch[row]]``.

    Args:
        cell_offsets: Integer image offsets, ``(E, 3)``.
        batch: System index per atom, ``(N,)``.
        row: Source atom per edge, ``(E,)``.
        cell: Lattice matrices, ``(S, 3, 3)``.

    Returns:
        ``(E, 3)``.
    """
    return _CellOffsets.apply(cell_offsets, batch, row, cell)


EE_BLOCK = 128
EE_WARPS = 4

_ENV_EXPONENT = 5.0


def _assert_envelope_exponent(envelope):
    """
    Fail loudly if the checkpoint's envelope exponent is not the one compiled in.

    ``_envelope`` unrolls ``t**p`` as ``t2*t2*t`` and differentiates it as
    ``5 * t4 * inner``, i.e. ``p = 5`` is baked into the *instructions*, while
    ``a``, ``b`` and ``c`` are read from the module at every call. ``p`` is a
    constructor argument of ``PolynomialEnvelope`` (``exponent: int = 5``), so a
    checkpoint trained with any other value would be evaluated with that
    checkpoint's coefficients and this file's power -- a smoothly wrong energy and
    a smoothly wrong force, with nothing out of range and nothing non-finite for
    any guard to catch. The single most silent constant in the variant.

    Same treatment as ``_assert_d55_pattern``, which checks the *other* piece of
    algebra that is compiled in rather than read.

    Args:
        envelope: ``backbone.envelope``.

    Raises:
        RuntimeError: If the exponent differs from the compiled-in 5.
    """
    # The default must NOT be the value being checked. Written as
    # ``getattr(envelope, "p", _ENV_EXPONENT)`` this guard degrades to a
    # tautology for any envelope class that names its exponent differently --
    # it becomes a comment, silently, which is the exact failure mode it exists
    # to prevent.
    p = getattr(envelope, "p", None)
    if p is None:
        raise UMANitroNotSupportedError(
            f"UMA Nitro: envelope {type(envelope).__name__} exposes no `p`, "
            f"so the exponent compiled into _envelope ({_ENV_EXPONENT}) cannot "
            f"be checked against it"
        )
    if float(p) != _ENV_EXPONENT:
        raise UMANitroNotSupportedError(
            f"UMA Nitro: checkpoint envelope exponent p={p} but _envelope "
            f"has p={_ENV_EXPONENT} unrolled into its instructions; regenerate "
            f"the polynomial and its derivative"
        )


_LMAX = 2
_M_SPLIT = (3, 4, 2)

# The same defect, one layer up. `_ln_silu_multi_bwd*` takes `L` from
# `weight.shape[0]` -- which is `mlp.num_layers + 1`, a checkpoint property --
# but carries exactly five gradient pointers, and both its kernels *fall
# through* past block 4: the fp32 one re-reads block 4's gradient, and the fp16
# one stores block 0's gradient into blocks 5..L-1's columns. The launcher's
# `pad[:5]` additionally drops `grads[5:]` before the kernel is reached.
#
# uma-s-1p2 is `num_layers = 4`, i.e. L = 5, exactly at the cap; the released
# K10L4 and K16L6 backbones are L = 11 and L = 17. Worse, only the *backward*
# is capped -- `_ln_silu_multi_fwd_kernel` is generic in L -- so a larger
# backbone would return a correct energy with wrong forces and stress, which is
# the mode every magnitude-based gate in this project is weakest at.
_LNS_MAX_BLOCKS = 5

_M_TO_L_LOADS = (
    (0, 0),
    (1, 2),
    (0, 1),
    (1, 0),
    (2, 1),
    (1, 3),
    (0, 2),
    (1, 1),
    (2, 0),
)


def _assert_m_to_l_table() -> None:
    """
    Fail loudly if the inverse-rotation kernel's inlined permute has drifted
    from the shared one.

    Raises:
        RuntimeError: If the ``(block, offset)`` pairs compiled into
            ``_invrot_segreduce_kernel`` disagree with ``M_TO_L_GATHER_IDX``
            split by the m-block sizes.
    """
    bounds, run = [], 0
    for size in _M_SPLIT:
        bounds.append(run)
        run += size
    want = []
    for m in M_TO_L_GATHER_IDX:
        block = max(i for i, start in enumerate(bounds) if m >= start)
        want.append((block, m - bounds[block]))
    if tuple(want) != _M_TO_L_LOADS:
        raise RuntimeError(
            f"UMA Nitro: M_TO_L_GATHER_IDX={M_TO_L_GATHER_IDX} with m-blocks "
            f"{_M_SPLIT} implies loads {tuple(want)}, but "
            f"_invrot_segreduce_kernel has {_M_TO_L_LOADS} inlined"
        )


def _assert_lns_blocks(blocks: int) -> None:
    """
    Fail loudly if the joint LayerNorm+SiLU backward is handed more blocks than
    its kernels carry pointers for.

    Args:
        blocks: ``weight.shape[0]``, i.e. the number of column blocks.

    Raises:
        RuntimeError: If it exceeds the compiled-in maximum.
    """
    if blocks > _LNS_MAX_BLOCKS:
        raise UMANitroNotSupportedError(
            f"UMA Nitro: ln_silu_multi backward got {blocks} column blocks "
            f"but its kernels unroll {_LNS_MAX_BLOCKS} gradient pointers; "
            f"blocks {_LNS_MAX_BLOCKS}.. would silently take block "
            f"{_LNS_MAX_BLOCKS - 1}'s gradient (fp32) or block 0's (fp16)"
        )


def _assert_lmax2(backbone) -> None:
    """
    Validate that the checkpoint matches the degree-2 kernel layout.
    """
    lmax = int(getattr(backbone, "lmax", -1))
    mmax = int(getattr(backbone, "mmax", -1))
    if lmax != _LMAX or mmax != _LMAX:
        raise UMANitroNotSupportedError(
            f"UMA Nitro: checkpoint has lmax={lmax}, mmax={mmax}; every "
            f"kernel here unrolls lmax=mmax={_LMAX} (9 coefficients, m-blocks "
            f"{_M_SPLIT}) into its instructions"
        )
    coeffs = int(getattr(backbone, "sph_feature_size", (lmax + 1) ** 2))
    if coeffs != (_LMAX + 1) ** 2:
        raise UMANitroNotSupportedError(
            f"UMA Nitro: sph_feature_size={coeffs}, expected " f"{(_LMAX + 1) ** 2}"
        )
    for i, block in enumerate(backbone.blocks):
        split = tuple(int(s) for s in block.edge_wise.so2_conv_2.m_split_sizes)
        if split != _M_SPLIT:
            raise UMANitroNotSupportedError(
                f"UMA Nitro: block {i} has m_split_sizes={split}; the conv1 "
                f"and conv2 backward kernels bake {_M_SPLIT} into their row "
                f"strides while the forward reads it from the checkpoint"
            )


@triton.jit
def _envelope(t, p_a, p_b, p_c):
    """
    The polynomial envelope ``1 + t^5 (a + t (b + c t))`` and its ``t`` derivative.

    ``p`` is 5.0 in the checkpoint, so the power is unrolled as four multiplies
    rather than an ``exp(5 log t)``, which would lose accuracy near ``t = 0``.
    Checked against the live module by ``_assert_envelope_exponent``: only ``a``,
    ``b`` and ``c`` travel from the checkpoint, so a different ``p`` would be
    silently mis-evaluated.
    """
    t2 = t * t
    t4 = t2 * t2
    t5 = t4 * t
    inner = p_a + t * (p_b + p_c * t)
    val = 1.0 + t5 * inner
    dval = 5.0 * t4 * inner + t5 * (p_b + 2.0 * p_c * t)
    return val, dval


@triton.jit
def _edge_emb_fwd_kernel(
    dist_ptr,
    offset_ptr,
    anum_ptr,
    ei_ptr,
    src_w_ptr,
    tgt_w_ptr,
    xedge_ptr,
    env_ptr,
    n_edges,
    edge_stride,
    coeff,
    inv_cutoff,
    p_a,
    p_b,
    p_c,
    NG: tl.constexpr,
    NC: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Write Gaussian and source/target embedding blocks into final columns.

    Stores remain inline to keep temporary live ranges short.
    """
    e = (tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    m = e < n_edges
    d = tl.load(dist_ptr + e, mask=m, other=0.0)

    # Gaussian distance expansion into columns [0, NG).
    j = tl.arange(0, NG)[None, :]
    diff = d[:, None] - tl.load(offset_ptr + j)
    tl.store(
        xedge_ptr + e[:, None] * WIDTH + j,
        tl.exp(coeff * diff * diff).to(xedge_ptr.dtype.element_ty),
        mask=m[:, None],
    )

    # Atom-type embeddings into [NG, NG+NC) and [NG+NC, NG+2NC). The double
    # indirection atomic_numbers[edge_index[k, e]] stays in registers.
    c = tl.arange(0, NC)[None, :]
    si = tl.load(anum_ptr + tl.load(ei_ptr + e, mask=m, other=0), mask=m, other=0)
    ti = tl.load(
        anum_ptr + tl.load(ei_ptr + edge_stride + e, mask=m, other=0), mask=m, other=0
    )
    tl.store(
        xedge_ptr + e[:, None] * WIDTH + NG + c,
        tl.load(src_w_ptr + si[:, None] * NC + c, mask=m[:, None], other=0.0).to(
            xedge_ptr.dtype.element_ty
        ),
        mask=m[:, None],
    )
    tl.store(
        xedge_ptr + e[:, None] * WIDTH + NG + NC + c,
        tl.load(tgt_w_ptr + ti[:, None] * NC + c, mask=m[:, None], other=0.0).to(
            xedge_ptr.dtype.element_ty
        ),
        mask=m[:, None],
    )

    t = d * inv_cutoff
    val, _ = _envelope(t, p_a, p_b, p_c)
    tl.store(env_ptr + e, tl.where(t < 1.0, val, 0.0), mask=m)


@triton.jit
def _edge_emb_bwd_kernel(
    gxedge_ptr,
    genv_ptr,
    dist_ptr,
    offset_ptr,
    gdist_ptr,
    n_edges,
    coeff,
    inv_cutoff,
    p_a,
    p_b,
    p_c,
    NG: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Reduce the Gaussian and envelope gradients onto ``edge_distance``.

    Reads only the ``NG`` Gaussian columns of the upstream gradient; the embedding
    columns carry no distance dependence.
    """
    e = (tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    m = e < n_edges
    d = tl.load(dist_ptr + e, mask=m, other=0.0)
    j = tl.arange(0, NG)[None, :]
    diff = d[:, None] - tl.load(offset_ptr + j)
    gauss = tl.exp(coeff * diff * diff)
    # ``.to(tl.float32)`` so one kernel serves an fp32 or an fp16 upstream
    # gradient; a no-op for fp32.
    g = tl.load(gxedge_ptr + e[:, None] * WIDTH + j, mask=m[:, None], other=0.0).to(
        tl.float32
    )
    gd = tl.sum(g * gauss * (2.0 * coeff) * diff, axis=1)

    t = d * inv_cutoff
    _, dval = _envelope(t, p_a, p_b, p_c)
    ge = tl.load(genv_ptr + e, mask=m, other=0.0)
    gd += tl.where(t < 1.0, ge * dval * inv_cutoff, 0.0)
    tl.store(gdist_ptr + e, gd, mask=m)


@triton_op("uma_nitro::edge_emb_fwd", mutates_args={"xedge", "env"})
def _edge_emb_fwd(
    dist: torch.Tensor,
    offset: torch.Tensor,
    atomic_numbers: torch.Tensor,
    edge_index: torch.Tensor,
    src_w: torch.Tensor,
    tgt_w: torch.Tensor,
    xedge: torch.Tensor,
    env: torch.Tensor,
    coeff: float,
    cutoff: float,
    p_a: float,
    p_b: float,
    p_c: float,
) -> None:
    """
    Launch the fused edge-embedding forward.

    Args:
        dist: Edge distances, ``(E,)``.
        offset: Gaussian centres, ``(NG,)``.
        atomic_numbers: Per-node atom types, ``(N,)``.
        edge_index: ``(2, E)``.
        src_w: Source embedding table, ``(T, NC)``.
        tgt_w: Target embedding table, ``(T, NC)``.
        xedge: Output ``(E, NG + 2 * NC)``; mutated.
        env: Output envelope ``(E,)``; mutated.
        coeff: Gaussian coefficient.
        cutoff: Radial cutoff.
        p_a: Envelope ``a``.
        p_b: Envelope ``b``.
        p_c: Envelope ``c``.
    """
    n = dist.shape[0]
    ng = offset.shape[0]
    nc = src_w.shape[1]
    wrap_triton(_edge_emb_fwd_kernel)[(triton.cdiv(n, EE_BLOCK),)](
        dist,
        offset,
        atomic_numbers,
        edge_index,
        src_w,
        tgt_w,
        xedge,
        env,
        n,
        edge_index.stride(0),
        coeff,
        1.0 / cutoff,
        p_a,
        p_b,
        p_c,
        NG=ng,
        NC=nc,
        WIDTH=ng + 2 * nc,
        BLOCK_N=EE_BLOCK,
        num_warps=EE_WARPS,
    )


@triton_op("uma_nitro::edge_emb_bwd", mutates_args={"gdist"})
def _edge_emb_bwd(
    gxedge: torch.Tensor,
    genv: torch.Tensor,
    dist: torch.Tensor,
    offset: torch.Tensor,
    gdist: torch.Tensor,
    coeff: float,
    cutoff: float,
    p_a: float,
    p_b: float,
    p_c: float,
) -> None:
    """
    Launch the fused edge-embedding backward.

    Args:
        gxedge: Gradient of ``xedge``, ``(E, NG + 2 * NC)``.
        genv: Gradient of the envelope, ``(E,)``.
        dist: Edge distances, ``(E,)``.
        offset: Gaussian centres, ``(NG,)``.
        gdist: Output gradient ``(E,)``; mutated.
        coeff: Gaussian coefficient.
        cutoff: Radial cutoff.
        p_a: Envelope ``a``.
        p_b: Envelope ``b``.
        p_c: Envelope ``c``.
    """
    n = dist.shape[0]
    ng = offset.shape[0]
    wrap_triton(_edge_emb_bwd_kernel)[(triton.cdiv(n, EE_BLOCK),)](
        gxedge,
        genv,
        dist,
        offset,
        gdist,
        n,
        coeff,
        1.0 / cutoff,
        p_a,
        p_b,
        p_c,
        NG=ng,
        WIDTH=gxedge.shape[1],
        BLOCK_N=EE_BLOCK,
        num_warps=EE_WARPS,
    )


class _EdgeEmbedding(torch.autograd.Function):
    """
    Autograd wrapper for the fused edge embedding and envelope.
    """

    @staticmethod
    def forward(
        ctx,
        dist,
        offset,
        atomic_numbers,
        edge_index,
        src_w,
        tgt_w,
        coeff,
        cutoff,
        p_a,
        p_b,
        p_c,
        narrow_grad=False,
        narrow_val=False,
    ):
        """
        Args:
            dist: Edge distances, ``(E,)``.
            offset: Gaussian centres, ``(NG,)``.
            atomic_numbers: Per-node atom types, ``(N,)``.
            edge_index: ``(2, E)``.
            src_w: Source embedding table.
            tgt_w: Target embedding table.
            coeff: Gaussian coefficient.
            cutoff: Radial cutoff.
            p_a: Envelope ``a``.
            p_b: Envelope ``b``.
            p_c: Envelope ``c``.
            narrow_grad: Return a third output, an fp16 gradient-channel token for
                ``x_edge``, and mark ``x_edge`` itself non-differentiable.
            narrow_val: Store ``x_edge`` itself as fp16.

        Returns:
            ``(x_edge, envelope)`` of shapes ``(E, NG + 2 * NC)`` and ``(E,)``,
            plus the fp16 token when ``narrow_grad``.
        """
        n = dist.shape[0]
        width = offset.shape[0] + 2 * src_w.shape[1]
        xedge = torch.empty(
            n,
            width,
            device=dist.device,
            dtype=torch.float16 if narrow_val else dist.dtype,
        )
        env = torch.empty(n, device=dist.device, dtype=dist.dtype)
        torch.ops.uma_nitro.edge_emb_fwd(
            dist,
            offset,
            atomic_numbers,
            edge_index,
            src_w,
            tgt_w,
            xedge,
            env,
            coeff,
            cutoff,
            p_a,
            p_b,
            p_c,
        )
        ctx.save_for_backward(dist, offset)
        ctx.consts = (coeff, cutoff, p_a, p_b, p_c)
        ctx.set_materialize_grads(False)
        ctx.narrow_grad = narrow_grad
        if not narrow_grad:
            return xedge, env
        # ``x_edge`` stays fp32 -- it is the merged head's forward A operand and is
        # not a narrowing candidate -- while its gradient rides the token. Two bytes.
        tok = torch.empty(1, device=dist.device, dtype=torch.float16).expand(n, width)
        ctx.mark_non_differentiable(xedge)
        return xedge, env, tok

    @staticmethod
    def backward(ctx, gxedge, genv, gtok=None):
        """
        Args:
            gxedge: Gradient of ``x_edge``, or None (always None when the token is
                in use, since ``x_edge`` is then non-differentiable).
            genv: Gradient of the envelope, or None.
            gtok: fp16 gradient of ``x_edge``, when the token is in use.

        Returns:
            A gradient for ``dist`` only; every other input is frozen or integral.
        """
        dist, offset = ctx.saved_tensors
        coeff, cutoff, p_a, p_b, p_c = ctx.consts
        n = dist.shape[0]
        if ctx.narrow_grad:
            gxedge = gtok
        width = gxedge.shape[1] if gxedge is not None else offset.shape[0]
        if gxedge is None:
            gxedge = torch.zeros(n, width, device=dist.device, dtype=dist.dtype)
        if genv is None:
            genv = torch.zeros(n, device=dist.device, dtype=dist.dtype)
        gdist = torch.empty_like(dist)
        torch.ops.uma_nitro.edge_emb_bwd(
            gxedge.contiguous(),
            genv.contiguous(),
            dist,
            offset,
            gdist,
            coeff,
            cutoff,
            p_a,
            p_b,
            p_c,
        )
        return (gdist,) + (None,) * 12


def edge_embedding(
    dist,
    offset,
    atomic_numbers,
    edge_index,
    src_w,
    tgt_w,
    coeff,
    cutoff,
    p_a,
    p_b,
    p_c,
    narrow_grad=False,
    narrow_val=False,
):
    """
    Fused edge embedding and radial envelope.

    Args:
        dist: Edge distances, ``(E,)``.
        offset: Gaussian centres, ``(NG,)``.
        atomic_numbers: Per-node atom types, ``(N,)``.
        edge_index: ``(2, E)``.
        src_w: Source embedding table, ``(T, NC)``.
        tgt_w: Target embedding table, ``(T, NC)``.
        coeff: Gaussian coefficient.
        cutoff: Radial cutoff.
        p_a: Envelope ``a``.
        p_b: Envelope ``b``.
        p_c: Envelope ``c``.

    Returns:
        ``(x_edge, envelope)``, plus an fp16 gradient token when ``narrow_grad``.
    """
    return _EdgeEmbedding.apply(
        dist,
        offset,
        atomic_numbers,
        edge_index,
        src_w,
        tgt_w,
        coeff,
        cutoff,
        p_a,
        p_b,
        p_c,
        narrow_grad,
        narrow_val,
    )


BLOCK_E = int(os.environ.get("UMA_NITRO_WFV_BLOCK", "16"))
WFV_WARPS = int(os.environ.get("UMA_NITRO_WFV_WARPS", "2"))
BLOCK_MOVE = int(os.environ.get("UMA_NITRO_WFV_MOVE", "32"))


@triton.jit
def _rn_tf32(x):
    """
    Round to TF32 with round-to-nearest instead of truncation.

    Pre-rounding avoids systematic truncation bias while preserving
    TF32-compatible inputs.
    """
    return tl.inline_asm_elementwise(
        "cvt.rn.tf32.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dot3(a, b, use_tf32: tl.constexpr):
    """
    Compute an fp32-accurate dot product with a three-term TF32 split.
    """
    if use_tf32:
        ahi = (a.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
        alo = _rn_tf32(a - ahi)
        bhi = (b.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
        blo = _rn_tf32(b - bhi)
        acc = tl.dot(ahi, bhi, input_precision="tf32")
        acc += tl.dot(ahi, blo, input_precision="tf32")
        acc += tl.dot(alo, bhi, input_precision="tf32")
        return acc
    return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _q2_tile(qw, qx, qy, qz):
    """
    Build the ``(BLOCK_E, 16)`` outer product ``q (x) q`` from four lane vectors.

    Returns:
        ``q2[:, i * 4 + j] = q[:, i] * q[:, j]``.
    """
    idx = tl.arange(0, 16)
    a = (idx // 4)[None, :]
    b = (idx % 4)[None, :]
    qa = tl.where(
        a == 0,
        qw[:, None],
        tl.where(a == 1, qx[:, None], tl.where(a == 2, qy[:, None], qz[:, None])),
    )
    qb = tl.where(
        b == 0,
        qw[:, None],
        tl.where(b == 1, qx[:, None], tl.where(b == 2, qy[:, None], qz[:, None])),
    )
    return qa * qb


@triton.jit
def _vec_to_quat(vx, vy, vz, gamma, EPS: tl.constexpr):
    """
    The ``v -> q`` half of the chain, matching ``_quaternion_wigner`` exactly.

    Returns:
        ``(qw, qx, qy, qz)`` plus every intermediate the backward needs.
    """
    vn = tl.sqrt(vx * vx + vy * vy + vz * vz)
    vn_safe = tl.maximum(vn, EPS)
    ex = vx / vn_safe
    ey = vy / vn_safe
    ez = vz / vn_safe

    # Chart 1: (1 + ey, -ez, 0, ex), chart 2: (-ez, 1 - ey, ex, 0).
    s1 = (1.0 + ey) * (1.0 + ey) + ez * ez + ex * ex
    n1 = tl.sqrt(tl.maximum(s1, EPS))
    q1w = (1.0 + ey) / n1
    q1x = -ez / n1
    q1y = 0.0 * ex
    q1z = ex / n1

    s2 = ez * ez + (1.0 - ey) * (1.0 - ey) + ex * ex
    n2 = tl.sqrt(tl.maximum(s2, EPS))
    q2w = -ez / n2
    q2x = (1.0 - ey) / n2
    q2y = ex / n2
    q2z = 0.0 * ex

    # Smoothstep blend weight, with the two flat tails the eager form forces.
    t = tl.minimum(tl.maximum((ey + 0.9) / 1.8, 0.0), 1.0)
    num = 2.0 * t - 1.0
    den = tl.maximum(t * (1.0 - t), EPS)
    sig = tl.sigmoid(num / den)
    inner = (t >= EPS) & (t <= 1.0 - EPS)
    s = tl.where(t < EPS, 0.0, tl.where(t > 1.0 - EPS, 1.0, sig))

    dot = q2w * q1w + q2x * q1x + q2y * q1y + q2z * q1z
    sgn = tl.where(dot < 0, -1.0, 1.0)
    q2w = sgn * q2w
    q2x = sgn * q2x
    q2y = sgn * q2y
    q2z = sgn * q2z

    bw = (1.0 - s) * q2w + s * q1w
    bx = (1.0 - s) * q2x + s * q1x
    by = (1.0 - s) * q2y + s * q1y
    bz = (1.0 - s) * q2z + s * q1z
    nb = tl.sqrt(tl.maximum(bw * bw + bx * bx + by * by + bz * bz, EPS))
    ew = bw / nb
    exq = bx / nb
    eyq = by / nb
    ezq = bz / nb

    # Compose with q_gamma = (cos(g/2), 0, sin(g/2), 0); the zeros collapse the
    # general Hamilton product to eight terms.
    cg = tl.cos(gamma * 0.5)
    sg = tl.sin(gamma * 0.5)
    qw = cg * ew - sg * eyq
    qx = cg * exq + sg * ezq
    qy = cg * eyq + sg * ew
    qz = cg * ezq - sg * exq
    return (
        qw,
        qx,
        qy,
        qz,
        ew,
        exq,
        eyq,
        ezq,
        nb,
        q1w,
        q1x,
        q1y,
        q1z,
        n1,
        q2w,
        q2x,
        q2y,
        q2z,
        n2,
        sgn,
        s,
        t,
        sig,
        inner,
        ex,
        ey,
        ez,
        vn_safe,
        cg,
        sg,
    )


@triton.jit
def _wig_fwd_kernel(
    v_ptr,
    gamma_ptr,
    d55_ptr,
    comp_ptr,
    n_edges,
    EPS: tl.constexpr,
    BLK: tl.constexpr,
):
    """
    ``v, gamma -> compact (E, 34)``: nine rotation entries then 25 ``d_l2`` entries.
    """
    e = (tl.program_id(0) * BLK + tl.arange(0, BLK)).to(tl.int64)
    m = e < n_edges
    vx = tl.load(v_ptr + e * 3 + 0, mask=m, other=1.0)
    vy = tl.load(v_ptr + e * 3 + 1, mask=m, other=0.0)
    vz = tl.load(v_ptr + e * 3 + 2, mask=m, other=0.0)
    gamma = tl.load(gamma_ptr + e, mask=m, other=0.0)
    qw, qx, qy, qz = _vec_to_quat(vx, vy, vz, gamma, EPS)[:4]

    x2 = qx * qx
    y2 = qy * qy
    z2 = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    j = tl.arange(0, 16)[None, :]
    rot = (
        tl.where(j == 0, (1 - 2 * (y2 + z2))[:, None], 0.0)
        + tl.where(j == 1, (2 * (xy - wz))[:, None], 0.0)
        + tl.where(j == 2, (2 * (xz + wy))[:, None], 0.0)
        + tl.where(j == 3, (2 * (xy + wz))[:, None], 0.0)
        + tl.where(j == 4, (1 - 2 * (x2 + z2))[:, None], 0.0)
        + tl.where(j == 5, (2 * (yz - wx))[:, None], 0.0)
        + tl.where(j == 6, (2 * (xz - wy))[:, None], 0.0)
        + tl.where(j == 7, (2 * (yz + wx))[:, None], 0.0)
        + tl.where(j == 8, (1 - 2 * (x2 + y2))[:, None], 0.0)
    )
    tl.store(comp_ptr + e[:, None] * 34 + j, rot, mask=m[:, None] & (j < 9))

    m0 = qw * qw
    m1 = qw * qx
    m2 = qw * qy
    m3 = qw * qz
    m4 = qx * qx
    m5 = qx * qy
    m6 = qx * qz
    m7 = qy * qy
    m8 = qy * qz
    m9 = qz * qz
    a0 = (
        m0 * m0 * tl.load(d55_ptr + 0 * 32 + 0)
        + m0 * m7 * tl.load(d55_ptr + 7 * 32 + 0)
        + m2 * m2 * tl.load(d55_ptr + 19 * 32 + 0)
        + m4 * m4 * tl.load(d55_ptr + 34 * 32 + 0)
        + m4 * m9 * tl.load(d55_ptr + 39 * 32 + 0)
        + m6 * m6 * tl.load(d55_ptr + 45 * 32 + 0)
        + m7 * m7 * tl.load(d55_ptr + 49 * 32 + 0)
        + m9 * m9 * tl.load(d55_ptr + 54 * 32 + 0)
    )
    a1 = (
        m0 * m1 * tl.load(d55_ptr + 1 * 32 + 1)
        + m0 * m8 * tl.load(d55_ptr + 8 * 32 + 1)
        + m1 * m4 * tl.load(d55_ptr + 13 * 32 + 1)
        + m1 * m7 * tl.load(d55_ptr + 16 * 32 + 1)
        + m1 * m9 * tl.load(d55_ptr + 18 * 32 + 1)
        + m2 * m3 * tl.load(d55_ptr + 20 * 32 + 1)
        + m2 * m5 * tl.load(d55_ptr + 22 * 32 + 1)
        + m3 * m6 * tl.load(d55_ptr + 30 * 32 + 1)
        + m4 * m8 * tl.load(d55_ptr + 38 * 32 + 1)
        + m5 * m6 * tl.load(d55_ptr + 41 * 32 + 1)
        + m7 * m8 * tl.load(d55_ptr + 50 * 32 + 1)
        + m8 * m9 * tl.load(d55_ptr + 53 * 32 + 1)
    )
    a2 = (
        m0 * m6 * tl.load(d55_ptr + 6 * 32 + 2)
        + m1 * m3 * tl.load(d55_ptr + 12 * 32 + 2)
        + m1 * m5 * tl.load(d55_ptr + 14 * 32 + 2)
        + m2 * m4 * tl.load(d55_ptr + 21 * 32 + 2)
        + m2 * m9 * tl.load(d55_ptr + 26 * 32 + 2)
        + m3 * m8 * tl.load(d55_ptr + 32 * 32 + 2)
        + m5 * m8 * tl.load(d55_ptr + 43 * 32 + 2)
        + m6 * m7 * tl.load(d55_ptr + 46 * 32 + 2)
    )
    a3 = (
        m0 * m3 * tl.load(d55_ptr + 3 * 32 + 3)
        + m0 * m5 * tl.load(d55_ptr + 5 * 32 + 3)
        + m1 * m2 * tl.load(d55_ptr + 11 * 32 + 3)
        + m1 * m6 * tl.load(d55_ptr + 15 * 32 + 3)
        + m2 * m8 * tl.load(d55_ptr + 25 * 32 + 3)
        + m3 * m4 * tl.load(d55_ptr + 28 * 32 + 3)
        + m3 * m7 * tl.load(d55_ptr + 31 * 32 + 3)
        + m3 * m9 * tl.load(d55_ptr + 33 * 32 + 3)
        + m4 * m5 * tl.load(d55_ptr + 35 * 32 + 3)
        + m5 * m7 * tl.load(d55_ptr + 42 * 32 + 3)
        + m5 * m9 * tl.load(d55_ptr + 44 * 32 + 3)
        + m6 * m8 * tl.load(d55_ptr + 47 * 32 + 3)
    )
    a4 = (
        m0 * m2 * tl.load(d55_ptr + 2 * 32 + 4)
        + m2 * m7 * tl.load(d55_ptr + 24 * 32 + 4)
        + m4 * m6 * tl.load(d55_ptr + 36 * 32 + 4)
        + m6 * m9 * tl.load(d55_ptr + 48 * 32 + 4)
    )
    a5 = (
        m0 * m1 * tl.load(d55_ptr + 1 * 32 + 5)
        + m0 * m8 * tl.load(d55_ptr + 8 * 32 + 5)
        + m1 * m4 * tl.load(d55_ptr + 13 * 32 + 5)
        + m1 * m7 * tl.load(d55_ptr + 16 * 32 + 5)
        + m1 * m9 * tl.load(d55_ptr + 18 * 32 + 5)
        + m2 * m3 * tl.load(d55_ptr + 20 * 32 + 5)
        + m2 * m5 * tl.load(d55_ptr + 22 * 32 + 5)
        + m3 * m6 * tl.load(d55_ptr + 30 * 32 + 5)
        + m4 * m8 * tl.load(d55_ptr + 38 * 32 + 5)
        + m5 * m6 * tl.load(d55_ptr + 41 * 32 + 5)
        + m7 * m8 * tl.load(d55_ptr + 50 * 32 + 5)
        + m8 * m9 * tl.load(d55_ptr + 53 * 32 + 5)
    )
    a6 = (
        m0 * m0 * tl.load(d55_ptr + 0 * 32 + 6)
        + m0 * m9 * tl.load(d55_ptr + 9 * 32 + 6)
        + m3 * m3 * tl.load(d55_ptr + 27 * 32 + 6)
        + m4 * m4 * tl.load(d55_ptr + 34 * 32 + 6)
        + m4 * m7 * tl.load(d55_ptr + 37 * 32 + 6)
        + m5 * m5 * tl.load(d55_ptr + 40 * 32 + 6)
        + m7 * m7 * tl.load(d55_ptr + 49 * 32 + 6)
        + m9 * m9 * tl.load(d55_ptr + 54 * 32 + 6)
    )
    a7 = (
        m0 * m3 * tl.load(d55_ptr + 3 * 32 + 7)
        + m0 * m5 * tl.load(d55_ptr + 5 * 32 + 7)
        + m1 * m2 * tl.load(d55_ptr + 11 * 32 + 7)
        + m1 * m6 * tl.load(d55_ptr + 15 * 32 + 7)
        + m2 * m8 * tl.load(d55_ptr + 25 * 32 + 7)
        + m3 * m4 * tl.load(d55_ptr + 28 * 32 + 7)
        + m3 * m7 * tl.load(d55_ptr + 31 * 32 + 7)
        + m3 * m9 * tl.load(d55_ptr + 33 * 32 + 7)
        + m4 * m5 * tl.load(d55_ptr + 35 * 32 + 7)
        + m5 * m7 * tl.load(d55_ptr + 42 * 32 + 7)
        + m5 * m9 * tl.load(d55_ptr + 44 * 32 + 7)
        + m6 * m8 * tl.load(d55_ptr + 47 * 32 + 7)
    )
    a8 = (
        m0 * m2 * tl.load(d55_ptr + 2 * 32 + 8)
        + m0 * m6 * tl.load(d55_ptr + 6 * 32 + 8)
        + m1 * m3 * tl.load(d55_ptr + 12 * 32 + 8)
        + m1 * m5 * tl.load(d55_ptr + 14 * 32 + 8)
        + m2 * m4 * tl.load(d55_ptr + 21 * 32 + 8)
        + m2 * m7 * tl.load(d55_ptr + 24 * 32 + 8)
        + m2 * m9 * tl.load(d55_ptr + 26 * 32 + 8)
        + m3 * m8 * tl.load(d55_ptr + 32 * 32 + 8)
        + m4 * m6 * tl.load(d55_ptr + 36 * 32 + 8)
        + m5 * m8 * tl.load(d55_ptr + 43 * 32 + 8)
        + m6 * m7 * tl.load(d55_ptr + 46 * 32 + 8)
        + m6 * m9 * tl.load(d55_ptr + 48 * 32 + 8)
    )
    a9 = (
        m0 * m3 * tl.load(d55_ptr + 3 * 32 + 9)
        + m0 * m5 * tl.load(d55_ptr + 5 * 32 + 9)
        + m1 * m2 * tl.load(d55_ptr + 11 * 32 + 9)
        + m1 * m6 * tl.load(d55_ptr + 15 * 32 + 9)
        + m2 * m8 * tl.load(d55_ptr + 25 * 32 + 9)
        + m3 * m4 * tl.load(d55_ptr + 28 * 32 + 9)
        + m3 * m7 * tl.load(d55_ptr + 31 * 32 + 9)
        + m3 * m9 * tl.load(d55_ptr + 33 * 32 + 9)
        + m4 * m5 * tl.load(d55_ptr + 35 * 32 + 9)
        + m5 * m7 * tl.load(d55_ptr + 42 * 32 + 9)
        + m5 * m9 * tl.load(d55_ptr + 44 * 32 + 9)
        + m6 * m8 * tl.load(d55_ptr + 47 * 32 + 9)
    )
    a10 = (
        m0 * m6 * tl.load(d55_ptr + 6 * 32 + 10)
        + m1 * m3 * tl.load(d55_ptr + 12 * 32 + 10)
        + m1 * m5 * tl.load(d55_ptr + 14 * 32 + 10)
        + m2 * m4 * tl.load(d55_ptr + 21 * 32 + 10)
        + m2 * m9 * tl.load(d55_ptr + 26 * 32 + 10)
        + m3 * m8 * tl.load(d55_ptr + 32 * 32 + 10)
        + m5 * m8 * tl.load(d55_ptr + 43 * 32 + 10)
        + m6 * m7 * tl.load(d55_ptr + 46 * 32 + 10)
    )
    a11 = (
        m0 * m3 * tl.load(d55_ptr + 3 * 32 + 11)
        + m0 * m5 * tl.load(d55_ptr + 5 * 32 + 11)
        + m1 * m2 * tl.load(d55_ptr + 11 * 32 + 11)
        + m1 * m6 * tl.load(d55_ptr + 15 * 32 + 11)
        + m2 * m8 * tl.load(d55_ptr + 25 * 32 + 11)
        + m3 * m4 * tl.load(d55_ptr + 28 * 32 + 11)
        + m3 * m7 * tl.load(d55_ptr + 31 * 32 + 11)
        + m3 * m9 * tl.load(d55_ptr + 33 * 32 + 11)
        + m4 * m5 * tl.load(d55_ptr + 35 * 32 + 11)
        + m5 * m7 * tl.load(d55_ptr + 42 * 32 + 11)
        + m5 * m9 * tl.load(d55_ptr + 44 * 32 + 11)
        + m6 * m8 * tl.load(d55_ptr + 47 * 32 + 11)
    )
    a12 = (
        m0 * m0 * tl.load(d55_ptr + 0 * 32 + 12)
        + m0 * m4 * tl.load(d55_ptr + 4 * 32 + 12)
        + m0 * m7 * tl.load(d55_ptr + 7 * 32 + 12)
        + m0 * m9 * tl.load(d55_ptr + 9 * 32 + 12)
        + m1 * m1 * tl.load(d55_ptr + 10 * 32 + 12)
        + m2 * m2 * tl.load(d55_ptr + 19 * 32 + 12)
        + m3 * m3 * tl.load(d55_ptr + 27 * 32 + 12)
        + m4 * m4 * tl.load(d55_ptr + 34 * 32 + 12)
        + m4 * m7 * tl.load(d55_ptr + 37 * 32 + 12)
        + m4 * m9 * tl.load(d55_ptr + 39 * 32 + 12)
        + m5 * m5 * tl.load(d55_ptr + 40 * 32 + 12)
        + m6 * m6 * tl.load(d55_ptr + 45 * 32 + 12)
        + m7 * m7 * tl.load(d55_ptr + 49 * 32 + 12)
        + m7 * m9 * tl.load(d55_ptr + 51 * 32 + 12)
        + m8 * m8 * tl.load(d55_ptr + 52 * 32 + 12)
        + m9 * m9 * tl.load(d55_ptr + 54 * 32 + 12)
    )
    a13 = (
        m0 * m1 * tl.load(d55_ptr + 1 * 32 + 13)
        + m0 * m8 * tl.load(d55_ptr + 8 * 32 + 13)
        + m1 * m4 * tl.load(d55_ptr + 13 * 32 + 13)
        + m1 * m7 * tl.load(d55_ptr + 16 * 32 + 13)
        + m1 * m9 * tl.load(d55_ptr + 18 * 32 + 13)
        + m2 * m3 * tl.load(d55_ptr + 20 * 32 + 13)
        + m2 * m5 * tl.load(d55_ptr + 22 * 32 + 13)
        + m3 * m6 * tl.load(d55_ptr + 30 * 32 + 13)
        + m4 * m8 * tl.load(d55_ptr + 38 * 32 + 13)
        + m5 * m6 * tl.load(d55_ptr + 41 * 32 + 13)
        + m7 * m8 * tl.load(d55_ptr + 50 * 32 + 13)
        + m8 * m9 * tl.load(d55_ptr + 53 * 32 + 13)
    )
    a14 = (
        m0 * m4 * tl.load(d55_ptr + 4 * 32 + 14)
        + m0 * m9 * tl.load(d55_ptr + 9 * 32 + 14)
        + m1 * m1 * tl.load(d55_ptr + 10 * 32 + 14)
        + m1 * m8 * tl.load(d55_ptr + 17 * 32 + 14)
        + m2 * m6 * tl.load(d55_ptr + 23 * 32 + 14)
        + m3 * m3 * tl.load(d55_ptr + 27 * 32 + 14)
        + m3 * m5 * tl.load(d55_ptr + 29 * 32 + 14)
        + m4 * m7 * tl.load(d55_ptr + 37 * 32 + 14)
        + m5 * m5 * tl.load(d55_ptr + 40 * 32 + 14)
        + m7 * m9 * tl.load(d55_ptr + 51 * 32 + 14)
        + m8 * m8 * tl.load(d55_ptr + 52 * 32 + 14)
    )
    a15 = (
        m0 * m3 * tl.load(d55_ptr + 3 * 32 + 15)
        + m0 * m5 * tl.load(d55_ptr + 5 * 32 + 15)
        + m1 * m2 * tl.load(d55_ptr + 11 * 32 + 15)
        + m1 * m6 * tl.load(d55_ptr + 15 * 32 + 15)
        + m2 * m8 * tl.load(d55_ptr + 25 * 32 + 15)
        + m3 * m4 * tl.load(d55_ptr + 28 * 32 + 15)
        + m3 * m7 * tl.load(d55_ptr + 31 * 32 + 15)
        + m3 * m9 * tl.load(d55_ptr + 33 * 32 + 15)
        + m4 * m5 * tl.load(d55_ptr + 35 * 32 + 15)
        + m5 * m7 * tl.load(d55_ptr + 42 * 32 + 15)
        + m5 * m9 * tl.load(d55_ptr + 44 * 32 + 15)
        + m6 * m8 * tl.load(d55_ptr + 47 * 32 + 15)
    )
    a16 = (
        m0 * m2 * tl.load(d55_ptr + 2 * 32 + 16)
        + m0 * m6 * tl.load(d55_ptr + 6 * 32 + 16)
        + m1 * m3 * tl.load(d55_ptr + 12 * 32 + 16)
        + m1 * m5 * tl.load(d55_ptr + 14 * 32 + 16)
        + m2 * m4 * tl.load(d55_ptr + 21 * 32 + 16)
        + m2 * m7 * tl.load(d55_ptr + 24 * 32 + 16)
        + m2 * m9 * tl.load(d55_ptr + 26 * 32 + 16)
        + m3 * m8 * tl.load(d55_ptr + 32 * 32 + 16)
        + m4 * m6 * tl.load(d55_ptr + 36 * 32 + 16)
        + m5 * m8 * tl.load(d55_ptr + 43 * 32 + 16)
        + m6 * m7 * tl.load(d55_ptr + 46 * 32 + 16)
        + m6 * m9 * tl.load(d55_ptr + 48 * 32 + 16)
    )
    a17 = (
        m0 * m1 * tl.load(d55_ptr + 1 * 32 + 17)
        + m0 * m8 * tl.load(d55_ptr + 8 * 32 + 17)
        + m1 * m4 * tl.load(d55_ptr + 13 * 32 + 17)
        + m1 * m7 * tl.load(d55_ptr + 16 * 32 + 17)
        + m1 * m9 * tl.load(d55_ptr + 18 * 32 + 17)
        + m2 * m3 * tl.load(d55_ptr + 20 * 32 + 17)
        + m2 * m5 * tl.load(d55_ptr + 22 * 32 + 17)
        + m3 * m6 * tl.load(d55_ptr + 30 * 32 + 17)
        + m4 * m8 * tl.load(d55_ptr + 38 * 32 + 17)
        + m5 * m6 * tl.load(d55_ptr + 41 * 32 + 17)
        + m7 * m8 * tl.load(d55_ptr + 50 * 32 + 17)
        + m8 * m9 * tl.load(d55_ptr + 53 * 32 + 17)
    )
    a18 = (
        m0 * m0 * tl.load(d55_ptr + 0 * 32 + 18)
        + m0 * m4 * tl.load(d55_ptr + 4 * 32 + 18)
        + m1 * m1 * tl.load(d55_ptr + 10 * 32 + 18)
        + m4 * m4 * tl.load(d55_ptr + 34 * 32 + 18)
        + m7 * m7 * tl.load(d55_ptr + 49 * 32 + 18)
        + m7 * m9 * tl.load(d55_ptr + 51 * 32 + 18)
        + m8 * m8 * tl.load(d55_ptr + 52 * 32 + 18)
        + m9 * m9 * tl.load(d55_ptr + 54 * 32 + 18)
    )
    a19 = (
        m0 * m1 * tl.load(d55_ptr + 1 * 32 + 19)
        + m0 * m8 * tl.load(d55_ptr + 8 * 32 + 19)
        + m1 * m4 * tl.load(d55_ptr + 13 * 32 + 19)
        + m1 * m7 * tl.load(d55_ptr + 16 * 32 + 19)
        + m1 * m9 * tl.load(d55_ptr + 18 * 32 + 19)
        + m2 * m3 * tl.load(d55_ptr + 20 * 32 + 19)
        + m2 * m5 * tl.load(d55_ptr + 22 * 32 + 19)
        + m3 * m6 * tl.load(d55_ptr + 30 * 32 + 19)
        + m4 * m8 * tl.load(d55_ptr + 38 * 32 + 19)
        + m5 * m6 * tl.load(d55_ptr + 41 * 32 + 19)
        + m7 * m8 * tl.load(d55_ptr + 50 * 32 + 19)
        + m8 * m9 * tl.load(d55_ptr + 53 * 32 + 19)
    )
    a20 = (
        m0 * m2 * tl.load(d55_ptr + 2 * 32 + 20)
        + m2 * m7 * tl.load(d55_ptr + 24 * 32 + 20)
        + m4 * m6 * tl.load(d55_ptr + 36 * 32 + 20)
        + m6 * m9 * tl.load(d55_ptr + 48 * 32 + 20)
    )
    a21 = (
        m0 * m3 * tl.load(d55_ptr + 3 * 32 + 21)
        + m0 * m5 * tl.load(d55_ptr + 5 * 32 + 21)
        + m1 * m2 * tl.load(d55_ptr + 11 * 32 + 21)
        + m1 * m6 * tl.load(d55_ptr + 15 * 32 + 21)
        + m2 * m8 * tl.load(d55_ptr + 25 * 32 + 21)
        + m3 * m4 * tl.load(d55_ptr + 28 * 32 + 21)
        + m3 * m7 * tl.load(d55_ptr + 31 * 32 + 21)
        + m3 * m9 * tl.load(d55_ptr + 33 * 32 + 21)
        + m4 * m5 * tl.load(d55_ptr + 35 * 32 + 21)
        + m5 * m7 * tl.load(d55_ptr + 42 * 32 + 21)
        + m5 * m9 * tl.load(d55_ptr + 44 * 32 + 21)
        + m6 * m8 * tl.load(d55_ptr + 47 * 32 + 21)
    )
    a22 = (
        m0 * m4 * tl.load(d55_ptr + 4 * 32 + 22)
        + m0 * m9 * tl.load(d55_ptr + 9 * 32 + 22)
        + m1 * m1 * tl.load(d55_ptr + 10 * 32 + 22)
        + m1 * m8 * tl.load(d55_ptr + 17 * 32 + 22)
        + m2 * m6 * tl.load(d55_ptr + 23 * 32 + 22)
        + m3 * m3 * tl.load(d55_ptr + 27 * 32 + 22)
        + m3 * m5 * tl.load(d55_ptr + 29 * 32 + 22)
        + m4 * m7 * tl.load(d55_ptr + 37 * 32 + 22)
        + m5 * m5 * tl.load(d55_ptr + 40 * 32 + 22)
        + m7 * m9 * tl.load(d55_ptr + 51 * 32 + 22)
        + m8 * m8 * tl.load(d55_ptr + 52 * 32 + 22)
    )
    a23 = (
        m0 * m1 * tl.load(d55_ptr + 1 * 32 + 23)
        + m0 * m8 * tl.load(d55_ptr + 8 * 32 + 23)
        + m1 * m4 * tl.load(d55_ptr + 13 * 32 + 23)
        + m1 * m7 * tl.load(d55_ptr + 16 * 32 + 23)
        + m1 * m9 * tl.load(d55_ptr + 18 * 32 + 23)
        + m2 * m3 * tl.load(d55_ptr + 20 * 32 + 23)
        + m2 * m5 * tl.load(d55_ptr + 22 * 32 + 23)
        + m3 * m6 * tl.load(d55_ptr + 30 * 32 + 23)
        + m4 * m8 * tl.load(d55_ptr + 38 * 32 + 23)
        + m5 * m6 * tl.load(d55_ptr + 41 * 32 + 23)
        + m7 * m8 * tl.load(d55_ptr + 50 * 32 + 23)
        + m8 * m9 * tl.load(d55_ptr + 53 * 32 + 23)
    )
    a24 = (
        m0 * m0 * tl.load(d55_ptr + 0 * 32 + 24)
        + m0 * m7 * tl.load(d55_ptr + 7 * 32 + 24)
        + m2 * m2 * tl.load(d55_ptr + 19 * 32 + 24)
        + m4 * m4 * tl.load(d55_ptr + 34 * 32 + 24)
        + m4 * m9 * tl.load(d55_ptr + 39 * 32 + 24)
        + m6 * m6 * tl.load(d55_ptr + 45 * 32 + 24)
        + m7 * m7 * tl.load(d55_ptr + 49 * 32 + 24)
        + m9 * m9 * tl.load(d55_ptr + 54 * 32 + 24)
    )
    tl.store(comp_ptr + e * 34 + 9, a0, mask=m)
    tl.store(comp_ptr + e * 34 + 10, a1, mask=m)
    tl.store(comp_ptr + e * 34 + 11, a2, mask=m)
    tl.store(comp_ptr + e * 34 + 12, a3, mask=m)
    tl.store(comp_ptr + e * 34 + 13, a4, mask=m)
    tl.store(comp_ptr + e * 34 + 14, a5, mask=m)
    tl.store(comp_ptr + e * 34 + 15, a6, mask=m)
    tl.store(comp_ptr + e * 34 + 16, a7, mask=m)
    tl.store(comp_ptr + e * 34 + 17, a8, mask=m)
    tl.store(comp_ptr + e * 34 + 18, a9, mask=m)
    tl.store(comp_ptr + e * 34 + 19, a10, mask=m)
    tl.store(comp_ptr + e * 34 + 20, a11, mask=m)
    tl.store(comp_ptr + e * 34 + 21, a12, mask=m)
    tl.store(comp_ptr + e * 34 + 22, a13, mask=m)
    tl.store(comp_ptr + e * 34 + 23, a14, mask=m)
    tl.store(comp_ptr + e * 34 + 24, a15, mask=m)
    tl.store(comp_ptr + e * 34 + 25, a16, mask=m)
    tl.store(comp_ptr + e * 34 + 26, a17, mask=m)
    tl.store(comp_ptr + e * 34 + 27, a18, mask=m)
    tl.store(comp_ptr + e * 34 + 28, a19, mask=m)
    tl.store(comp_ptr + e * 34 + 29, a20, mask=m)
    tl.store(comp_ptr + e * 34 + 30, a21, mask=m)
    tl.store(comp_ptr + e * 34 + 31, a22, mask=m)
    tl.store(comp_ptr + e * 34 + 32, a23, mask=m)
    tl.store(comp_ptr + e * 34 + 33, a24, mask=m)


@triton.jit
def _wig_expand_kernel(comp_ptr, w_ptr, wi_ptr, env_ptr, n_edges, BLK: tl.constexpr):
    """
    Expand the compact Wigner representation into block-diagonal tensors.

    The inverse is transposed during the same pass, with the radial envelope folded
    into it.
    """
    e = (tl.program_id(0) * BLK + tl.arange(0, BLK)).to(tl.int64)
    m = e < n_edges
    p = tl.arange(0, 128)[None, :]
    row = p // 9
    col = p % 9
    live = m[:, None] & (p < 81)

    in_rot = (row >= 1) & (row <= 3) & (col >= 1) & (col <= 3)
    in_d = (row >= 4) & (col >= 4)
    src = tl.where(
        in_rot,
        (row - 1) * 3 + (col - 1),
        tl.where(in_d, 9 + (row - 4) * 5 + (col - 4), 0),
    )
    val = tl.load(
        comp_ptr + e[:, None] * 34 + src, mask=live & (in_rot | in_d), other=0.0
    )
    val = tl.where(p == 0, 1.0, val)
    tl.store(w_ptr + e[:, None] * 81 + p, val, mask=live)

    # Transposed: read the (col, row) slot instead.
    srct = tl.where(
        in_rot,
        (col - 1) * 3 + (row - 1),
        tl.where(in_d, 9 + (col - 4) * 5 + (row - 4), 0),
    )
    valt = tl.load(
        comp_ptr + e[:, None] * 34 + srct, mask=live & (in_rot | in_d), other=0.0
    )
    valt = tl.where(p == 0, 1.0, valt)
    env = tl.load(env_ptr + e, mask=m, other=0.0)
    tl.store(wi_ptr + e[:, None] * 81 + p, valt * env[:, None], mask=live)


@triton.jit
def _wig_gather_kernel(
    gw_ptr,
    gwi_ptr,
    comp_ptr,
    env_ptr,
    dcomp_ptr,
    denv_ptr,
    n_edges,
    BLK: tl.constexpr,
):
    """
    Reduce full Wigner output gradients into the compact representation.

    The same pass also differentiates the envelope folded into the inverse tensor.
    """
    e = (tl.program_id(0) * BLK + tl.arange(0, BLK)).to(tl.int64)
    m = e < n_edges
    s = tl.arange(0, 64)[None, :]
    live = m[:, None] & (s < 34)
    row = tl.where(s < 9, 1 + s // 3, 4 + (s - 9) // 5)
    col = tl.where(s < 9, 1 + s % 3, 4 + (s - 9) % 5)
    env = tl.load(env_ptr + e, mask=m, other=0.0)
    g = tl.load(gw_ptr + e[:, None] * 81 + row * 9 + col, mask=live, other=0.0)
    gi = tl.load(gwi_ptr + e[:, None] * 81 + col * 9 + row, mask=live, other=0.0)
    tl.store(dcomp_ptr + e[:, None] * 34 + s, g + gi * env[:, None], mask=live)

    comp = tl.load(comp_ptr + e[:, None] * 34 + s, mask=live, other=0.0)
    denv = tl.load(gwi_ptr + e * 81, mask=m, other=0.0) + tl.sum(gi * comp, axis=1)
    tl.store(denv_ptr + e, denv, mask=m)


@triton.jit
def _wig_bwd_kernel(
    v_ptr,
    gamma_ptr,
    csym_ptr,
    dcomp_ptr,
    dv_ptr,
    n_edges,
    EPS: tl.constexpr,
    BLK: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Differentiate the compact Wigner form while recomputing its quaternion.
    """
    e = (tl.program_id(0) * BLK + tl.arange(0, BLK)).to(tl.int64)
    m = e < n_edges
    vx = tl.load(v_ptr + e * 3 + 0, mask=m, other=1.0)
    vy = tl.load(v_ptr + e * 3 + 1, mask=m, other=0.0)
    vz = tl.load(v_ptr + e * 3 + 2, mask=m, other=0.0)
    gamma = tl.load(gamma_ptr + e, mask=m, other=0.0)
    (
        qw,
        qx,
        qy,
        qz,
        ew,
        exq,
        eyq,
        ezq,
        nb,
        q1w,
        q1x,
        q1y,
        q1z,
        n1,
        q2w,
        q2x,
        q2y,
        q2z,
        n2,
        sgn,
        s,
        t,
        sig,
        inner,
        ex,
        ey,
        ez,
        vn_safe,
        cg,
        sg,
    ) = _vec_to_quat(vx, vy, vz, gamma, EPS)

    g = tl.arange(0, 16)[None, :]
    grot = tl.load(
        dcomp_ptr + e[:, None] * 34 + g, mask=m[:, None] & (g < 9), other=0.0
    )
    g00 = tl.sum(tl.where(g == 0, grot, 0.0), axis=1)
    g01 = tl.sum(tl.where(g == 1, grot, 0.0), axis=1)
    g02 = tl.sum(tl.where(g == 2, grot, 0.0), axis=1)
    g10 = tl.sum(tl.where(g == 3, grot, 0.0), axis=1)
    g11 = tl.sum(tl.where(g == 4, grot, 0.0), axis=1)
    g12 = tl.sum(tl.where(g == 5, grot, 0.0), axis=1)
    g20 = tl.sum(tl.where(g == 6, grot, 0.0), axis=1)
    g21 = tl.sum(tl.where(g == 7, grot, 0.0), axis=1)
    g22 = tl.sum(tl.where(g == 8, grot, 0.0), axis=1)

    # Rotation-matrix backward, collected per quaternion component.
    dqw = 2.0 * (qx * (g21 - g12) + qy * (g02 - g20) + qz * (g10 - g01))
    dqx = 2.0 * (
        -2.0 * qx * (g11 + g22) + qy * (g01 + g10) + qz * (g02 + g20) + qw * (g21 - g12)
    )
    dqy = 2.0 * (
        -2.0 * qy * (g00 + g22) + qx * (g01 + g10) + qz * (g12 + g21) + qw * (g02 - g20)
    )
    dqz = 2.0 * (
        -2.0 * qz * (g00 + g11) + qx * (g02 + g20) + qy * (g12 + g21) + qw * (g10 - g01)
    )

    # d_l2 backward. d_l2[k] = sum_ab sum_cd C[k,ab,cd] q2[ab] q2[cd], so
    # dq2[ab] = sum_k gd[k] * sum_cd (C[k,ab,cd] + C[k,cd,ab]) q2[cd].
    q2 = _q2_tile(qw, qx, qy, qz)
    k32 = tl.arange(0, 32)[None, :]
    gd = tl.load(
        dcomp_ptr + e[:, None] * 34 + 9 + k32, mask=m[:, None] & (k32 < 25), other=0.0
    )
    # Contract the symmetrized coefficients against gd with one MMA, then a single
    # 16-wide reduction, instead of 25 reductions over (BLK, 16, 16) tiles.
    cs = tl.load(
        csym_ptr + tl.arange(0, 32)[:, None] * 256 + tl.arange(0, 256)[None, :]
    )
    cg2 = tl.reshape(_dot3(gd, cs, USE_TF32), (BLK, 16, 16))
    dq2 = tl.sum(cg2 * q2[:, None, :], axis=2)

    # dq[i] += sum_j (dq2[i,j] + dq2[j,i]) * q[j], read off the 4x4 block.
    i16 = tl.arange(0, 16)
    a = (i16 // 4)[None, :]
    b = (i16 % 4)[None, :]
    qb = tl.where(
        b == 0,
        qw[:, None],
        tl.where(b == 1, qx[:, None], tl.where(b == 2, qy[:, None], qz[:, None])),
    )
    qa = tl.where(
        a == 0,
        qw[:, None],
        tl.where(a == 1, qx[:, None], tl.where(a == 2, qy[:, None], qz[:, None])),
    )
    row_contrib = dq2 * qb
    col_contrib = dq2 * qa
    for i in tl.static_range(4):
        acc = tl.sum(tl.where(a == i, row_contrib, 0.0), axis=1) + tl.sum(
            tl.where(b == i, col_contrib, 0.0), axis=1
        )
        if i == 0:
            dqw += acc
        elif i == 1:
            dqx += acc
        elif i == 2:
            dqy += acc
        else:
            dqz += acc

    # Compose backward: the conjugate of q_gamma.
    dew = cg * dqw + sg * dqy
    dexq = cg * dqx - sg * dqz
    deyq = -sg * dqw + cg * dqy
    dezq = sg * dqx + cg * dqz

    # normalize(b) backward.
    proj = ew * dew + exq * dexq + eyq * deyq + ezq * dezq
    dbw = (dew - ew * proj) / nb
    dbx = (dexq - exq * proj) / nb
    dby = (deyq - eyq * proj) / nb
    dbz = (dezq - ezq * proj) / nb

    # Blend backward.
    dq1w = s * dbw
    dq1x = s * dbx
    dq1y = s * dby
    dq1z = s * dbz
    dq2w = (1.0 - s) * dbw
    dq2x = (1.0 - s) * dbx
    dq2y = (1.0 - s) * dby
    dq2z = (1.0 - s) * dbz
    ds = dbw * (q1w - q2w) + dbx * (q1x - q2x) + dby * (q1y - q2y) + dbz * (q1z - q2z)

    # The sign flip is piecewise constant, so it just carries through.
    dq2w = sgn * dq2w
    dq2x = sgn * dq2x
    dq2y = sgn * dq2y
    dq2z = sgn * dq2z

    # normalize backward for both charts, then map onto (ex, ey, ez).
    p1 = q1w * dq1w + q1x * dq1x + q1y * dq1y + q1z * dq1z
    da1w = (dq1w - q1w * p1) / n1
    da1x = (dq1x - q1x * p1) / n1
    da1z = (dq1z - q1z * p1) / n1

    q2wr = sgn * q2w
    q2xr = sgn * q2x
    q2yr = sgn * q2y
    p2 = q2wr * dq2w + q2xr * dq2x + q2yr * dq2y
    # q2 raw components are (-ez, 1-ey, ex, 0) / n2; sgn cancels in the projection
    # because both the value and the gradient carry it.
    r2w = -ez / n2
    r2x = (1.0 - ey) / n2
    r2y = ex / n2
    p2 = r2w * dq2w + r2x * dq2x + r2y * dq2y
    da2w = (dq2w - r2w * p2) / n2
    da2x = (dq2x - r2x * p2) / n2
    da2y = (dq2y - r2y * p2) / n2

    dex = da1z + da2y
    dey = da1w - da2x
    dez = -da1x - da2w

    # Smoothstep backward: sigmoid(u) with u = (2t-1)/(t(1-t)), flat outside.
    den = tl.maximum(t * (1.0 - t), EPS)
    du_dt = (2.0 * den - (2.0 * t - 1.0) * (1.0 - 2.0 * t)) / (den * den)
    ds_dt = tl.where(inner, sig * (1.0 - sig) * du_dt, 0.0)
    in_ramp = ((ey + 0.9) / 1.8 > 0.0) & ((ey + 0.9) / 1.8 < 1.0)
    dey += tl.where(in_ramp, ds * ds_dt / 1.8, 0.0)

    # normalize(v) backward.
    pv = ex * dex + ey * dey + ez * dez
    tl.store(dv_ptr + e * 3 + 0, (dex - ex * pv) / vn_safe, mask=m)
    tl.store(dv_ptr + e * 3 + 1, (dey - ey * pv) / vn_safe, mask=m)
    tl.store(dv_ptr + e * 3 + 2, (dez - ez * pv) / vn_safe, mask=m)


@triton_op("uma_nitro::wigner_fwd", mutates_args={"comp", "wigner", "wigner_inv"})
def _wigner_fwd(
    v: torch.Tensor,
    gamma: torch.Tensor,
    d55: torch.Tensor,
    env: torch.Tensor,
    comp: torch.Tensor,
    wigner: torch.Tensor,
    wigner_inv: torch.Tensor,
    eps: float,
) -> None:
    """
    Launch the compact forward and the block-diagonal expansion.

    Args:
        v: Edge distance vectors, ``(E, 3)``.
        gamma: Random roll angle per edge, ``(E,)``.
        d55: Quartic monomial coefficients, ``(55, 32)``.
        env: Radial envelope ``(E,)``, folded into ``wigner_inv``.
        comp: Compact output ``(E, 34)``; mutated.
        wigner: Output ``(E, 9, 9)``; mutated.
        wigner_inv: Envelope-scaled transposed output ``(E, 9, 9)``; mutated.
        eps: Floor used by the normalizations.
    """
    n = v.shape[0]
    grid = (triton.cdiv(n, BLOCK_E),)
    wrap_triton(_wig_fwd_kernel)[grid](
        v, gamma, d55, comp, n, EPS=eps, BLK=BLOCK_E, num_warps=WFV_WARPS
    )
    wrap_triton(_wig_expand_kernel)[(triton.cdiv(n, BLOCK_MOVE),)](
        comp, wigner, wigner_inv, env, n, BLK=BLOCK_MOVE, num_warps=4
    )


@triton_op("uma_nitro::wigner_bwd", mutates_args={"dcomp", "dv", "denv"})
def _wigner_bwd(
    grad_w: torch.Tensor,
    grad_wi: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
    csym: torch.Tensor,
    comp: torch.Tensor,
    env: torch.Tensor,
    dcomp: torch.Tensor,
    dv: torch.Tensor,
    denv: torch.Tensor,
    eps: float,
) -> None:
    """
    Launch the compact gradient gather and the chain backward.

    Args:
        grad_w: Gradient of ``wigner``, ``(E, 9, 9)``.
        grad_wi: Gradient of the envelope-scaled ``wigner_inv``, ``(E, 9, 9)``.
        v: Edge distance vectors, ``(E, 3)``.
        gamma: Random roll angle per edge, ``(E,)``.
        csym: Symmetrized coefficients, column-padded to ``(32, 256)``.
        comp: The forward's compact ``(E, 34)`` values.
        env: Radial envelope ``(E,)``.
        dcomp: Scratch ``(E, 34)``; mutated.
        dv: Output gradient ``(E, 3)``; mutated.
        denv: Output envelope gradient ``(E,)``; mutated.
        eps: Floor used by the normalizations.
    """
    n = v.shape[0]
    grid = (triton.cdiv(n, BLOCK_E),)
    wrap_triton(_wig_gather_kernel)[(triton.cdiv(n, BLOCK_MOVE),)](
        grad_w, grad_wi, comp, env, dcomp, denv, n, BLK=BLOCK_MOVE, num_warps=4
    )
    wrap_triton(_wig_bwd_kernel)[grid](
        v,
        gamma,
        csym,
        dcomp,
        dv,
        n,
        EPS=eps,
        BLK=BLOCK_E,
        USE_TF32=_SUPPORTS_TF32_ROUND,
        num_warps=WFV_WARPS,
    )


_COEF_CACHE: dict = {}


_D55_NZ = "10000010000010000010000010000000010001000000010001010001000000000000100010000000100010000000000000010001010100010000010000000000000000000000101000100010000000000001000101010001000001000000000000100000101000001000000000000000100000000000100000000000100000000100010000000100010100010000000000000010000010100000001000000000000000000000101000100010000000000001000101010001000001000000000000100000101000001000000000000000010001000000010001010001000000000010000010100000100000000000000000010001010100010000010000000000010001000000010001010001000000000000000000000010000000100000000001000100000001000101000100000000100000000000100000000000100000000100010000000100010100010000000000100000101000001000000000000000010001000000010001010001000000000000000000000010000000100000000000001000100000001000100000000000000100010101000100000100000000000010000010100000100000000000000000000010000010100000001000000000000100010101000100000100000000000000000000000010000000100000000001000100000001000101000100000000000100010101000100000100000000000010000010100000100000000000000000010001010100010000010000000000100000100000100000100000100000000001000101010001000001000000000000001000100000001000100000000000000000100000101000000010000000000100010000000100010100010000000010000000000010000000000010000000000000100000101000000010000000000100010000000100010100010000000000010001010100010000010000000000001000001010000010000000000000000001000101010001000001000000000010000000000010000000000010000000001000001010000010000000000000000001000101010001000001000000000000001000100000001000100000000000100000100000100000100000100000000100010000000100010100010000000000000000000010100010001000000000000000000000101000100010000000000100010000000100010100010000000010000010000010000010000010000000"


def _assert_d55_pattern(d55):
    """
    Fail loudly if the coefficient sparsity differs from the pattern baked into
    _wig_fwd_kernel. Called once per process from _coefficients.
    """
    got = "".join("1" if v else "0" for v in (d55 != 0).flatten().tolist())
    if got != _D55_NZ:
        raise UMANitroNotSupportedError(
            "wigner d55 sparsity differs from the pattern compiled into "
            "_wig_fwd_kernel; regenerate the sparse contraction"
        )


def _coefficients(c):
    """
    Build the two padded coefficient matrices the kernels contract against.

    Args:
        c: ``l=2`` coefficients viewing as ``(25, 256)``.

    Returns:
        ``(d55, csym)``: ``d55`` is ``(55, 32)`` holding one coefficient per
        quartic monomial for the forward, and ``csym`` is ``(32, 256)`` holding
        ``C[k, ab, cd] + C[k, cd, ab]`` for the backward's MMA. Both are padded
        from 25 columns (resp. rows) to 32.
    """
    key = (c.data_ptr(), c.shape, c.dtype, c.device)
    hit = _COEF_CACHE.get(key)
    if hit is not None:
        return hit
    cc = c.reshape(25, 256).to(torch.float32)
    # Collapse the 16x16 (ab, cd) index pairs onto the 10 quadratic monomials, then
    # fold to the upper triangle so each of the 55 quartic products carries one
    # coefficient. Generated with the same pair enumeration the kernel unrolls.
    quad = [(i, j) for i in range(4) for j in range(i, 4)]
    midx = {p: n for n, p in enumerate(quad)}
    # Accumulate in fp64: each folded coefficient sums up to four originals, and
    # rounding that sum in fp32 perturbs the function itself rather than just its
    # evaluation.
    full = torch.zeros(10, 10, 25, device=c.device, dtype=torch.float64)
    grid = cc.reshape(25, 16, 16).double()
    for ab in range(16):
        a, b = divmod(ab, 4)
        i = midx[(min(a, b), max(a, b))]
        for cd in range(16):
            cx, dx = divmod(cd, 4)
            j = midx[(min(cx, dx), max(cx, dx))]
            full[i, j] += grid[:, ab, cd]
    acc64 = torch.zeros(55, 32, device=c.device, dtype=torch.float64)
    n = 0
    for i in range(10):
        for j in range(i, 10):
            acc64[n, :25] = full[i, j] + (full[j, i] if j > i else 0.0)
            n += 1
    d55 = acc64.to(torch.float32)
    ct = torch.zeros(256, 32, device=c.device, dtype=torch.float32)
    ct[:, :25] = cc.t()
    sym = cc.reshape(25, 16, 16) + cc.reshape(25, 16, 16).transpose(1, 2)
    csym = torch.zeros(32, 256, device=c.device, dtype=torch.float32)
    csym[:25] = sym.reshape(25, 256)
    out = (d55.contiguous(), csym.contiguous())
    _assert_d55_pattern(out[0] if isinstance(out, tuple) else out)
    _COEF_CACHE[key] = out
    return out


class _WignerFromVec(torch.autograd.Function):
    """
    Autograd wrapper for the fused edge-vector -> Wigner chain.
    """

    @staticmethod
    def forward(ctx, v, gamma, c, eps, env):
        """
        Args:
            v: Edge distance vectors, ``(E, 3)``.
            gamma: Random roll angle per edge, ``(E,)``.
            c: ``l=2`` coefficients, any shape viewing as ``(25, 16, 16)``.
            eps: Floor used by the normalizations.
            env: Radial envelope ``(E,)``, folded into ``wigner_inv``.

        Returns:
            ``(wigner, wigner_inv * env)``, each ``(E, 9, 9)``.
        """
        n = v.shape[0]
        comp = torch.empty(n, 34, device=v.device, dtype=v.dtype)
        wigner = torch.empty(n, 9, 9, device=v.device, dtype=v.dtype)
        wigner_inv = torch.empty(n, 9, 9, device=v.device, dtype=v.dtype)
        d55, csym = _coefficients(c)
        torch.ops.uma_nitro.wigner_fwd(
            v, gamma, d55, env, comp, wigner, wigner_inv, eps
        )
        ctx.save_for_backward(v, gamma, csym, comp, env)
        ctx.eps = eps
        return wigner, wigner_inv

    @staticmethod
    def backward(ctx, grad_w, grad_wi):
        """
        Args:
            grad_w: Gradient of ``wigner``.
            grad_wi: Gradient of the envelope-scaled ``wigner_inv``.

        Returns:
            Gradients for ``(v, gamma, c, eps, env)``; ``gamma`` and ``c`` are
            frozen and ``eps`` is a float.
        """
        v, gamma, csym, comp, env = ctx.saved_tensors
        n = v.shape[0]
        dcomp = torch.empty(n, 34, device=v.device, dtype=v.dtype)
        dv = torch.empty_like(v)
        denv = torch.empty_like(env)
        torch.ops.uma_nitro.wigner_bwd(
            grad_w.contiguous(),
            grad_wi.contiguous(),
            v,
            gamma,
            csym,
            comp,
            env,
            dcomp,
            dv,
            denv,
            ctx.eps,
        )
        return dv, None, None, None, denv


def wigner_from_vec(edge_distance_vec, gamma, c_l2, eps, env):
    """
    Fused ``edge vector -> (wigner, wigner_inv * envelope)``.

    Args:
        edge_distance_vec: Edge distance vectors, ``(E, 3)``.
        gamma: Random roll angle per edge, ``(E,)``.
        c_l2: ``l=2`` coefficients viewing as ``(25, 256)``.
        eps: Floor used by the normalizations.
        env: Radial envelope ``(E,)``.

    Returns:
        ``(wigner, wigner_inv * envelope)``, each ``(E, 9, 9)``.
    """
    return _WignerFromVec.apply(edge_distance_vec, gamma, c_l2, eps, env)


@dataclass(frozen=True)
class BlockLayout:
    """
    Describes one ``(E, P, C)`` buffer partitioned into per-m coefficient blocks.
    """

    num_edges: int
    channels: int = 128
    block_sizes: tuple[int, ...] = (3, 4, 2)
    coeff_pad: int | None = None
    # Leading columns before the coefficient blocks, for conv1's gating scalars.
    # Not decoration: conv1's m=0 GEMM emits 640 columns that split as
    # (256 gating, 384 m=0 coefficients), gating first -- so a [gating | blocks]
    # buffer is exactly that GEMM's natural output and one out= write covers both.
    prefix: int = 0

    @property
    def num_coeffs(self) -> int:
        """
        Number of real coefficients, i.e. ``(lmax + 1) ** 2``.
        """
        return sum(self.block_sizes)

    @property
    def coeff_stride(self) -> int:
        """
        Coefficient slots per edge, padded or not. This is ``P``.
        """
        return self.coeff_pad if self.coeff_pad is not None else self.num_coeffs

    @property
    def offsets(self) -> tuple[int, ...]:
        """
        CSR-style block offsets over coefficients, e.g. ``(0, 3, 7, 9)``.
        """
        out, running = [0], 0
        for size in self.block_sizes:
            running += size
            out.append(running)
        return tuple(out)

    @property
    def coeff_base(self) -> int:
        """
        Element offset of coefficient 0 within a row.
        """
        return self.prefix

    @property
    def row_stride(self) -> int:
        """
        Elements between consecutive edges.
        """
        return self.prefix + self.coeff_stride * self.channels

    def prefix_view(self, buf):
        """
        The leading columns, e.g. conv1's gating scalars.

        Args:
            buf: A buffer from ``allocate``.

        Returns:
            An ``(E, prefix)`` strided view.
        """
        return buf[:, : self.prefix]

    def allocate(self, device="cuda", dtype=torch.float32, fill=None):
        """
        Allocate one buffer for this layout.

        Args:
            device: Torch device string.
            dtype: Element dtype.
            fill: If given, ``full`` with this value instead of ``empty``, which
                makes padded lanes detectable in tests.

        Returns:
            A ``(E, P * C)`` tensor.
        """
        shape = (self.num_edges, self.row_stride)
        if fill is None:
            return torch.empty(shape, device=device, dtype=dtype)
        return torch.full(shape, fill, device=device, dtype=dtype)

    def coeff_view(self, buf):
        """
        View the buffer as ``(E, num_coeffs, C)``, excluding padded lanes.

        Strided when padded, which every consumer here tolerates: cuBLAS takes
        it via ``lda`` and Triton via explicit offsets.

        Args:
            buf: A buffer from ``allocate``.

        Returns:
            A strided view.
        """
        coeffs = buf[:, self.coeff_base :]
        return coeffs.reshape(self.num_edges, self.coeff_stride, self.channels)[
            :, : self.num_coeffs, :
        ]

    def block(self, buf, m):
        """
        Return one coefficient block as a strided GEMM view.
        """
        start = self.coeff_base + self.offsets[m] * self.channels
        width = self.block_sizes[m] * self.channels
        return buf[:, start : start + width]

    def block_offsets_elements(self) -> tuple[int, ...]:
        """
        Per-m start offsets in elements, for passing to a kernel as constants.

        Returns:
            One offset per block.
        """
        return tuple(self.coeff_base + o * self.channels for o in self.offsets[:-1])

    def describe(self) -> str:
        """
        Return a human-readable layout summary.
        """
        waste = self.coeff_stride / self.num_coeffs
        return (
            f"E={self.num_edges} C={self.channels} P={self.coeff_stride} "
            f"blocks={self.block_sizes} offsets={self.offsets} "
            f"prefix={self.prefix} row_stride={self.row_stride} mem={waste:.2f}x"
        )


@triton.jit
def _segreduce_tgt_fp16_kernel(
    ge_ptr,
    scale_ptr,
    off_ptr,
    out_ptr,
    num_nodes,
    CHANNELS: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    ESPLIT: tl.constexpr,
    NSPLIT: tl.constexpr,
    CBLK: tl.constexpr,
    CPG: tl.constexpr,
    NGRP: tl.constexpr,
    ACC: tl.constexpr = 0,
):
    """
    Reduce an fp16 per-edge gradient onto target nodes.

    Each power-of-two scale is undone before fp32 accumulation. ESPLIT programs
    may share a node and combine their partitions atomically.
    """
    # Channel split as well as edge split. This kernel reads only ``scale[e]`` per edge,
    # so splitting the channel axis costs no redundant scalar loads -- unlike
    # _invrot_segreduce, where re-reading 35 Wigner values per edge per split cancelled
    # the occupancy gain and left NSPLIT useless there.
    _lane = tl.program_id(0) // ESPLIT
    grp = _lane % NGRP
    _lane //= NGRP
    node = _lane // NSPLIT
    chan = (_lane % NSPLIT) * CBLK + tl.arange(0, CBLK)
    part = tl.program_id(0) % ESPLIT
    # First coefficient this program owns. The nine accumulators are mutually
    # independent -- each coefficient's sum over the node's edges is its own chain --
    # so handing a program only CPG of them multiplies the program count without
    # touching a single sum's term order, and without narrowing a load the way a
    # channel split does. The only duplicated read is the per-edge scalar.
    cb = grp * CPG
    # Contiguous (E, 9, C) buffer: the target half is its own allocation now, so the
    # nine loads per edge are one sequential 2304-byte run instead of nine 256-byte
    # chunks separated by 256-byte gaps.
    row = 9 * CHANNELS
    while node < num_nodes:
        start = tl.load(off_ptr + node)
        end = tl.load(off_ptr + node + 1)
        a0 = tl.zeros((CBLK,), dtype=tl.float32)
        a1 = tl.zeros((CBLK,), dtype=tl.float32)
        a2 = tl.zeros((CBLK,), dtype=tl.float32)
        a3 = tl.zeros((CBLK,), dtype=tl.float32)
        a4 = tl.zeros((CBLK,), dtype=tl.float32)
        a5 = tl.zeros((CBLK,), dtype=tl.float32)
        a6 = tl.zeros((CBLK,), dtype=tl.float32)
        a7 = tl.zeros((CBLK,), dtype=tl.float32)
        a8 = tl.zeros((CBLK,), dtype=tl.float32)
        for e in range(start + part, end, ESPLIT):
            b = e * row + chan
            s = tl.load(scale_ptr + e)
            a0 += tl.load(ge_ptr + b + (cb + 0) * CHANNELS).to(tl.float32) * s
            if CPG > 1:
                a1 += tl.load(ge_ptr + b + (cb + 1) * CHANNELS).to(tl.float32) * s
            if CPG > 2:
                a2 += tl.load(ge_ptr + b + (cb + 2) * CHANNELS).to(tl.float32) * s
            if CPG > 3:
                a3 += tl.load(ge_ptr + b + (cb + 3) * CHANNELS).to(tl.float32) * s
            if CPG > 4:
                a4 += tl.load(ge_ptr + b + (cb + 4) * CHANNELS).to(tl.float32) * s
            if CPG > 5:
                a5 += tl.load(ge_ptr + b + (cb + 5) * CHANNELS).to(tl.float32) * s
            if CPG > 6:
                a6 += tl.load(ge_ptr + b + (cb + 6) * CHANNELS).to(tl.float32) * s
            if CPG > 7:
                a7 += tl.load(ge_ptr + b + (cb + 7) * CHANNELS).to(tl.float32) * s
            if CPG > 8:
                a8 += tl.load(ge_ptr + b + (cb + 8) * CHANNELS).to(tl.float32) * s
        o = node * 9 * CHANNELS + chan
        # ACC forces the accumulating form at ESPLIT == 1, for the case where the
        # conv1 backward has already scattered the *source* half into ``out``
        # (C1B_SCATTER == 1) and this reduction must add rather than overwrite.
        if ESPLIT == 1 and not ACC:
            tl.store(out_ptr + o + (cb + 0) * CHANNELS, a0)
            if CPG > 1:
                tl.store(out_ptr + o + (cb + 1) * CHANNELS, a1)
            if CPG > 2:
                tl.store(out_ptr + o + (cb + 2) * CHANNELS, a2)
            if CPG > 3:
                tl.store(out_ptr + o + (cb + 3) * CHANNELS, a3)
            if CPG > 4:
                tl.store(out_ptr + o + (cb + 4) * CHANNELS, a4)
            if CPG > 5:
                tl.store(out_ptr + o + (cb + 5) * CHANNELS, a5)
            if CPG > 6:
                tl.store(out_ptr + o + (cb + 6) * CHANNELS, a6)
            if CPG > 7:
                tl.store(out_ptr + o + (cb + 7) * CHANNELS, a7)
            if CPG > 8:
                tl.store(out_ptr + o + (cb + 8) * CHANNELS, a8)
        else:
            tl.atomic_add(out_ptr + o + (cb + 0) * CHANNELS, a0)
            if CPG > 1:
                tl.atomic_add(out_ptr + o + (cb + 1) * CHANNELS, a1)
            if CPG > 2:
                tl.atomic_add(out_ptr + o + (cb + 2) * CHANNELS, a2)
            if CPG > 3:
                tl.atomic_add(out_ptr + o + (cb + 3) * CHANNELS, a3)
            if CPG > 4:
                tl.atomic_add(out_ptr + o + (cb + 4) * CHANNELS, a4)
            if CPG > 5:
                tl.atomic_add(out_ptr + o + (cb + 5) * CHANNELS, a5)
            if CPG > 6:
                tl.atomic_add(out_ptr + o + (cb + 6) * CHANNELS, a6)
            if CPG > 7:
                tl.atomic_add(out_ptr + o + (cb + 7) * CHANNELS, a7)
            if CPG > 8:
                tl.atomic_add(out_ptr + o + (cb + 8) * CHANNELS, a8)
        node += GRID_STRIDE // NSPLIT


@triton_op("uma_nitro::segreduce_tgt_fp16", mutates_args={"out"})
def _segreduce_tgt_fp16_op(
    grad_edge: torch.Tensor,
    scale: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
    channels: int,
) -> None:
    """
    Launch the fp16 node-major target reduction.

    Args:
        grad_edge: ``(E, 9, C)`` fp16 per-edge gradient, target half.
        scale: Per-edge reciprocal scale, ``(E,)`` fp32.
        offsets: CSR row pointers over targets, ``(N + 1,)``.
        out: ``(N, 9, C)`` fp32 destination; overwritten.
        channels: ``C``.
    """
    nodes = out.shape[0]
    if "segtgt" not in _ABLATIONS:
        _ngrp = _coeff_groups(SCT_CPG, "SCT_CPG")
        wrap_triton(_segreduce_tgt_fp16_kernel)[
            (nodes * SCT_ESPLIT * SCT_NSPLIT * _ngrp,)
        ](
            grad_edge,
            scale,
            offsets,
            out,
            nodes,
            CHANNELS=channels,
            GRID_STRIDE=nodes * SCT_NSPLIT,
            ESPLIT=SCT_ESPLIT,
            NSPLIT=SCT_NSPLIT,
            CBLK=channels // SCT_NSPLIT,
            num_warps=SCT_WARPS,
            num_stages=SCT_STAGES,
            CPG=SCT_CPG,
            NGRP=_ngrp,
            ACC=1 if (_scatter_mode() & 1) else 0,
        )


SCT_CHUNK = int(os.environ.get("UMA_NITRO_SCT_CHUNK", "0"))


@triton.jit
def _segreduce_tgt_bal_kernel(
    ge_ptr,
    scale_ptr,
    tgt_ptr,
    out_ptr,
    num_edges,
    CHANNELS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """
    Reduce fixed-size chunks of sorted edges onto target nodes.

    A chunk flushes its fp32 accumulators when the target changes and at the end.
    """
    chan = tl.arange(0, CHANNELS)
    row = 9 * CHANNELS
    e0 = tl.program_id(0).to(tl.int64) * CHUNK
    e1 = tl.minimum(e0 + CHUNK, num_edges)
    a0 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a1 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a2 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a3 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a4 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a5 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a6 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a7 = tl.zeros((CHANNELS,), dtype=tl.float32)
    a8 = tl.zeros((CHANNELS,), dtype=tl.float32)
    cur = tl.load(tgt_ptr + e0, mask=e0 < num_edges, other=0)
    for e in range(e0, e1):
        node = tl.load(tgt_ptr + e)
        if node != cur:
            o = cur * row + chan
            tl.atomic_add(out_ptr + o + 0 * CHANNELS, a0)
            tl.atomic_add(out_ptr + o + 1 * CHANNELS, a1)
            tl.atomic_add(out_ptr + o + 2 * CHANNELS, a2)
            tl.atomic_add(out_ptr + o + 3 * CHANNELS, a3)
            tl.atomic_add(out_ptr + o + 4 * CHANNELS, a4)
            tl.atomic_add(out_ptr + o + 5 * CHANNELS, a5)
            tl.atomic_add(out_ptr + o + 6 * CHANNELS, a6)
            tl.atomic_add(out_ptr + o + 7 * CHANNELS, a7)
            tl.atomic_add(out_ptr + o + 8 * CHANNELS, a8)
            a0 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a1 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a2 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a3 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a4 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a5 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a6 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a7 = tl.zeros((CHANNELS,), dtype=tl.float32)
            a8 = tl.zeros((CHANNELS,), dtype=tl.float32)
            cur = node
        b = e * row + chan
        sc = tl.load(scale_ptr + e)
        a0 += tl.load(ge_ptr + b + 0 * CHANNELS).to(tl.float32) * sc
        a1 += tl.load(ge_ptr + b + 1 * CHANNELS).to(tl.float32) * sc
        a2 += tl.load(ge_ptr + b + 2 * CHANNELS).to(tl.float32) * sc
        a3 += tl.load(ge_ptr + b + 3 * CHANNELS).to(tl.float32) * sc
        a4 += tl.load(ge_ptr + b + 4 * CHANNELS).to(tl.float32) * sc
        a5 += tl.load(ge_ptr + b + 5 * CHANNELS).to(tl.float32) * sc
        a6 += tl.load(ge_ptr + b + 6 * CHANNELS).to(tl.float32) * sc
        a7 += tl.load(ge_ptr + b + 7 * CHANNELS).to(tl.float32) * sc
        a8 += tl.load(ge_ptr + b + 8 * CHANNELS).to(tl.float32) * sc
    if e0 < num_edges:
        o = cur * row + chan
        tl.atomic_add(out_ptr + o + 0 * CHANNELS, a0)
        tl.atomic_add(out_ptr + o + 1 * CHANNELS, a1)
        tl.atomic_add(out_ptr + o + 2 * CHANNELS, a2)
        tl.atomic_add(out_ptr + o + 3 * CHANNELS, a3)
        tl.atomic_add(out_ptr + o + 4 * CHANNELS, a4)
        tl.atomic_add(out_ptr + o + 5 * CHANNELS, a5)
        tl.atomic_add(out_ptr + o + 6 * CHANNELS, a6)
        tl.atomic_add(out_ptr + o + 7 * CHANNELS, a7)
        tl.atomic_add(out_ptr + o + 8 * CHANNELS, a8)


@triton_op("uma_nitro::segreduce_tgt_bal", mutates_args={"out"})
def _segreduce_tgt_bal(
    grad_edge: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    out: torch.Tensor,
    channels: int,
) -> None:
    """
    Launch the load-balanced target reduction.

    Args:
        grad_edge: ``(E, 9, C)`` fp16 target half.
        scale: Per-edge reciprocal scale, ``(E,)``.
        target: Target node per edge, ``(E,)``, sorted ascending.
        out: ``(N, 9, C)`` fp32, pre-zeroed; mutated.
        channels: ``C``.
    """
    edges = grad_edge.shape[0]
    grid = triton.cdiv(edges, SCT_CHUNK)
    wrap_triton(_segreduce_tgt_bal_kernel)[(grid,)](
        grad_edge,
        scale,
        target,
        out,
        edges,
        CHANNELS=channels,
        CHUNK=SCT_CHUNK,
        num_warps=SCT_WARPS,
    )


def segreduce_tgt_fp16(grad_edge, scale, offsets, num_nodes, channels, out=None):
    """
    Reduce an fp16 ``grad_edge``'s target half onto nodes with no atomics.

    Args:
        grad_edge: ``(E, 9, C)`` fp16 per-edge gradient, target half.
        scale: Per-edge reciprocal scale, ``(E,)`` fp32.
        offsets: CSR row pointers over targets, ``(N + 1,)``.
        num_nodes: Node count.
        channels: ``C``.
        out: Optional ``(N, 9, C)`` fp32 destination; overwritten.

    Returns:
        ``(N, 9, C)`` fp32.
    """
    if out is None:
        alloc = torch.zeros if SCT_ESPLIT > 1 else torch.empty
        out = alloc(
            num_nodes, 9, channels, device=grad_edge.device, dtype=torch.float32
        )
    torch.ops.uma_nitro.segreduce_tgt_fp16(grad_edge, scale, offsets, out, channels)
    return out


# ============================================================================
# Embedded CUTLASS dense GEMM
# ============================================================================
# BEGIN COPIED/ADAPTED NVIDIA CUTLASS SOURCE
#
# ruff: noqa
# Project: NVIDIA CUTLASS (https://github.com/NVIDIA/cutlass)
# Upstream: v4.7.0, commit dcf215af68a2d08d305076c152a06f201728cd53
# Original path:
#   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py
# Copied/adapted symbol: the complete dense-GEMM kernel class, named
#   HopperWgmmaGemmKernel here, including every method through the END marker.
#
# FairChem modifications:
# - renamed the class for the local namespace;
# - removed the example CLI, benchmark driver, and __main__ entry point;
# - added optional bias and per-row rescaling to the epilogue; and
# - added an optional ``epilogue_op`` constexpr before output conversion.
#
# Linting is disabled for this block so it remains readily comparable to the
# upstream implementation. The full upstream BSD-3-Clause notice follows.
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Hopper batched dense GEMM adapted from the NVIDIA CUTLASS CuTe DSL example.

The kernel uses TMA transfers, WGMMA, clustered multicast, and a staged
pipeline. The retained implementation adds an optional accumulator epilogue for
UMA Nitro. See ``UMA_NITRO_LICENSE_PROVENANCE.md`` for the upstream source.
"""


# The upstream example's argument parsing lived here and is removed: it referenced
# ``argparse``, which this file does not import, so it could only ever raise.

# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////


class HopperWgmmaGemmKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float16
          A and B must have the same data type
        - Float8E4M3FN/Float8E5M2
          A and B can have different types (Float8E4M3FN/Float8E5M2)
          only support k-major layout
        - Int8/Uint8
          A and B can have different types (Int8/Uint8)
          only support k-major layout

    :note: Supported accumulation types:
        - Float32/Float16 (for all floating point inputs)
        - Int32 (for Int8/Uint8 inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = None,
        bias: cute.Tensor = None,
        rscale: cute.Tensor = None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mn)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.epilogue_op = epilogue_op
        self.mBias = bias
        self.mRScale = rscale

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epilogue_op,
            self.mBias,
            self.mRScale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        epilogue_op: cutlass.Constexpr = None,
        mBias: cute.Tensor = None,
        mRScale: cute.Tensor = None,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        # CTA Swizzle to promote L2 data reuse
        group_size_m = 8
        s_shape = (
            (group_size_m, cdimx // group_size_m),
            cdimy,
        )
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

        # Deal with the tail part
        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]

        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        #  Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, RestK)
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        # (bN, bK, RestK)
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        # (bM, bN)
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mnl)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma_partition,
            gB_for_tma_partition,
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Make fragments
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cluster wait
        # ///////////////////////////////////////////////////////////////////////////////
        # cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        # /////////////////////////////////////////////////////////////////////////////
        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if warp_idx == 0:
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1

        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            # Wait for A/B buffer to be ready
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            # /////////////////////////////////////////////////////////////////////////////
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA
            # /////////////////////////////////////////////////////////////////////////////
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )

            cute.nvgpu.warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )
            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        # /////////////////////////////////////////////////////////////////////////////
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mn) > 1:
            # Wait for all threads in the cluster to finish, avoid early release of smem
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            # For cluster that has a single thread block, it might have more than one warp groups.
            # Wait for all warp groups in the thread block to finish, because smem for tensor A in
            # the mainloop is reused in the epilogue.
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        # Per-column bias for the fused epilogue. Rather than extract an (m, n)
        # coordinate -- partition_C returns a *hierarchical* coordinate, so naive
        # crd[1] indexing yields an inner mode (it produced n % 2, not n) -- the
        # bias arrives already shaped (M, N, L) with stride (0, 1, 0), i.e. an
        # M-broadcast view. Pushing it through the same local_tile / partition_C /
        # retile chain as the accumulator makes its flat register indexing
        # identical to tRS_rAcc's by construction, which the bit-exact identity
        # epilogue already validates.
        if cutlass.const_expr(mBias is not None or mRScale is not None):
            # The fragment's element order is defined by the *copy* partitioning
            # (tRS_rD's own layout comes from thr_copy_r2s.partition_S(sC)), not by
            # thr_mma.partition_C. Building coordinates from partition_C instead
            # yields values that are individually right but land at the wrong
            # output columns. Coordinates are per epilogue subtile and get the
            # subtile's own N offset added below.
            tRS_cEpi = thr_copy_r2s.partition_S(
                cute.make_identity_tensor(self.epi_tile)
            )
            n_tile_base = pid_n * self.tile_shape_mnk[1]
            m_tile_base = pid_m * self.tile_shape_mnk[0]

        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        # Initialize tma store c_pipeline
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from accumulators to D registers
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            if cutlass.const_expr(mRScale is not None):
                # Per-row unscale, applied before the bias to match fp16_mm.
                sc_frag = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
                epi_crd = epi_tile_layout.get_hier_coord(epi_idx)
                m_epi_base = m_tile_base + epi_crd[0] * self.epi_tile[0]
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    sc_frag[epi_v] = mRScale[m_epi_base + tRS_cEpi[epi_v][0]]
                acc_vec = acc_vec * sc_frag.load()
            if cutlass.const_expr(mBias is not None):
                # Gather bias[n] per register. size_tRS_rD is small and the bias is
                # N floats, so this stays L1-resident.
                bias_frag = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
                epi_crd_n = epi_tile_layout.get_hier_coord(epi_idx)
                n_epi_base = n_tile_base + epi_crd_n[1] * self.epi_tile[1]
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    bias_frag[epi_v] = mBias[n_epi_base + tRS_cEpi[epi_v][1]]
                acc_vec = acc_vec + bias_frag.load()
            if cutlass.const_expr(epilogue_op is not None):
                acc_vec = epilogue_op(acc_vec)
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # barrier for sync
            pipeline.sync(barrier_id=1)

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()

            pipeline.sync(barrier_id=1)

        if warp_idx == 0:
            c_pipeline.producer_tail()

        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        epi_stage = 4
        # epi_smem will reuse smem ab.
        epi_bytes = 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl)
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))
        return grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Uint8,
            cutlass.Int8,
        }
        if a_dtype not in valid_ab_dtypes:
            is_valid = False
        if b_dtype not in valid_ab_dtypes:
            is_valid = False

        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        if a_dtype.width != b_dtype.width:
            is_valid = False
        if not a_dtype.is_same_kind(b_dtype):
            is_valid = False

        # for 8-bit types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and A type
        if a_dtype not in acc_ab_compatibility[acc_dtype]:
            is_valid = False

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[acc_dtype]:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid


# END COPIED/ADAPTED NVIDIA CUTLASS SOURCE


@cute.kernel
def _gather_v4_kernel(
    src: cute.Tensor,
    idx: cute.Tensor,
    out: cute.Tensor,
    num_edges: cutlass.Int32,
    ROW: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
):
    """
    Gather rows with aligned vector loads and stores.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    chunks: cutlass.Constexpr = ROW // 4
    per_thread: cutlass.Constexpr = (chunks + THREADS - 1) // THREADS
    edge = bidx
    while edge < num_edges:
        node = idx[edge]
        for r in cutlass.range_constexpr(per_thread):
            k = tidx + r * THREADS
            if k < chunks:
                off = k * 4
                sfrag = cute.make_tensor(
                    src.iterator + node * ROW + off, cute.make_layout(4)
                )
                dfrag = cute.make_tensor(
                    out.iterator + edge * ROW + off, cute.make_layout(4)
                )
                dfrag.store(sfrag.load())
        edge += gdim


@cute.jit
def gather_v4(
    src: cute.Tensor,
    idx: cute.Tensor,
    out: cute.Tensor,
    num_edges: cutlass.Int32,
    grid: cutlass.Constexpr,
    ROW: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    """
    Launch the vector-4 gather.

    The stream is explicit: `.launch()` without one does not necessarily use the
    stream that a CUDA graph capture is recording on, and work issued to a
    non-capturing stream during capture shows up as an illegal memory access. That
    failure vanished under CUDA_LAUNCH_BLOCKING=1, which is the tell for a stream
    race rather than a bad pointer.
    """
    _gather_v4_kernel(src, idx, out, num_edges, ROW, THREADS).launch(
        grid=[grid, 1, 1], block=[THREADS, 1, 1], stream=stream
    )


@cute.kernel
def _scatter_v4_fp16_kernel(
    src: cute.Tensor,
    idx: cute.Tensor,
    scale: cute.Tensor,
    out: cute.Tensor,
    num_edges: cutlass.Int32,
    NUM_SPH: cutlass.Constexpr,
    C: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
):
    """
    Scatter an fp16 source into an fp32 destination with vector atomics.

    Each source row is converted in registers and multiplied by its scale.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    chunks: cutlass.Constexpr = (NUM_SPH * C) // 4
    per_thread: cutlass.Constexpr = (chunks + THREADS - 1) // THREADS

    edge = bidx
    while edge < num_edges:
        node = idx[edge]
        sc = scale[edge]
        for r in cutlass.range_constexpr(per_thread):
            k = tidx + r * THREADS
            if k < chunks:
                dst_off = k * 4
                s = edge * (NUM_SPH * C) + dst_off
                frag = cute.make_tensor(src.iterator + s, cute.make_layout(4))
                cute.arch.atomic_add(
                    out.iterator + node * (NUM_SPH * C) + dst_off,
                    frag.load().to(cutlass.Float32) * sc,
                )
        edge += gdim


@cute.jit
def scatter_v4_fp16(
    src: cute.Tensor,
    idx: cute.Tensor,
    scale: cute.Tensor,
    out: cute.Tensor,
    num_edges: cutlass.Int32,
    grid: cutlass.Constexpr,
    NUM_SPH: cutlass.Constexpr,
    C: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    """
    Launch the fp16 vector-4 scatter, on an explicit stream (see gather_v4).
    """
    _scatter_v4_fp16_kernel(
        src, idx, scale, out, num_edges, NUM_SPH, C, THREADS
    ).launch(grid=[grid, 1, 1], block=[THREADS, 1, 1], stream=stream)


SV4_GRID = int(os.environ.get("UMA_NITRO_SV4_GRID", "105472"))
THREADS = int(os.environ.get("UMA_NITRO_SV4_THREADS", "128"))


@lru_cache(maxsize=16)
def _compiled_fp16(num_sph, channels, grid):
    """
    Compile and cache the fp16 scatter for a shape signature.
    """
    dev = torch.device("cuda")
    edges = 1
    # ``_scatter_v4_fp16_kernel`` splits ``NUM_SPH * C`` into ``// 4`` fragments and
    # visits exactly that many, so a row that is not a multiple of four silently
    # drops its tail -- a wrong gradient with nothing out of range for any guard to
    # notice. Checked once per compiled signature rather than assumed.
    if (num_sph * channels) % 4:
        raise AssertionError(
            f"UMA Nitro: vector scatter needs NUM_SPH * C divisible by 4, "
            f"got {num_sph} * {channels} = {num_sph * channels}"
        )
    src = torch.empty(edges * num_sph * channels, device=dev, dtype=torch.float16)
    idx = torch.empty(edges, device=dev, dtype=torch.int32)
    sc = torch.empty(edges, device=dev)
    out = torch.empty(1, device=dev)
    return cute.compile(
        scatter_v4_fp16,
        from_dlpack(src, assumed_align=16),
        from_dlpack(idx, assumed_align=16),
        from_dlpack(sc, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        cutlass.Int32(edges),
        grid,
        num_sph,
        channels,
        THREADS,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def scatter_src_v4_fp16(
    grad_edge: torch.Tensor,
    index: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Scatter-add the source half of an fp16 ``grad_edge`` with vector-4 atomics.

    Args:
        grad_edge: ``(E, 9, C)`` fp16 source half, read in full.
        index: Source node per edge, ``(E,)`` int32.
        scale: Per-edge reciprocal scale, ``(E,)`` fp32.
        out: ``(N, 9, C)`` fp32; accumulated in place.
    """
    edges, num_sph, src_c = grad_edge.shape
    channels = out.shape[2]
    if src_c != channels:
        raise AssertionError(
            f"UMA Nitro: vector scatter assumes the source row is the "
            f"destination width, got src_c={src_c} channels={channels}"
        )
    fn = _compiled_fp16(num_sph, channels, min(edges, SV4_GRID))
    fn(
        from_dlpack(grad_edge.reshape(-1), assumed_align=16),
        from_dlpack(index, assumed_align=16),
        from_dlpack(scale, assumed_align=16),
        from_dlpack(out.reshape(-1), assumed_align=16),
        cutlass.Int32(edges),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _scatter_src_v4_fp16_torch(
    grad_edge: torch.Tensor,
    index: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Scatter-add with portable PyTorch operations on pre-SM90 GPUs.
    """
    out.index_add_(
        0,
        index,
        grad_edge.float() * scale[:, None, None],
    )


scatter_src_v4_fp16 = torch.library.custom_op(
    "uma_nitro::scatter_src_v4_fp16", mutates_args={"out"}, device_types="cuda"
)(scatter_src_v4_fp16 if _USE_CUTEDSL else _scatter_src_v4_fp16_torch)


@scatter_src_v4_fp16.register_fake
def _(grad_edge, index, scale, out):
    """
    Nothing to describe: the op only mutates ``out``.
    """
    return None


@lru_cache(maxsize=16)
def _compiled_gather(row, grid):
    """
    Compile and cache the gather for one shape signature.

    ``edges`` is not part of the key, for the same reason as ``_compiled_fp16``.
    """
    dev = torch.device("cuda")
    edges = 1
    # Same ``// 4`` tail as the scatter: ``_gather_v4_kernel`` copies ``ROW // 4``
    # fragments and any remainder is never written, leaving whatever the
    # destination happened to hold.
    if row % 4:
        raise AssertionError(
            f"UMA Nitro: vector gather needs ROW divisible by 4, got {row}"
        )
    src = torch.empty(row, device=dev)
    idx = torch.empty(edges, device=dev, dtype=torch.int32)
    out = torch.empty(edges * row, device=dev)
    return cute.compile(
        gather_v4,
        from_dlpack(src, assumed_align=16),
        from_dlpack(idx, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        cutlass.Int32(edges),
        grid,
        row,
        THREADS,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def gather_node_v4(src: torch.Tensor, index: torch.Tensor, out: torch.Tensor) -> None:
    """
    Gather node rows to edges with vector-4 copies.

    Args:
        src: ``(N, 9, C)``.
        index: Node per edge, ``(E,)`` int32.
        out: ``(E, 9, C)``; mutated.
    """
    row = src.shape[1] * src.shape[2]
    fn = _compiled_gather(row, min(index.shape[0], SV4_GRID))
    fn(
        from_dlpack(src.reshape(-1), assumed_align=16),
        from_dlpack(index, assumed_align=16),
        from_dlpack(out.reshape(-1), assumed_align=16),
        cutlass.Int32(index.shape[0]),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _gather_node_v4_torch(
    src: torch.Tensor, index: torch.Tensor, out: torch.Tensor
) -> None:
    """
    Gather node rows with portable PyTorch operations on pre-SM90 GPUs.
    """
    torch.index_select(src, 0, index, out=out)


gather_node_v4 = torch.library.custom_op(
    "uma_nitro::gather_node_v4", mutates_args={"out"}, device_types="cuda"
)(gather_node_v4 if _USE_CUTEDSL else _gather_node_v4_torch)


@gather_node_v4.register_fake
def _(src, index, out):
    """
    Nothing to describe: the op only mutates ``out``.
    """
    return None


_TILE_BY_KN = {
    (384, 384): (128, 128),
    (512, 512): (128, 256),
    (256, 256): (128, 128),
    (768, 640): (128, 128),
    (1024, 512): (128, 256),
    (512, 256): (128, 256),
    (640, 768): (128, 256),
    (512, 1024): (128, 256),
    (256, 512): (128, 256),
    (128, 1536): (128, 256),
    (640, 288): (128, 128),
}
_TILE_BY_N = {256: (128, 128), 384: (128, 128), 512: (128, 256), 640: (128, 256)}
_DEFAULT_TILE = (128, 128)


def _tile_for(n, k=None):
    """
    Select the configured CTA tile for a GEMM shape.
    """
    if k is not None and (k, n) in _TILE_BY_KN:
        return _TILE_BY_KN[(k, n)]
    return _TILE_BY_N.get(n, _DEFAULT_TILE)


def _as_cute(t, dtype, dynamic=True):
    """
    Wrap a torch tensor as a cute tensor; 2-D inputs gain the required L mode.

    The kernel is MxNxKxL and indexes mode 2 as the batch, so a rank-2 input dies
    in ``_compute_grid`` with "expects rank(tiler) <= rank(input)".
    """
    ct = from_dlpack(t.unsqueeze(-1) if t.dim() == 2 else t, assumed_align=16)
    ct.element_type = dtype
    return ct.mark_layout_dynamic(leading_dim=1) if dynamic else ct


def _pack(mA, mB, mC, stream, mBias, mScale):
    """
    Build the argument list, truncating trailing ``None`` optional params.

    ``cute.compile`` drops trailing ``None`` from the compiled program's runtime
    signature. Passing them anyway packs more arguments than the buffer holds and
    segfaults inside ``run_compiled_program`` at launch, with no Python traceback
    unless faulthandler is on.
    """
    args = [mA, mB, mC, stream, None, mBias, mScale]
    while len(args) > 4 and args[-1] is None:
        args.pop()
    return args


def _check_gemm_operands(m, n, k, a_stride, c_stride, cct):
    """
    Refuse a shape the vendored kernel does not actually support.

    ``HopperWgmmaGemmKernel.is_valid_dtypes`` and ``is_valid_tensor_alignment``
    came in with the upstream example and lost their only caller when its ``run()``
    driver was dropped -- so two static methods that exist to reject an unsupported
    problem sat here as documentation. The cost of calling them is one evaluation
    per distinct compiled signature, i.e. nothing.

    The operand conventions are fixed by ``_as_cute``: everything is passed with
    ``mark_layout_dynamic(leading_dim=1)``, so A ``(M, K)`` and B ``(N, K)`` are
    k-major and C ``(M, N)`` is n-major.

    The vendored alignment check sees only *logical* shapes, which for this call
    site means it constrains K and N and never touches M. That leaves the thing
    that can actually break unexamined: A is frequently a strided row-slice of a
    wider buffer and C a column block of one, and ``_as_cute`` asserts
    ``assumed_align=16`` on the base pointer -- which only holds for every row if
    the row stride is a whole number of 16-byte lines. So the strides are checked
    here as well.

    Args:
        m: Rows of A and C.
        n: Rows of B, columns of C.
        k: Contiguous extent of A and B.
        a_stride: A's row stride in elements.
        c_stride: C's row stride in elements.
        cct: C's cutlass element type.

    Raises:
        AssertionError: If the dtypes, the logical alignment or the row strides
            are outside what the kernel supports.
    """
    if not HopperWgmmaGemmKernel.is_valid_dtypes(
        cutlass.Float16, cutlass.Float16, cutlass.Float32, cct, "k", "k"
    ):
        raise AssertionError(
            f"UMA Nitro: CuTeDSL GEMM rejects fp16 x fp16 -> {cct} with an "
            f"fp32 accumulator"
        )
    if not HopperWgmmaGemmKernel.is_valid_tensor_alignment(
        m, n, k, 1, cutlass.Float16, cct, "k", "k", "n"
    ):
        raise AssertionError(
            f"UMA Nitro: CuTeDSL GEMM rejects m={m} n={n} k={k} on 16-byte "
            f"alignment of the major modes"
        )
    for name, stride, width in (("a", a_stride, 16), ("c", c_stride, cct.width)):
        per_line = 16 * 8 // width
        if stride % per_line:
            raise AssertionError(
                f"UMA Nitro: {name} row stride {stride} is not a multiple of "
                f"{per_line} elements, so rows after the first are not 16-byte "
                f"aligned and `assumed_align=16` is a lie"
            )


@lru_cache(maxsize=128)
def _cdg_compiled(m, n, k, a_stride, c_stride, tile, has_bias, has_scale, c_half=False):
    """
    Compile and cache a kernel for one shape, stride and epilogue signature.

    Compilation is ~0.3 s per distinct signature and happens during warmup.

    ``c_half`` selects an fp16 destination. The kernel reads C's element type off
    the tensor and its epilogue already ends in ``acc_vec.to(self.c_dtype)``, so
    this costs one extra compiled signature and nothing at runtime. Verified
    bit-identical to rounding the fp32-C result, with and without the ``rscale``
    epilogue, so the only difference is the intended rounding.
    """
    dev = torch.device("cuda")
    cdt = torch.float16 if c_half else torch.float32
    cct = cutlass.Float16 if c_half else cutlass.Float32
    a = torch.empty((m, a_stride), device=dev, dtype=torch.float16)[:, :k]
    w = torch.empty((n, k), device=dev, dtype=torch.float16)
    c = torch.empty((m, c_stride), device=dev, dtype=cdt)[:, :n]
    bias = torch.empty(n, device=dev) if has_bias else None
    scale = torch.empty(m, device=dev) if has_scale else None
    kernel = HopperWgmmaGemmKernel(cutlass.Float32, tile, (1, 1))
    _check_gemm_operands(m, n, k, a_stride, c_stride, cct)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = _pack(
        _as_cute(a, cutlass.Float16),
        _as_cute(w, cutlass.Float16),
        _as_cute(c, cct),
        stream,
        None if bias is None else _as_cute(bias, cutlass.Float32, False),
        None if scale is None else _as_cute(scale, cutlass.Float32, False),
    )
    return cute.compile(kernel, *args)


def _cutedsl_mm_op(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
    rscale: torch.Tensor | None = None,
) -> None:
    """
    Opaque wrapper so the launch never runs under fake-tensor tracing.

    AOTAutograd traces the span's hand-written backward, where shapes arrive as
    SymInts; an ``lru_cache`` keyed on them raises "unhashable type: non-nested
    SymInt" (the same reason ``span_block``'s ``_layout`` is deliberately not
    memoized). Wrapping the launch in a custom op keeps the compile cache on the
    runtime path only, and Dynamo treats the whole thing as one node.
    """
    _launch(a, w, out, bias, rscale)


def _torch_mm_op(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
    rscale: torch.Tensor | None = None,
) -> None:
    """
    Apply the CuTe GEMM contract with portable PyTorch operations.
    """
    if out.dtype == torch.float32:
        torch.mm(a, w.t(), out=out, out_dtype=torch.float32)
        result = out
    else:
        result = torch.mm(a, w.t(), out_dtype=torch.float32)
    if rscale is not None:
        result.mul_(rscale[:, None])
    if bias is not None:
        result.add_(bias)
    if result is not out:
        out.copy_(result)


_cutedsl_mm_op = torch.library.custom_op(
    "uma_nitro_gemm::mm", mutates_args={"out"}, device_types="cuda"
)(_cutedsl_mm_op if _USE_HOPPER_GEMM else _torch_mm_op)


@_cutedsl_mm_op.register_fake
def _(a, w, out, bias=None, rscale=None):
    """
    Nothing to describe: the op only mutates ``out``.
    """
    return None


def cutedsl_mm(a, w, out, bias=None, rscale=None):
    """
    Compute ``out = (a @ w.t()) * rscale[:, None] + bias`` into an fp32 result.

    Args:
        a: fp16 ``(M, K)``, contiguous in K; may be a strided row-slice.
        w: fp16 ``(N, K)``, contiguous.
        out: fp32 ``(M, N)``, contiguous in N; may be a column block of a wider
            buffer. Mutated.
        bias: Optional fp32 ``(N,)``, added last.
        rscale: Optional fp32 ``(M,)`` per-row scale, applied before the bias.

    Returns:
        ``out``.
    """
    torch.ops.uma_nitro_gemm.mm(a, w, out, bias, rscale)
    return out


def _launch_tile(a, w, out, bias, rscale, tile):
    """
    Launch at an explicit CTA tile, for tile sweeps that bypass ``_tile_for``.
    """
    return _launch(a, w, out, bias, rscale, tile)


def _launch(a, w, out, bias=None, rscale=None, tile=None):
    """
    Compile-cache lookup and kernel launch, on real tensors only.
    """
    m, k = a.shape
    n = w.shape[0]
    c_half = out.dtype == torch.float16
    compiled = _cdg_compiled(
        m,
        n,
        k,
        a.stride(0),
        out.stride(0),
        tile or _tile_for(n, k),
        bias is not None,
        rscale is not None,
        c_half,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        *_pack(
            _as_cute(a, cutlass.Float16),
            _as_cute(w, cutlass.Float16),
            _as_cute(out, cutlass.Float16 if c_half else cutlass.Float32),
            stream,
            None if bias is None else _as_cute(bias, cutlass.Float32, False),
            None if rscale is None else _as_cute(rscale, cutlass.Float32, False),
        )
    )


def fp16_mm_like(
    a,
    w,
    out=None,
    a_row_stride=None,
    bias=None,
    out_row_stride=None,
    out_col_base=0,
    rscale=None,
):
    """
    ``fp16_mm``-signature shim so a variant can swap the GEMM with one import.

    ``cutedsl_mm`` takes the destination as a *view* rather than a
    stride/column-base pair, so the column block is materialized here. ``a`` is
    already a strided view at every call site, so its own ``stride(0)`` carries
    what ``a_row_stride`` passed explicitly; a mismatch means the caller built the
    view differently and is worth failing on.

    Args:
        a: fp16 activations ``(M, K)``.
        w: fp16 weight ``(N, K)``.
        out: Optional fp32 destination.
        a_row_stride: Checked against ``a.stride(0)`` if given.
        bias: Optional fp32 ``(N,)``.
        out_row_stride: Row stride of the destination buffer.
        out_col_base: Column offset within the destination.
        rscale: Optional fp32 ``(M,)`` per-row scale.

    Returns:
        The fp32 result, or the column-block view that was written.
    """
    n = w.shape[0]
    if a_row_stride is not None and a.stride(0) != a_row_stride:
        raise AssertionError(
            f"a.stride(0)={a.stride(0)} != a_row_stride={a_row_stride}; the "
            "caller's view does not carry the intended stride"
        )
    if out is None:
        view = torch.empty(a.shape[0], n, device=a.device, dtype=torch.float32)
    elif out_row_stride is not None or out_col_base:
        view = out[:, out_col_base : out_col_base + n]
    else:
        view = out
    cutedsl_mm(a, w, view, bias=bias, rscale=rscale)
    return view


# ============================================================================
# Remaining `import ... as` aliases from the original module boundaries.
# ============================================================================
fp16_mm = fp16_mm_like


def _red_v4_ok(tile_width, num_warps):
    """
    Whether ``_red_add_v4`` is legal for this tile shape and launch config.

    Args:
        tile_width: Elements in the tile handed to the atomic.
        num_warps: The kernel's ``num_warps``.

    Returns:
        True when every thread owns a whole multiple of four contiguous elements.
    """
    per_thread, rem = divmod(tile_width, 32 * num_warps)
    return _SUPPORTS_VECTOR_ATOMICS and rem == 0 and per_thread % 4 == 0


@triton.jit
def _red_add_v4(ptr, val):
    """
    Add an fp32 tile with reduction-only vector atomics.

    The tile and pointer must support aligned four-element groups. Callers use
    _red_v4_ok to enforce the launch constraints.
    """
    return tl.inline_asm_elementwise(
        "red.global.add.v4.f32 [$4], {$8, $9, $10, $11};",
        "=r,=r,=r,=r,l,l,l,l,f,f,f,f",
        [ptr.to(tl.uint64, bitcast=False), val],
        dtype=tl.int32,
        is_pure=False,
        pack=4,
    )


@triton.jit
def wigner_conv1_fused_bwd_fp16_kernel(
    gm0_ptr,  # [E, 3*2C] grad wrt m0 buffer
    gm1_ptr,  # [E, 4*2C]
    gm2_ptr,  # [E, 2*2C]
    wigner_ptr,  # [E, 81]
    radial_ptr,  # [E, 6C] conv1 radial
    x_ptr,  # [N, 9, C] node feats (re-gathered for x recompute)
    edge_index_ptr,  # [2, E]
    ge_src_ptr,  # [E, 9, C] out: per-edge grad wrt x, source half (for the scatter)
    ge_tgt_ptr,  # [E, 9, C] out: per-edge grad wrt x, target half (for the reduce)
    gx_ptr,  # [N, 9, C] out, SCATTER only: node grad_x, accumulated with atomics
    gwig_ptr,  # [E, 81] out: grad wrt wigner (block-diagonal)
    grad_rad_ptr,  # [E, 6C] out: grad wrt radial
    scale_ptr,  # [E] out: per-edge power-of-two scale applied to grad_edge
    rad_scale_ptr,  # [E] out: the same, for grad_rad, when RAD_FP16
    dmu_ptr,  # [3, E] in: per-block unscale for gm0/gm1/gm2, when DM_FP16
    dmu_stride,
    num_edges,
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
    DM_ABL: tl.constexpr = 0,
    RAD_ABL: tl.constexpr = 0,
    RAD_FP16: tl.constexpr = 0,
    DM_FP16: tl.constexpr = 0,
    MTREE: tl.constexpr = 1,
    SCATTER: tl.constexpr = 0,
    RED: tl.constexpr = 0,
    X_ABL: tl.constexpr = 0,
    EV_S: tl.constexpr = "",
    EV_R: tl.constexpr = "",
):
    """
    Fused conv1+wigner backward. For each edge:
      1. re-gather L-major x (src & tgt) from x_full + edge_index
      2. recompute rotated y-values from x + wigner
      3. grad_radial = sum over reuse of (grad_m * y)
      4. grad wrt rotated y (M-major) g_y = grad_m * radial ; permute M->L to g_l
      5. grad_x = W^T @ g_l  (block-diagonal) -> per-edge L-major buffer
      6. grad_W = g_l @ x_l^T  (block-diagonal outer product)
    """
    edge_id = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    C2 = 2 * C
    m0_row = 3 * C2
    m1_row = 4 * C2
    m2_row = 2 * C2
    RAD = 6 * C2  # 1536 for C=128

    while edge_id < num_edges:
        w_base = edge_id * 81

        # ---- re-gather L-major x (src & tgt) from x_full + edge_index ----
        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)
        _rx0 = edge_id % 64 if X_ABL == 2 else idx0
        _rx1 = (edge_id + 1) % 64 if X_ABL == 2 else idx1
        s_base = _rx0 * x_stride_n + c * x_stride_c
        t_base = _rx1 * x_stride_n + c * x_stride_c
        xs = (
            tl.load(
                x_ptr + s_base + 0 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 1 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 2 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 3 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 4 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 5 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 6 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 7 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + s_base + 8 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
        )
        xt = (
            tl.load(
                x_ptr + t_base + 0 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 1 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 2 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 3 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 4 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 5 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 6 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 7 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                x_ptr + t_base + 8 * x_stride_m,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
        )

        # ---- load wigner block-diagonal (once; reused by y recompute + dx) ----
        # kept explicit so the y-recompute and grad_x below match the original
        # kernel's FMA contraction bit-for-bit (grad_x is 1-ULP sensitive)
        w00 = tl.load(wigner_ptr + w_base + 0)
        w11 = tl.load(wigner_ptr + w_base + 1 * 9 + 1)
        w12 = tl.load(wigner_ptr + w_base + 1 * 9 + 2)
        w13 = tl.load(wigner_ptr + w_base + 1 * 9 + 3)
        w21 = tl.load(wigner_ptr + w_base + 2 * 9 + 1)
        w22 = tl.load(wigner_ptr + w_base + 2 * 9 + 2)
        w23 = tl.load(wigner_ptr + w_base + 2 * 9 + 3)
        w31 = tl.load(wigner_ptr + w_base + 3 * 9 + 1)
        w32 = tl.load(wigner_ptr + w_base + 3 * 9 + 2)
        w33 = tl.load(wigner_ptr + w_base + 3 * 9 + 3)
        w44 = tl.load(wigner_ptr + w_base + 4 * 9 + 4)
        w45 = tl.load(wigner_ptr + w_base + 4 * 9 + 5)
        w46 = tl.load(wigner_ptr + w_base + 4 * 9 + 6)
        w47 = tl.load(wigner_ptr + w_base + 4 * 9 + 7)
        w48 = tl.load(wigner_ptr + w_base + 4 * 9 + 8)
        w54 = tl.load(wigner_ptr + w_base + 5 * 9 + 4)
        w55 = tl.load(wigner_ptr + w_base + 5 * 9 + 5)
        w56 = tl.load(wigner_ptr + w_base + 5 * 9 + 6)
        w57 = tl.load(wigner_ptr + w_base + 5 * 9 + 7)
        w58 = tl.load(wigner_ptr + w_base + 5 * 9 + 8)
        w64 = tl.load(wigner_ptr + w_base + 6 * 9 + 4)
        w65 = tl.load(wigner_ptr + w_base + 6 * 9 + 5)
        w66 = tl.load(wigner_ptr + w_base + 6 * 9 + 6)
        w67 = tl.load(wigner_ptr + w_base + 6 * 9 + 7)
        w68 = tl.load(wigner_ptr + w_base + 6 * 9 + 8)
        w74 = tl.load(wigner_ptr + w_base + 7 * 9 + 4)
        w75 = tl.load(wigner_ptr + w_base + 7 * 9 + 5)
        w76 = tl.load(wigner_ptr + w_base + 7 * 9 + 6)
        w77 = tl.load(wigner_ptr + w_base + 7 * 9 + 7)
        w78 = tl.load(wigner_ptr + w_base + 7 * 9 + 8)
        w84 = tl.load(wigner_ptr + w_base + 8 * 9 + 4)
        w85 = tl.load(wigner_ptr + w_base + 8 * 9 + 5)
        w86 = tl.load(wigner_ptr + w_base + 8 * 9 + 6)
        w87 = tl.load(wigner_ptr + w_base + 8 * 9 + 7)
        w88 = tl.load(wigner_ptr + w_base + 8 * 9 + 8)

        # ---- recompute rotated y (L-major) from x + wigner ----
        x0s, x1s, x2s, x3s, x4s, x5s, x6s, x7s, x8s = xs
        x0t, x1t, x2t, x3t, x4t, x5t, x6t, x7t, x8t = xt
        y0s = w00 * x0s
        y0t = w00 * x0t
        y1s = w11 * x1s + w12 * x2s + w13 * x3s
        y2s = w21 * x1s + w22 * x2s + w23 * x3s
        y3s = w31 * x1s + w32 * x2s + w33 * x3s
        y1t = w11 * x1t + w12 * x2t + w13 * x3t
        y2t = w21 * x1t + w22 * x2t + w23 * x3t
        y3t = w31 * x1t + w32 * x2t + w33 * x3t
        y4s = w44 * x4s + w45 * x5s + w46 * x6s + w47 * x7s + w48 * x8s
        y5s = w54 * x4s + w55 * x5s + w56 * x6s + w57 * x7s + w58 * x8s
        y6s = w64 * x4s + w65 * x5s + w66 * x6s + w67 * x7s + w68 * x8s
        y7s = w74 * x4s + w75 * x5s + w76 * x6s + w77 * x7s + w78 * x8s
        y8s = w84 * x4s + w85 * x5s + w86 * x6s + w87 * x7s + w88 * x8s
        y4t = w44 * x4t + w45 * x5t + w46 * x6t + w47 * x7t + w48 * x8t
        y5t = w54 * x4t + w55 * x5t + w56 * x6t + w57 * x7t + w58 * x8t
        y6t = w64 * x4t + w65 * x5t + w66 * x6t + w67 * x7t + w68 * x8t
        y7t = w74 * x4t + w75 * x5t + w76 * x6t + w77 * x7t + w78 * x8t
        y8t = w84 * x4t + w85 * x5t + w86 * x6t + w87 * x7t + w88 * x8t

        rbb = (
            0
            if RAD_ABL == 1
            else ((edge_id % 64) * RAD if RAD_ABL == 2 else edge_id * RAD)
        )
        r0s = tl.load(
            radial_ptr + rbb + 0 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        r0t = tl.load(
            radial_ptr + rbb + 0 * C2 + C + c,
            mask=c_mask,
            other=0.0,
            eviction_policy=EV_S,
        )
        r1s = tl.load(
            radial_ptr + rbb + 1 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        r1t = tl.load(
            radial_ptr + rbb + 1 * C2 + C + c,
            mask=c_mask,
            other=0.0,
            eviction_policy=EV_S,
        )
        r2s = tl.load(
            radial_ptr + rbb + 2 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        r2t = tl.load(
            radial_ptr + rbb + 2 * C2 + C + c,
            mask=c_mask,
            other=0.0,
            eviction_policy=EV_S,
        )
        ra_s = tl.load(
            radial_ptr + rbb + 3 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        ra_t = tl.load(
            radial_ptr + rbb + 3 * C2 + C + c,
            mask=c_mask,
            other=0.0,
            eviction_policy=EV_S,
        )
        rb_s = tl.load(
            radial_ptr + rbb + 4 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        rb_t = tl.load(
            radial_ptr + rbb + 4 * C2 + C + c,
            mask=c_mask,
            other=0.0,
            eviction_policy=EV_S,
        )
        rc_s = tl.load(
            radial_ptr + rbb + 5 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        rc_t = tl.load(
            radial_ptr + rbb + 5 * C2 + C + c,
            mask=c_mask,
            other=0.0,
            eviction_policy=EV_S,
        )

        # ---- load grad_m (M-major packed) ----
        # DM_ABL pins every edge to row 0, so dm0/dm1/dm2 stop being a DRAM stream
        # and become an L2-resident broadcast. Wrong numbers on purpose: this is the
        # ceiling for any fusion that stops materializing them.
        _ea = edge_id % 64 if DM_ABL == 2 else 0
        mb = (edge_id if not DM_ABL else _ea) * m0_row
        g_m0s = tl.load(
            gm0_ptr + mb + 0 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m0t = tl.load(
            gm0_ptr + mb + 0 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m1s = tl.load(
            gm0_ptr + mb + 1 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m1t = tl.load(
            gm0_ptr + mb + 1 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m2s = tl.load(
            gm0_ptr + mb + 2 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m2t = tl.load(
            gm0_ptr + mb + 2 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        m1b = (edge_id if not DM_ABL else _ea) * m1_row
        g_m3s = tl.load(
            gm1_ptr + m1b + 0 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m3t = tl.load(
            gm1_ptr + m1b + 0 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m4s = tl.load(
            gm1_ptr + m1b + 1 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m4t = tl.load(
            gm1_ptr + m1b + 1 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m5s = tl.load(
            gm1_ptr + m1b + 2 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m5t = tl.load(
            gm1_ptr + m1b + 2 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m6s = tl.load(
            gm1_ptr + m1b + 3 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m6t = tl.load(
            gm1_ptr + m1b + 3 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        m2b = (edge_id if not DM_ABL else _ea) * m2_row
        g_m7s = tl.load(
            gm2_ptr + m2b + 0 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m7t = tl.load(
            gm2_ptr + m2b + 0 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m8s = tl.load(
            gm2_ptr + m2b + 1 * C2 + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )
        g_m8t = tl.load(
            gm2_ptr + m2b + 1 * C2 + C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
        )

        if DM_FP16:
            # One exact power of two per GEMM block, emitted by the kernel that
            # wrote this GEMM's A operand. Applied on load so every expression
            # below sees the fp32 value it saw before, up to the fp16 rounding:
            # both this factor and the scale the epilogue applied are powers of
            # two, so the pair is lossless.
            u0 = tl.load(dmu_ptr + 0 * dmu_stride + edge_id)
            u1 = tl.load(dmu_ptr + 1 * dmu_stride + edge_id)
            u2 = tl.load(dmu_ptr + 2 * dmu_stride + edge_id)
            g_m0s = g_m0s.to(tl.float32) * u0
            g_m0t = g_m0t.to(tl.float32) * u0
            g_m1s = g_m1s.to(tl.float32) * u0
            g_m1t = g_m1t.to(tl.float32) * u0
            g_m2s = g_m2s.to(tl.float32) * u0
            g_m2t = g_m2t.to(tl.float32) * u0
            g_m3s = g_m3s.to(tl.float32) * u1
            g_m3t = g_m3t.to(tl.float32) * u1
            g_m4s = g_m4s.to(tl.float32) * u1
            g_m4t = g_m4t.to(tl.float32) * u1
            g_m5s = g_m5s.to(tl.float32) * u1
            g_m5t = g_m5t.to(tl.float32) * u1
            g_m6s = g_m6s.to(tl.float32) * u1
            g_m6t = g_m6t.to(tl.float32) * u1
            g_m7s = g_m7s.to(tl.float32) * u2
            g_m7t = g_m7t.to(tl.float32) * u2
            g_m8s = g_m8s.to(tl.float32) * u2
            g_m8t = g_m8t.to(tl.float32) * u2

        # ---- grad_radial = sum over reuse of grad_m * y (y in M order) ----
        # M0<-y0 (r0), M1<-y2 (r1), M2<-y6 (r2); blk a (r3): M3<-y3,M5<-y1;
        # blk b (r4): M4<-y7,M6<-y5; blk c (r5): M7<-y8,M8<-y4
        gr_s = (
            g_m0s * y0s,
            g_m1s * y2s,
            g_m2s * y6s,
            g_m3s * y3s + g_m5s * y1s,
            g_m4s * y7s + g_m6s * y5s,
            g_m7s * y8s + g_m8s * y4s,
        )
        gr_t = (
            g_m0t * y0t,
            g_m1t * y2t,
            g_m2t * y6t,
            g_m3t * y3t + g_m5t * y1t,
            g_m4t * y7t + g_m6t * y5t,
            g_m7t * y8t + g_m8t * y4t,
        )
        if RAD_FP16:
            if MTREE:
                rv = tl.abs(tl.where(c_mask, gr_s[0], 0.0))
                rv = tl.maximum(rv, tl.abs(tl.where(c_mask, gr_t[0], 0.0)))
                for r in tl.static_range(1, 6):
                    rv = tl.maximum(rv, tl.abs(tl.where(c_mask, gr_s[r], 0.0)))
                    rv = tl.maximum(rv, tl.abs(tl.where(c_mask, gr_t[r], 0.0)))
                rmx = tl.maximum(tl.max(rv), 0.0)
            else:
                rmx = 0.0
                for r in tl.static_range(6):
                    rmx = tl.maximum(
                        rmx, tl.max(tl.abs(tl.where(c_mask, gr_s[r], 0.0)))
                    )
                    rmx = tl.maximum(
                        rmx, tl.max(tl.abs(tl.where(c_mask, gr_t[r], 0.0)))
                    )
            rsc = tl.exp2(tl.floor(tl.log2(FP16_SCALE_TARGET / tl.maximum(rmx, 1e-30))))
            tl.store(rad_scale_ptr + edge_id, 1.0 / rsc)
            for r in tl.static_range(6):
                tl.store(
                    grad_rad_ptr + rbb + r * C2 + c,
                    (gr_s[r] * rsc).to(tl.float16),
                    mask=c_mask,
                )
                tl.store(
                    grad_rad_ptr + rbb + r * C2 + C + c,
                    (gr_t[r] * rsc).to(tl.float16),
                    mask=c_mask,
                )
        else:
            for r in tl.static_range(6):
                tl.store(grad_rad_ptr + rbb + r * C2 + c, gr_s[r], mask=c_mask)
                tl.store(grad_rad_ptr + rbb + r * C2 + C + c, gr_t[r], mask=c_mask)

        # ---- grad wrt rotated y (put back into L order) g_y = grad_m * radial ----
        gys = (
            g_m0s * r0s,  # L0 (M0)
            g_m5s * ra_s,  # L1 (M5)
            g_m1s * r1s,  # L2 (M1)
            g_m3s * ra_s,  # L3 (M3)
            g_m8s * rc_s,  # L4 (M8)
            g_m6s * rb_s,  # L5 (M6)
            g_m2s * r2s,  # L6 (M2)
            g_m4s * rb_s,  # L7 (M4)
            g_m7s * rc_s,  # L8 (M7)
        )
        gyt = (
            g_m0t * r0t,
            g_m5t * ra_t,
            g_m1t * r1t,
            g_m3t * ra_t,
            g_m8t * rc_t,
            g_m6t * rb_t,
            g_m2t * r2t,
            g_m4t * rb_t,
            g_m7t * rc_t,
        )

        # ---- grad_x = W^T @ g_l (block-diagonal); g_l is L-major grad wrt y ----
        # explicit expressions reusing the single wigner load above; matches the
        # original bit-for-bit (grad_x FMA contraction is 1-ULP sensitive)
        gy0s, gy1s, gy2s, gy3s, gy4s, gy5s, gy6s, gy7s, gy8s = gys
        gy0t, gy1t, gy2t, gy3t, gy4t, gy5t, gy6t, gy7t, gy8t = gyt
        dxs = (
            w00 * gy0s,
            w11 * gy1s + w21 * gy2s + w31 * gy3s,
            w12 * gy1s + w22 * gy2s + w32 * gy3s,
            w13 * gy1s + w23 * gy2s + w33 * gy3s,
            w44 * gy4s + w54 * gy5s + w64 * gy6s + w74 * gy7s + w84 * gy8s,
            w45 * gy4s + w55 * gy5s + w65 * gy6s + w75 * gy7s + w85 * gy8s,
            w46 * gy4s + w56 * gy5s + w66 * gy6s + w76 * gy7s + w86 * gy8s,
            w47 * gy4s + w57 * gy5s + w67 * gy6s + w77 * gy7s + w87 * gy8s,
            w48 * gy4s + w58 * gy5s + w68 * gy6s + w78 * gy7s + w88 * gy8s,
        )
        dxt = (
            w00 * gy0t,
            w11 * gy1t + w21 * gy2t + w31 * gy3t,
            w12 * gy1t + w22 * gy2t + w32 * gy3t,
            w13 * gy1t + w23 * gy2t + w33 * gy3t,
            w44 * gy4t + w54 * gy5t + w64 * gy6t + w74 * gy7t + w84 * gy8t,
            w45 * gy4t + w55 * gy5t + w65 * gy6t + w75 * gy7t + w85 * gy8t,
            w46 * gy4t + w56 * gy5t + w66 * gy6t + w76 * gy7t + w86 * gy8t,
            w47 * gy4t + w57 * gy5t + w67 * gy6t + w77 * gy7t + w87 * gy8t,
            w48 * gy4t + w58 * gy5t + w68 * gy6t + w78 * gy7t + w88 * gy8t,
        )

        gb = edge_id * 9 * C
        if SCATTER & 1:
            sxb = idx0 * (9 * C) + c
            for i in tl.static_range(9):
                if RED:
                    _red_add_v4(gx_ptr + sxb + i * C, dxs[i])
                else:
                    tl.atomic_add(gx_ptr + sxb + i * C, dxs[i], mask=c_mask)
        if SCATTER & 2:
            txb = idx1 * (9 * C) + c
            for i in tl.static_range(9):
                if RED:
                    _red_add_v4(gx_ptr + txb + i * C, dxt[i])
                else:
                    tl.atomic_add(gx_ptr + txb + i * C, dxt[i], mask=c_mask)
        # Eighteen cross-lane reductions collapsed to one; see the grad_rad
        # scale above for why this is bit-identical and why it is worth doing.
        # Only the halves that still go through the fp16 buffer need the scale.
        if SCATTER != 3:
            if MTREE:
                dv = tl.zeros([BLOCK_C], dtype=tl.float32)
                for i in tl.static_range(9):
                    if not (SCATTER & 1):
                        dv = tl.maximum(dv, tl.abs(tl.where(c_mask, dxs[i], 0.0)))
                    if not (SCATTER & 2):
                        dv = tl.maximum(dv, tl.abs(tl.where(c_mask, dxt[i], 0.0)))
                mx = tl.maximum(tl.max(dv), 0.0)
            else:
                mx = 0.0
                for i in tl.static_range(9):
                    if not (SCATTER & 1):
                        mx = tl.maximum(
                            mx, tl.max(tl.abs(tl.where(c_mask, dxs[i], 0.0)))
                        )
                    if not (SCATTER & 2):
                        mx = tl.maximum(
                            mx, tl.max(tl.abs(tl.where(c_mask, dxt[i], 0.0)))
                        )
            sc = tl.exp2(tl.floor(tl.log2(FP16_SCALE_TARGET / tl.maximum(mx, 1e-30))))
            tl.store(scale_ptr + edge_id, 1.0 / sc)
            for i in tl.static_range(9):
                if not (SCATTER & 1):
                    tl.store(
                        ge_src_ptr + gb + i * C + c,
                        (dxs[i] * sc).to(tl.float16),
                        mask=c_mask,
                    )
                if not (SCATTER & 2):
                    tl.store(
                        ge_tgt_ptr + gb + i * C + c,
                        (dxt[i] * sc).to(tl.float16),
                        mask=c_mask,
                    )

        _p = tl.arange(0, 128)
        _prow = _p // 9
        _pcol = _p % 9
        _inb = (
            ((_prow == 0) & (_pcol == 0))
            | ((_prow >= 1) & (_prow <= 3) & (_pcol >= 1) & (_pcol <= 3))
            | ((_prow >= 4) & (_pcol >= 4))
        )
        tl.store(
            gwig_ptr + w_base + _p,
            tl.zeros((128,), dtype=tl.float32),
            mask=(_p < 81) & (_inb == 0),
        )
        _wig_dw_store2(gwig_ptr, w_base, gys, xs, gyt, xt)

        edge_id += GRID_E_STRIDE


@triton_op(
    "uma_nitro::conv1_bwd_fp16",
    mutates_args={"ge_src", "ge_tgt", "gx", "gwig", "grad_rad", "scale", "rad_scale"},
)
def conv1_bwd_fp16(
    gm0: torch.Tensor,
    gm1: torch.Tensor,
    gm2: torch.Tensor,
    wigner_flat: torch.Tensor,
    radial: torch.Tensor,
    x_full: torch.Tensor,
    edge_index: torch.Tensor,
    ge_src: torch.Tensor,
    ge_tgt: torch.Tensor,
    gx: torch.Tensor,
    gwig: torch.Tensor,
    grad_rad: torch.Tensor,
    scale: torch.Tensor,
    rad_scale: torch.Tensor,
    dm_unscale: torch.Tensor,
    channels: int,
) -> None:
    """
    Launch the fused conv1 backward with a scaled-fp16 edge gradient.
    """
    edges = edge_index.shape[1]
    wrap_triton(wigner_conv1_fused_bwd_fp16_kernel)[(edges,)](
        gm0,
        gm1,
        gm2,
        wigner_flat,
        radial,
        x_full,
        edge_index,
        ge_src,
        ge_tgt,
        gx,
        gwig,
        grad_rad,
        scale,
        rad_scale,
        dm_unscale,
        dm_unscale.stride(0),
        edges,
        x_full.stride(0),
        x_full.stride(1),
        x_full.stride(2),
        edge_index.stride(0),
        C=channels,
        BLOCK_C=channels,
        GRID_E_STRIDE=edges,
        num_warps=C1B_WARPS,
        num_stages=C1B_STAGES,
        X_ABL=2 if "xbwd64" in _ABLATIONS else 0,
        EV_S=_EV_S,
        EV_R=_ev_r(edges),
        DM_ABL=(
            2
            if "dmread64" in _ABLATIONS
            else 1
            if ("dmread" in _ABLATIONS or "dmgemm" in _ABLATIONS)
            else 0
        ),
        RAD_ABL=(
            1 if "radread" in _ABLATIONS else 2 if "radread64" in _ABLATIONS else 0
        ),
        RAD_FP16=1 if grad_rad.dtype == torch.float16 else 0,
        DM_FP16=1 if gm0.dtype == torch.float16 else 0,
        MTREE=C1B_MTREE,
        SCATTER=_scatter_mode() if gx.numel() > 1 else 0,
        # The inline-PTX reduction-only atomic needs a 4-aligned, unmasked tile
        # *and* four contiguous elements per thread, which is the third
        # precondition `_red_v4_ok` exists for and this site was not checking:
        # `channels % 4` alone passes at C1B_WARPS=2 (128 // 64 = 2, not a
        # multiple of 4), where the failure is a deferred
        # `CUDA error: misaligned address` surfacing in whatever kernel runs
        # next. This is the most expensive kernel in the trace and therefore
        # the first thing a num_warps sweep touches.
        RED=1 if (C1B_RED and _red_v4_ok(channels, C1B_WARPS)) else 0,
    )


GRID_E = 2048
IC2_WARPS = 1


@triton.jit
def wigner_inv_conv2_fused_bwd_fp16_kernel(
    dy_ptr,  # [E, 9, C] grad wrt x_rotated (L-major), or [N, 9, C] when GMODE
    tgt_ptr,  # [E] int32 target node per edge; only read when GMODE == 1
    g0_ptr,  # [E, 3*C] saved conv2 GEMM buffers (for dW recompute of x_l)
    g1_ptr,  # [E, 4*C]
    g2_ptr,  # [E, 2*C]
    W_ptr,  # [E, 81] inverse-wigner
    dg0_ptr,  # [E, 3*C] out: grad wrt g0
    dg1_ptr,  # [E, 4*C] out: grad wrt g1
    dg2_ptr,  # [E, 2*C] out: grad wrt g2
    dw_ptr,  # [E, 81] out: grad wrt wigner (block-diagonal)
    scale_ptr,  # [1] in: scalar power-of-two scale applied to dg0/dg1/dg2
    rscale_ptr,  # [E] out: what the *consumer* multiplies d0/d1/d2 by
    dbeta_ptr,  # [E] out: the GEMM epilogue's per-edge power of two, when DBROW
    CS_MAX,  # max_n sum_k |v[n, k]| over conv2's three backward weights
    num_edges,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,  # == C (all channels; needed for tl.sum over C in dW)
    GRID_E_STRIDE: tl.constexpr,
    GMODE: tl.constexpr = 0,
    G_ABL: tl.constexpr = 0,
    DBROW: tl.constexpr = 0,
    DBETA_CAP: tl.constexpr = 1.0,
    EV_S: tl.constexpr = "",
    EV_R: tl.constexpr = "",
):
    """
    Fused conv2-buffer + inv-wigner backward. For each edge:
      1. load dy (L-major) and W (block-diagonal)
      2. dx_l = W^T @ dy  (block-diagonal) -> scatter to g0/g1/g2 (L->M layout)
      3. recompute x_l from saved GEMM buffers (M->L)
      4. dW = dy @ x_l^T  (block-diagonal outer product)
    """
    edge_id = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    sc = tl.load(scale_ptr)

    g0_row = 3 * C
    g1_row = 4 * C
    g2_row = 2 * C

    # The three conv2 backward GEMMs undo the scale in their epilogue, which reads a
    # per-row (E,) fp32 vector -- a stride-0 expand() faults, because the epilogue loads
    # it as a vector fragment past the end of a 4-byte allocation. Filling it here is one
    # scalar store on a kernel that already visits every edge, and replaces
    # `dscale.reciprocal().expand(E).contiguous()`: two launches per layer, eight per
    # iteration, of which the reciprocal computed a single element.
    rsc = 1.0 / sc

    while edge_id < num_edges:
        w_base = edge_id * 81
        if GMODE == 1:
            dy_base = tl.load(tgt_ptr + edge_id).to(tl.int64) * (9 * C)
        elif GMODE == 2:
            dy_base = 0 * edge_id
        elif GMODE == 3:
            # As GMODE 2, but cycling 64 node rows so the addresses stay
            # edge-varying: the row-0 form camps on one L2 slice and reports the
            # wrong sign on a wide multi-load read.
            dy_base = (edge_id % 64) * (9 * C)
        else:
            dy_base = edge_id * 9 * C
        if not DBROW:
            tl.store(rscale_ptr + edge_id, rsc)

        # ---- load dy (L-major) ----
        dy = (
            tl.load(
                dy_ptr + dy_base + 0 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 1 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 2 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 3 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 4 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 5 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 6 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 7 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
            tl.load(
                dy_ptr + dy_base + 8 * C + c,
                mask=c_mask,
                other=0.0,
                eviction_policy=EV_R,
            ),
        )

        # ---- dx_l = W^T @ dy (block-diagonal) ----
        dx = _wig_rotT9(W_ptr, w_base, *dy)

        if DBROW:
            _a = tl.abs(dx[0])
            _a = tl.maximum(_a, tl.abs(dx[1]))
            _a = tl.maximum(_a, tl.abs(dx[2]))
            _a = tl.maximum(_a, tl.abs(dx[3]))
            _a = tl.maximum(_a, tl.abs(dx[4]))
            _a = tl.maximum(_a, tl.abs(dx[5]))
            _a = tl.maximum(_a, tl.abs(dx[6]))
            _a = tl.maximum(_a, tl.abs(dx[7]))
            _a = tl.maximum(_a, tl.abs(dx[8]))
            dgmax = tl.max(tl.where(c_mask, _a, 0.0)) * sc
            # 32768 rather than 65504: the target has to survive the rounding of
            # the division and of ``floor(log2(.))``, and half the range costs one
            # binade of a quantity with 1e7 of dynamic range to spare.
            bound = tl.maximum(dgmax * CS_MAX, 1e-30)
            beta = tl.minimum(DBETA_CAP, tl.exp2(tl.floor(tl.log2(32768.0 / bound))))
            tl.store(dbeta_ptr + edge_id, beta)
            # Both factors are exact powers of two, so the round trip is lossless.
            # ``rsc`` is at most 2^-4 and ``1 / beta`` at most 2^8 on any weight
            # this model can hold, so the product cannot overflow either.
            tl.store(rscale_ptr + edge_id, rsc / beta)

        # ---- scatter dx_l (L-major) into GEMM-buffer layout (L->M) ----
        g0b = edge_id * g0_row
        g1b = edge_id * g1_row
        g2b = edge_id * g2_row

        # L0->M0=g0[0], L2->M1=g0[1], L6->M2=g0[2]
        tl.store(dg0_ptr + g0b + 0 * C + c, dx[0] * sc, mask=c_mask)
        tl.store(dg0_ptr + g0b + 1 * C + c, dx[2] * sc, mask=c_mask)
        tl.store(dg0_ptr + g0b + 2 * C + c, dx[6] * sc, mask=c_mask)
        # L3->M3=g1[0], L7->M4=g1[1], L1->M5=g1[2], L5->M6=g1[3]
        tl.store(dg1_ptr + g1b + 0 * C + c, dx[3] * sc, mask=c_mask)
        tl.store(dg1_ptr + g1b + 1 * C + c, dx[7] * sc, mask=c_mask)
        tl.store(dg1_ptr + g1b + 2 * C + c, dx[1] * sc, mask=c_mask)
        tl.store(dg1_ptr + g1b + 3 * C + c, dx[5] * sc, mask=c_mask)
        # L8->M7=g2[0], L4->M8=g2[1]
        tl.store(dg2_ptr + g2b + 0 * C + c, dx[8] * sc, mask=c_mask)
        tl.store(dg2_ptr + g2b + 1 * C + c, dx[4] * sc, mask=c_mask)

        # ---- recompute x_l from saved GEMM buffers (M->L) ----
        # G_ABL: read g row 0 for every edge; the dg stores keep their real bases.
        _ga = edge_id % 64 if G_ABL == 2 else 0
        xg0b = g0b if not G_ABL else _ga * 3 * C
        xg1b = g1b if not G_ABL else _ga * 4 * C
        xg2b = g2b if not G_ABL else _ga * 2 * C
        x = (
            tl.load(
                g0_ptr + xg0b + 0 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L0<-M0
            tl.load(
                g1_ptr + xg1b + 2 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L1<-M5
            tl.load(
                g0_ptr + xg0b + 1 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L2<-M1
            tl.load(
                g1_ptr + xg1b + 0 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L3<-M3
            tl.load(
                g2_ptr + xg2b + 1 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L4<-M8
            tl.load(
                g1_ptr + xg1b + 3 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L5<-M6
            tl.load(
                g0_ptr + xg0b + 2 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L6<-M2
            tl.load(
                g1_ptr + xg1b + 1 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L7<-M4
            tl.load(
                g2_ptr + xg2b + 0 * C + c, mask=c_mask, other=0.0, eviction_policy=EV_S
            ),  # L8<-M7
        )

        _p = tl.arange(0, 128)
        _prow = _p // 9
        _pcol = _p % 9
        _inb = (
            ((_prow == 0) & (_pcol == 0))
            | ((_prow >= 1) & (_prow <= 3) & (_pcol >= 1) & (_pcol <= 3))
            | ((_prow >= 4) & (_pcol >= 4))
        )
        tl.store(
            dw_ptr + w_base + _p,
            tl.zeros((128,), dtype=tl.float32),
            mask=(_p < 81) & (_inb == 0),
        )
        _wig_dw_store1(dw_ptr, w_base, dy, x)

        edge_id += GRID_E_STRIDE


@triton_op(
    "uma_nitro::invrot_conv2_bwd_fp16",
    mutates_args={"dg0", "dg1", "dg2", "dw", "rscale", "dbeta"},
)
def _ic2_op(
    dy: torch.Tensor,
    tgt: torch.Tensor,
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    wigner: torch.Tensor,
    dg0: torch.Tensor,
    dg1: torch.Tensor,
    dg2: torch.Tensor,
    dw: torch.Tensor,
    scale: torch.Tensor,
    rscale: torch.Tensor,
    dbeta: torch.Tensor,
    cs_max: float,
    channels: int,
    grid: int = GRID_E,
    num_edges: int = 0,
    gmode: int = 0,
    dbrow: bool = False,
) -> None:
    """
    Launch the fp16 per-row-scaled inverse-Wigner backward.
    """
    wrap_triton(wigner_inv_conv2_fused_bwd_fp16_kernel)[(grid,)](
        dy,
        tgt,
        g0,
        g1,
        g2,
        wigner,
        dg0,
        dg1,
        dg2,
        dw,
        scale,
        rscale,
        dbeta,
        cs_max,
        num_edges or dy.shape[0],
        C=channels,
        BLOCK_C=triton.next_power_of_2(channels),
        GRID_E_STRIDE=grid,
        GMODE=gmode,
        G_ABL=2 if "gbwd64" in _ABLATIONS else 1 if "gbwd" in _ABLATIONS else 0,
        DBROW=1 if dbrow else 0,
        DBETA_CAP=D_BETA,
        EV_S=_EV_S,
        EV_R=_ev_r(num_edges or dy.shape[0]),
        num_warps=IC2_WARPS,
        num_stages=IC2_STAGES,
    )


IC2_GATHER = os.environ.get("UMA_NITRO_IC2_GATHER", "1") == "1"
IC2_ABL = os.environ.get("UMA_NITRO_IC2_ABL", "0") == "1"


def invrot_conv2_bwd_fp16(
    dy,
    g0,
    g1,
    g2,
    wigner,
    scale,
    channels,
    grid=GRID_E,
    tgt=None,
    edges=0,
    cs_max=0.0,
):
    """
    Inverse-Wigner backward returning fp16 dg plus a reciprocal per-row scale.

    Args:
        dy: Gathered node gradient, ``(E, 9, C)``.
        g0: Saved conv2 m=0 output.
        g1: Saved conv2 m=1 output.
        g2: Saved conv2 m=2 output.
        wigner: Inverse Wigner with the envelope folded in, ``(E, 81)``.
        scale: 1-element fp32 tensor, the power-of-two scale to apply.
        channels: ``C``.
        cs_max: ``max_n sum_k |v[n, k]|`` over the three conv2 backward weights.
            Nonzero turns on the per-edge ``d0/d1/d2`` scale.

    Returns:
        ``(dg0, dg1, dg2, dw, rscale, dbeta)``. ``dg*`` are fp16; ``rscale`` is
        what the *consumer* multiplies ``d*`` by and ``dbeta`` what the GEMM
        epilogue multiplies its accumulator by. Without ``cs_max`` the second is a
        1-element placeholder and the first is plain ``1 / scale``.
    """
    gmode = 2 if IC2_ABL else (1 if tgt is not None else 0)
    if gmode == 1 and "dybwd64" in _ABLATIONS:
        gmode = 3
    edges, dev = (edges or dy.shape[0]), dy.device
    dg0 = torch.empty((edges, 3 * channels), device=dev, dtype=torch.float16)
    dg1 = torch.empty((edges, 4 * channels), device=dev, dtype=torch.float16)
    dg2 = torch.empty((edges, 2 * channels), device=dev, dtype=torch.float16)
    # Not zeros: the kernel writes the off-block entries itself.
    dw = torch.empty((edges, 81), device=dev, dtype=torch.float32)
    if os.environ.get("UMA_NITRO_POISON_WIG") == "1":
        dw.fill_(float("nan"))
    rscale = torch.empty(edges, device=dev, dtype=torch.float32)
    dbrow = cs_max > 0.0
    dbeta = torch.empty(edges if dbrow else 1, device=dev, dtype=torch.float32)
    torch.ops.uma_nitro.invrot_conv2_bwd_fp16(
        dy.contiguous(),
        dy if tgt is None else tgt,
        g0,
        g1,
        g2,
        wigner,
        dg0,
        dg1,
        dg2,
        dw,
        scale,
        rscale,
        dbeta,
        cs_max,
        channels,
        grid,
        edges,
        gmode,
        dbrow,
    )
    return dg0, dg1, dg2, dw, rscale, dbeta


BLK_GRID_STRIDE = int(os.environ.get("UMA_NITRO_BLK_GRID", "32768"))
DEFAULT_BLOCK_E = 16


@triton.jit
def _gate_kernel(
    x_ptr,
    gating_ptr,
    expand_ptr,
    out_ptr,
    bias_ptr,
    num_edges,
    NUM_COEFFS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COEFF_BASE: tl.constexpr,
    GATING_STRIDE: tl.constexpr,
    GATING_BASE: tl.constexpr,
    BIAS_COEFFS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    GRID: tl.constexpr,
    X_ABL: tl.constexpr = 0,
    PRELOAD: tl.constexpr = 0,
    EV_S2: tl.constexpr = "",
    EV_R2: tl.constexpr = "",
    OUT_ROW_STRIDE: tl.constexpr = 0,
):
    """
    Apply the SO2 scalar activation and vector gates.

    The kernel may also apply the m=0 bias while the affected values are resident.
    """
    chan = tl.arange(0, CHANNELS)
    tile = tl.program_id(0)
    while tile * BLOCK_E < num_edges:
        edge = (tile * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
        alive = edge < num_edges
        row = edge[:, None] * ROW_STRIDE + COEFF_BASE + chan[None, :]
        # Gating may live in this same buffer -- conv1's m=0 GEMM emits it as a
        # 256-column prefix ahead of the m=0 coefficients -- or in a separate
        # tensor. The caller picks by passing the pointer and its stride/base.
        gate_row = edge[:, None] * GATING_STRIDE + GATING_BASE + chan[None, :]
        # X_ABL: read row 0 for every edge, write normally. Prices ``buf``'s read.
        rrow = row
        rgate = gate_row
        if X_ABL:
            _xa = (edge % 64 if X_ABL == 2 else edge * 0)[:, None]
            rrow = _xa * ROW_STRIDE + COEFF_BASE + chan[None, :]
            rgate = _xa * GATING_STRIDE + GATING_BASE + chan[None, :]

        # The two gate scalars are shared across all 8 vector coefficients, so
        # load and activate them once rather than per coefficient.
        r0 = tl.load(
            gating_ptr + rgate, mask=alive[:, None], other=0.0, eviction_policy=EV_S2
        )
        r1 = tl.load(
            gating_ptr + rgate + CHANNELS,
            mask=alive[:, None],
            other=0.0,
            eviction_policy=EV_S2,
        )
        if BIAS_COEFFS > 0:
            r0 += tl.load(bias_ptr + GATING_BASE + chan)[None, :]
            r1 += tl.load(bias_ptr + GATING_BASE + CHANNELS + chan)[None, :]
        g0 = tl.sigmoid(r0)
        g1 = tl.sigmoid(r1)

        if PRELOAD:
            tl.static_assert(NUM_COEFFS == 9, "PRELOAD unrolls nine coefficients")
            xv = (
                tl.load(
                    x_ptr + rrow + 0 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 1 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 2 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 3 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 4 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 5 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 6 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 7 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
                tl.load(
                    x_ptr + rrow + 8 * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ),
            )
            tl.debug_barrier()
            orow = edge[:, None] * OUT_ROW_STRIDE + COEFF_BASE + chan[None, :]
            for coeff in tl.static_range(NUM_COEFFS):
                xi = xv[coeff]
                if coeff < BIAS_COEFFS:
                    xi += tl.load(bias_ptr + COEFF_BASE + coeff * CHANNELS + chan)[
                        None, :
                    ]
                if coeff == 0:
                    v = xi * tl.sigmoid(xi)
                else:
                    which_l = tl.load(expand_ptr + (coeff - 1))
                    v = xi * tl.where(which_l == 0, g0, g1)
                tl.store(out_ptr + orow + coeff * CHANNELS, v, mask=alive[:, None])
        else:
            # coefficient 0 is the scalar: SiLU
            x0 = tl.load(
                x_ptr + rrow, mask=alive[:, None], other=0.0, eviction_policy=EV_S2
            )
            if BIAS_COEFFS > 0:
                x0 += tl.load(bias_ptr + COEFF_BASE + chan)[None, :]
            tl.store(out_ptr + row, x0 * tl.sigmoid(x0), mask=alive[:, None])

            for coeff in tl.static_range(1, NUM_COEFFS):
                which_l = tl.load(expand_ptr + (coeff - 1))
                gate = tl.where(which_l == 0, g0, g1)
                xi = tl.load(
                    x_ptr + rrow + coeff * CHANNELS, mask=alive[:, None], other=0.0
                )
                if coeff < BIAS_COEFFS:
                    xi += tl.load(bias_ptr + COEFF_BASE + coeff * CHANNELS + chan)[
                        None, :
                    ]
                tl.store(
                    out_ptr + row + coeff * CHANNELS,
                    xi * gate,
                    mask=alive[:, None],
                )
        tile += GRID


@triton_op("uma_nitro_block::gate", mutates_args={"out"})
def _gate_op(
    x: torch.Tensor,
    gating: torch.Tensor,
    expand: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor,
    num_coeffs: int,
    channels: int,
    row_stride: int,
    coeff_base: int,
    gating_stride: int,
    gating_base: int,
    bias_coeffs: int,
    block_e: int,
    out_row_stride: int = 0,
) -> None:
    """
    Launch the gate kernel.

    Args:
        x: Input buffer, ``(E, P * C)``.
        gating: Gating scalars, ``(E, num_l * C)``.
        expand: Coefficient-to-l map, length ``num_coeffs - 1``.
        out: Output buffer, ``(E, P * C)``; mutated.
        num_coeffs: Real coefficient count.
        channels: Channels per coefficient.
        coeff_stride: Coefficient slots per edge, i.e. ``P``.
        num_l: Number of gate scalar blocks.
        block_e: Edges per program.
    """
    _bg = min(x.shape[0], BLK_GRID_STRIDE)
    wrap_triton(_gate_kernel)[(_bg,)](
        x,
        gating,
        expand,
        out,
        bias,
        x.shape[0],
        NUM_COEFFS=num_coeffs,
        CHANNELS=channels,
        ROW_STRIDE=row_stride,
        COEFF_BASE=coeff_base,
        GATING_STRIDE=gating_stride,
        GATING_BASE=gating_base,
        BIAS_COEFFS=bias_coeffs,
        BLOCK_E=block_e,
        # GRID must equal the launch grid: the loop advances by GRID, so a GRID
        # larger than the program count would leave a tail of edges unvisited.
        PRELOAD=1 if (GATE_PRELOAD or GATE_PACK) else 0,
        EV_S2=_EV_S2,
        EV_R2=_EV_R2,
        OUT_ROW_STRIDE=out_row_stride or row_stride,
        GRID=_bg,
        num_warps=GATE_WARPS,
        num_stages=GATE_STAGES,
        X_ABL=2 if "bufgate64" in _ABLATIONS else 1 if "bufgate" in _ABLATIONS else 0,
    )


def gate(
    layout,
    x,
    gating=None,
    expand=None,
    block_e=DEFAULT_BLOCK_E,
    out=None,
    gating_base=0,
    bias=None,
):
    """
    Apply the gate over a block layout in one kernel launch.

    Args:
        layout: A ``BlockLayout``.
        x: Input buffer from ``layout.allocate``.
        gating: Gating scalars, ``(E, num_l * C)``.
        expand: Coefficient-to-l map.
        block_e: Edges per program.
        out: Optional output buffer; allocated if omitted.

    Returns:
        The output buffer.
    """
    if out is None:
        # In-place is safe: each coefficient is read once and written once, with
        # no cross-coefficient dependency, and the gating prefix is never written.
        out = x
    if gating is None:
        gating, gating_stride = x, layout.row_stride
    else:
        gating_stride = gating.shape[1]
    # A zero-element placeholder keeps the schema simple when there is no bias;
    # BIAS_COEFFS=0 means the kernel never dereferences it.
    bias_coeffs = 0 if bias is None else layout.block_sizes[0]
    if bias is None:
        bias = x
    torch.ops.uma_nitro_block.gate(
        x,
        gating,
        expand,
        out,
        bias,
        layout.num_coeffs,
        layout.channels,
        layout.row_stride,
        layout.coeff_base,
        gating_stride,
        gating_base,
        bias_coeffs,
        block_e,
        out.stride(0),
    )
    return out


@triton.jit
def _gate_bwd_split_fp16_kernel(
    x_ptr,
    y_ptr,
    d0_ptr,
    d1_ptr,
    d2_ptr,
    dbuf_ptr,
    scale_ptr,
    dyscale_ptr,
    expand_ptr,
    bias_ptr,
    dmsc_ptr,
    CS0,
    CS1,
    CS2,
    num_edges,
    NUM_COEFFS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COEFF_BASE: tl.constexpr,
    GATING_BASE: tl.constexpr,
    BIAS_COEFFS: tl.constexpr,
    OFF1: tl.constexpr,
    OFF2: tl.constexpr,
    TARGET_EXP: tl.constexpr,
    HAS_DYSCALE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    GRID: tl.constexpr,
    X_ABL: tl.constexpr = 0,
    DYBETA: tl.constexpr = 1.0,
    EV_S2: tl.constexpr = "",
    EV_R2: tl.constexpr = "",
    DM_SCALE: tl.constexpr = 0,
    DM_TARGET: tl.constexpr = 14.0,
    Y_ROW_STRIDE: tl.constexpr = 0,
    IDENT: tl.constexpr = 0,
    MAX_KROW: tl.constexpr = 113.0,
    DYROW: tl.constexpr = 0,
):
    """
    Compute the SO2 gate backward into scaled fp16 rows.

    Each row receives an exact power-of-two scale before storage.
    """
    chan = tl.arange(0, CHANNELS)
    # Undo the producer's scalar scale on load. Free: these values are being read
    # anyway, so no separate unscale pass over d0/d1/d2 is needed.
    dysc_g = tl.load(dyscale_ptr) * DYBETA if (HAS_DYSCALE and not DYROW) else 1.0
    tile = tl.program_id(0)
    while tile * BLOCK_E < num_edges:
        edge = (tile * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
        alive = edge < num_edges
        # ``DYROW`` is a constexpr, so exactly one of these exists in the compiled
        # kernel and ``dysc`` has one type. Naming the pre-loop scalar separately
        # matters: reassigning it inside the loop makes it loop-carried, and
        # Triton rejects the type change from ``fp32`` to ``[BLOCK_E, 1] fp32``.
        if DYROW:
            # The producer's factor is per edge, not global: ``d0/d1/d2`` carry an
            # exact power of two derived from that edge's own bound. One extra
            # ``(BLOCK_E,)`` load against a scalar one.
            dysc = tl.load(dyscale_ptr + edge, mask=alive, other=0.0)[:, None]
        else:
            dysc = dysc_g
        yrow = edge[:, None] * (Y_ROW_STRIDE or ROW_STRIDE) + COEFF_BASE + chan[None, :]
        row = edge[:, None] * ROW_STRIDE + COEFF_BASE + chan[None, :]
        gate_row = edge[:, None] * ROW_STRIDE + GATING_BASE + chan[None, :]
        # X_ABL: read `buf` at row 0 for every edge; writes are unchanged.
        rrow = row
        rgate = gate_row
        if X_ABL == 1 or X_ABL == 2:
            _xa = (edge % 64 if X_ABL == 2 else edge * 0)[:, None]
            rrow = _xa * ROW_STRIDE + COEFF_BASE + chan[None, :]
            rgate = _xa * ROW_STRIDE + GATING_BASE + chan[None, :]
        # X_ABL == 3 thins *only* the coefficient reads (coeff >= 1), leaving the
        # gating prefix and coefficient 0 exact. That is precisely what the
        # ``dgs = (1-s) * sum(dy * y)`` identity would remove from this kernel: it
        # needs the pre-activation of coefficient 0 (SiLU is not invertible) and the
        # gating scalars, but the eight gated coefficients only ever enter as
        # ``dyi * xi``, which the identity rewrites in terms of the *post*-gate
        # values this kernel's sibling already wrote to ``gbuf``.
        crow = rrow
        if X_ABL == 3:
            crow = (edge % 64)[:, None] * ROW_STRIDE + COEFF_BASE + chan[None, :]

        r0 = tl.load(
            x_ptr + rgate, mask=alive[:, None], other=0.0, eviction_policy=EV_S2
        )
        r1 = tl.load(
            x_ptr + rgate + CHANNELS,
            mask=alive[:, None],
            other=0.0,
            eviction_policy=EV_S2,
        )
        if BIAS_COEFFS > 0:
            r0 += tl.load(bias_ptr + GATING_BASE + chan)[None, :]
            r1 += tl.load(bias_ptr + GATING_BASE + CHANNELS + chan)[None, :]
        s0 = tl.sigmoid(r0)
        s1 = tl.sigmoid(r1)

        x0 = tl.load(
            x_ptr + rrow, mask=alive[:, None], other=0.0, eviction_policy=EV_S2
        )
        if BIAS_COEFFS > 0:
            x0 += tl.load(bias_ptr + COEFF_BASE + chan)[None, :]
        dy0 = (
            tl.load(
                d0_ptr + edge[:, None] * (OFF1 * CHANNELS) + chan[None, :],
                mask=alive[:, None],
                other=0.0,
                eviction_policy=EV_S2,
            )
            * dysc
        )
        sig0 = tl.sigmoid(x0)
        # tuple concatenation: Triton's frontend supports neither list.append
        # nor dict literals, but a tuple rebuilt each step traces fine.
        out = (dy0 * sig0 * (1.0 + x0 * (1.0 - sig0)),)

        acc0 = tl.zeros((BLOCK_E, CHANNELS), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_E, CHANNELS), dtype=tl.float32)
        for coeff in tl.static_range(1, NUM_COEFFS):
            which_l = tl.load(expand_ptr + (coeff - 1))
            gate = tl.where(which_l == 0, s0, s1)
            if coeff < OFF1:
                src = d0_ptr + edge[:, None] * (OFF1 * CHANNELS) + coeff * CHANNELS
            elif coeff < OFF2:
                src = (
                    d1_ptr
                    + edge[:, None] * ((OFF2 - OFF1) * CHANNELS)
                    + (coeff - OFF1) * CHANNELS
                )
            else:
                src = (
                    d2_ptr
                    + edge[:, None] * ((NUM_COEFFS - OFF2) * CHANNELS)
                    + (coeff - OFF2) * CHANNELS
                )
            if IDENT:
                xi = tl.load(
                    y_ptr + (yrow if X_ABL == 0 else crow) + coeff * CHANNELS,
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                ).to(tl.float32)
            else:
                xi = tl.load(
                    x_ptr + crow + coeff * CHANNELS, mask=alive[:, None], other=0.0
                )
                if coeff < BIAS_COEFFS:
                    xi += tl.load(bias_ptr + COEFF_BASE + coeff * CHANNELS + chan)[
                        None, :
                    ]
            dyi = (
                tl.load(
                    src + chan[None, :],
                    mask=alive[:, None],
                    other=0.0,
                    eviction_policy=EV_S2,
                )
                * dysc
            )
            prod = dyi * xi
            acc0 += tl.where(which_l == 0, prod, 0.0)
            acc1 += tl.where(which_l == 1, prod, 0.0)
            out = out + (dyi * gate,)

        if IDENT:
            dgs0 = acc0 * (1.0 - s0)
            dgs1 = acc1 * (1.0 - s1)
        else:
            dgs0 = acc0 * s0 * (1.0 - s0)
            dgs1 = acc1 * s1 * (1.0 - s1)

        # --- per-row max over the whole [gating | coefficients] row ---
        # Accumulated as three partials, one per *GEMM block*, then folded. The fold
        # is exact and order-free, so ``rmax`` -- and therefore ``dbuf`` and
        # ``scale`` -- is bit-identical to the single running maximum this replaced.
        # The partials are what conv1's backward GEMMs need to emit fp16: block ``b``
        # of this row is the whole A operand of GEMM ``b``, so
        # ``rmax_b * scale * colsum_b`` bounds that GEMM's accumulator row, and no
        # other kernel is in a position to know it.
        rmax0 = tl.max(tl.abs(dgs0), axis=1)
        rmax0 = tl.maximum(rmax0, tl.max(tl.abs(dgs1), axis=1))
        rmax1 = tl.zeros((BLOCK_E,), dtype=tl.float32)
        rmax2 = tl.zeros((BLOCK_E,), dtype=tl.float32)
        for coeff in tl.static_range(NUM_COEFFS):
            cmax = tl.max(tl.abs(out[coeff]), axis=1)
            if coeff < OFF1:
                rmax0 = tl.maximum(rmax0, cmax)
            elif coeff < OFF2:
                rmax1 = tl.maximum(rmax1, cmax)
            else:
                rmax2 = tl.maximum(rmax2, cmax)
        rmax = tl.maximum(rmax0, tl.maximum(rmax1, rmax2))
        # exact power of two: floor(log2) then exp2, so scale and unscale are lossless
        expo = tl.floor(tl.log2(tl.where(rmax > 0.0, rmax, 1.0)))
        krow = tl.minimum(tl.where(rmax > 0.0, TARGET_EXP - expo, 0.0), MAX_KROW)
        scale = tl.exp2(krow)
        tl.store(scale_ptr + edge, 1.0 / scale, mask=alive)

        if DM_SCALE:
            rb = (rmax0, rmax1, rmax2)
            cs = (CS0, CS1, CS2)
            for b in tl.static_range(3):
                bound = tl.maximum(rb[b] * scale * cs[b], 1e-30)
                kb = DM_TARGET - tl.floor(tl.log2(bound))
                tl.store(dmsc_ptr + b * num_edges + edge, tl.exp2(kb), mask=alive)
                tl.store(
                    dmsc_ptr + (3 + b) * num_edges + edge,
                    tl.exp2(-krow - kb),
                    mask=alive,
                )

        sc = scale[:, None]
        tl.store(dbuf_ptr + gate_row, dgs0 * sc, mask=alive[:, None])
        tl.store(dbuf_ptr + gate_row + CHANNELS, dgs1 * sc, mask=alive[:, None])
        for coeff in tl.static_range(NUM_COEFFS):
            tl.store(
                dbuf_ptr + row + coeff * CHANNELS, out[coeff] * sc, mask=alive[:, None]
            )
        tile += GRID


@triton_op(
    "uma_nitro_block::gate_bwd_split_fp16", mutates_args={"dbuf", "scale", "dmsc"}
)
def _gate_bwd_split_fp16_op(
    x: torch.Tensor,
    y: torch.Tensor,
    d0: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    dbuf: torch.Tensor,
    scale: torch.Tensor,
    dyscale: torch.Tensor,
    expand: torch.Tensor,
    bias: torch.Tensor,
    dmsc: torch.Tensor,
    colsums: list[float],
    dm_scale: bool,
    num_coeffs: int,
    channels: int,
    row_stride: int,
    coeff_base: int,
    gating_base: int,
    bias_coeffs: int,
    off1: int,
    off2: int,
    target_exp: int,
    has_dyscale: bool,
    block_e: int,
    dybeta: float = 1.0,
    y_row_stride: int = 0,
    dyrow: bool = False,
) -> None:
    """
    Launch the fp16 per-row-scaled gate backward.
    """
    _bg = min(x.shape[0], BLK_GRID_STRIDE)
    wrap_triton(_gate_bwd_split_fp16_kernel)[(_bg,)](
        x,
        y,
        d0,
        d1,
        d2,
        dbuf,
        scale,
        dyscale,
        expand,
        bias,
        dmsc,
        colsums[0],
        colsums[1],
        colsums[2],
        x.shape[0],
        NUM_COEFFS=num_coeffs,
        CHANNELS=channels,
        ROW_STRIDE=row_stride,
        COEFF_BASE=coeff_base,
        GATING_BASE=gating_base,
        BIAS_COEFFS=bias_coeffs,
        OFF1=off1,
        OFF2=off2,
        TARGET_EXP=target_exp,
        MAX_KROW=GATE_MAX_KROW,
        # Passed explicitly. ``DM_TARGET_EXP`` was a module global that nothing
        # read -- the live value was the kernel's own ``DM_TARGET`` default, so a
        # scale target had two sources of truth and editing the documented one did
        # nothing. Identical value, so bit-identical.
        DM_TARGET=DM_TARGET_EXP,
        HAS_DYSCALE=has_dyscale,
        BLOCK_E=block_e,
        # GRID must equal the launch grid: the loop advances by GRID, so a GRID
        # larger than the program count would leave a tail of edges unvisited.
        GRID=_bg,
        num_warps=GATE_WARPS,
        num_stages=GATE_STAGES,
        X_ABL=(
            3
            if "bufbwdc64" in _ABLATIONS
            else 2
            if "bufbwd64" in _ABLATIONS
            else 1
            if "bufbwd" in _ABLATIONS
            else 0
        ),
        DYBETA=dybeta,
        EV_S2=_EV_S2,
        EV_R2=_EV_R2,
        DM_SCALE=1 if dm_scale else 0,
        IDENT=1 if GATE_IDENT else 0,
        Y_ROW_STRIDE=y_row_stride,
        DYROW=1 if dyrow else 0,
    )


def gate_bwd_split_fp16(
    layout,
    x,
    d0,
    d1,
    d2,
    dbuf,
    scale,
    expand=None,
    bias=None,
    block_e=2,
    target_exp=13,
    dyscale=None,
    dybeta=1.0,
    dmsc=None,
    colsums=None,
    y=None,
    dyrow=False,
):
    """
    Gate backward writing fp16 with an exact per-row power-of-two scale.

    Args:
        layout: A ``BlockLayout``.
        x: Saved pre-gate buffer.
        y: Saved *post*-gate fp16 buffer, for the ``GATE_IDENT`` path. Only the
            eight gated coefficients are read from it; the gating scalars and
            coefficient 0's pre-activation still come from ``x``.
        d0: conv2 m=0 backward GEMM output.
        d1: conv2 m=1 backward GEMM output.
        d2: conv2 m=2 backward GEMM output.
        dbuf: fp16 destination in the full layout.
        scale: ``(E,)`` fp32, receives the reciprocal scale per row.
        expand: Coefficient-to-l map.
        bias: conv1 m=0 bias, matching what the forward applied.
        block_e: Edges per program; small because a whole row lives in registers.
        target_exp: Row max is scaled to about ``2 ** target_exp``.
        dyscale: Optional tensor; ``d0/d1/d2`` are multiplied by it on load,
            undoing a producer's scale without a separate pass. One element when
            ``dyrow`` is false, ``(E,)`` when it is true.
        dyrow: Read ``dyscale`` per edge rather than once. The conv2 backward
            derives its headroom factor from each edge's own bound, so the factor
            that has to be undone here is per edge too.
        dybeta: Extra scalar folded into ``dyscale``. Lets the producer apply a
            constant power-of-two headroom factor without needing its own
            per-row tensor for it.
        dmsc: Optional ``(6, E)`` fp32 destination for conv1's backward per-block
            scales; rows 0-2 are the GEMM epilogue's factor, rows 3-5 the
            consumer's. Mutated only when ``colsums`` is also given.
        colsums: ``max_n sum_k |w[n, k]|`` for the three backward weights.

    Returns:
        ``(dbuf, scale)``.
    """
    bias_coeffs = 0 if bias is None else layout.block_sizes[0]
    torch.ops.uma_nitro_block.gate_bwd_split_fp16(
        x,
        x if y is None else y,
        d0,
        d1,
        d2,
        dbuf,
        scale,
        scale if dyscale is None else dyscale,
        expand,
        x if bias is None else bias,
        _dm_dummy(x.device) if dmsc is None else dmsc,
        [0.0, 0.0, 0.0] if colsums is None else list(colsums),
        colsums is not None and dmsc is not None,
        layout.num_coeffs,
        layout.channels,
        layout.row_stride,
        layout.coeff_base,
        0,
        bias_coeffs,
        layout.offsets[1],
        layout.offsets[2],
        target_exp,
        dyscale is not None,
        block_e,
        dybeta,
        0 if y is None else y.stride(0),
        dyrow,
    )
    return dbuf, scale


_NO_PAD = -1

GATE_IDENT = os.environ.get("UMA_NITRO_GATE_IDENT", "1") == "1"

FP16_D = os.environ.get("UMA_NITRO_FP16_D", "1") == "1"

D_BETA = 2.0**-4
_DBETA_VEC = {}

DBETA_ROW = os.environ.get("UMA_NITRO_DBETA_ROW", "1") == "1"
_D_COLSUM = {}


def _d_colsum(v0, v1, v2):
    """
    ``max_n sum_k |v[n, k]|`` over conv2's three transposed backward weights.

    One number for all three, matching the single shared scale ``d0/d1/d2``
    already carry. The weights are frozen, so it is computed once and cached.

    Args:
        v0: conv2's m=0 transposed fp16 weight, ``(N, K)``.
        v1: The m=1 weight.
        v2: The m=2 weight.

    Returns:
        A float.
    """
    key = (v0.data_ptr(), v1.data_ptr(), v2.data_ptr())
    cs = _D_COLSUM.get(key)
    if cs is None:
        cs = max(float(v.float().abs().sum(dim=1).max()) for v in (v0, v1, v2))
        _D_COLSUM[key] = cs
    return cs


def _dbeta_bound(v0, v1, v2, beta=None):
    """
    Worst-case stored ``|d|`` for a *constant* headroom factor.

    Kept so the unsafety of the constant path is stated in code rather than only
    in a comment, and so ``UMA_NITRO_DBETA_ROW=0`` reports what it is giving up.

    Args:
        v0: conv2's m=0 transposed fp16 weight.
        v1: The m=1 weight.
        v2: The m=2 weight.
        beta: The constant factor; defaults to ``D_BETA``.

    Returns:
        ``(worst, ok)`` -- the bound on the stored magnitude and whether it fits.
    """
    worst = (
        3.0
        * _FP16_SCALE_TARGET
        * _d_colsum(v0, v1, v2)
        * (D_BETA if beta is None else beta)
    )
    return worst, worst <= 65504.0


def _dbeta_check(v0, v1, v2):
    """
    Validate the constant D_BETA path against fp16 range.

    Raises:
        RuntimeError: If the live weights make the constant scaling unsafe.
    """
    worst, ok = _dbeta_bound(v0, v1, v2)
    if not ok:
        raise RuntimeError(
            f"UMA Nitro: UMA_NITRO_DBETA_ROW=0 selects the constant D_BETA="
            f"{D_BETA:g}, whose worst-case stored |d| on these weights is "
            f"{worst:.3e} against fp16's 65504 -- it would store inf and return "
            f"NaN forces with a finite energy. Leave UMA_NITRO_DBETA_ROW=1 (the "
            f"per-edge bound), or set UMA_NITRO_FP16_D=0."
        )


FP16_DM = os.environ.get("UMA_NITRO_FP16_DM", "1") == "1"

GATE_MAX_KROW = 113.0

# Scaled accumulator rows land at about ``2 ** DM_TARGET`` divided by the bound's
# slack. 14 is the top of fp16's normal range on an exact power of two.
DM_TARGET_EXP = _FP16_SCALE_EXP
_DM_DUMMY = {}
_DM_COLSUMS = {}


def _dm_dummy(device):
    """
    Cached 1-element placeholder for the per-block scale buffer when it is unused.

    ``dmsc`` is in the op's ``mutates_args``, so the ``dyscale`` trick of passing
    another live tensor would declare a false mutation on it.

    Args:
        device: CUDA device.

    Returns:
        A 1-element fp32 tensor.
    """
    key = str(device)
    vec = _DM_DUMMY.get(key)
    if vec is None:
        vec = torch.empty(1, device=device, dtype=torch.float32)
        _DM_DUMMY[key] = vec
    return vec


def _dm_colsums(w0, w1, w2):
    """
    ``max_n sum_k |w[n, k]|`` for conv1's three transposed backward weights.

    The GEMM is ``a (M, K) @ w (N, K).t()``, so this is the operator norm that turns
    the A operand's row maximum into a bound on the accumulator's row maximum. The
    weights are frozen, so it is computed once per module and cached.

    Args:
        w0: conv1's m=0 transposed fp16 weight.
        w1: The m=1 weight.
        w2: The m=2 weight.

    Returns:
        A list of three floats.
    """
    key = (w0.data_ptr(), w1.data_ptr(), w2.data_ptr())
    cs = _DM_COLSUMS.get(key)
    if cs is None:
        cs = [float(w.float().abs().sum(dim=1).max()) for w in (w0, w1, w2)]
        _DM_COLSUMS[key] = cs
    return cs


def _dbeta_vec(edges, device):
    """
    Return a cached per-row vector containing D_BETA.

    The GEMM epilogue performs vector loads, so this uses real storage rather
    than a stride-zero expansion.
    """
    key = (edges, str(device))
    vec = _DBETA_VEC.get(key)
    if vec is None:
        vec = torch.full((edges,), D_BETA, device=device, dtype=torch.float32)
        _DBETA_VEC[key] = vec
    return vec


def _layout(num_edges, channels, prefix, block_sizes, coeff_pad):
    """
    Rebuild a ``BlockLayout`` from the primitives carried through the schema.

    Deliberately not memoized: under fake-tensor tracing ``num_edges`` is a SymInt,
    which is unhashable, so an ``lru_cache`` here raises
    ``TypeError: unhashable type: non-nested SymInt``.
    """
    return BlockLayout(
        num_edges=num_edges,
        channels=channels,
        block_sizes=tuple(block_sizes),
        prefix=prefix,
        coeff_pad=None if coeff_pad == _NO_PAD else coeff_pad,
    )


_SEG_GRID = 4096


def _conv1_fwd(x_full, edge_index, wigner_flat, radial, m0, m1, m2, channels):
    """
    Launch the fused conv1 forward with one program per edge.

    Args:
        x_full: Node features, ``(N, 9, C)``.
        edge_index: ``(2, E)``.
        wigner_flat: ``(E, 81)``.
        radial: ``(E, 12C)``.
        m0: Output ``(E, 6C)``; mutated.
        m1: Output ``(E, 8C)``; mutated.
        m2: Output ``(E, 4C)``; mutated.
        channels: Channel count ``C``.
    """
    edges = edge_index.shape[1]
    wrap_triton(wigner_conv1_fused_fwd_kernel)[(edges,)](
        x_full,
        edge_index,
        wigner_flat,
        radial,
        m0,
        m1,
        m2,
        edges,
        channels,
        x_full.stride(0),
        x_full.stride(1),
        x_full.stride(2),
        edge_index.stride(0),
        BLOCK_C=channels,
        GRID_E_STRIDE=edges,
        num_warps=C1F_WARPS,
        num_stages=C1F_STAGES,
        X_ABL=2 if "xfwd64" in _ABLATIONS else 0,
        EV_S=_EV_S,
        EV_R=_ev_r(edges),
    )


@torch.library.custom_op(
    "uma_nitro_span::block_span", mutates_args=(), device_types="cuda"
)
def _conv1_span(
    x_full: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    radial: torch.Tensor,
    rad_tok: torch.Tensor,
    rad_tok_s: torch.Tensor,
    wigner_inv_envelope: torch.Tensor,
    node_offsets: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    b0: torch.Tensor,
    vb0: torch.Tensor,
    w0h: torch.Tensor,
    w1h: torch.Tensor,
    w2h: torch.Tensor,
    v0h: torch.Tensor,
    v1h: torch.Tensor,
    v2h: torch.Tensor,
    w0h_t: torch.Tensor,
    w1h_t: torch.Tensor,
    w2h_t: torch.Tensor,
    v0h_t: torch.Tensor,
    v1h_t: torch.Tensor,
    v2h_t: torch.Tensor,
    w0t: torch.Tensor,
    w1t: torch.Tensor,
    w2t: torch.Tensor,
    v0t: torch.Tensor,
    v1t: torch.Tensor,
    v2t: torch.Tensor,
    expand: torch.Tensor,
    channels: int,
    prefix: int,
    block_sizes: list[int],
    coeff_pad: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Pack, conv1, gate and conv2 in one node.

    ``rad_tok``/``rad_tok_s`` are unread here. They are the autograd channel that
    carries ``grad_rad`` back to the radial fc3 backward *in fp16*: a gradient's
    dtype is pinned to its tensor's, so an fp32 ``radial`` can only ever receive an
    fp32 gradient. Two zero-cost tokens -- one-element storages expanded to
    ``(E, 12C)`` fp16 and ``(E,)`` fp32 -- give the backward two extra output slots
    with the right metadata, so the narrow gradient travels through autograd
    normally instead of through a side channel, and ``radial`` itself is marked
    non-differentiable by its producer.

    Returns:
        The updated node features, then the buffers the backward needs.
    """
    edges = edge_index.shape[1]
    layout = _layout(edges, channels, prefix, block_sizes, coeff_pad)
    c2 = 2 * channels
    dev = x_full.device

    # --- pack, straight into fp16: internal, so no gradient dtype is implied ---
    m0 = torch.empty((edges, 3 * c2), device=dev, dtype=torch.float16)
    m1 = torch.empty((edges, 4 * c2), device=dev, dtype=torch.float16)
    m2 = torch.empty((edges, 2 * c2), device=dev, dtype=torch.float16)
    _conv1_fwd(
        x_full.contiguous(),
        edge_index,
        wigner.reshape(edges, 81).contiguous(),
        radial.contiguous(),
        m0,
        m1,
        m2,
        channels,
    )

    # --- conv1: fp16 in, fp32 accumulate, fp32 straight into the block layout ---
    buf = layout.allocate(device=dev, dtype=torch.float32)
    stride = layout.row_stride
    head = layout.prefix + layout.block_sizes[0] * channels
    # Disjoint column ranges of one `buf` -- [0, head), then the two coeff blocks --
    # so the three writes cannot overlap and the GEMMs are mutually independent.
    # Unlike the other parallel triples these write into a buffer allocated on the
    # *calling* stream rather than allocating their own; safe because the join
    # precedes any use, and under capture nothing in the pool is ever freed.
    _m0, _m1, _m2 = (
        (_abl_row0(m0), _abl_row0(m1), _abl_row0(m2))
        if "mread" in _ABLATIONS
        else (m0, m1, m2)
    )
    _parallel(
        [
            lambda: fp16_mm(_m0, w0h, out=buf, out_row_stride=stride, out_col_base=0),
            lambda: fp16_mm(
                _m1,
                w1h,
                out=buf,
                out_row_stride=stride,
                out_col_base=layout.coeff_base + layout.offsets[1] * channels,
            ),
            lambda: fp16_mm(
                _m2,
                w2h,
                out=buf,
                out_row_stride=stride,
                out_col_base=layout.coeff_base + layout.offsets[2] * channels,
            ),
        ]
    )

    if GATE_PACK:
        gbuf = buf.view(torch.float16).narrow(1, layout.row_stride, layout.row_stride)
    else:
        gbuf = layout.allocate(device=dev, dtype=torch.float16)
    gate(layout, buf, out=gbuf, expand=expand, bias=b0)

    _b0, _b1, _b2 = (
        layout.block(gbuf, 0),
        layout.block(gbuf, 1),
        layout.block(gbuf, 2),
    )
    _astride = gbuf.stride(0)
    if "gbufread" in _ABLATIONS:
        _b0, _b1, _b2 = _abl_row0(_b0), _abl_row0(_b1), _abl_row0(_b2)
        _astride = 0
    g0, g1, g2 = _parallel(
        [
            lambda: fp16_mm(_b0, v0h, bias=vb0, a_row_stride=_astride),
            lambda: fp16_mm(_b1, v1h, a_row_stride=_astride),
            lambda: fp16_mm(_b2, v2h, a_row_stride=_astride),
        ]
    )

    out = torch.zeros_like(residual) if SEG_ESPLIT > 1 else torch.empty_like(residual)
    wig_inv = wigner_inv_envelope.reshape(edges, 81).contiguous()
    _nodes = residual.shape[0]
    _seg_grid = min(_nodes, SEG_GRID) * SEG_NSPLIT * SEG_ESPLIT
    wrap_triton(_invrot_segreduce_kernel)[(_seg_grid,)](
        g0,
        g1,
        g2,
        wig_inv,
        node_offsets,
        residual,
        out,
        _nodes,
        C=channels,
        GRID=_seg_grid,
        NSPLIT=SEG_NSPLIT,
        CBLK=channels // SEG_NSPLIT,
        ESPLIT=SEG_ESPLIT,
        num_warps=SEG_WARPS,
        num_stages=SEG_STAGES,
        G_ABL=2 if "gfwd64" in _ABLATIONS else 1 if "gfwd" in _ABLATIONS else 0,
        RED=SEG_RED and _red_v4_ok(channels // SEG_NSPLIT, SEG_WARPS),
        EV_S2=_EV_S2,
        EV_R2=_EV_R2,
    )
    # gbuf leaves the op only so the backward can apply the ``dgs = (1-s)*sum(dy*y)``
    # identity against the post-gate fp16 values instead of re-reading the fp32
    # pre-gate ones. See GATE_IDENT. Off, this slot is an empty placeholder, so the
    # buffer stays a forward-local temporary and its memory is reused per layer.
    # A custom op may not return two outputs that alias, so under GATE_PACK the
    # slot is empty and the backward rebuilds the view from ``buf``.
    return (
        out,
        buf,
        g0,
        g1,
        g2,
        gbuf if (GATE_IDENT and not GATE_PACK) else buf.new_empty(0),
    )


@_conv1_span.register_fake
def _(
    x_full,
    edge_index,
    wigner,
    radial,
    wigner_inv_envelope,
    node_offsets,
    residual,
    target,
    b0,
    vb0,
    w0h,
    w1h,
    w2h,
    v0h,
    v1h,
    v2h,
    w0h_t,
    w1h_t,
    w2h_t,
    v0h_t,
    v1h_t,
    v2h_t,
    w0t,
    w1t,
    w2t,
    v0t,
    v1t,
    v2t,
    expand,
    channels,
    prefix,
    block_sizes,
    coeff_pad,
):
    """
    Shape function, so Dynamo can trace the op without running it.
    """
    edges = edge_index.shape[1]
    layout = _layout(edges, channels, prefix, block_sizes, coeff_pad)
    return (
        torch.empty_like(residual),
        x_full.new_empty(edges, layout.row_stride),
        x_full.new_empty(edges, v0h.shape[0]),
        x_full.new_empty(edges, v1h.shape[0]),
        x_full.new_empty(edges, v2h.shape[0]),
        x_full.new_empty(
            (edges, layout.row_stride) if (GATE_IDENT and not GATE_PACK) else (0,),
            dtype=torch.float16,
        ),
    )


def _setup_context(ctx, inputs, output):
    """
    Save the pre-gate buffer, the pack's inputs, and the backward weights.
    """
    (
        x_full,
        edge_index,
        wigner,
        radial,
        _rad_tok,
        _rad_tok_s,
        wigner_inv_envelope,
        node_offsets,
        _residual,
        target,
        b0,
        _vb0,
        _w0h,
        _w1h,
        _w2h,
        _v0h,
        _v1h,
        _v2h,
        w0h_t,
        w1h_t,
        w2h_t,
        v0h_t,
        v1h_t,
        v2h_t,
        w0t,
        w1t,
        w2t,
        v0t,
        v1t,
        v2t,
        expand,
        channels,
        prefix,
        blocks,
        pad,
    ) = inputs
    ctx.mark_non_differentiable(output[1], output[2], output[3], output[4], output[5])
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        output[1],
        output[2],
        output[3],
        output[4],
        output[5],
        wigner_inv_envelope,
        node_offsets,
        target,
        b0,
        w0h_t,
        w1h_t,
        w2h_t,
        v0h_t,
        v1h_t,
        v2h_t,
        x_full,
        edge_index,
        wigner,
        radial,
        w0t,
        w1t,
        w2t,
        v0t,
        v1t,
        v2t,
        expand,
    )
    ctx.layout_args = (channels, prefix, blocks, pad)
    # An fp16 token means the caller wants grad_rad narrow and routed through the
    # token slots. Off, the tokens are empty fp32 placeholders and grad_rad goes
    # back through `radial` as before.
    ctx.rad_fp16 = _rad_tok.dtype == torch.float16


_SPAN_I32 = [None, None]


def _edge_index_i32(edge_index):
    """
    Return ``(source, target)`` as contiguous int32 rows and cache them.

    Args:
        edge_index: ``(2, E)`` int64.

    Returns:
        A pair of ``(E,)`` int32 tensors.
    """
    ei32 = edge_index.to(torch.int32)
    pair = (ei32[0], ei32[1])
    _SPAN_I32[0], _SPAN_I32[1] = pair
    return pair


def _span_backward(ctx, grad_out, d_buf, d_g0, d_g1, d_g2, d_gbuf):
    """
    conv2 GEMMs, gate backward, conv1 GEMMs, then the fused wigner backward.

    Returns:
        Gradients for ``x_full``, ``wigner``, ``radial``,
        ``wigner_inv_envelope`` and ``residual``; None elsewhere.
    """
    del d_buf, d_g0, d_g1, d_g2, d_gbuf  # implementation-detail outputs
    (
        buf,
        g0,
        g1,
        g2,
        gbuf,
        wig_inv,
        node_offsets,
        target,
        b0,
        w0h_t,
        w1h_t,
        w2h_t,
        v0h_t,
        v1h_t,
        v2h_t,
        x_full,
        edge_index,
        wigner,
        radial,
        w0t,
        w1t,
        w2t,
        v0t,
        v1t,
        v2t,
        expand,
    ) = ctx.saved_tensors
    channels = ctx.layout_args[0]
    layout = _layout(buf.shape[0], *ctx.layout_args)
    edges, nodes, c2 = buf.shape[0], x_full.shape[0], 2 * channels
    del nodes
    dev, dt = buf.device, buf.dtype
    tail = (None,) * 26

    if torch.is_grad_enabled():
        raise NotImplementedError(
            "uma_nitro_span::block_span does not support double backward; compute "
            "Hessians with a variant that keeps the composable path."
        )

    # --- invrot backward: gather the node gradient back to edges, then the fused
    # inverse-rotate backward. dg* are internal to this op now, which is the point
    # of the merge -- as separate autograd outputs they were pinned to g*'s dtype.
    grad_out = grad_out.contiguous()
    _fuse_gather = IC2_GATHER or IC2_ABL
    grad_edge = (
        None
        if _fuse_gather
        else torch.empty((edges, 9, channels), device=dev, dtype=grad_out.dtype)
    )
    _gather_view = grad_out.view(-1, 9, channels)
    # int32 edge endpoints, converted once per graph in _backbone_forward rather than
    # twice per layer here. `target` is `edge_index[1]`, so both rows come from one
    # (2, E) cast: 8 aten::copy_ launches per iteration became 1.
    _src32, _target32 = _SPAN_I32
    if _target32 is None or _target32.shape[0] != edges:
        _src32, _target32 = _edge_index_i32(edge_index)
    _parts = torch.empty(AMX_GRID, device=dev, dtype=torch.float32)
    dscale_g = torch.empty(1, device=dev, dtype=torch.float32)
    # Both branches only *read* grad_out and write disjoint destinations -- the gather
    # fills grad_edge, the scale reduction fills _parts/dscale_g -- and both are first
    # consumed together by invrot_conv2_bwd_fp16 below. pow2_scale is two kernels, the
    # second a single-program launch, so it is almost entirely scheduling latency and
    # hides completely under the gather.
    if _fuse_gather:
        torch.ops.uma_nitro.pow2_scale(grad_out, _parts, dscale_g)
    else:
        _parallel(
            [
                lambda: torch.ops.uma_nitro.gather_node_v4(
                    _gather_view, _target32, grad_edge
                ),
                lambda: torch.ops.uma_nitro.pow2_scale(grad_out, _parts, dscale_g),
            ]
        )
    _dbrow = FP16_D and DBETA_ROW
    if FP16_D and not _dbrow:
        _dbeta_check(v0h_t, v1h_t, v2h_t)
    dg0, dg1, dg2, dwig, dg_rscale, dg_beta = invrot_conv2_bwd_fp16(
        _gather_view if _fuse_gather else grad_edge,
        g0,
        g1,
        g2,
        wig_inv,
        dscale_g,
        channels,
        grid=edges,
        tgt=_target32 if _fuse_gather else None,
        edges=edges,
        cs_max=_d_colsum(v0h_t, v1h_t, v2h_t) if _dbrow else 0.0,
    )
    if FP16_D:
        _bvec = dg_beta if _dbrow else _dbeta_vec(edges, dev)
        d0, d1, d2 = (
            torch.empty(edges, v0h_t.shape[0], device=dev, dtype=torch.float16),
            torch.empty(edges, v1h_t.shape[0], device=dev, dtype=torch.float16),
            torch.empty(edges, v2h_t.shape[0], device=dev, dtype=torch.float16),
        )
        _parallel(
            [
                lambda: fp16_mm(dg0, v0h_t, out=d0, rscale=_bvec),
                lambda: fp16_mm(dg1, v1h_t, out=d1, rscale=_bvec),
                lambda: fp16_mm(dg2, v2h_t, out=d2, rscale=_bvec),
            ]
        )
    else:
        d0, d1, d2 = _parallel(
            [
                lambda: fp16_mm(dg0, v0h_t, rscale=dg_rscale),
                lambda: fp16_mm(dg1, v1h_t, rscale=dg_rscale),
                lambda: fp16_mm(dg2, v2h_t, rscale=dg_rscale),
            ]
        )

    # dbuf in fp16 with an exact per-row power-of-two scale, computed by the kernel
    # that already reads every element of every row.
    dbuf = layout.allocate(device=dev, dtype=torch.float16)
    dscale = torch.empty(edges, device=dev, dtype=torch.float32)
    _chunk = 0 if _ABLATIONS else min(C1B_CHUNK, edges) if C1B_CHUNK else 0
    dm_fp16 = FP16_DM and not _chunk and "dmgemm" not in _ABLATIONS
    _dmsc = torch.empty(6, edges, device=dev, dtype=torch.float32) if dm_fp16 else None
    gate_bwd_split_fp16(
        layout,
        buf,
        d0,
        d1,
        d2,
        dbuf,
        dscale,
        y=(_gpack(buf, layout) if GATE_PACK else gbuf) if GATE_IDENT else None,
        expand=expand,
        bias=b0,
        block_e=4,
        dyscale=dg_rscale if FP16_D else None,
        dybeta=1.0 if _dbrow else 1.0 / D_BETA,
        dyrow=_dbrow,
        dmsc=_dmsc,
        colsums=_dm_colsums(w0h_t, w1h_t, w2h_t) if dm_fp16 else None,
    )

    head = layout.prefix + layout.block_sizes[0] * channels
    stride = layout.row_stride
    if _chunk:
        dm0 = dm1 = dm2 = None  # produced inside the chunk loop below
    elif "dmgemm" in _ABLATIONS:
        # Ceiling for a fusion that also absorbs the GEMM: not reachable (the FLOPs
        # have to happen somewhere), but it brackets "dmread" from above.
        dm0, dm1, dm2 = (
            torch.empty(1, w0h_t.shape[0], device=dev, dtype=torch.float32),
            torch.empty(1, w1h_t.shape[0], device=dev, dtype=torch.float32),
            torch.empty(1, w2h_t.shape[0], device=dev, dtype=torch.float32),
        )
    else:
        _a0, _a1, _a2 = dbuf[:, :head], layout.block(dbuf, 1), layout.block(dbuf, 2)
        if "dbufread" in _ABLATIONS:
            _a0, _a1, _a2 = _abl_row0(_a0), _abl_row0(_a1), _abl_row0(_a2)
            stride = 0
        if dm_fp16:
            dm0, dm1, dm2 = (
                torch.empty(edges, w0h_t.shape[0], device=dev, dtype=torch.float16),
                torch.empty(edges, w1h_t.shape[0], device=dev, dtype=torch.float16),
                torch.empty(edges, w2h_t.shape[0], device=dev, dtype=torch.float16),
            )
            _parallel(
                [
                    lambda: fp16_mm(
                        _a0, w0h_t, out=dm0, a_row_stride=stride, rscale=_dmsc[0]
                    ),
                    lambda: fp16_mm(
                        _a1, w1h_t, out=dm1, a_row_stride=stride, rscale=_dmsc[1]
                    ),
                    lambda: fp16_mm(
                        _a2, w2h_t, out=dm2, a_row_stride=stride, rscale=_dmsc[2]
                    ),
                ]
            )
        else:
            dm0, dm1, dm2 = _parallel(
                [
                    lambda: fp16_mm(_a0, w0h_t, a_row_stride=stride, rscale=dscale),
                    lambda: fp16_mm(_a1, w1h_t, a_row_stride=stride, rscale=dscale),
                    lambda: fp16_mm(_a2, w2h_t, a_row_stride=stride, rscale=dscale),
                ]
            )

    _sc = 0 if (("segtgt" in _ABLATIONS) or SCT_CHUNK) else _scatter_mode()
    nodes = grad_out.shape[0]
    _ged = torch.empty(1, 1, 1, device=dev, dtype=torch.float16)
    ge_src = (
        _ged
        if _sc & 1
        else torch.empty((edges, 9, channels), device=dev, dtype=torch.float16)
    )
    ge_tgt = (
        _ged
        if _sc & 2
        else torch.empty((edges, 9, channels), device=dev, dtype=torch.float16)
    )
    gescale = torch.empty(1 if _sc == 3 else edges, device=dev, dtype=torch.float32)
    grad_x = (
        torch.zeros(nodes, 9, channels, device=dev, dtype=torch.float32)
        if _sc
        else torch.empty(1, device=dev, dtype=torch.float32)
    )
    # Only the block-diagonal entries are written, so this must start zeroed.
    # Not zeros: the kernel writes the off-block entries itself.
    gwig = torch.empty((edges, 81), device=dev, dtype=dt)
    if os.environ.get("UMA_NITRO_POISON_WIG") == "1":
        gwig.fill_(float("nan"))
    rad_fp16 = ctx.rad_fp16
    grad_rad = torch.empty(
        (edges, 6 * c2), device=dev, dtype=torch.float16 if rad_fp16 else dt
    )
    rad_rscale = torch.empty(edges if rad_fp16 else 1, device=dev, dtype=torch.float32)
    _wig_flat = wigner.reshape(edges, 81).contiguous()
    _rad = radial.contiguous()
    _xf = x_full.contiguous()
    if _chunk:
        dm0c = torch.empty(_chunk, w0h_t.shape[0], device=dev, dtype=torch.float32)
        dm1c = torch.empty(_chunk, w1h_t.shape[0], device=dev, dtype=torch.float32)
        dm2c = torch.empty(_chunk, w2h_t.shape[0], device=dev, dtype=torch.float32)
        for _s in range(0, edges, _chunk):
            _e = min(_s + _chunk, edges)
            _n = _e - _s
            _v0, _v1, _v2 = dm0c[:_n], dm1c[:_n], dm2c[:_n]
            _db = dbuf[_s:_e]
            _ds = dscale[_s:_e]
            fp16_mm(_db[:, :head], w0h_t, out=_v0, a_row_stride=stride, rscale=_ds)
            fp16_mm(
                _db[:, head : head + layout.block_sizes[1] * channels],
                w1h_t,
                out=_v1,
                a_row_stride=stride,
                rscale=_ds,
            )
            fp16_mm(
                _db[:, head + layout.block_sizes[1] * channels :],
                w2h_t,
                out=_v2,
                a_row_stride=stride,
                rscale=_ds,
            )
            torch.ops.uma_nitro.conv1_bwd_fp16(
                _v0,
                _v1,
                _v2,
                _wig_flat[_s:_e],
                _rad[_s:_e],
                _xf,
                edge_index[:, _s:_e],
                ge_src[_s:_e],
                ge_tgt[_s:_e],
                grad_x,
                gwig[_s:_e],
                grad_rad[_s:_e],
                gescale[_s:_e],
                rad_rscale[_s:_e] if rad_fp16 else rad_rscale,
                _dm_dummy(dev),
                channels,
            )
    else:
        torch.ops.uma_nitro.conv1_bwd_fp16(
            dm0.contiguous(),
            dm1.contiguous(),
            dm2.contiguous(),
            _wig_flat,
            _rad,
            _xf,
            edge_index,
            ge_src,
            ge_tgt,
            grad_x,
            gwig,
            grad_rad,
            gescale,
            rad_rscale,
            _dmsc[3:6] if dm_fp16 else _dm_dummy(dev),
            channels,
        )
    nodes = grad_out.shape[0]
    if _sc & 2:
        pass  # target half already accumulated by the conv1 backward
    elif SCT_CHUNK:
        grad_x = torch.zeros(nodes, 9, channels, device=dev, dtype=torch.float32)
        torch.ops.uma_nitro.segreduce_tgt_bal(ge_tgt, gescale, target, grad_x, channels)
    else:
        grad_x = segreduce_tgt_fp16(
            ge_tgt,
            gescale,
            node_offsets,
            nodes,
            channels,
            out=grad_x if _sc & 1 else None,
        )
    if not (_sc & 1):
        torch.ops.uma_nitro.scatter_src_v4_fp16(ge_src, _src32, gescale, grad_x)
    # residual's gradient is identity: the forward adds it into the scatter.
    return (
        grad_x,
        None,
        gwig.view_as(wigner),
        None if rad_fp16 else grad_rad,
        grad_rad if rad_fp16 else None,
        rad_rscale if rad_fp16 else None,
        dwig.view_as(wig_inv),
        None,
        grad_out,
        None,
        *tail,
    )


torch.library.register_autograd(
    "uma_nitro_span::block_span", _span_backward, setup_context=_setup_context
)


def block_span(
    x_full,
    edge_index,
    wigner,
    radial,
    wigner_inv_envelope,
    node_offsets,
    residual,
    target,
    b0,
    vb0,
    hs,
    hts,
    ts,
    expand,
    layout,
    rad_tok=None,
    rad_tok_s=None,
):
    """
    Run pack + conv1 + gate + conv2 as one node.

    Args:
        x_full: Node features ``(N, 9, C)``.
        edge_index: ``(2, E)``.
        wigner: Wigner rotations ``(E, 9, 9)``.
        radial: conv1 radial embedding ``(E, 6*2C)``.
        rad_tok: Optional ``(E, 6*2C)`` fp16 autograd token carrying the scaled
            ``grad_rad``; ``rad_tok_s`` carries its per-edge reciprocal scale. When
            either is None the gradient goes back through ``radial`` in fp32.
        b0: conv1 m=0 bias.
        vb0: conv2 m=0 bias.
        hts: the six transposed weights in fp16 (conv1 then conv2), for the
            backward GEMMs.
        hs: The six fp16 weights (conv1 m0/m1/m2 then conv2 m0/m1/m2).
        ts: The six pre-transposed fp32 weights, for the backward.
        expand: Gate coefficient-to-l map.
        layout: A ``BlockLayout``.

    Returns:
        The updated node features, ``(N, 9, C)``.
    """
    if rad_tok is None or rad_tok_s is None:
        rad_tok = rad_tok_s = None
    _res = torch.ops.uma_nitro_span.block_span(
        x_full,
        edge_index,
        wigner,
        radial,
        radial.new_empty(0) if rad_tok is None else rad_tok,
        radial.new_empty(0) if rad_tok_s is None else rad_tok_s,
        wigner_inv_envelope,
        node_offsets,
        residual,
        target,
        b0,
        vb0,
        *hs,
        *hts,
        *ts,
        expand,
        layout.channels,
        layout.prefix,
        list(layout.block_sizes),
        _NO_PAD if layout.coeff_pad is None else layout.coeff_pad,
    )

    if os.environ.get("UMA_NITRO_PROBE_GRAD") == "1":
        print(
            "PROBE requires_grad:",
            [t.requires_grad for t in _res],
            "shapes:",
            [tuple(t.shape) for t in _res],
            flush=True,
        )
    out, *_ = _res
    return out


BF_BLOCK = 1024


@triton.jit
def _bucket_fill_direct_kernel(
    bucket_ptr,
    ok_ptr,
    counter_ptr,
    flat_ptr,
    n,
    cell_capacity,
    num_buckets,
    flag_ptr,
    BLOCK_N: tl.constexpr,
):
    """
    Build the bucket table with a counting sort, no argsort at all.

    The sort-based path exists only to give each atom a slot within its bucket.
    An atomic increment of a per-bucket counter does the same in one pass, which
    removes the ``argsort`` (CUB radix, 3M keys), the gather of ``source_bucket``
    into sorted order, and the ``searchsorted``/``cummax`` that derived the slot.

    Which atom lands in which slot becomes nondeterministic, but that was already
    arbitrary -- it followed the sort permutation. It only matters when a bucket
    overflows ``cell_capacity``, and then the sort-based version dropped an equally
    arbitrary subset.
    """
    off = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = off < n
    bucket = tl.load(bucket_ptr + off, mask=m, other=0)
    ok = tl.load(ok_ptr + off, mask=m, other=0)
    valid = m & (ok != 0)
    slot = tl.atomic_add(counter_ptr + bucket, 1, mask=valid)
    keep = valid & (slot < cell_capacity)
    row = tl.where(keep, bucket, num_buckets)
    slot_c = tl.minimum(tl.maximum(slot, 0), cell_capacity - 1)
    tl.store(flat_ptr + row * cell_capacity + slot_c, off, mask=m)
    # `counter[:num_buckets].max() > cell_capacity` folded in. `slot` is the
    # pre-increment count, so some lane sees `slot >= cell_capacity` exactly when
    # the final count exceeds it -- the same test, without the reduction. Masked,
    # so a non-overflowing build issues no atomic.
    tl.atomic_max(
        flag_ptr + off * 0,
        tl.full((BLOCK_N,), 1, tl.int64),
        mask=valid & (slot >= cell_capacity),
    )


@triton_op("uma_nitro::bucket_fill_direct", mutates_args={"counter", "flat", "flag"})
def bucket_fill_direct(
    source_bucket: torch.Tensor,
    source_ok: torch.Tensor,
    counter: torch.Tensor,
    flat: torch.Tensor,
    flag: torch.Tensor,
    cell_capacity: int,
    num_buckets: int,
) -> None:
    """
    Fill the bucket table directly, without sorting.

    Args:
        source_bucket: Hashed bucket per source atom, ``(M,)``.
        source_ok: Validity flag per source atom, ``(M,)``.
        counter: Per-bucket occupancy, ``(num_buckets + 1,)``, pre-zeroed; mutated.
        flat: Table ``((num_buckets + 1) * cell_capacity,)`` pre-filled with -1;
            mutated.
        cell_capacity: Slots per bucket.
        num_buckets: Bucket count; row ``num_buckets`` is the sentinel.
    """
    n = source_bucket.shape[0]
    wrap_triton(_bucket_fill_direct_kernel)[(triton.cdiv(n, BF_BLOCK),)](
        source_bucket,
        source_ok,
        counter,
        flat,
        n,
        cell_capacity,
        num_buckets,
        flag,
        BLOCK_N=BF_BLOCK,
        num_warps=4,
    )


FC_BLOCK = 512
# Instantiated as tl.constexpr, not annotated: a plain module global is rejected
# from inside a @triton.jit function.
GOLDEN = tl.constexpr(-7046029254386353131)
FMIX_A = tl.constexpr(-49064778989728563)
FMIX_B = tl.constexpr(-4265267296055464877)
SIGN_MASK = tl.constexpr(0x7FFFFFFFFFFFFFFF)


@triton.jit
def _fc_hash_cells(cx, cy, cz, system, num_buckets):
    """
    Hash integer cell coordinates and a system id into a bucket.

    The multiply-add chain avoids structural cancellation on lattice inputs.
    """
    h = cx * GOLDEN + cy
    h = h * GOLDEN + cz
    h = h * GOLDEN + system
    h = h ^ ((h & SIGN_MASK) >> 33)
    h = h * FMIX_A
    h = h ^ ((h & SIGN_MASK) >> 33)
    h = h * FMIX_B
    h = h ^ ((h & SIGN_MASK) >> 33)
    # Python's remainder is non-negative for a positive divisor; the modulus is a
    # power of two, so a mask gives the same answer without a division.
    return h & (num_buckets - 1)


@triton.jit
def _candidates_kernel(
    pos_ptr,
    batch_ptr,
    offsets_ptr,
    table_ptr,
    scc_ptr,
    spos_ptr,
    ssys_ptr,
    keep_ptr,
    safe_ptr,
    n_total,
    n_cells,
    capacity,
    num_buckets,
    inv_res,
    r_sq,
    BLOCK_N: tl.constexpr,
):
    """
    One pass per (atom, neighbor cell, bucket slot) triple.

    Emits ``keep`` and the clamped candidate index ``safe``; every other
    intermediate the eager chain materializes stays in registers.
    """
    tid = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = tid < n_total
    per_atom = n_cells * capacity
    atom = tid // per_atom
    rem = tid % per_atom
    cellj = rem // capacity
    slot = rem % capacity

    px = tl.load(pos_ptr + atom * 3 + 0, mask=m, other=0.0)
    py = tl.load(pos_ptr + atom * 3 + 1, mask=m, other=0.0)
    pz = tl.load(pos_ptr + atom * 3 + 2, mask=m, other=0.0)
    sysid = tl.load(batch_ptr + atom, mask=m, other=0)

    # floor(pos / res) via multiply by the reciprocal, matching the eager
    # `torch.floor(pos / grid_resolution)`.
    tcx = tl.floor(px * inv_res).to(tl.int64)
    tcy = tl.floor(py * inv_res).to(tl.int64)
    tcz = tl.floor(pz * inv_res).to(tl.int64)
    lcx = tcx + tl.load(offsets_ptr + cellj * 3 + 0, mask=m, other=0)
    lcy = tcy + tl.load(offsets_ptr + cellj * 3 + 1, mask=m, other=0)
    lcz = tcz + tl.load(offsets_ptr + cellj * 3 + 2, mask=m, other=0)

    bucket = _fc_hash_cells(lcx, lcy, lcz, sysid, num_buckets)
    cand = tl.load(table_ptr + bucket * capacity + slot, mask=m, other=-1)
    present = cand >= 0
    safe = tl.maximum(cand, 0)

    # A hashed bucket can hold atoms from another cell or another system, so both
    # are re-checked. The cell check is not an optimization: two distinct neighbor
    # cells hashing to one bucket would otherwise emit the same edge twice.
    ccx = tl.load(scc_ptr + safe * 3 + 0, mask=m, other=0)
    ccy = tl.load(scc_ptr + safe * 3 + 1, mask=m, other=0)
    ccz = tl.load(scc_ptr + safe * 3 + 2, mask=m, other=0)
    cell_match = (ccx == lcx) & (ccy == lcy) & (ccz == lcz)
    same_system = tl.load(ssys_ptr + safe, mask=m, other=-1) == sysid

    dx = px - tl.load(spos_ptr + safe * 3 + 0, mask=m, other=0.0)
    dy = py - tl.load(spos_ptr + safe * 3 + 1, mask=m, other=0.0)
    dz = pz - tl.load(spos_ptr + safe * 3 + 2, mask=m, other=0.0)
    dist_sq = dx * dx + dy * dy + dz * dz

    keep = present & cell_match & same_system & (dist_sq <= r_sq) & (dist_sq != 0.0)
    tl.store(keep_ptr + tid, keep.to(tl.int1), mask=m)
    tl.store(safe_ptr + tid, safe, mask=m)


@triton_op("uma_nitro::fused_candidates", mutates_args={"keep", "safe"})
def fused_candidates(
    pos: torch.Tensor,
    batch: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    table: torch.Tensor,
    source_cell_coord: torch.Tensor,
    source_pos: torch.Tensor,
    source_system: torch.Tensor,
    keep: torch.Tensor,
    safe: torch.Tensor,
    grid_resolution: float,
    radius: float,
) -> None:
    """
    Compute the candidate keep mask and clamped candidate indices in one pass.

    Args:
        pos: Target atom positions, ``(N, 3)``.
        batch: System index per target atom, ``(N,)``.
        neighbor_offsets: Integer cell offsets, ``(C, 3)``.
        table: Bucket table, ``(num_buckets + 1, capacity)``.
        source_cell_coord: Grid coordinates per source atom, ``(M, 3)``.
        source_pos: Source atom positions including image shifts, ``(M, 3)``.
        source_system: System index per source atom, ``(M,)``.
        keep: Output mask, ``(N * C * capacity,)`` bool; mutated.
        safe: Output candidate indices, ``(N * C * capacity,)`` int64; mutated.
        grid_resolution: Cell size.
        radius: Neighbor cutoff.
    """
    n_total = keep.shape[0]
    num_buckets = table.shape[0] - 1
    wrap_triton(_candidates_kernel)[(triton.cdiv(n_total, FC_BLOCK),)](
        pos,
        batch,
        neighbor_offsets,
        table,
        source_cell_coord,
        source_pos,
        source_system,
        keep,
        safe,
        n_total,
        neighbor_offsets.shape[0],
        table.shape[1],
        num_buckets,
        1.0 / grid_resolution,
        radius * radius,
        BLOCK_N=FC_BLOCK,
        num_warps=4,
    )


def plan_capacities(
    num_atoms: int, rep_max: int = 1, edges_per_atom: int = 112
) -> dict:
    """
    Return conservative radius-graph capacities for a system size.
    """
    return {
        "rep_max": rep_max,
        "cell_capacity": 24,
        "edge_capacity": num_atoms * edges_per_atom,
    }


REP_NAN_GUARD = os.environ.get("UMA_NITRO_REP_NAN_GUARD", "1") == "1"


@triton.jit
def _replication_kernel(
    cell_ptr,
    pbc_ptr,
    out_ptr,
    flag_ptr,
    radius,
    rep_max,
    n_sys,
    BLOCK: tl.constexpr,
    NAN_GUARD: tl.constexpr = 1,
):
    """
    Per-axis image count for one system per lane.

    Written out rather than looped: Triton has no list comprehensions and cannot index
    a Python list with a traced value, so the three axes are unrolled by hand over the
    same (i, j) pairs the eager version uses.
    """
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = b < n_sys
    a00 = tl.load(cell_ptr + b * 9 + 0, mask=m, other=0.0)
    a01 = tl.load(cell_ptr + b * 9 + 1, mask=m, other=0.0)
    a02 = tl.load(cell_ptr + b * 9 + 2, mask=m, other=0.0)
    a10 = tl.load(cell_ptr + b * 9 + 3, mask=m, other=0.0)
    a11 = tl.load(cell_ptr + b * 9 + 4, mask=m, other=0.0)
    a12 = tl.load(cell_ptr + b * 9 + 5, mask=m, other=0.0)
    a20 = tl.load(cell_ptr + b * 9 + 6, mask=m, other=0.0)
    a21 = tl.load(cell_ptr + b * 9 + 7, mask=m, other=0.0)
    a22 = tl.load(cell_ptr + b * 9 + 8, mask=m, other=0.0)

    x0 = a11 * a22 - a12 * a21
    x1 = a12 * a20 - a10 * a22
    x2 = a10 * a21 - a11 * a20
    volume = a00 * x0 + a01 * x1 + a02 * x2

    d0 = x0 / volume
    d1 = x1 / volume
    d2 = x2 / volume
    inv0 = tl.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

    y0 = a21 * a02 - a22 * a01
    y1 = a22 * a00 - a20 * a02
    y2 = a20 * a01 - a21 * a00
    e0 = y0 / volume
    e1 = y1 / volume
    e2 = y2 / volume
    inv1 = tl.sqrt(e0 * e0 + e1 * e1 + e2 * e2)

    z0 = a01 * a12 - a02 * a11
    z1 = a02 * a10 - a00 * a12
    z2 = a00 * a11 - a01 * a10
    f0 = z0 / volume
    f1 = z1 / volume
    f2 = z2 / volume
    inv2 = tl.sqrt(f0 * f0 + f1 * f1 + f2 * f2)

    p0 = tl.load(pbc_ptr + b * 3 + 0, mask=m, other=0)
    p1 = tl.load(pbc_ptr + b * 3 + 1, mask=m, other=0)
    p2 = tl.load(pbc_ptr + b * 3 + 2, mask=m, other=0)
    r0 = tl.where(p0 != 0, tl.ceil(radius * inv0), 0.0)
    r1 = tl.where(p1 != 0, tl.ceil(radius * inv1), 0.0)
    r2 = tl.where(p2 != 0, tl.ceil(radius * inv2), 0.0)
    tl.store(out_ptr + b * 3 + 0, r0, mask=m)
    tl.store(out_ptr + b * 3 + 1, r1, mask=m)
    tl.store(out_ptr + b * 3 + 2, r2, mask=m)
    # `rep.max() > rep_max` folded in: a masked atomic, so in the ordinary
    # non-overflowing case no atomic is issued at all. Replaces a `max` reduction
    # and a `gt` on the host-visible chain.
    #
    # The NaN terms are not decoration. ``volume`` is zero for a singular cell --
    # a two-dimensional periodic system whose third lattice vector is zero is the
    # easy way to get one -- and then ``x0 / volume`` is ``0 / 0``, ``inv`` is NaN,
    # and ``ceil(radius * NaN)`` is NaN. **``NaN > rep_max`` is false**, so without
    # this the guard stays silent and a NaN image count is stored, which the
    # downstream ``.long()`` turns into an arbitrary integer: the graph is built
    # with the wrong number of images and nothing says so. A degenerate periodic
    # cell is ill-posed, so the right behaviour is the loud assert this flag
    # already drives. Non-periodic axes are unaffected -- the ``where`` above has
    # already replaced them with an exact 0.
    bad = m & ((r0 > rep_max) | (r1 > rep_max) | (r2 > rep_max))
    if NAN_GUARD:
        bad = bad | (m & ((r0 != r0) | (r1 != r1) | (r2 != r2)))
    tl.atomic_max(flag_ptr + b * 0, tl.full((BLOCK,), 1, tl.int64), mask=bad)


@triton_op("uma_nitro::replication", mutates_args={"out", "flag"})
def _replication_fused(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    out: torch.Tensor,
    flag: torch.Tensor,
    radius: float,
    rep_max: int,
) -> None:
    """
    Compute periodic image counts for every cell axis.

    The unrolled arithmetic follows the same axis pairs as the reference path.
    """
    n = cell.shape[0]
    wrap_triton(_replication_kernel)[(triton.cdiv(n, 64),)](
        cell,
        pbc,
        out,
        flag,
        radius,
        rep_max,
        n,
        BLOCK=64,
        num_warps=1,
        NAN_GUARD=1 if REP_NAN_GUARD else 0,
    )


def _replication(cell, pbc, radius, batch_size, flag=None, rep_max=1 << 30):
    """
    Return how many periodic images each axis needs, as a float tensor.

    Args:
        cell: Unit cells, shape ``(B, 3, 3)``.
        pbc: Periodicity flags, shape ``(B, 3)``.
        radius: Cutoff radius.
        batch_size: Number of systems.

    Returns:
        Shape ``(B, 3)``; identical to the baseline's ``rep``.
    """
    out = torch.empty(batch_size, 3, device=cell.device, dtype=cell.dtype)
    if flag is None:
        flag = torch.zeros(1, dtype=torch.int64, device=cell.device)
    torch.ops.uma_nitro.replication(
        cell.contiguous(),
        torch.atleast_2d(pbc).contiguous(),
        out,
        flag,
        radius,
        rep_max,
    )
    return out


@triton.jit
def _hash_cells_fused_kernel(
    coords_ptr, sys_ptr, out_ptr, n, num_buckets, BLOCK: tl.constexpr
):
    """
    The whole bucket-hash chain for one coordinate triple per lane, in one launch.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    cx = tl.load(coords_ptr + i * 3 + 0, mask=m, other=0)
    cy = tl.load(coords_ptr + i * 3 + 1, mask=m, other=0)
    cz = tl.load(coords_ptr + i * 3 + 2, mask=m, other=0)
    sysid = tl.load(sys_ptr + i, mask=m, other=0)
    h = _fc_hash_cells(cx, cy, cz, sysid, num_buckets)
    tl.store(out_ptr + i, h, mask=m)


@triton_op("uma_nitro::hash_cells_fused", mutates_args={"out"})
def _hash_cells_fused(
    coords: torch.Tensor,
    system: torch.Tensor,
    out: torch.Tensor,
    num_buckets: int,
) -> None:
    """
    Hash coordinate triples and system ids into graph buckets.

    The candidate path uses the same device hash to keep lookup behavior identical.
    """
    n = coords.shape[0]
    wrap_triton(_hash_cells_fused_kernel)[(triton.cdiv(n, 512),)](
        coords, system, out, n, num_buckets, BLOCK=512, num_warps=4
    )


def _hash_cells(coords, system, num_buckets):
    """
    Hash integer grid coordinates plus a system id into a bucket index.

    Collisions are safe: they only surface extra candidates, which the distance
    test rejects. They never hide a true neighbor, provided bucket occupancy
    stays under ``cell_capacity``.

    Args:
        coords: Integer grid coordinates, shape ``(..., 3)``.
        system: System index broadcastable to ``coords[..., 0]``.
        num_buckets: Bucket count; a host int.

    Returns:
        Int64 bucket indices with the shape of ``coords[..., 0]``.
    """
    golden = -7046029254386353131  # 0x9E3779B97F4A7C15
    h = coords[..., 0]
    h = h * golden + coords[..., 1]
    h = h * golden + coords[..., 2]
    h = h * golden + system
    # Avalanche (murmur3 fmix64) so a power-of-two modulus, which keeps only
    # low bits, still sees the high bits. Masking the sign bit before shifting
    # stands in for a logical shift, which torch lacks on signed int64.
    for multiplier in (-49064778989728563, -4265267296055464877):
        h = h ^ ((h & 0x7FFFFFFFFFFFFFFF) >> 33)
        h = h * multiplier
    h = h ^ ((h & 0x7FFFFFFFFFFFFFFF) >> 33)
    return torch.remainder(h, num_buckets)


_IMAGE_SCAFFOLD: dict = {}


def _image_scaffold(num_atoms, rep_max, device, dtype):
    """
    Return the shape-only index tensors for the periodic-image expansion.

    Args:
        num_atoms: Atom count.
        rep_max: Half-width of the image grid, so the grid is ``(2*rep_max+1)^3``.
        device: Device to allocate on.
        dtype: Floating dtype for the offsets, matching the cell.

    Returns:
        ``(image_offsets, source_atom, image_index, source_offset)``.
    """
    key = (num_atoms, rep_max, str(device), dtype)
    cached = _IMAGE_SCAFFOLD.get(key)
    if cached is None:
        axis = torch.arange(-rep_max, rep_max + 1, device=device, dtype=dtype)
        image_offsets = torch.cartesian_prod(axis, axis, axis)
        num_images = image_offsets.shape[0]
        # Source atoms are (atom, image) pairs laid out as atom * num_images + image.
        source_atom = torch.repeat_interleave(
            torch.arange(num_atoms, device=device), num_images
        )
        image_index = torch.arange(num_atoms * num_images, device=device) % num_images
        source_offset = image_offsets[image_index]
        cached = (image_offsets, source_atom, image_index, source_offset)
        _IMAGE_SCAFFOLD[key] = cached
    return cached


# The neighbour-cell offset grid, cached per span. Same argument as
# `_image_scaffold`: a pure function of `span`, so identical on every replay, yet
# `arange` + `cartesian_prod` were re-running each step on a handful of integers.
_NEIGHBOR_GRID: dict = {}


def _neighbor_grid(span, device):
    """
    Return the ``(2*span+1)^3`` integer neighbour-cell offsets.

    Args:
        span: Half-width of the neighbour-cell grid.
        device: Device to allocate on.

    Returns:
        An ``int64`` tensor of shape ``((2*span+1)**3, 3)``.
    """
    key = (span, str(device))
    grid = _NEIGHBOR_GRID.get(key)
    if grid is None:
        naxis = torch.arange(-span, span + 1, device=device, dtype=torch.int64)
        grid = torch.cartesian_prod(naxis, naxis, naxis)
        _NEIGHBOR_GRID[key] = grid
    return grid


# `repeat_interleave(arange(n), k)` is a pure function of (n, k) -- and it lowers to a
# cumsum plus a gather, so it is several launches for a value that never changes.
_REPEAT_ARANGE: dict = {}


def _repeat_arange(count, repeats, device):
    """
    Return ``arange(count).repeat_interleave(repeats)``, cached per shape.

    Args:
        count: Number of distinct values.
        repeats: How many times each value repeats.
        device: Device to allocate on.

    Returns:
        An ``int64`` tensor of length ``count * repeats``.
    """
    key = (count, repeats, str(device))
    out = _REPEAT_ARANGE.get(key)
    if out is None:
        out = torch.repeat_interleave(torch.arange(count, device=device), repeats)
        _REPEAT_ARANGE[key] = out
    return out


GB_BLOCK = 512


@triton.jit
def _source_prep_kernel(
    pos_ptr,
    cell_ptr,
    batch_ptr,
    sa_ptr,
    soff_ptr,
    rep_ptr,
    spos_ptr,
    scoord_ptr,
    ssys_ptr,
    sok_ptr,
    n,
    grid_resolution,
    BLOCK: tl.constexpr,
):
    """
    The whole periodic-image source pass for one (atom, image) pair per lane.

    Reproduces, term for term:
    ``source_system = batch[source_atom]``, ``shift = source_offset @ cell[sys]``,
    ``source_pos = pos[source_atom] + shift``,
    ``source_cell_coord = floor(source_pos / grid_resolution).to(int64)`` and
    ``source_ok = all(|source_offset| <= rep[sys])`` -- the last being exactly
    ``image_ok[source_system, image_index]``, since ``source_offset`` *is*
    ``image_offsets[image_index]``.

    The division is written as a division, not a multiply by a reciprocal, so the
    floor lands on the same cell as the eager chain.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(sa_ptr + i, mask=m, other=0)
    sysid = tl.load(batch_ptr + a, mask=m, other=0)
    o0 = tl.load(soff_ptr + i * 3 + 0, mask=m, other=0.0)
    o1 = tl.load(soff_ptr + i * 3 + 1, mask=m, other=0.0)
    o2 = tl.load(soff_ptr + i * 3 + 2, mask=m, other=0.0)
    cb = cell_ptr + sysid * 9
    for j in tl.static_range(3):
        c0 = tl.load(cb + 0 * 3 + j, mask=m, other=0.0)
        c1 = tl.load(cb + 1 * 3 + j, mask=m, other=0.0)
        c2 = tl.load(cb + 2 * 3 + j, mask=m, other=0.0)
        p = tl.load(pos_ptr + a * 3 + j, mask=m, other=0.0)
        v = p + (o0 * c0 + o1 * c1 + o2 * c2)
        tl.store(spos_ptr + i * 3 + j, v, mask=m)
        tl.store(
            scoord_ptr + i * 3 + j,
            tl.floor(v / grid_resolution).to(tl.int64),
            mask=m,
        )
    r0 = tl.load(rep_ptr + sysid * 3 + 0, mask=m, other=0.0)
    r1 = tl.load(rep_ptr + sysid * 3 + 1, mask=m, other=0.0)
    r2 = tl.load(rep_ptr + sysid * 3 + 2, mask=m, other=0.0)
    ok = (tl.abs(o0) <= r0) & (tl.abs(o1) <= r1) & (tl.abs(o2) <= r2)
    tl.store(ssys_ptr + i, sysid, mask=m)
    tl.store(sok_ptr + i, ok, mask=m)


@triton_op(
    "uma_nitro::source_prep",
    mutates_args={"source_pos", "source_cell_coord", "source_system", "source_ok"},
)
def _source_prep(
    pos: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
    source_atom: torch.Tensor,
    source_offset: torch.Tensor,
    rep: torch.Tensor,
    source_pos: torch.Tensor,
    source_cell_coord: torch.Tensor,
    source_system: torch.Tensor,
    source_ok: torch.Tensor,
    grid_resolution: float,
) -> None:
    """
    Launch the fused periodic-image source pass.

    Args:
        pos: Positions, ``(N, 3)``.
        cell: Unit cells, ``(B, 3, 3)``.
        batch: System index per atom, ``(N,)``.
        source_atom: Atom index per (atom, image) pair, ``(M,)``.
        source_offset: Image offset per pair, ``(M, 3)``.
        rep: Per-system image counts, ``(B, 3)``.
        source_pos: Output shifted positions, ``(M, 3)``; mutated.
        source_cell_coord: Output grid coordinates, ``(M, 3)`` int64; mutated.
        source_system: Output system index, ``(M,)`` int64; mutated.
        source_ok: Output validity flag, ``(M,)`` bool; mutated.
        grid_resolution: Bucket edge length.
    """
    n = source_atom.shape[0]
    wrap_triton(_source_prep_kernel)[(triton.cdiv(n, GB_BLOCK),)](
        pos,
        cell,
        batch,
        source_atom,
        source_offset,
        rep,
        source_pos,
        source_cell_coord,
        source_system,
        source_ok,
        n,
        grid_resolution,
        BLOCK=GB_BLOCK,
        num_warps=4,
    )


@triton.jit
def _table_init_kernel(flat_ptr, counter_ptr, n_flat, n_counter, BLOCK: tl.constexpr):
    """
    ``flat.fill_(-1)`` and ``counter.zero_()`` in one launch.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(flat_ptr + i, tl.full((BLOCK,), -1, tl.int64), mask=i < n_flat)
    tl.store(counter_ptr + i, tl.zeros((BLOCK,), tl.int32), mask=i < n_counter)


@triton_op("uma_nitro::table_init", mutates_args={"flat", "counter"})
def table_init(flat: torch.Tensor, counter: torch.Tensor) -> None:
    """
    Initialize the bucket table and its occupancy counters.

    Args:
        flat: Table storage, ``((num_buckets + 1) * cell_capacity,)`` int64.
        counter: Per-bucket occupancy, ``(num_buckets + 1,)`` int32.
    """
    n = flat.shape[0]
    wrap_triton(_table_init_kernel)[(triton.cdiv(n, 1024),)](
        flat, counter, n, counter.shape[0], BLOCK=1024, num_warps=4
    )


@triton.jit
def _count_atom_kernel(keep_ptr, cnt_ptr, row_len, BLOCK: tl.constexpr):
    """
    Count retained graph candidates for each atom.
    """
    a = tl.program_id(0)
    base = a.to(tl.int64) * row_len
    off = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.int64)
    j = 0
    while j < row_len:
        idx = j + off
        m = idx < row_len
        k = tl.load(keep_ptr + base + idx, mask=m, other=0)
        acc += tl.where(m & (k != 0), 1, 0).to(tl.int64)
        j += BLOCK
    tl.store(cnt_ptr + a, tl.sum(acc, axis=0))


@triton_op("uma_nitro::count_atom", mutates_args={"cnt"})
def count_atom(keep: torch.Tensor, cnt: torch.Tensor, row_len: int) -> None:
    """
    Count kept candidates per atom.

    Args:
        keep: Boolean keep flag per candidate, ``(N * row_len,)``.
        cnt: Output count per atom, ``(N,)`` int64; mutated.
        row_len: Candidates per atom.
    """
    wrap_triton(_count_atom_kernel)[(cnt.shape[0],)](
        keep, cnt, row_len, BLOCK=1024, num_warps=4
    )


@triton.jit
def _compact_base_kernel(
    keep_ptr, base_ptr, gather_ptr, row_len, cap, BLOCK: tl.constexpr
):
    """
    Invert the compaction rank from a per-atom base, without sorting.

    Every kept candidate gets its own slot, so this is a masked scatter with no
    duplicate targets, no atomics and no radix passes. Slots at or past the live
    count keep the zero fill; index 0 is always a valid candidate and those edges
    are dead, which is what ``node_offsets`` bounds.
    """
    a = tl.program_id(0)
    rbase = a.to(tl.int64) * row_len
    gbase = tl.load(base_ptr + a)
    off = tl.arange(0, BLOCK)
    carry = tl.zeros((BLOCK,), dtype=tl.int64)
    j = 0
    while j < row_len:
        idx = j + off
        m = idx < row_len
        k = m & (tl.load(keep_ptr + rbase + idx, mask=m, other=0) != 0)
        kk = tl.where(k, 1, 0).to(tl.int64)
        p = gbase + (tl.cumsum(kk, axis=0) - kk) + carry
        tl.store(gather_ptr + p, rbase + idx, mask=k & (p < cap))
        carry = carry + tl.sum(kk, axis=0)
        j += BLOCK


@triton_op("uma_nitro::compact_base", mutates_args={"gather"})
def compact_base(
    keep: torch.Tensor,
    base: torch.Tensor,
    gather: torch.Tensor,
    row_len: int,
    cap: int,
) -> None:
    """
    Scatter kept candidate indices into their compacted slots.

    Args:
        keep: Boolean keep flag per candidate, ``(N * row_len,)``.
        base: Exclusive prefix sum of the per-atom counts, ``(N,)``.
        gather: Result ``(edge_capacity,)``, pre-filled with zeros; mutated.
        row_len: Candidates per atom.
        cap: Edge capacity.
    """
    wrap_triton(_compact_base_kernel)[(base.shape[0],)](
        keep, base, gather, row_len, cap, BLOCK=1024, num_warps=4
    )


@triton.jit
def _edge_emit_kernel(
    gather_ptr,
    safe_ptr,
    sa_ptr,
    tgt_ptr,
    soff_ptr,
    ei_ptr,
    scell_ptr,
    cap,
    BLOCK: tl.constexpr,
):
    """
    ``stack([source_atom[safe[gather]], target_index[gather]])`` and
    ``source_offset[safe[gather]]`` in one launch instead of five.
    """
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < cap
    g = tl.load(gather_ptr + e, mask=m, other=0)
    s = tl.load(safe_ptr + g, mask=m, other=0)
    tl.store(ei_ptr + e, tl.load(sa_ptr + s, mask=m, other=0), mask=m)
    tl.store(ei_ptr + cap + e, tl.load(tgt_ptr + g, mask=m, other=0), mask=m)
    for k in tl.static_range(3):
        tl.store(
            scell_ptr + e * 3 + k,
            tl.load(soff_ptr + s * 3 + k, mask=m, other=0.0),
            mask=m,
        )


@triton_op("uma_nitro::edge_emit", mutates_args={"edge_index", "source_cell"})
def edge_emit(
    gather: torch.Tensor,
    safe: torch.Tensor,
    source_atom: torch.Tensor,
    target_index: torch.Tensor,
    source_offset: torch.Tensor,
    edge_index: torch.Tensor,
    source_cell: torch.Tensor,
) -> None:
    """
    Materialize the compacted edge list.

    Args:
        gather: Candidate index per output slot, ``(cap,)``.
        safe: Source-pair index per candidate.
        source_atom: Atom index per source pair.
        target_index: Target atom per candidate.
        source_offset: Image offset per source pair, ``(M, 3)``.
        edge_index: Output ``(2, cap)`` int64; mutated.
        source_cell: Output ``(cap, 3)``; mutated.
    """
    cap = gather.shape[0]
    wrap_triton(_edge_emit_kernel)[(triton.cdiv(cap, GB_BLOCK),)](
        gather,
        safe,
        source_atom,
        target_index,
        source_offset,
        edge_index,
        source_cell,
        cap,
        BLOCK=GB_BLOCK,
        num_warps=4,
    )


@triton.jit
def _graph_scan_kernel(
    cnt_ptr,
    batch_ptr,
    base_ptr,
    offsets_ptr,
    nedges_ptr,
    flag_ptr,
    num_atoms,
    edge_capacity,
    BLOCK: tl.constexpr,
):
    """
    The per-atom scan, the CSR row pointers, the per-system count and the capacity
    test, in one program.

    The eager tail was ``zeros`` + ``cumsum`` (two CUB launches over the candidate
    array) + ``index_add_`` plus a scalar ``add``/``gt``. One program carrying two
    running totals does all of it over ``num_atoms`` entries.

    Both the uncapped base (which the compaction scatter needs) and the capped
    degree (which is what the CSR offsets and the edge counts report) come out of
    the same scan: a candidate survives iff ``base[a] + local_rank < capacity``, so
    the capped degree is ``clamp(capacity - base[a], 0, cnt[a])`` exactly.
    """
    z = tl.arange(0, 1)
    off = tl.arange(0, BLOCK)
    tl.store(offsets_ptr + z, tl.zeros((1,), dtype=tl.int64))
    carry = tl.zeros((BLOCK,), dtype=tl.int64)
    ccarry = tl.zeros((BLOCK,), dtype=tl.int64)
    j = 0
    while j < num_atoms:
        idx = j + off
        m = idx < num_atoms
        v = tl.load(cnt_ptr + idx, mask=m, other=0)
        incl = tl.cumsum(v, axis=0) + carry
        excl = incl - v
        tl.store(base_ptr + idx, excl, mask=m)
        room = tl.maximum(tl.minimum(edge_capacity - excl, v), 0)
        tl.store(offsets_ptr + 1 + idx, tl.cumsum(room, axis=0) + ccarry, mask=m)
        b = tl.load(batch_ptr + idx, mask=m, other=0)
        tl.atomic_add(nedges_ptr + b, room, mask=m)
        carry = carry + tl.sum(v, axis=0)
        ccarry = ccarry + tl.sum(room, axis=0)
        j += BLOCK
    total = tl.max(carry, axis=0)
    tl.atomic_max(
        flag_ptr + z,
        tl.where(total > edge_capacity, 1, 0).to(tl.int64)
        + tl.zeros((1,), dtype=tl.int64),
    )


@triton_op("uma_nitro::graph_scan", mutates_args={"base", "offsets", "n_edges", "flag"})
def graph_scan(
    cnt: torch.Tensor,
    batch: torch.Tensor,
    base: torch.Tensor,
    offsets: torch.Tensor,
    n_edges: torch.Tensor,
    flag: torch.Tensor,
    edge_capacity: int,
) -> None:
    """
    Launch the per-atom scan / CSR / edge-count / overflow finalizer.

    Args:
        cnt: Uncapped kept-candidate count per atom, ``(N,)`` int64.
        batch: System index per atom, ``(N,)``.
        base: Output exclusive prefix sum of ``cnt``, ``(N,)`` int64; mutated.
        offsets: Output CSR row pointers, ``(N + 1,)`` int64; mutated.
        n_edges: Output per-system edge count, ``(B,)`` int64, pre-zeroed; mutated.
        flag: Overflow flag, ``(1,)`` int64; mutated.
        edge_capacity: Edge capacity.
    """
    wrap_triton(_graph_scan_kernel)[(1,)](
        cnt,
        batch,
        base,
        offsets,
        n_edges,
        flag,
        cnt.shape[0],
        edge_capacity,
        BLOCK=1024,
        num_warps=4,
    )


@triton.jit
def _edge_geom_fwd_kernel(
    pos_ptr,
    row_ptr,
    col_ptr,
    off_ptr,
    nlive_ptr,
    dv_ptr,
    dist_ptr,
    num_edges,
    radius,
    BLOCK_E: tl.constexpr,
):
    """
    ``where(live, pos[row] - pos[col] + offsets, e0 * radius)`` and its norm.

    Eleven launches in the eager form -- two gathers, a sub, an add, the ``where``
    with its ``basis.to`` and ``* radius``, the norm, and the ``arange``/``sum``/``lt``
    that built ``live``. The live mask is derived in-kernel from the CSR total, which
    is already on the device as ``node_offsets[-1]``.
    """
    e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    m = e < num_edges
    nlive = tl.load(nlive_ptr)
    lv = e.to(tl.int64) < nlive
    r = tl.load(row_ptr + e, mask=m, other=0)
    c = tl.load(col_ptr + e, mask=m, other=0)
    x = (
        tl.load(pos_ptr + r * 3 + 0, mask=m, other=0.0)
        - tl.load(pos_ptr + c * 3 + 0, mask=m, other=0.0)
    ) + tl.load(off_ptr + e * 3 + 0, mask=m, other=0.0)
    y = (
        tl.load(pos_ptr + r * 3 + 1, mask=m, other=0.0)
        - tl.load(pos_ptr + c * 3 + 1, mask=m, other=0.0)
    ) + tl.load(off_ptr + e * 3 + 1, mask=m, other=0.0)
    zz = (
        tl.load(pos_ptr + r * 3 + 2, mask=m, other=0.0)
        - tl.load(pos_ptr + c * 3 + 2, mask=m, other=0.0)
    ) + tl.load(off_ptr + e * 3 + 2, mask=m, other=0.0)
    x = tl.where(lv, x, radius)
    y = tl.where(lv, y, 0.0)
    zz = tl.where(lv, zz, 0.0)
    tl.store(dv_ptr + e * 3 + 0, x, mask=m)
    tl.store(dv_ptr + e * 3 + 1, y, mask=m)
    tl.store(dv_ptr + e * 3 + 2, zz, mask=m)
    tl.store(dist_ptr + e, tl.sqrt(x * x + y * y + zz * zz), mask=m)


@triton_op("uma_nitro::edge_geom_fwd", mutates_args={"dv", "dist"})
def _edge_geom_fwd_op(
    pos: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    offsets: torch.Tensor,
    n_live: torch.Tensor,
    dv: torch.Tensor,
    dist: torch.Tensor,
    radius: float,
) -> None:
    """
    Launch the fused edge-geometry forward.

    Args:
        pos: Positions, ``(N, 3)``.
        row: Source atom per edge, ``(E,)``.
        col: Target atom per edge, ``(E,)``.
        offsets: Periodic shift per edge, ``(E, 3)``.
        n_live: Scalar int64 live-edge count.
        dv: Output edge vectors, ``(E, 3)``; mutated.
        dist: Output edge lengths, ``(E,)``; mutated.
        radius: Cutoff, written into the dead rows.
    """
    edges = dist.shape[0]
    wrap_triton(_edge_geom_fwd_kernel)[(triton.cdiv(edges, EG_BLOCK),)](
        pos,
        row,
        col,
        offsets,
        n_live,
        dv,
        dist,
        edges,
        radius,
        BLOCK_E=EG_BLOCK,
        num_warps=EG_WARPS,
    )


def padded_radius_graph(
    pos,
    cell,
    natoms,
    batch,
    pbc,
    radius,
    rep_max: int = 1,
    cell_capacity: int = 32,
    edge_capacity: int = 160_000,
    return_offsets: bool = False,
    fused: bool = False,
):
    """
    Build a radius graph with fully static internal shapes.

    Matches ``baseline._radius_graph`` as a set of (source, target, offset)
    triples. Edge *order* differs, because compaction is by cumulative sum
    rather than ``masked_select``.

    Args:
        pos: Positions, shape ``(N, 3)``.
        cell: Unit cells, shape ``(B, 3, 3)``.
        natoms: Atoms per system, shape ``(B,)``.
        batch: System index per atom, shape ``(N,)``.
        pbc: Periodicity flags, shape ``(B, 3)``.
        radius: Cutoff radius, a host float.
        rep_max: Periodic image replication per axis.
        cell_capacity: Atoms retained per grid bucket.
        edge_capacity: Total edge slots.
        fused: Use the two-kernel candidate test in ``fused_build`` instead of the
            ~15-kernel eager chain. Same edge set and same CSR offsets (verified
            exactly); the win is distinct kernel count, not time.

    Returns:
        ``(edge_index, source_cell, n_edges, overflow)`` where ``edge_index`` is
        ``(2, edge_capacity)``, ``source_cell`` is ``(edge_capacity, 3)``,
        ``n_edges`` is ``(B,)``, and ``overflow`` is a bool tensor that is True
        if any capacity was exceeded. Slots beyond ``n_edges.sum()`` are unset.
    """
    device = pos.device
    num_atoms = pos.shape[0]
    batch_size = cell.shape[0]
    grid_resolution = radius / 1.99

    zbuf = torch.zeros(edge_capacity + batch_size + 1, dtype=torch.int64, device=device)
    gather = zbuf[:edge_capacity]
    n_edges = zbuf[edge_capacity : edge_capacity + batch_size]
    flag = zbuf[edge_capacity + batch_size :]
    rep = _replication(cell, pbc, radius, batch_size, flag, rep_max)

    # --- periodic images: fixed (2*rep_max+1)^3 grid, masked per system ------
    image_offsets, source_atom, image_index, source_offset = _image_scaffold(
        num_atoms, rep_max, device, cell.dtype
    )
    num_images = image_offsets.shape[0]
    num_source = source_atom.shape[0]
    source_pos = torch.empty(num_source, 3, device=device, dtype=pos.dtype)
    source_cell_coord = torch.empty(num_source, 3, device=device, dtype=torch.int64)
    source_system = torch.empty(num_source, device=device, dtype=torch.int64)
    source_ok = torch.empty(num_source, device=device, dtype=torch.bool)
    with torch.no_grad():
        torch.ops.uma_nitro.source_prep(
            pos,
            cell,
            batch,
            source_atom,
            source_offset,
            rep,
            source_pos,
            source_cell_coord,
            source_system,
            source_ok,
            grid_resolution,
        )

    # --- bin source atoms into hashed buckets --------------------------------
    num_buckets = 1 << max(4, (num_source * 4 - 1).bit_length())
    source_bucket = torch.empty(
        source_cell_coord.shape[0], dtype=torch.int64, device=device
    )
    torch.ops.uma_nitro.hash_cells_fused(
        source_cell_coord, source_system, source_bucket, num_buckets
    )

    # int32 keys: CUB radix-sorts one pass per 8-bit digit, so an int64 key costs
    # 8 passes where these need 3-4. source_bucket < num_buckets (2**24 at this
    # size) and rank <= edge_capacity, both far inside int32. The returned indices
    # are int64 either way, and only the ordering is used.
    # Counting sort instead of argsort + gather + searchsorted + fill: one atomic
    # pass assigns each atom a slot in its bucket. See bucket_fill.py.
    flat = torch.empty(
        (num_buckets + 1) * cell_capacity, dtype=torch.int64, device=device
    )
    counter = torch.empty(num_buckets + 1, dtype=torch.int32, device=device)
    torch.ops.uma_nitro.table_init(flat, counter)
    torch.ops.uma_nitro.bucket_fill_direct(
        source_bucket, source_ok, counter, flat, flag, cell_capacity, num_buckets
    )
    table = flat.view(num_buckets + 1, cell_capacity)

    if fused:
        raise NotImplementedError(
            "the fused candidate path is not supported by UMA Nitro"
        )
    # --- gather candidates for each target atom ------------------------------
    span = int(radius / grid_resolution) + 1
    neighbor_offsets = _neighbor_grid(span, device)
    num_neighbor_cells = neighbor_offsets.shape[0]

    n_triples = num_atoms * num_neighbor_cells * cell_capacity
    keep = torch.empty(n_triples, dtype=torch.bool, device=device)
    safe = torch.empty(n_triples, dtype=torch.int64, device=device)
    torch.ops.uma_nitro.fused_candidates(
        pos,
        batch,
        neighbor_offsets,
        table,
        source_cell_coord,
        source_pos,
        source_system,
        keep,
        safe,
        grid_resolution,
        radius,
    )
    target_index = _repeat_arange(num_atoms, num_neighbor_cells * cell_capacity, device)

    row_len = num_neighbor_cells * cell_capacity
    cnt = torch.empty(num_atoms, dtype=torch.int64, device=device)
    torch.ops.uma_nitro.count_atom(keep, cnt, row_len)
    base = torch.empty(num_atoms, dtype=torch.int64, device=device)
    offsets = torch.empty(num_atoms + 1, dtype=torch.int64, device=device)
    torch.ops.uma_nitro.graph_scan(
        cnt, batch, base, offsets, n_edges, flag, edge_capacity
    )
    torch.ops.uma_nitro.compact_base(keep, base, gather, row_len, edge_capacity)

    # `stack([source_atom[safe[gather]], target_index[gather]])` plus
    # `source_offset[safe[gather]]` as one kernel: the eager form built the
    # intermediate `safe_edges` and then ran four more advanced indexings and a
    # `cat` over it, five launches for the same gathers.
    edge_index = torch.empty(2, edge_capacity, dtype=torch.int64, device=device)
    source_cell_out = torch.empty(
        edge_capacity, 3, device=device, dtype=source_offset.dtype
    )
    torch.ops.uma_nitro.edge_emit(
        gather,
        safe,
        source_atom,
        target_index,
        source_offset,
        edge_index,
        source_cell_out,
    )

    # An integer flag rather than a bool: the three producers raise it with a
    # masked `atomic_max`, and the caller compares it against zero, which is the
    # same one launch the old `~overflow` cost.
    overflow = flag[0]

    if return_offsets:
        return edge_index, source_cell_out, n_edges, overflow, offsets
    return edge_index, source_cell_out, n_edges, overflow


# ``_compiled_padded`` and ``radius_graph_padded_compiled_entry`` are removed:
# both called ``torch.compile``, which this file does not use, and neither was
# reachable -- ``_build_graph`` passes ``graph_core=None``.


# ============================================================================
# Aliases the original modules established through `import ... as`. Kept so the
# merged bodies below read exactly as they did across files.
# ============================================================================
_fused_eq_norm = equivariant_rms_norm


COEFF_PAD = (
    int(os.environ["UMA_NITRO_COEFF_PAD"])
    if "UMA_NITRO_COEFF_PAD" in os.environ
    else None
)

# Launch config for the node-major conv2 inverse-rotate reduction, swept in-model.
SEG_GRID = int(os.environ.get("UMA_NITRO_SEG_GRID", "4096"))
SEG_WARPS = int(os.environ.get("UMA_NITRO_SEG_WARPS", "1"))
SEG_NSPLIT = int(os.environ.get("UMA_NITRO_SEG_NSPLIT", "1"))
STAGES = int(os.environ.get("UMA_NITRO_STAGES", "3"))
# Per-kernel launch knobs, exposed for coordinate-descent tuning (the Gauss-Seidel
# hill-climb Inductor uses: walk one field at a time applying each improvement
# immediately, x2/div2 neighbours for blocks and warps, +/-1 radius 2 for num_stages).
C1B_WARPS = int(os.environ.get("UMA_NITRO_C1B_WARPS", "1"))
C1B_STAGES = int(os.environ.get("UMA_NITRO_C1B_STAGES", "3"))
C1B_MTREE = int(os.environ.get("UMA_NITRO_C1B_MTREE", "1"))
C1B_SCATTER = bool(int(os.environ.get("UMA_NITRO_C1B_SCATTER", "1")))
# Bitmask selecting fused halves: 1 = source, 2 = target, 3 = both.
C1B_SCATTER_MODE = int(os.environ.get("UMA_NITRO_C1B_SCATTER_MODE", "3"))
# Use the inline-PTX ``red.global.add.v4.f32`` rather than ``tl.atomic_add`` for
# a reduction-only vector atomic.
C1B_RED = int(os.environ.get("UMA_NITRO_C1B_RED", "1"))
# Load every coefficient before storing any, in the forward gate. Only needed if the
# gate's output is to alias its input's row tail; see PRELOAD in _gate_kernel.
GATE_PRELOAD = int(os.environ.get("UMA_NITRO_GATE_PRELOAD", "0"))
# Alias the fp16 post-gate buffer into the dead tail of ``buf``'s own row, so
# GATE_IDENT costs no extra saved memory. Implies GATE_PRELOAD. See the comment at
# the allocation site.
GATE_PACK = os.environ.get("UMA_NITRO_GATE_PACK", "1") == "1" and GATE_IDENT


def _gpack(buf, layout):
    """
    The fp16 post-gate view of ``buf``'s row tail.

    Args:
        buf: The ``(E, row_stride)`` fp32 pre-gate buffer.
        layout: Its ``BlockLayout``.

    Returns:
        An ``(E, row_stride)`` fp16 view with row stride ``2 * row_stride``.
    """
    return buf.view(torch.float16).narrow(1, layout.row_stride, layout.row_stride)


def _scatter_mode():
    """
    Return the enabled source/target scatter bitmask.
    """
    return C1B_SCATTER_MODE if C1B_SCATTER else 0


C1B_CHUNK = int(os.environ.get("UMA_NITRO_C1B_CHUNK", "0"))
C1F_WARPS = int(os.environ.get("UMA_NITRO_C1F_WARPS", "1"))
C1F_STAGES = int(os.environ.get("UMA_NITRO_C1F_STAGES", "3"))
IC2_STAGES = int(os.environ.get("UMA_NITRO_IC2_STAGES", "3"))
GATE_WARPS = int(os.environ.get("UMA_NITRO_GATE_WARPS", "4"))
GATE_STAGES = int(os.environ.get("UMA_NITRO_GATE_STAGES", "3"))
SEG_ESPLIT = int(os.environ.get("UMA_NITRO_SEG_ESPLIT", "4"))
SEG_RED = int(os.environ.get("UMA_NITRO_SEG_RED", "1"))
SCT_ESPLIT = int(os.environ.get("UMA_NITRO_SCT_ESPLIT", "1"))
SCT_WARPS = int(os.environ.get("UMA_NITRO_SCT_WARPS", "4"))
SCT_NSPLIT = int(os.environ.get("UMA_NITRO_SCT_NSPLIT", "1"))
SCT_STAGES = int(os.environ.get("UMA_NITRO_SCT_STAGES", "3"))
NODE_CSPLIT = False
NODE_CSPLIT_MAX = int(os.environ.get("UMA_NITRO_NODE_CSPLIT_MAX", "512"))
SCT_CPG = int(os.environ.get("UMA_NITRO_SCT_CPG", "1"))
FED_CPG = int(os.environ.get("UMA_NITRO_FED_CPG", "1"))
FED_STAGES = int(os.environ.get("UMA_NITRO_FED_STAGES", "3"))
SEG_STAGES = int(os.environ.get("UMA_NITRO_SEG_STAGES", "3"))

# Append the edge-degree node reduction to the radial fork's edge-degree branch.
M0_FORK = os.environ.get("UMA_NITRO_M0_FORK", "1") == "1"

REGIME_TUNE = os.environ.get("UMA_NITRO_REGIME_TUNE", "1") == "1"

_REGIME_PINNED = {
    "SEG_ESPLIT": "UMA_NITRO_SEG_ESPLIT" in os.environ,
    "SCT_WARPS": "UMA_NITRO_SCT_WARPS" in os.environ,
    "FED_CPG": "UMA_NITRO_FED_CPG" in os.environ,
}

# Node-count threshold for coefficient splitting in the edge-degree reduction.
FED_CPG_MAX_NODES = int(os.environ.get("UMA_NITRO_FED_CPG_MAX_NODES", "1024"))


def _pow2_within(value, lo, hi):
    """
    Return the power of two in [lo, hi] nearest value.
    """
    best = lo
    out = lo
    while out <= hi:
        if abs(math.log2(out) - math.log2(max(value, 1e-9))) < abs(
            math.log2(best) - math.log2(max(value, 1e-9))
        ):
            best = out
        out *= 2
    return best


REGIME_FEW_NODES = int(os.environ.get("UMA_NITRO_REGIME_FEW_NODES", "256"))

# Degree threshold for reducing the edge split on sparse graphs.
REGIME_SPARSE_DEGREE = int(os.environ.get("UMA_NITRO_REGIME_SPARSE_DEG", "35"))

REGIME_EDGES_PER_SPLIT = float(os.environ.get("UMA_NITRO_REGIME_EPS", "3"))
REGIME_ESPLIT_MAX = int(os.environ.get("UMA_NITRO_REGIME_ESPLIT_MAX", "16"))
# Select the single-axis edge-split rule.
REGIME_ESPLIT_ONEAXIS = int(os.environ.get("UMA_NITRO_REGIME_ONEAXIS", "1"))


def _apply_regime(num_nodes, num_edges):
    """
    Set launch parameters for the current node and edge counts.

    The parameters are captured in a graph keyed by input shapes and capacities.
    """
    if not REGIME_TUNE:
        return
    degree = num_edges / max(1, num_nodes)
    warps = 2
    # One axis, not two: see REGIME_EDGES_PER_SPLIT. Only valid while SEG_RED is on;
    # with the fetching atomic the old two-branch rule is the fitted optimum.
    esplit = (
        _pow2_within(degree / REGIME_EDGES_PER_SPLIT, 2, REGIME_ESPLIT_MAX)
        if (SEG_RED and REGIME_ESPLIT_ONEAXIS)
        else (
            8
            if num_nodes <= REGIME_FEW_NODES
            else 2
            if degree <= REGIME_SPARSE_DEGREE
            else 4
        )
    )
    scope = globals()
    scope["NODE_CSPLIT"] = num_nodes <= NODE_CSPLIT_MAX
    for name, value in (
        ("SEG_ESPLIT", esplit),
        ("SCT_WARPS", warps),
        ("FED_CPG", 1 if num_nodes <= FED_CPG_MAX_NODES else 9),
    ):
        if not _REGIME_PINNED[name]:
            scope[name] = value


STREAM_FORK = os.environ.get("UMA_NITRO_STREAM_FORK", "1") == "1"

_SEQ_FORK = False
_MAX_FORK = [0]
_SIDE_STREAMS: list = []

# Side-stream outputs must be associated with the consuming stream before their
# Python references are released; see ``_record_uses``.


def _side_streams(count):
    """
    Return ``count`` cached side streams, creating them on first use.

    Cached rather than created per call because a fresh stream during capture would
    not be reusable on replay, and because stream creation is not free.

    Args:
        count: How many side streams are needed.

    Returns:
        A list of ``count`` streams.
    """
    while len(_SIDE_STREAMS) < count:
        _SIDE_STREAMS.append(torch.cuda.Stream())
    return _SIDE_STREAMS[:count]


def _parallel(thunks):
    """
    Run independent thunks concurrently on side streams and join.

    Only correct for thunks with no data dependence on each other. Each runs on its
    own stream, all of them ordered after the calling stream's current work and
    joined back before returning, so callers see ordinary sequential semantics.

    Outputs allocated inside a thunk belong to that thunk's stream as far as the
    caching allocator is concerned, while every consumer runs on ``main`` after the
    join. That is a cross-stream lifetime the allocator cannot see, so each one is
    declared with ``record_stream`` -- see ``_record_uses``, and the bug it fixes.

    Args:
        thunks: Callables to run. The first runs on the calling stream, so ``n``
            thunks need ``n - 1`` side streams.

    Returns:
        A list of each thunk's return value, in order.
    """
    if _SEQ_FORK and len(thunks) > 1:
        # Record the width so `_capture` can give exactly those streams their
        # cuBLAS workspace before capture begins, which is the only per-stream
        # state the warmup would otherwise have established.
        _MAX_FORK[0] = max(_MAX_FORK[0], len(thunks) - 1)
    if not STREAM_FORK or _SEQ_FORK or len(thunks) < 2:
        return [thunk() for thunk in thunks]
    main = torch.cuda.current_stream()
    streams = _side_streams(len(thunks) - 1)
    fork = torch.cuda.Event()
    fork.record(main)
    for stream in streams:
        stream.wait_event(fork)
    results = [None] * len(thunks)
    for index, thunk in enumerate(thunks):
        if index == 0:
            results[index] = thunk()
        else:
            with torch.cuda.stream(streams[index - 1]):
                results[index] = thunk()
    for stream in streams:
        join = torch.cuda.Event()
        join.record(stream)
        main.wait_event(join)
    # Thunk 0 ran on `main`, so only side-stream results need declaring.
    _record_uses(results[1:], main)
    return results


def _record_uses(values, stream):
    """
    Associate side-stream outputs with their consuming stream.

    This prevents allocator reuse before asynchronous consumers have finished.
    """
    for value in values:
        if isinstance(value, torch.Tensor):
            value.record_stream(stream)
        elif isinstance(value, (tuple, list)):
            _record_uses(value, stream)


EDGES_PER_ATOM = int(os.environ.get("UMA_NITRO_EDGES_PER_ATOM", "128"))

EDGE_BUCKET_BITS = int(os.environ.get("UMA_NITRO_EDGE_BUCKET_BITS", "8"))
EDGE_QUANTUM_MIN = int(os.environ.get("UMA_NITRO_EDGE_QUANTUM_MIN", "64"))

EDGE_MARGIN = float(os.environ.get("UMA_NITRO_EDGE_MARGIN", "1.30"))

_LIVE_EDGES: dict = {}

# Positions sampled when a system has no usable cell, so the density signature costs
# O(1) host work instead of a reduction over every atom of a large molecule.
GEOM_SAMPLES = 4096

GEOM_BUCKET = 1.005


def _composition_key(data):
    """
    Return an exact, order-independent composition signature.
    """
    z = data.atomic_numbers.reshape(-1)
    if z.dtype.is_floating_point:
        z = z.to(torch.int64)
    if not z.is_cpu:
        # `forward` is handed a CPU batch and stages it itself, so this
        # is the uncommon path -- but `.numpy()` raises on a device tensor where
        # the density estimate above would merely have synchronized, and a
        # confusing TypeError is a bad way to learn that. Two attribute reads on
        # the path that matters.
        z = z.cpu()
    return np.bincount(z.numpy()).tobytes()


def _geometry_key(data):
    """
    Return a host-side key for radius-graph capacity reuse.

    The key combines a quantized density estimate and an exact composition
    signature.
    """
    volume = 0.0
    cell = getattr(data, "cell", None)
    if cell is not None and cell.numel() >= 9:
        # Written out rather than torch.det: this is nine multiplies on host floats,
        # and it avoids pulling in a linalg backend for a 3x3.
        m = cell.reshape(-1, 3, 3)[0].tolist()
        volume = abs(
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
    periodic = bool(getattr(data, "pbc", None) is not None and data.pbc.any())
    if not periodic or volume <= 0.0:
        pos = data.pos
        step = max(1, pos.shape[0] // GEOM_SAMPLES)
        sample = pos[::step]
        extent = (sample.amax(0) - sample.amin(0)).clamp_min(1e-6)
        volume = float(extent.prod())
    density = data.pos.shape[0] / volume
    return (
        int(math.log(density) / math.log(GEOM_BUCKET)),
        _composition_key(data),
    )


# The graph builder writes excess edges into a discard row and raises this flag,
# so an under-capacity replay is memory-safe until the result is returned. Read the
# flag before returning, replace the graph, and retry instead of enqueueing a CUDA
# assertion that would poison the process. Value 2 remains a fatal numerical error.
_NONFINITE_OUTPUT_MSG = (
    "UMA Nitro: non-finite model output detected in energy, forces, or stress."
)
_CAPACITY_RECOVERY_RETRIES = 2


class _GraphCapacityOverflow(Exception):
    """
    Signal that graph capacity must be remeasured and replayed.
    """

    def __init__(self, cache_key: tuple, natoms: int) -> None:
        super().__init__(cache_key, natoms)
        self.cache_key = cache_key
        self.natoms = natoms


def _check_graph_flag(flag: torch.Tensor, cache_key: tuple, natoms: int) -> None:
    """
    Retry capacity overflow while keeping numerical failures fatal.
    """
    value = int(flag.item())
    if value == 1:
        raise _GraphCapacityOverflow(cache_key, natoms)
    if value != 0:
        raise RuntimeError(_NONFINITE_OUTPUT_MSG)


EVICT = os.environ.get("UMA_NITRO_EVICT", "1") == "1"
_EV_S = (
    "evict_first" if (EVICT and os.environ.get("UMA_NITRO_EVICT_S", "0") == "1") else ""
)
_EV_R_ON = EVICT and os.environ.get("UMA_NITRO_EVICT_R", "1") == "1"

EV_R_MIN_EDGES = int(os.environ.get("UMA_NITRO_EVICT_R_MIN", "32768"))


def _ev_r(edges: int) -> str:
    """
    The gather-side eviction policy for a launch over ``edges`` edges.

    Args:
        edges: Number of edges the kernel will visit.

    Returns:
        ``"evict_last"`` when the hint is enabled and the stream is large enough
        to be worth defending against, else ``""``.
    """
    return "evict_last" if (_EV_R_ON and edges >= EV_R_MIN_EDGES) else ""


EVICT2 = os.environ.get("UMA_NITRO_EVICT2", "0") == "1"
_EV_S2 = "evict_first" if EVICT2 else ""
_EV_R2 = "evict_last" if EVICT2 else ""

FINITE_GUARD = os.environ.get("UMA_NITRO_FINITE_GUARD", "1") == "1" and not _ABLATIONS


@triton.jit
def _finite_guard_kernel(
    f_ptr,
    e_ptr,
    s_ptr,
    flag_ptr,
    nf,
    ne,
    ns,
    BLOCK: tl.constexpr,
):
    """
    Raise the graph-build flag to 2 if any output entry is not finite.

    ``(v - v) != 0`` is true for both NaN and +/-inf and false for everything
    else, which is one subtract and one compare rather than two range tests.

    Args:
        f_ptr: Forces.
        e_ptr: Energy.
        s_ptr: Stress.
        flag_ptr: The capacity guard's int64 flag.
        nf: Number of force entries.
        ne: Number of energy entries.
        ns: Number of stress entries.
        BLOCK: Lanes per program.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(f_ptr + off, mask=off < nf, other=0.0)
    bad = (v - v) != 0.0
    if pid == 0:
        o2 = tl.arange(0, BLOCK)
        base = 0
        while base < ne:
            a = tl.load(e_ptr + base + o2, mask=base + o2 < ne, other=0.0)
            bad = bad | ((a - a) != 0.0)
            base += BLOCK
        base = 0
        while base < ns:
            b = tl.load(s_ptr + base + o2, mask=base + o2 < ns, other=0.0)
            bad = bad | ((b - b) != 0.0)
            base += BLOCK
    tl.atomic_max(flag_ptr + off * 0, tl.full((BLOCK,), 2, tl.int64), mask=bad)


@triton_op("uma_nitro::finite_guard", mutates_args={"flag"})
def _finite_guard(
    forces: torch.Tensor,
    energy: torch.Tensor,
    stress: torch.Tensor,
    flag: torch.Tensor,
) -> None:
    """
    Fold an output finiteness check into the capacity flag.

    Args:
        forces: ``(N, 3)`` fp32.
        energy: ``(B,)`` fp32.
        stress: ``(B, 3, 3)`` fp32, or a 1-element placeholder.
        flag: The int64 flag the capacity guard raises; mutated.
    """
    nf = forces.numel()
    blk = 1024
    wrap_triton(_finite_guard_kernel)[(max(1, triton.cdiv(nf, blk)),)](
        forces,
        energy,
        stress,
        flag,
        nf,
        energy.numel(),
        stress.numel(),
        BLOCK=blk,
        num_warps=4,
    )


_OVERFLOW_HELP = """
UMA Nitro radius-graph capacity overflow.

No incomplete result was returned. The cached capacity and captured graph were
discarded before remeasurement and retry. If overflows persist, increase
UMA_NITRO_EDGE_MARGIN. Increase UMA_NITRO_EDGES_PER_ATOM only when the initial probe
itself cannot find a sufficient capacity.
"""


def _capacity_cache_key(natoms, dataset_name, geom_key):
    """
    Return the key used by the measured-capacity cache.
    """
    return (natoms, dataset_name, geom_key)


def _capacity_high_water(
    backbone, natoms, dataset_name, geom_key, proposed, live_edges
):
    """
    Prevent graph capacity from shrinking within a stable model scope.
    """
    high_water = getattr(backbone, "_uma_nitro_capacity_high_water", None)
    if high_water is None:
        high_water = {}
        backbone._uma_nitro_capacity_high_water = high_water
    _density_bucket, composition = geom_key
    stable_key = (natoms, dataset_name, composition)
    previous = high_water.get(stable_key)
    if previous is None:
        selected = proposed.copy()
    else:
        selected = {
            "edge_capacity": (
                previous["edge_capacity"]
                if live_edges <= previous["edge_capacity"]
                else proposed["edge_capacity"]
            ),
            "cell_capacity": max(previous["cell_capacity"], proposed["cell_capacity"]),
            "rep_max": max(previous["rep_max"], proposed["rep_max"]),
        }
    high_water[stable_key] = selected.copy()
    return selected


def _measured_caps(backbone, tensors, dataset_name, geom_key):
    """
    Measure and cache graph capacities for an input geometry.

    The probe grows overflowing dimensions and records a bounded capacity. A
    model-local high-water mark prevents capacity shrinkage across changing cells.
    """
    natoms = tensors["pos"].shape[0]
    key = _capacity_cache_key(natoms, dataset_name, geom_key)
    cached = _LIVE_EDGES.get(key)
    if cached is None:
        # An overflowed probe reports a *truncated* edge count, so every capacity has
        # to fit before the count is trusted. The flag is a single bool covering all
        # three, so the retry grows all three rather than reading which one tripped;
        # over-provisioning during the probe costs nothing, since the probe's
        # capacities are discarded except for the ones that had to grow.
        per_atom, cell_cap, rep = EDGES_PER_ATOM, 24, 1
        for _ in range(8):
            probe = plan_capacities(natoms, rep_max=rep, edges_per_atom=per_atom)
            probe["cell_capacity"] = cell_cap
            with torch.no_grad():
                built = _graph_padded(
                    tensors["pos"],
                    tensors["cell"],
                    tensors["natoms"],
                    tensors["batch"],
                    tensors["pbc"],
                    backbone.cutoff,
                    probe,
                )
            if not bool(built[4]):
                break
            per_atom, cell_cap, rep = per_atom * 2, cell_cap * 2, rep + 1
        else:
            raise UMANitroNotSupportedError(
                f"UMA Nitro: probe build still overflowing at {per_atom} "
                f"edges/atom, cell_capacity {cell_cap}, rep_max {rep} for "
                f"{natoms} atoms.\n{_OVERFLOW_HELP}"
            )
        live = int(built[-1][-1])
        del built
        cached = (live, cell_cap, rep)
        _LIVE_EDGES[key] = cached
    live, cell_cap, rep = cached

    quantum = max(EDGE_QUANTUM_MIN, 1 << max(0, live.bit_length() - EDGE_BUCKET_BITS))
    want = max(int(live * EDGE_MARGIN), live + 1)
    caps = plan_capacities(natoms, rep_max=rep)
    caps["cell_capacity"] = cell_cap
    caps["edge_capacity"] = -(-want // quantum) * quantum
    caps = _capacity_high_water(backbone, natoms, dataset_name, geom_key, caps, live)
    if os.environ.get("UMA_NITRO_CAP_DEBUG"):
        print(
            f"[uma_nitro] natoms={natoms} live={live} cap={caps['edge_capacity']} "
            f"pad={caps['edge_capacity'] / live:.4f} q={quantum} "
            f"cell_cap={cell_cap} rep_max={rep}"
        )
    return caps


def _embedding(module: nn.Embedding, indices: torch.Tensor) -> torch.Tensor:
    return F.embedding(
        indices,
        module.weight,
        module.padding_idx,
        module.max_norm,
        module.norm_type,
        module.scale_grad_by_freq,
        module.sparse,
    )


def _linear(module: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    return F.linear(inputs, module.weight, module.bias)


def _sequential_warm(module: nn.Sequential, start: int = 0) -> None:
    """
    Populate ``_sequential``'s transposed-weight caches before a stream fork.

    ``_sequential`` builds ``layer._uma_nitro_wt`` lazily, on the first call that reaches
    the layer. Under CUDA graph capture the first call *is* the capture, so a
    ``_sequential`` invoked from inside a forked thunk would record its
    ``.t().contiguous()`` on a side stream while every subsequent replay reads that
    buffer from the main one -- a cross-stream read with no event ordering it. Warming
    here keeps the copy on the calling stream, ahead of the fork.

    Idempotent and free after the first call, so callers can invoke it unconditionally.

    Args:
        module: The ``nn.Sequential`` ``_sequential`` will be called on.
        start: First layer index that will be executed, matching ``_sequential``.
    """
    if not TN_INNER:
        return
    for layer in list(module._modules.values())[start:]:
        if isinstance(layer, nn.Linear) and not layer.weight.requires_grad:
            if getattr(layer, "_uma_nitro_wt", None) is None:
                layer._uma_nitro_wt = layer.weight.t().contiguous()


def _sequential(
    module: nn.Sequential,
    inputs: torch.Tensor,
    start: int = 0,
    stop: int | None = None,
) -> torch.Tensor:
    layers = list(module._modules.values())
    if stop is not None:
        layers = layers[:stop]
    i = start
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, nn.Linear):
            if TN_INNER and not layer.weight.requires_grad:
                wt = getattr(layer, "_uma_nitro_wt", None)
                if wt is None:
                    wt = layer.weight.t().contiguous()
                    layer._uma_nitro_wt = wt
                inputs = _LinearFrozenTN.apply(inputs, layer.weight, layer.bias, wt)
            else:
                inputs = _linear(layer, inputs)
        elif isinstance(layer, nn.LayerNorm):
            fuse = i + 1 < len(layers) and isinstance(layers[i + 1], nn.SiLU)
            if fuse and inputs.dim() == 2 and inputs.is_contiguous():
                inputs = ln_silu(inputs, layer.weight, layer.bias, layer.eps)
                i += 2
                continue
            inputs = F.layer_norm(
                inputs,
                layer.normalized_shape,
                layer.weight,
                layer.bias,
                layer.eps,
            )
        elif isinstance(layer, nn.SiLU):
            inputs = F.silu(inputs)
        else:
            raise TypeError(f"unsupported UMA Nitro layer: {type(layer)!r}")
        i += 1
    return inputs


def _graph_padded(pos, cell, natoms, batch, pbc, radius, caps):
    """
    Build edge distances from the fixed-capacity radius graph.

    Returns a graph of constant width ``edge_capacity`` regardless of how many
    edges actually exist, so no shape anywhere downstream depends on data and
    nothing has to be read back to the host.

    Padding slots are neutralized by giving them a distance of exactly
    ``radius``. The envelope is ``where(distance / cutoff < 1, ..., 0)``, so at
    exactly the cutoff it evaluates to zero -- and every edge-to-node path in
    the backbone is multiplied by ``wigner_inv_envelope``. That zeroes padded
    edges' contribution in the forward pass *and* their gradient in the
    backward pass, which matters because the fused Wigner op's backward
    scatters from edges back to nodes. No dummy node row is needed.

    Args:
        pos: Positions, shape ``(N, 3)``.
        cell: Unit cells, shape ``(B, 3, 3)``.
        natoms: Atoms per system, shape ``(B,)``.
        batch: System index per atom, shape ``(N,)``.
        pbc: Periodicity flags, shape ``(B, 3)``.
        radius: Cutoff radius.
        caps: Capacity keyword arguments from ``plan_capacities``.

    Returns:
        ``(edge_index, edge_distance, edge_distance_vec, n_edges, overflow)``,
        all of width ``edge_capacity``.
    """
    edge_index, cell_offsets, n_edges, overflow, node_offsets = padded_radius_graph(
        pos, cell, natoms, batch, pbc, radius, return_offsets=True, **caps
    )
    # Bundled so the CSR invariants are checkable in one place; set
    # UMA_NITRO_CHECK_CSR=1 to assert them. A violation is silent otherwise --
    # correct energy, wrong gradients.
    graph = EdgeCSR(edge_index, node_offsets, node_offsets[-1], cell_offsets, n_edges)
    row, col = graph.source, graph.target
    offsets = cell_offset_product(cell_offsets, batch, row, cell)

    distance_vectors, edge_distance = _EdgeGeom.apply(
        pos, row, col, offsets, node_offsets[-1], radius
    )
    return (
        edge_index,
        edge_distance,
        distance_vectors,
        n_edges,
        overflow,
        node_offsets,
    )


EG_BLOCK = int(os.environ.get("UMA_NITRO_EG_BLOCK", "256"))
EG_WARPS = int(os.environ.get("UMA_NITRO_EG_WARPS", "8"))


@triton.jit
def _edge_geom_bwd_kernel(
    gdv_ptr,
    gdist_ptr,
    dv_ptr,
    dist_ptr,
    row_ptr,
    col_ptr,
    nlive_ptr,
    gpos_ptr,
    goff_ptr,
    num_edges,
    HAS_GDV: tl.constexpr,
    HAS_GDIST: tl.constexpr,
    BLOCK_E: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    Adjoint of ``where(live, pos[row] - pos[col] + offsets, e0 * radius)`` and its norm.

    Reproduces the eleven-launch eager chain term by term:
    ``LinalgVectorNormBackward0`` is ``grad_dist * masked_fill(dv / dist, dist == 0, 0)``,
    ``WhereBackward0`` zeroes the dead rows, ``SubBackward0`` negates the ``col`` side,
    and each ``IndexSelectBackward0`` was a ``(N, 3)`` zeros plus an ``index_add_``. The
    two scatters land in one buffer here, which changes only the atomic order the
    ``index_add_`` pair already had.
    """
    base = tl.program_id(0).to(tl.int64) * BLOCK_E
    nlive = tl.load(nlive_ptr)
    while base < num_edges:
        e = base + tl.arange(0, BLOCK_E)
        m = e < num_edges
        d = tl.load(dist_ptr + e, mask=m, other=1.0)
        gd = tl.load(gdist_ptr + e, mask=m, other=0.0) if HAS_GDIST else 0.0
        keep = e.to(tl.int64) < nlive
        r = tl.load(row_ptr + e, mask=m, other=0)
        c = tl.load(col_ptr + e, mask=m, other=0)
        for k in tl.static_range(3):
            v = tl.load(dv_ptr + e * 3 + k, mask=m, other=0.0)
            g = tl.load(gdv_ptr + e * 3 + k, mask=m, other=0.0) if HAS_GDV else 0.0
            if HAS_GDIST:
                g += gd * tl.where(d == 0.0, 0.0, v / d)
            g = tl.where(keep, g, 0.0)
            tl.store(goff_ptr + e * 3 + k, g, mask=m)
            tl.atomic_add(gpos_ptr + r * 3 + k, g, mask=m)
            tl.atomic_add(gpos_ptr + c * 3 + k, -g, mask=m)
        base += GRID_S * BLOCK_E


@triton_op("uma_nitro::edge_geom_bwd", mutates_args={"grad_pos", "grad_off"})
def _edge_geom_bwd_op(
    gdv: torch.Tensor | None,
    gdist: torch.Tensor | None,
    dv: torch.Tensor,
    dist: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    n_live: torch.Tensor,
    grad_pos: torch.Tensor,
    grad_off: torch.Tensor,
) -> None:
    """
    Launch the fused edge-geometry adjoint.
    """
    edges = dist.shape[0]
    grid = min(triton.cdiv(edges, EG_BLOCK), 32768)
    wrap_triton(_edge_geom_bwd_kernel)[(grid,)](
        dv if gdv is None else gdv,
        dist if gdist is None else gdist,
        dv,
        dist,
        row,
        col,
        n_live,
        grad_pos,
        grad_off,
        edges,
        HAS_GDV=gdv is not None,
        HAS_GDIST=gdist is not None,
        BLOCK_E=EG_BLOCK,
        GRID_S=grid,
        num_warps=EG_WARPS,
    )


class _EdgeGeom(torch.autograd.Function):
    """
    Compute edge vectors and lengths with a custom adjoint.
    """

    @staticmethod
    def forward(ctx, pos, row, col, offsets, n_live, radius):
        """
        Args:
            pos: Positions, ``(N, 3)``.
            row: Source atom per edge, ``(E,)``.
            col: Target atom per edge, ``(E,)``.
            offsets: Periodic shift per edge, ``(E, 3)``.
            n_live: Scalar int64 live-edge count, i.e. ``node_offsets[-1]``.
            radius: Cutoff, used for the dead rows.

        Returns:
            ``(distance_vectors, edge_distance)``.
        """
        edges = row.shape[0]
        dv = torch.empty(edges, 3, device=pos.device, dtype=pos.dtype)
        dist = torch.empty(edges, device=pos.device, dtype=pos.dtype)
        torch.ops.uma_nitro.edge_geom_fwd(
            pos, row, col, offsets, n_live, dv, dist, radius
        )
        ctx.save_for_backward(dv, dist, row, col, n_live)
        ctx.natoms = pos.shape[0]
        ctx.set_materialize_grads(False)
        return dv, dist

    @staticmethod
    def backward(ctx, grad_dv, grad_dist):
        """
        Args:
            grad_dv: Gradient of the edge vectors, ``(E, 3)`` or ``None``.
            grad_dist: Gradient of the edge lengths, ``(E,)`` or ``None``.

        Returns:
            Gradients for ``pos`` and ``offsets``; ``None`` for the rest.
        """
        dv, dist, row, col, n_live = ctx.saved_tensors
        if grad_dv is None and grad_dist is None:
            return (None,) * 6
        grad_pos = torch.zeros((ctx.natoms, 3), device=dv.device, dtype=dv.dtype)
        grad_off = torch.empty_like(dv)
        torch.ops.uma_nitro.edge_geom_bwd(
            None if grad_dv is None else grad_dv.contiguous(),
            None if grad_dist is None else grad_dist.contiguous(),
            dv,
            dist,
            row,
            col,
            n_live,
            grad_pos,
            grad_off,
        )
        return grad_pos, None, None, grad_off, None, None


def _quaternion_wigner(edge_distance_vec, c_l2, env):
    """
    Compute Wigner tensors and the radial envelope in one fused operation.
    """
    if os.environ.get("UMA_NITRO_EAGER_WIGNER") == "1":
        w, wi = _quaternion_wigner_eager(edge_distance_vec, c_l2)
        return w, wi * env.reshape(-1, 1, 1)
    # `* (2 * pi)` rather than `* 2 * math.pi`: doubling is exact in binary, so
    # fp32(2*pi) == 2 * fp32(pi) and the two forms are bit-identical -- one launch
    # instead of two.
    gamma = torch.rand(
        edge_distance_vec.shape[0],
        dtype=edge_distance_vec.dtype,
        device=edge_distance_vec.device,
    ) * (2.0 * math.pi)
    return wigner_from_vec(
        edge_distance_vec,
        gamma,
        c_l2,
        torch.finfo(edge_distance_vec.dtype).eps,
        env,
    )


def _quaternion_wigner_eager(edge_distance_vec, c_l2):
    edge_normalized = F.normalize(edge_distance_vec, dim=-1)
    gamma = (
        torch.rand(
            edge_normalized.shape[0],
            dtype=edge_normalized.dtype,
            device=edge_normalized.device,
        )
        * 2
        * math.pi
    )
    ex = edge_normalized[..., 0]
    ey = edge_normalized[..., 1]
    ez = edge_normalized[..., 2]

    q_chart1 = torch.stack([1.0 + ey, -ez, torch.zeros_like(ex), ex], dim=-1)
    q_chart1 = q_chart1 / torch.sqrt(
        torch.clamp(
            torch.sum(q_chart1**2, dim=-1, keepdim=True),
            min=torch.finfo(ex.dtype).eps,
        )
    )
    q_chart2 = torch.stack([-ez, 1.0 - ey, ex, torch.zeros_like(ex)], dim=-1)
    q_chart2 = q_chart2 / torch.sqrt(
        torch.clamp(
            torch.sum(q_chart2**2, dim=-1, keepdim=True),
            min=torch.finfo(ex.dtype).eps,
        )
    )
    t_clamped = ((ey + 0.9) / 1.8).clamp(0, 1)
    eps = torch.finfo(ey.dtype).eps
    numerator = 2.0 * t_clamped - 1.0
    denominator = t_clamped * (1.0 - t_clamped)
    t_smooth = torch.sigmoid(numerator / denominator.clamp(min=eps))
    t_smooth = torch.where(t_clamped < eps, torch.zeros_like(t_smooth), t_smooth)
    t_smooth = torch.where(t_clamped > 1 - eps, torch.ones_like(t_smooth), t_smooth)
    dot = (q_chart2 * q_chart1).sum(dim=-1, keepdim=True)
    q_chart2 = torch.where(dot < 0, -q_chart2, q_chart2)
    q_edge_to_y = F.normalize(
        (1.0 - t_smooth.unsqueeze(-1)) * q_chart2 + t_smooth.unsqueeze(-1) * q_chart1,
        dim=-1,
    )

    half_gamma = gamma / 2
    q_gamma = torch.stack(
        [
            torch.cos(half_gamma),
            torch.zeros_like(gamma),
            torch.sin(half_gamma),
            torch.zeros_like(gamma),
        ],
        dim=-1,
    )
    w1, x1, y1, z1 = q_gamma.unbind(-1)
    w2, x2, y2, z2 = q_edge_to_y.unbind(-1)
    q = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    w, x, y, z = q.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rotation = torch.stack(
        [
            torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1),
        ],
        dim=-2,
    )
    q2 = q.unsqueeze(-1) * q.unsqueeze(-2)
    q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3)
    n_edges = q.shape[0]
    c_flat = c_l2.to(dtype=q.dtype, device=q.device).reshape(25, 256)
    d_l2 = (q4.reshape(n_edges, 256) @ c_flat.t()).reshape(n_edges, 5, 5)
    # Build the block-diagonal Wigner with one cat instead of zeros + three
    # slice-assignments, whose backwards are index_put/copy_ chains.
    wigner = torch.zeros(n_edges, 9, 9, dtype=q.dtype, device=q.device)
    wigner[:, 0, 0] = 1.0
    wigner[:, 1:4, 1:4] = rotation
    wigner[:, 4:9, 4:9] = d_l2
    return wigner, wigner.transpose(1, 2).contiguous()


def _equivariant_rms_norm(module, node_input, row0=None):
    """
    Apply the fused equivariant RMS norm.
    """
    return _fused_eq_norm(module, node_input, row0)


NG_GRID = int(os.environ.get("UMA_NITRO_NG_GRID", "4096"))
NG_WARPS = int(os.environ.get("UMA_NITRO_NG_WARPS", "4"))


@triton.jit
def _node_gate_fwd_kernel(
    x_ptr,
    gs_ptr,
    exp_ptr,
    out_ptr,
    num_nodes,
    P: tl.constexpr,
    C: tl.constexpr,
    LMAX: tl.constexpr,
    GRID: tl.constexpr,
    CSPL: tl.constexpr,
):
    """
    Apply scalar SiLU and vector sigmoid gates in one pass.
    """
    c = tl.arange(0, C)
    # One coefficient row per program. Every row is independent -- the scalar row gets
    # a SiLU, the rest a gate multiply -- so there is no reduction to preserve and the
    # split just raises the grid from ``N`` to ``N * P``.
    if CSPL == 1:
        node = tl.program_id(0)
        while node < num_nodes:
            xb = node * P * C
            gb = node * LMAX * C
            x0 = tl.load(x_ptr + xb + c)
            tl.store(out_ptr + xb + c, x0 * tl.sigmoid(x0))
            for i in tl.static_range(1, P):
                l = tl.load(exp_ptr + (i - 1))
                s = tl.sigmoid(tl.load(gs_ptr + gb + l * C + c))
                tl.store(out_ptr + xb + i * C + c, tl.load(x_ptr + xb + i * C + c) * s)
            node += GRID
    else:
        i = tl.program_id(0) % P
        node = tl.program_id(0) // P
        while node < num_nodes:
            xb = node * P * C
            gb = node * LMAX * C
            if i == 0:
                x0 = tl.load(x_ptr + xb + c)
                tl.store(out_ptr + xb + c, x0 * tl.sigmoid(x0))
            else:
                l = tl.load(exp_ptr + (i - 1))
                s = tl.sigmoid(tl.load(gs_ptr + gb + l * C + c))
                tl.store(out_ptr + xb + i * C + c, tl.load(x_ptr + xb + i * C + c) * s)
            node += GRID


@triton.jit
def _node_gate_bwd_kernel(
    dy_ptr,
    x_ptr,
    gs_ptr,
    exp_ptr,
    dx_ptr,
    dgs_ptr,
    num_nodes,
    dy_stride_n,
    dy_stride_p,
    P: tl.constexpr,
    C: tl.constexpr,
    LMAX: tl.constexpr,
    GRID: tl.constexpr,
):
    """
    Gradients of the node gate for both the features and the gating scalars.

    ``dgs[l] = s(1-s) * sum_{i : expand[i] = l} dy_i * x_i`` -- a reduction over the
    coefficient axis only, so it stays inside one program, and one program per node is
    all the parallelism this kernel can have. See the launcher for why a per-degree
    split is not taken.
    """
    c = tl.arange(0, C)
    node = tl.program_id(0)
    while node < num_nodes:
        xb = node * P * C
        gb = node * LMAX * C
        db = node * dy_stride_n + c
        x0 = tl.load(x_ptr + xb + c)
        dy0 = tl.load(dy_ptr + db)
        s0 = tl.sigmoid(x0)
        tl.store(dx_ptr + xb + c, dy0 * s0 * (1.0 + x0 * (1.0 - s0)))
        acc0 = tl.zeros((C,), dtype=tl.float32)
        acc1 = tl.zeros((C,), dtype=tl.float32)
        acc2 = tl.zeros((C,), dtype=tl.float32)
        acc3 = tl.zeros((C,), dtype=tl.float32)
        for i in tl.static_range(1, P):
            li = tl.load(exp_ptr + (i - 1))
            dyi = tl.load(dy_ptr + db + i * dy_stride_p)
            s = tl.sigmoid(tl.load(gs_ptr + gb + li * C + c))
            tl.store(dx_ptr + xb + i * C + c, dyi * s)
            prod = dyi * tl.load(x_ptr + xb + i * C + c)
            acc0 += tl.where(li == 0, prod, 0.0)
            if LMAX > 1:
                acc1 += tl.where(li == 1, prod, 0.0)
            if LMAX > 2:
                acc2 += tl.where(li == 2, prod, 0.0)
            if LMAX > 3:
                acc3 += tl.where(li == 3, prod, 0.0)
        for l in tl.static_range(LMAX):
            acc = acc0
            if l == 1:
                acc = acc1
            if l == 2:
                acc = acc2
            if l == 3:
                acc = acc3
            s = tl.sigmoid(tl.load(gs_ptr + gb + l * C + c))
            sg = s * (1.0 - s)
            tl.store(dgs_ptr + gb + l * C + c, acc * sg)
        node += GRID


@triton_op("uma_nitro::node_gate_fwd", mutates_args={"out"})
def _node_gate_fwd(
    x: torch.Tensor, gs: torch.Tensor, expand: torch.Tensor, out: torch.Tensor
) -> None:
    """
    Launch the fused node gate forward.

    Args:
        x: Node features, ``(N, P, C)``.
        gs: Raw gating scalars, ``(N, LMAX * C)``.
        expand: Degree index per non-scalar coefficient, ``(P - 1,)``.
        out: Output ``(N, P, C)``; mutated.
    """
    nodes, coeffs, channels = x.shape
    grid = min(nodes, NG_GRID)
    if "ngate" not in _ABLATIONS:
        _cs = coeffs if NODE_CSPLIT else 1
        wrap_triton(_node_gate_fwd_kernel)[(grid * _cs,)](
            x,
            gs,
            expand,
            out,
            nodes,
            P=coeffs,
            C=channels,
            LMAX=gs.numel() // (nodes * channels),
            GRID=grid,
            CSPL=_cs,
            num_warps=NG_WARPS,
        )


@triton_op("uma_nitro::node_gate_bwd", mutates_args={"dx", "dgs"})
def _node_gate_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    gs: torch.Tensor,
    expand: torch.Tensor,
    dx: torch.Tensor,
    dgs: torch.Tensor,
) -> None:
    """
    Launch the fused node gate backward.

    Args:
        dy: Upstream gradient, ``(N, P, C)``.
        x: Saved node features, ``(N, P, C)``.
        gs: Saved raw gating scalars, ``(N, LMAX * C)``.
        expand: Degree index per non-scalar coefficient, ``(P - 1,)``.
        dx: Output gradient for ``x``; mutated.
        dgs: Output gradient for ``gs``; mutated.
    """
    nodes, coeffs, channels = x.shape
    grid = min(nodes, NG_GRID)
    if "ngate" not in _ABLATIONS:
        wrap_triton(_node_gate_bwd_kernel)[(grid,)](
            dy,
            x,
            gs,
            expand,
            dx,
            dgs,
            nodes,
            dy.stride(0),
            dy.stride(1),
            P=coeffs,
            C=channels,
            LMAX=gs.numel() // (nodes * channels),
            GRID=grid,
            num_warps=NG_WARPS,
        )


class _NodeGate(torch.autograd.Function):
    """
    Autograd wrapper for the fused node gate.
    """

    @staticmethod
    def forward(ctx, gs, x, expand):
        """
        Args:
            gs: Raw gating scalars, ``(N, LMAX * C)`` or ``(N, 1, LMAX * C)``.
            x: Node features, ``(N, P, C)``.
            expand: Degree index per non-scalar coefficient.

        Returns:
            ``(N, P, C)``.
        """
        ctx.gs_shape = gs.shape
        gs = gs.reshape(x.shape[0], -1).contiguous()
        x = x.contiguous()
        out = torch.empty_like(x)
        torch.ops.uma_nitro.node_gate_fwd(x, gs, expand, out)
        ctx.save_for_backward(x, gs, expand)
        return out

    @staticmethod
    def backward(ctx, dy):
        """
        Args:
            dy: Upstream gradient, ``(N, P, C)``.

        Returns:
            Gradients for ``(gs, x, expand)``.
        """
        x, gs, expand = ctx.saved_tensors
        dx = torch.empty_like(x)
        dgs = torch.empty_like(gs)
        torch.ops.uma_nitro.node_gate_bwd(dy, x, gs, expand, dx, dgs)
        # Must match the input's shape, which may carry a squeezed middle axis.
        return dgs.reshape(ctx.gs_shape), dx, None


def _gate(module, gating_scalars, input_tensors):
    if os.environ.get("UMA_NITRO_NO_NODEGATE") == "1":
        gating_scalars = torch.sigmoid(gating_scalars).view(
            gating_scalars.shape[0], module.lmax, module.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=module.expand_index
        )
        scalars, vectors = input_tensors.split((1, input_tensors.shape[1] - 1), 1)
        return torch.cat((F.silu(scalars), vectors * gating_scalars), dim=1)
    return _NodeGate.apply(gating_scalars, input_tensors, module.expand_index)


def _so3_warm(module):
    """
    Populate ``_so3_linear``'s cached degree-expanded weight before a stream fork.

    The cache is written on first call, and under CUDA graph capture the first call is
    the capture itself. Populating it on the calling stream keeps the `index_select`
    that builds it out of a forked branch, where it would be recorded on a side stream
    and then read from the main one.

    Args:
        module: An SO3 linear module.
    """
    if getattr(module, "_uma_nitro_wexp", None) is None:
        weight = torch.index_select(module.weight, dim=0, index=module.expand_index)
        if not module.weight.requires_grad:
            module._uma_nitro_wexp = weight
    # The bias constant is built lazily too, and its `zeros` plus slice write would
    # otherwise be recorded on whichever stream first reaches `_so3_linear` -- a
    # forked one, under `_parallel`.
    _so3_bias9(module)


def _so3_linear(module, inputs, residual=None, pre_t=False):
    # The expansion maps (3, in, out) -> (9, in, out) by degree and the weight is
    # frozen at inference, so it is the same tensor every call: 8 index_selects per
    # iteration for a value that never changes.
    weight = getattr(module, "_uma_nitro_wexp", None)
    if weight is None:
        weight = torch.index_select(module.weight, dim=0, index=module.expand_index)
        if not module.weight.requires_grad:
            module._uma_nitro_wexp = weight
    inputs_t = inputs if pre_t else inputs.transpose(0, 1)
    out = torch.bmm(inputs_t, weight.transpose(1, 2))
    return _T01Bias.apply(out, _so3_bias9(module), residual)


def _so3_bias9(module):
    """
    Return the cached ``(1, P, out)`` bias constant, zero outside ``l = 0``.

    Replaces the in-place ``out[:, 0:1, :] = out.narrow(1, 0, 1) + bias``, which cost
    a CopySlices node -- clone the base, allocate zeros, two copies -- twice per
    layer. Adding exact zeros outside ``l = 0`` changes nothing.

    Args:
        module: An SO3 linear module.

    Returns:
        The cached bias, shape ``(1, P, out_features)``.
    """
    b9 = getattr(module, "_uma_nitro_bias9", None)
    if b9 is None:
        coeffs = module.expand_index.shape[0]
        b9 = torch.zeros(
            1,
            coeffs,
            module.out_features,
            device=module.bias.device,
            dtype=module.bias.dtype,
        )
        b9[0, 0] = module.bias.reshape(-1)
        module._uma_nitro_bias9 = b9
    return b9


@triton.jit
def _csd_cat_kernel(
    charge_ptr,
    spin_ptr,
    cw_ptr,
    sw_ptr,
    dw_ptr,
    out_ptr,
    c_off,
    s_off,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    ``cat((charge_emb, spin_emb, dataset_emb), dim=1)`` for one system, in one launch.

    The index arithmetic, the two ``.long()`` casts and all three table lookups are
    gathers, so this is bit-identical to the chain it replaces; only the ``cat`` is
    avoided, by writing each block straight into its column range.
    """
    d = tl.arange(0, BLOCK)
    ci = (tl.load(charge_ptr) + c_off).to(tl.int64)
    si = (tl.load(spin_ptr) + s_off).to(tl.int64)
    m1 = d < D1
    tl.store(out_ptr + d, tl.load(cw_ptr + ci * D1 + d, mask=m1, other=0.0), mask=m1)
    m2 = d < D2
    tl.store(
        out_ptr + D1 + d, tl.load(sw_ptr + si * D2 + d, mask=m2, other=0.0), mask=m2
    )
    m3 = d < D3
    tl.store(out_ptr + D1 + D2 + d, tl.load(dw_ptr + d, mask=m3, other=0.0), mask=m3)


@triton_op("uma_nitro::csd_cat", mutates_args={"out"})
def _csd_cat(
    charge: torch.Tensor,
    spin: torch.Tensor,
    charge_w: torch.Tensor,
    spin_w: torch.Tensor,
    dataset_w: torch.Tensor,
    out: torch.Tensor,
    charge_offset: int,
    spin_offset: int,
) -> None:
    """
    Build the concatenated charge/spin/dataset embedding for a single system.

    Args:
        charge: Total charge, ``(1,)``.
        spin: Spin multiplicity, ``(1,)``.
        charge_w: Charge embedding table.
        spin_w: Spin embedding table.
        dataset_w: The dataset embedding row, ``(D3,)``.
        out: Output ``(1, D1 + D2 + D3)``; mutated.
        charge_offset: Index offset for the charge table.
        spin_offset: Index offset for the spin table.
    """
    d1 = charge_w.shape[1]
    d2 = spin_w.shape[1]
    d3 = dataset_w.shape[0]
    wrap_triton(_csd_cat_kernel)[(1,)](
        charge,
        spin,
        charge_w,
        spin_w,
        dataset_w,
        out,
        charge_offset,
        spin_offset,
        D1=d1,
        D2=d2,
        D3=d3,
        BLOCK=triton.next_power_of_2(max(d1, d2, d3)),
        num_warps=4,
    )


@triton.jit
def _node_init_kernel(
    csd_ptr,
    batch_ptr,
    sw_ptr,
    z_ptr,
    sys_ptr,
    x_ptr,
    num_nodes,
    P: tl.constexpr,
    C: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    ``silu(csd)[batch]`` and the initial ``(N, P, C)`` node features, in one pass.

    The eager form is ``silu`` -> ``index_select`` -> ``zeros`` -> ``embedding`` ->
    two slice-assign ``copy_`` -- seven launches. ``silu`` is evaluated per node
    rather than per system, which is redundant arithmetic on ``C`` floats and
    returns exactly the same value, and lets the gather and the row-0 sum happen in
    the same registers.
    """
    c = tl.arange(0, C)
    n = tl.program_id(0)
    while n < num_nodes:
        b = tl.load(batch_ptr + n)
        v = tl.load(csd_ptr + b * C + c)
        # `x / (1 + exp(-x))`, which is what ATen's silu kernel computes, rather
        # than `x * sigmoid(x)`: the latter rounds twice and this value lands in
        # row 0 of every node and in every block's norm epilogue.
        sil = v / (1.0 + tl.exp(-v))
        tl.store(sys_ptr + n * C + c, sil)
        z = tl.load(z_ptr + n)
        base = n * (P * C)
        tl.store(x_ptr + base + c, tl.load(sw_ptr + z * C + c) + sil)
        for i in tl.static_range(1, P):
            tl.store(x_ptr + base + i * C + c, 0.0)
        n += GRID_S


@triton_op("uma_nitro::node_init", mutates_args={"sys_emb", "x_message"})
def _node_init(
    csd: torch.Tensor,
    batch: torch.Tensor,
    sphere_w: torch.Tensor,
    atomic_numbers: torch.Tensor,
    sys_emb: torch.Tensor,
    x_message: torch.Tensor,
) -> None:
    """
    Launch the fused per-node embedding init.

    Args:
        csd: Pre-activation per-system embedding, ``(B, C)``.
        batch: System index per atom, ``(N,)``.
        sphere_w: Atom-type embedding table.
        atomic_numbers: Atomic number per atom, ``(N,)`` int64.
        sys_emb: Output ``silu(csd)[batch]``, ``(N, C)``; mutated.
        x_message: Output node features ``(N, P, C)``; mutated in full.
    """
    nodes, coeffs, channels = x_message.shape
    grid = min(nodes, 8192)
    wrap_triton(_node_init_kernel)[(grid,)](
        csd,
        batch,
        sphere_w,
        atomic_numbers,
        sys_emb,
        x_message,
        nodes,
        P=coeffs,
        C=channels,
        GRID_S=grid,
        num_warps=4,
    )


@triton.jit
def _t01_bias_kernel(
    y_ptr,
    b_ptr,
    r_ptr,
    out_ptr,
    num_nodes,
    HAS_RES: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    GRID_S: tl.constexpr,
    CSPL: tl.constexpr,
):
    """
    ``y.transpose(0, 1) + bias`` (plus an optional residual), one *coefficient row*
    per program.

    One program per node is 0.76 waves at 100 atoms and the kernel is then floored by
    a single memory latency. There is no reduction here at all -- every output element
    is an independent two- or three-term add -- so splitting the coefficient axis is
    bit-identical and multiplies the program count ninefold while keeping every load a
    full ``C``-wide row.
    """
    c = tl.arange(0, C)
    if CSPL == 1:
        n = tl.program_id(0)
        while n < num_nodes:
            base = n * (P * C)
            for i in tl.static_range(P):
                v = tl.load(y_ptr + (i * num_nodes + n) * C + c) + tl.load(
                    b_ptr + i * C + c
                )
                if HAS_RES:
                    v += tl.load(r_ptr + base + i * C + c)
                tl.store(out_ptr + base + i * C + c, v)
            n += GRID_S
    else:
        i = tl.program_id(0) % P
        n = tl.program_id(0) // P
        while n < num_nodes:
            v = tl.load(y_ptr + (i * num_nodes + n) * C + c) + tl.load(
                b_ptr + i * C + c
            )
            if HAS_RES:
                v += tl.load(r_ptr + n * (P * C) + i * C + c)
            tl.store(out_ptr + n * (P * C) + i * C + c, v)
            n += GRID_S


@triton_op("uma_nitro::t01_bias", mutates_args={"out"})
def _t01_bias(
    y: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    has_res: bool,
) -> None:
    """
    Launch the transpose-plus-bias epilogue for a degree-batched SO3 linear.

    Args:
        y: The bmm output, ``(P, N, C)`` contiguous.
        bias: The degree-expanded bias, ``(1, P, C)``.
        residual: Residual to add, ``(N, P, C)``; ignored when ``has_res`` is False.
        out: Output ``(N, P, C)``; mutated.
        has_res: Whether to add the residual.
    """
    coeffs, nodes, channels = y.shape
    grid = min(nodes, 8192)
    if "t01" not in _ABLATIONS:
        _cs = coeffs if NODE_CSPLIT else 1
        wrap_triton(_t01_bias_kernel)[(grid * _cs,)](
            y,
            bias,
            residual,
            out,
            nodes,
            HAS_RES=has_res,
            P=coeffs,
            C=channels,
            GRID_S=grid,
            CSPL=_cs,
            num_warps=4,
        )


class _T01Bias(torch.autograd.Function):
    """
    ``bmm_out.transpose(0, 1).contiguous() + bias`` with an optional residual.

    The transposing copy and the bias add were two launches per SO3 linear, eight
    per iteration, and the residual add another four. Fusing them costs nothing in
    the backward: ``d/d(bmm_out)`` is the same transposed *view* that
    ``TransposeBackward`` already produced, and ``d/d(residual)`` is the identity,
    so no kernel runs on the way back. The bias is a frozen cached constant.
    """

    @staticmethod
    def forward(ctx, y, bias, residual):
        """
        Args:
            y: The bmm output, ``(P, N, C)``.
            bias: Degree-expanded bias, ``(1, P, C)``.
            residual: Optional ``(N, P, C)`` addend.

        Returns:
            ``(N, P, C)``.
        """
        coeffs, nodes, channels = y.shape
        out = torch.empty(nodes, coeffs, channels, device=y.device, dtype=y.dtype)
        torch.ops.uma_nitro.t01_bias(
            y, bias, y if residual is None else residual, out, residual is not None
        )
        ctx.has_res = residual is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: ``(N, P, C)``.

        Returns:
            The transposed view for ``y``, ``None`` for the bias, and the same
            gradient again for the residual.
        """
        return (
            grad_out.transpose(0, 1),
            None,
            grad_out if ctx.has_res else None,
        )


@triton.jit
def _row0_split_bwd_kernel(
    gt_ptr,
    g0_ptr,
    out_ptr,
    num_nodes,
    HAS_G0: tl.constexpr,
    HAS_GT: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    ``g_t.transpose(0, 1)`` with ``g_row0`` scattered into row 0, in one pass.
    """
    c = tl.arange(0, C)
    n = tl.program_id(0)
    while n < num_nodes:
        base = n * (P * C)
        for i in tl.static_range(P):
            if HAS_GT:
                v = tl.load(gt_ptr + (i * num_nodes + n) * C + c)
            else:
                v = tl.zeros((C,), dtype=out_ptr.dtype.element_ty)
            if i == 0:
                if HAS_G0:
                    v += tl.load(g0_ptr + n * C + c)
            tl.store(out_ptr + base + i * C + c, v)
        n += GRID_S


@triton_op("uma_nitro::row0_split_bwd", mutates_args={"out"})
def _row0_split_bwd(
    gt: torch.Tensor,
    g0: torch.Tensor,
    out: torch.Tensor,
    has_gt: bool,
    has_g0: bool,
) -> None:
    """
    Launch the joined row-0 / transposed-view adjoint.

    Args:
        gt: Gradient of the ``(P, N, C)`` view.
        g0: Gradient of the ``(N, C)`` row-0 view.
        out: Output ``(N, P, C)``; mutated in full.
        has_gt: Whether ``gt`` is a real gradient.
        has_g0: Whether ``g0`` is a real gradient.
    """
    nodes, coeffs, channels = out.shape
    grid = min(nodes, 8192)
    wrap_triton(_row0_split_bwd_kernel)[(grid,)](
        gt,
        g0,
        out,
        nodes,
        HAS_G0=has_g0,
        HAS_GT=has_gt,
        P=coeffs,
        C=channels,
        GRID_S=grid,
        num_warps=4,
    )


class _Row0Split(torch.autograd.Function):
    """
    Hand a node tensor to its two consumers as views, and join their gradients.

    ``_spectral_atomwise`` reads ``inputs`` twice: the scalar MLP takes the ``l = 0``
    row and the SO3 linear takes the degree-major transpose. As two independent
    reads that costs **two** launches on the way back -- ``_row0_bwd`` writing a
    dense ``(N, P, C)`` that is zero outside row 0, and then an ``add`` folding it
    into the bmm's gradient. Owning both views in one node makes the join a single
    pass: the transposed gradient is read and row 0 gets the scalar branch added
    while it is still in registers.

    Both outputs are views, so the forward is free.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Args:
            x: Node features, ``(N, P, C)``.

        Returns:
            ``(x[:, 0, :], x.transpose(0, 1))``, both views.
        """
        ctx.shape = x.shape
        ctx.set_materialize_grads(False)
        return x.select(1, 0), x.transpose(0, 1)

    @staticmethod
    def backward(ctx, g_row0, g_t):
        """
        Args:
            g_row0: Gradient of the ``(N, C)`` row, or ``None``.
            g_t: Gradient of the ``(P, N, C)`` view, or ``None``.

        Returns:
            One ``(N, P, C)`` gradient.
        """
        if g_row0 is None and g_t is None:
            return None
        ref = g_t if g_t is not None else g_row0
        out = torch.empty(ctx.shape, device=ref.device, dtype=ref.dtype)
        torch.ops.uma_nitro.row0_split_bwd(
            out if g_t is None else g_t.contiguous(),
            out if g_row0 is None else g_row0.contiguous(),
            out,
            g_t is not None,
            g_row0 is not None,
        )
        return out


def _spectral_atomwise(module, inputs, residual=None):
    _so3_warm(module.so3_linear_1)
    _sequential_warm(module.scalar_mlp)
    # One node owns both reads of `inputs`, so their gradients join in a single
    # kernel instead of `_row0_bwd` plus an `add`.
    row0, inputs_t = _Row0Split.apply(inputs)
    gating_scalars, inputs = _parallel(
        [
            lambda: _sequential(module.scalar_mlp, row0),
            lambda: _so3_linear(module.so3_linear_1, inputs_t, pre_t=True),
        ]
    )
    inputs = _gate(module.act, gating_scalars, inputs)
    return _so3_linear(module.so3_linear_2, inputs, residual)


def _half_transposed(conv1, conv2):
    """
    Cache fp16 ``(in, out)`` copies of conv1's three weights.

    The backward GEMMs are ``A @ Wt.t()`` with ``A`` in fp16, so the weight must
    match; a static fp16 copy costs nothing at runtime.

    Args:
        conv1: The conv1 SO2 module.
        conv2: The conv2 SO2 module.

    Returns:
        Six fp16 transposed weights, conv1's then conv2's.
    """
    out = []
    for mod, attr in (
        (conv1.fc_m0, "weight"),
        (conv1.so2_m_conv[0], "_w_block"),
        (conv1.so2_m_conv[1], "_w_block"),
        (conv2.fc_m0, "weight"),
        (conv2.so2_m_conv[0], "_w_block"),
        (conv2.so2_m_conv[1], "_w_block"),
    ):
        cached = getattr(mod, "_uma_nitro_wht", None)
        if cached is None:
            cached = getattr(mod, attr).t().contiguous().half().contiguous()
            mod._uma_nitro_wht = cached
        out.append(cached)
    return out


def _half_weights(conv1, conv2):
    """
    Cache fp16 copies of the six SO2 weights.
    """
    out = []
    for mod, attr in (
        (conv1.fc_m0, "weight"),
        (conv1.so2_m_conv[0], "_w_block"),
        (conv1.so2_m_conv[1], "_w_block"),
        (conv2.fc_m0, "weight"),
        (conv2.so2_m_conv[0], "_w_block"),
        (conv2.so2_m_conv[1], "_w_block"),
    ):
        cached = getattr(mod, "_uma_nitro_wh", None)
        if cached is None:
            cached = getattr(mod, attr).half().contiguous()
            mod._uma_nitro_wh = cached
        out.append(cached)
    return out


def _transposed_weights(conv1, conv2):
    """
    Cache ``(in, out)`` copies of the six SO2 weights, for the backward GEMMs.

    Args:
        conv1: The conv1 SO2 module.
        conv2: The conv2 SO2 module.

    Returns:
        Six transposed weights in the order the span op expects.
    """
    out = []
    for mod, attr in (
        (conv1.fc_m0, "weight"),
        (conv1.so2_m_conv[0], "_w_block"),
        (conv1.so2_m_conv[1], "_w_block"),
        (conv2.fc_m0, "weight"),
        (conv2.so2_m_conv[0], "_w_block"),
        (conv2.so2_m_conv[1], "_w_block"),
    ):
        cached = getattr(mod, "_uma_nitro_wt", None)
        if cached is None:
            cached = getattr(mod, attr).t().contiguous()
            mod._uma_nitro_wt = cached
        out.append(cached)
    return out


@triton.jit
def _sum5_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    out_ptr,
    n,
    N: tl.constexpr,
    BLK: tl.constexpr,
):
    """
    Sum up to five equally-shaped flat tensors in one pass.
    """
    off = (tl.program_id(0) * BLK + tl.arange(0, BLK)).to(tl.int64)
    m = off < n
    acc = tl.load(a_ptr + off, mask=m, other=0.0) + tl.load(
        b_ptr + off, mask=m, other=0.0
    )
    if N > 2:
        acc += tl.load(c_ptr + off, mask=m, other=0.0)
    if N > 3:
        acc += tl.load(d_ptr + off, mask=m, other=0.0)
    if N > 4:
        acc += tl.load(e_ptr + off, mask=m, other=0.0)
    tl.store(out_ptr + off, acc, mask=m)


@triton_op("uma_nitro::sum5", mutates_args={"out"})
def _sum5(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    out: torch.Tensor,
    count: int,
) -> None:
    """
    Launch the fused n-way sum.

    Args:
        a, b, c, d, e: Inputs; entries past ``count`` are ignored (pass ``a``).
        out: Output; mutated.
        count: How many of the inputs are live, 2 to 5.
    """
    n = out.numel()
    wrap_triton(_sum5_kernel)[(triton.cdiv(n, S5_BLK),)](
        a,
        b,
        c,
        d,
        e,
        out,
        n,
        N=count,
        BLK=S5_BLK,
        num_warps=S5_WARPS,
    )


class _Fanout(torch.autograd.Function):
    """
    Return tensor aliases and combine their gradients in one kernel.
    """

    @staticmethod
    def forward(ctx, t, n):
        """
        Args:
            t: The tensor to fan out.
            n: Number of consumers, at most 5.

        Returns:
            A tuple of ``n`` aliases.
        """
        assert 2 <= n <= 5, "sum5 handles 2..5 contributions"
        ctx.n = n
        return tuple(t.view_as(t) for _ in range(n))

    @staticmethod
    def backward(ctx, *grads):
        """
        Args:
            grads: One gradient per alias; any may be None.

        Returns:
            Their sum, and None for ``n``.
        """
        live = [g for g in grads if g is not None]
        if not live:
            return None, None
        if len(live) == 1:
            return live[0], None
        live = [g.contiguous() for g in live]
        pad = live + [live[0]] * (5 - len(live))
        out = torch.empty_like(live[0])
        torch.ops.uma_nitro.sum5(*pad[:5], out, len(live))
        return out, None


def _block(
    module,
    inputs,
    radial,
    edge_index,
    wigner,
    wigner_inv_envelope,
    sys_node_embedding,
    node_offsets,
):
    """
    Run one eSCN-MD block through the fused edgewise path.
    """
    rad_tok = rad_tok_s = None
    if isinstance(radial, tuple):
        radial, rad_tok, rad_tok_s = radial
    residual = inputs
    # sys_node_embedding folded into the norm's epilogue rather than written back with
    # ``inputs[:, 0, :] += ...``. That slice write cost a CopySlices node per block.
    inputs = _equivariant_rms_norm(module.norm_1, inputs, row0=sys_node_embedding)

    edge_wise = module.edge_wise
    sphere_channels = inputs.shape[2]
    conv1, conv2 = edge_wise.so2_conv_1, edge_wise.so2_conv_2
    layout = BlockLayout(
        num_edges=edge_index.shape[1],
        channels=sphere_channels,
        block_sizes=tuple(conv2.m_split_sizes),
        prefix=conv1.extra_m0_output_channels,
        coeff_pad=COEFF_PAD,
    )
    # out= is illegal under autograd, and m0/m1/m2 descend from pos, so the span
    # is a custom op with a hand-written backward -- a custom op, not an
    # autograd.Function, because Dynamo skips the latter to eager when its
    # forward writes out= into a strided view (gb0242). That is cheap here
    # only because inference freezes every weight, leaving dx = grad @ W per GEMM.
    for conv in (*conv1.so2_m_conv, *conv2.so2_m_conv):
        if conv._w_block is None:
            conv._build_w_block()
    ts = _transposed_weights(conv1, conv2)
    # fp16 copies of all six SO2 weights for the fp16 forward GEMMs.
    hs = _half_weights(conv1, conv2)
    # transposed weights in fp16, for the per-row-scaled backward GEMMs.
    hts = _half_transposed(conv1, conv2)

    inputs = block_span(
        inputs,
        edge_index,
        wigner,
        radial,
        wigner_inv_envelope,
        node_offsets,
        residual,
        edge_index[1],
        conv1.fc_m0.bias,
        conv2.fc_m0.bias,
        hs,
        hts,
        ts,
        edge_wise.act.expand_index,
        layout,
        rad_tok,
        rad_tok_s,
    )
    del sphere_channels

    residual = inputs
    inputs = _equivariant_rms_norm(module.norm_2, inputs)
    # The block residual rides into the last SO3 linear's epilogue rather than
    # costing its own `(N, P, C)` add.
    return _spectral_atomwise(module.atom_wise, inputs, residual)


@triton.jit
def _add_block_kernel(
    x_ptr,
    d_ptr,
    out_ptr,
    num_nodes,
    col_start,
    width,
    ROW: tl.constexpr,
    C: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    ``out = x`` with ``d`` added into ``out[:, 0, col_start:col_start + width]``.
    """
    c = tl.arange(0, C)
    node = tl.program_id(0)
    while node < num_nodes:
        base = node * ROW
        for i in tl.static_range(ROW // C):
            v = tl.load(x_ptr + base + i * C + c)
            if i == 0:
                rel = c - col_start
                inb = (rel >= 0) & (rel < width)
                v += tl.load(d_ptr + node * width + rel, mask=inb, other=0.0)
            tl.store(out_ptr + base + i * C + c, v)
        node += GRID_S


@triton_op("uma_nitro::add_block_row0", mutates_args={"out"})
def _add_block_row0(
    x: torch.Tensor,
    d: torch.Tensor,
    out: torch.Tensor,
    col_start: int,
) -> None:
    """
    Launch the row-0 block add.

    Args:
        x: Node features, ``(N, P, C)``.
        d: Addend for the ``l=0`` row's column block, ``(N, W)``.
        out: Output ``(N, P, C)``; mutated.
        col_start: First column of the block.
    """
    nodes, coeffs, channels = x.shape
    grid = min(nodes, 4096)
    wrap_triton(_add_block_kernel)[(grid,)](
        x,
        d,
        out,
        nodes,
        col_start,
        d.shape[1],
        ROW=coeffs * channels,
        C=channels,
        GRID_S=grid,
        num_warps=4,
    )


class _AddBlockRow0(torch.autograd.Function):
    """
    ``out = x`` with ``d`` added into the ``l=0`` row's column block.

    Replaces ``out = x.clone(); out[:, 0, a:b] += d``. That form costs a CopySlices
    node -- clone the base, allocate zeros, two copies -- once per layer. Here the
    forward is a single pass and **the backward is free**: d(out)/d(x) is the identity
    and d(out)/d(d) is a narrow view of the incoming gradient, so no kernel runs.
    """

    @staticmethod
    def forward(ctx, x, d, col_start):
        """
        Args:
            x: Node features, ``(N, P, C)``.
            d: Addend, ``(N, W)``.
            col_start: First column of the block.

        Returns:
            ``(N, P, C)``.
        """
        out = torch.empty_like(x)
        torch.ops.uma_nitro.add_block_row0(x, d, out, col_start)
        ctx.slice = (col_start, d.shape[1])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: Gradient of the output, ``(N, P, C)``.

        Returns:
            Gradients for ``(x, d, col_start)``; both are views, no kernel.
        """
        start, width = ctx.slice
        return grad_out, grad_out[:, 0, start : start + width], None


@triton.jit
def _bc_reduce_kernel(
    x_ptr,
    batch_ptr,
    sums_ptr,
    num_nodes,
    col_start,
    ROW: tl.constexpr,
    W: tl.constexpr,
    WPAD: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    Per-system sum of the ``l=0`` row's charge column block.

    ``sums`` must arrive zeroed; the accumulation is ``tl.atomic_add``, which is the
    same non-deterministic reduction order ``index_add_`` already had.
    """
    w = tl.arange(0, WPAD)
    m = w < W
    node = tl.program_id(0)
    while node < num_nodes:
        b = tl.load(batch_ptr + node)
        v = tl.load(x_ptr + node * ROW + col_start + w, mask=m, other=0.0)
        tl.atomic_add(sums_ptr + b * W + w, v, mask=m)
        node += GRID_S


@triton.jit
def _bc_apply_kernel(
    x_ptr,
    sums_ptr,
    charge_ptr,
    natoms_ptr,
    batch_ptr,
    out_ptr,
    num_nodes,
    col_start,
    SUB_TARGET: tl.constexpr,
    ROW: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    GRID_S: tl.constexpr,
    CSPL: tl.constexpr,
):
    """
    ``out = x`` with the per-system correction subtracted from the ``l=0`` block.

    With ``SUB_TARGET`` the correction is ``(sums - charge) / natoms`` -- the forward.
    Without it, it is ``sums / natoms`` -- the backward, where the target is constant
    and so drops out.
    """
    c = tl.arange(0, C)
    rel = c - col_start
    inb = (rel >= 0) & (rel < W)
    # One coefficient row per program, not one node: eight of the nine rows are a
    # straight copy and the ninth is a copy minus a broadcast scalar, so there is
    # nothing to reduce and the split is bit-identical. At 100 atoms it takes the grid
    # from 100 blocks (0.76 waves) to 900. The rows with ``i != 0`` skip the batch,
    # sums, charge and natoms loads entirely -- ``i`` is uniform within a program.
    if CSPL == 1:
        node = tl.program_id(0)
        while node < num_nodes:
            b = tl.load(batch_ptr + node)
            s = tl.load(sums_ptr + b * W + rel, mask=inb, other=0.0)
            if SUB_TARGET:
                s = s - tl.load(charge_ptr + b).to(tl.float32)
            corr = tl.math.div_rn(s, tl.load(natoms_ptr + b).to(tl.float32))
            corr = tl.where(inb, corr, 0.0)
            base = node * ROW
            for i in tl.static_range(ROW // C):
                v = tl.load(x_ptr + base + i * C + c)
                if i == 0:
                    v -= corr
                tl.store(out_ptr + base + i * C + c, v)
            node += GRID_S
    else:
        i = tl.program_id(0) % (ROW // C)
        node = tl.program_id(0) // (ROW // C)
        while node < num_nodes:
            b = tl.load(batch_ptr + node)
            s = tl.load(sums_ptr + b * W + rel, mask=inb, other=0.0)
            if SUB_TARGET:
                s = s - tl.load(charge_ptr + b).to(tl.float32)
            corr = tl.math.div_rn(s, tl.load(natoms_ptr + b).to(tl.float32))
            corr = tl.where(inb, corr, 0.0)
            base = node * ROW
            v = tl.load(x_ptr + base + i * C + c)
            if i == 0:
                v -= corr
            tl.store(out_ptr + base + i * C + c, v)
            node += GRID_S


@triton_op("uma_nitro::bc_reduce", mutates_args={"sums"})
def _bc_reduce_op(
    x: torch.Tensor,
    batch: torch.Tensor,
    sums: torch.Tensor,
    col_start: int,
) -> None:
    """
    Launch the per-system charge-block reduction.
    """
    nodes, coeffs, channels = x.shape
    grid = min(nodes, 4096)
    width = sums.shape[1]
    if "bc" not in _ABLATIONS:
        wrap_triton(_bc_reduce_kernel)[(grid,)](
            x,
            batch,
            sums,
            nodes,
            col_start,
            ROW=coeffs * channels,
            W=width,
            WPAD=triton.next_power_of_2(width),
            GRID_S=grid,
            num_warps=1,
        )


@triton_op("uma_nitro::bc_apply", mutates_args={"out"})
def _bc_apply_op(
    x: torch.Tensor,
    sums: torch.Tensor,
    charge: torch.Tensor,
    natoms: torch.Tensor,
    batch: torch.Tensor,
    out: torch.Tensor,
    col_start: int,
    sub_target: bool,
) -> None:
    """
    Launch the charge-balance apply pass.
    """
    nodes, coeffs, channels = x.shape
    grid = min(nodes, 4096)
    if "bc" not in _ABLATIONS:
        _cs = coeffs if NODE_CSPLIT else 1
        wrap_triton(_bc_apply_kernel)[(grid * _cs,)](
            x,
            sums,
            charge,
            natoms,
            batch,
            out,
            nodes,
            col_start,
            SUB_TARGET=sub_target,
            ROW=coeffs * channels,
            C=channels,
            W=sums.shape[1],
            GRID_S=grid,
            CSPL=_cs,
            num_warps=4,
        )


class _BalanceCharge(torch.autograd.Function):
    """
    Apply charge balancing as one autograd node.

    Forward and backward each reduce the scalar coefficient block per system and
    apply the corresponding correction.
    """

    @staticmethod
    def forward(ctx, inputs, charge, natoms, batch, col_start, fwd_sums, bwd_sums):
        """
        Args:
            inputs: Node features, ``(N, P, C)``.
            charge: Per-system target charge, ``(B,)``.
            natoms: Atoms per system, ``(B,)``.
            batch: System index per node, ``(N,)``.
            col_start: First charge channel.
            fwd_sums: Zeroed ``(B, W)`` accumulator for this call's forward.
            bwd_sums: Zeroed ``(B, W)`` accumulator for this call's backward.

        Returns:
            ``(N, P, C)`` with the charge correction applied.
        """
        torch.ops.uma_nitro.bc_reduce(inputs, batch, fwd_sums, col_start)
        out = torch.empty_like(inputs)
        torch.ops.uma_nitro.bc_apply(
            inputs, fwd_sums, charge, natoms, batch, out, col_start, True
        )
        # Not save_for_backward: the scratch slots are written in place (that is the
        # point of sharing one fill), and saved tensors are version-checked.
        ctx.save_for_backward(natoms, batch)
        ctx.bwd_sums = bwd_sums
        ctx.col_start = col_start
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: Gradient of the output, ``(N, P, C)``.

        Returns:
            Gradient for ``inputs``; ``None`` for the rest.
        """
        natoms, batch = ctx.saved_tensors
        sums = ctx.bwd_sums
        grad_out = grad_out.contiguous()
        torch.ops.uma_nitro.bc_reduce(grad_out, batch, sums, ctx.col_start)
        dx = torch.empty_like(grad_out)
        torch.ops.uma_nitro.bc_apply(
            grad_out, sums, natoms, natoms, batch, dx, ctx.col_start, False
        )
        return dx, None, None, None, None, None, None


def _bc_scratch(backbone, inputs, natoms, nlayers):
    """
    One zeroed ``(2 * nlayers, B, W)`` accumulator pool for every charge-balance call.

    Each reduction needs its destination to start at zero, and ``torch.zeros`` on a
    ``(1, 3)`` tensor is still a whole ``aten::fill_`` launch: 8 per iteration, forward
    and backward, for 24 bytes. Every slot is written by exactly one call, so one fill
    covers all of them.

    Args:
        backbone: The backbone, for the charge channel range.
        inputs: A node tensor, for device and dtype.
        natoms: Atoms per system, for ``B``.
        nlayers: Number of blocks.

    Returns:
        ``(2 * nlayers, B, W)`` zeros.
    """
    width = backbone.charge_channel_end - backbone.charge_channel_start
    return torch.zeros(
        (2 * nlayers, natoms.shape[0], width),
        device=inputs.device,
        dtype=inputs.dtype,
    )


def _balance_charge(backbone, inputs, charge, natoms, batch, scratch, i):
    if scratch.shape[-1] == 0:
        return inputs
    nl = scratch.shape[0] // 2
    return _BalanceCharge.apply(
        inputs,
        charge,
        natoms,
        batch,
        backbone.charge_channel_start,
        scratch[i],
        scratch[nl + i],
    )


@triton.jit
def _ln_silu_multi_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    num_rows,
    eps,
    x_stride,
    L: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    LayerNorm+SiLU over every ``C``-wide column block of a row, in one launch.

    ``_LNSiLUSplit`` ran one kernel per block, so each read a 512-byte chunk out of a
    2560-byte row and the five launches walked the same rows five times. One program per
    row block reads each row once. Every reduction is still within a block's channels.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        for l in tl.static_range(L):
            off = rows[:, None] * x_stride + l * C + c[None, :]
            x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
            mean = tl.sum(x, axis=1) / C
            d = tl.where(m[:, None], x - mean[:, None], 0.0)
            rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=1) / C + eps)
            w = tl.load(w_ptr + l * C + c)[None, :]
            bb = tl.load(b_ptr + l * C + c)[None, :]
            y = d * rstd[:, None] * w + bb
            tl.store(
                out_ptr + l * num_rows * C + rows[:, None] * C + c[None, :],
                (y * tl.sigmoid(y)).to(out_ptr.dtype.element_ty),
                mask=m[:, None],
            )
            tl.store(mean_ptr + l * num_rows + rows, mean, mask=m)
            tl.store(rstd_ptr + l * num_rows + rows, rstd, mask=m)
        blk += GRID_S


@triton_op("uma_nitro::ln_silu_multi_fwd", mutates_args={"out", "mean", "rstd"})
def _ln_silu_multi_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
) -> None:
    """
    Launch the multi-block fused LayerNorm+SiLU forward.

    Args:
        x: Input, ``(R, L * C)``.
        weight: Per-block weights, ``(L, C)``.
        bias: Per-block biases, ``(L, C)``.
        out: Output, ``(L, R, C)`` so each block is contiguous; mutated.
        mean: Saved means, ``(L, R)``; mutated.
        rstd: Saved reciprocal stds, ``(L, R)``; mutated.
        eps: LayerNorm epsilon.
    """
    rows = x.shape[0]
    blocks, channels = weight.shape
    g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_ln_silu_multi_fwd_kernel)[(g,)](
        x,
        weight,
        bias,
        out,
        mean,
        rstd,
        rows,
        eps,
        x.stride(0),
        L=blocks,
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=g,
        num_warps=LNS_WARPS,
    )


@triton.jit
def _ln_silu_multi_bwd_kernel(
    d0,
    d1,
    d2,
    d3,
    d4,
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    num_rows,
    x_stride,
    dx_stride,
    L: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    SiLU'+LayerNorm dx for every column block of a row, in one launch.

    The five per-block backwards each walked the same rows, so the joint buffer was read
    five times. One program per row block reads it once. Up to five upstream gradients
    are passed as separate pointers because autograd hands back one tensor per output.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        for l in tl.static_range(L):
            off = rows[:, None] * x_stride + l * C + c[None, :]
            flat = rows[:, None] * C + c[None, :]
            x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + l * num_rows + rows, mask=m, other=0.0)
            rstd = tl.load(rstd_ptr + l * num_rows + rows, mask=m, other=0.0)
            xhat = (x - mean[:, None]) * rstd[:, None]
            w = tl.load(w_ptr + l * C + c)[None, :]
            y = xhat * w + tl.load(b_ptr + l * C + c)[None, :]
            sg = tl.sigmoid(y)
            if l == 0:
                g = tl.load(d0 + flat, mask=m[:, None], other=0.0)
            elif l == 1:
                g = tl.load(d1 + flat, mask=m[:, None], other=0.0)
            elif l == 2:
                g = tl.load(d2 + flat, mask=m[:, None], other=0.0)
            elif l == 3:
                g = tl.load(d3 + flat, mask=m[:, None], other=0.0)
            else:
                g = tl.load(d4 + flat, mask=m[:, None], other=0.0)
            dxhat = g * sg * (1.0 + y * (1.0 - sg)) * w
            c1 = tl.sum(dxhat, axis=1) / C
            c2 = tl.sum(dxhat * xhat, axis=1) / C
            dx = rstd[:, None] * (dxhat - c1[:, None] - xhat * c2[:, None])
            tl.store(
                dx_ptr + rows[:, None] * dx_stride + l * C + c[None, :],
                dx,
                mask=m[:, None],
            )
        blk += GRID_S


@triton_op("uma_nitro::ln_silu_multi_bwd", mutates_args={"dx"})
def _ln_silu_multi_bwd(
    grads: list[torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
) -> None:
    """
    Launch the multi-block fused SiLU'+LayerNorm backward.

    Args:
        grads: One contiguous ``(R, C)`` upstream gradient per column block, at most 5.
        x: Joint pre-norm buffer, ``(R, L * C)``.
        weight: Per-block weights, ``(L, C)``.
        bias: Per-block biases, ``(L, C)``.
        mean: Saved means, ``(L, R)``.
        rstd: Saved reciprocal stds, ``(L, R)``.
        dx: Joint output gradient, ``(R, L * C)``; mutated.
    """
    rows = x.shape[0]
    blocks, channels = weight.shape
    _assert_lns_blocks(blocks)
    pad = list(grads) + [grads[0]] * (5 - len(grads))
    g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_ln_silu_multi_bwd_kernel)[(g,)](
        *pad[:5],
        x,
        weight,
        bias,
        mean,
        rstd,
        dx,
        rows,
        x.stride(0),
        dx.stride(0),
        L=blocks,
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=g,
        num_warps=LNS_WARPS,
    )


class _LNSiLUSplit(torch.autograd.Function):
    """
    Apply LayerNorm and SiLU to column blocks as one autograd node.

    The backward writes each block into one joint gradient buffer.
    """

    @staticmethod
    def forward(ctx, h_all, ln_weight, ln_bias, eps, hidden, num_layers):
        """
        Args:
            h_all: Joint pre-norm buffer, ``(R, num_layers * hidden)``.
            ln_weight: Per-layer LayerNorm weights, ``(num_layers, hidden)``.
            ln_bias: Per-layer LayerNorm biases, ``(num_layers, hidden)``.
            eps: LayerNorm epsilon.
            hidden: Columns per layer.
            num_layers: Number of column blocks.

        Returns:
            A tuple of ``num_layers`` tensors, each ``(R, hidden)``.
        """
        rows = h_all.shape[0]
        joint = torch.empty(
            (num_layers, rows, hidden), device=h_all.device, dtype=h_all.dtype
        )
        mean = torch.empty(num_layers, rows, device=h_all.device, dtype=torch.float32)
        rstd = torch.empty(num_layers, rows, device=h_all.device, dtype=torch.float32)
        torch.ops.uma_nitro.ln_silu_multi_fwd(
            h_all, ln_weight, ln_bias, joint, mean, rstd, eps
        )
        ctx.save_for_backward(h_all, ln_weight, ln_bias, mean, rstd)
        ctx.dims = (hidden, num_layers)
        # Column views: the consumers are all GEMMs, which take a row stride for free.
        return tuple(joint[i] for i in range(num_layers))

    @staticmethod
    def backward(ctx, *grads):
        """
        Args:
            grads: One upstream gradient per column block.

        Returns:
            The joint gradient for ``h_all``; the norm parameters are frozen.
        """
        hidden, num_layers = ctx.dims
        h_all, ln_weight, ln_bias, mean, rstd = ctx.saved_tensors
        grad_h = torch.empty_like(h_all)
        torch.ops.uma_nitro.ln_silu_multi_bwd(
            [g.contiguous() for g in grads],
            h_all,
            ln_weight,
            ln_bias,
            mean,
            rstd,
            grad_h,
        )
        return grad_h, None, None, None, None, None


def _unified_radials(module, x_edge):
    h_all = F.linear(x_edge, module.W1_cat, module.b1_cat)
    if os.environ.get("UMA_NITRO_NO_LNSILU") == "1":
        h_per_layer = h_all.split(module.hidden_features, dim=1)
    else:
        h_per_layer = _LNSiLUSplit.apply(
            h_all,
            module.ln1_weight,
            module.ln1_bias,
            module.ln_eps,
            module.hidden_features,
            module.num_layers,
        )
    outputs = []
    for i in range(module.num_layers):
        if os.environ.get("UMA_NITRO_NO_LNSILU") == "1":
            hidden = F.silu(
                F.layer_norm(
                    h_per_layer[i],
                    (module.hidden_features,),
                    module.ln1_weight[i],
                    module.ln1_bias[i],
                    module.ln_eps,
                )
            )
            hidden = F.linear(hidden, module.fc2_weight[i], module.fc2_bias[i])
            hidden = F.silu(
                F.layer_norm(
                    hidden,
                    (module.hidden_features,),
                    module.ln2_weight[i],
                    module.ln2_bias[i],
                    module.ln_eps,
                )
            )
        else:
            hidden = h_per_layer[i]
            hidden = F.linear(hidden, module.fc2_weight[i], module.fc2_bias[i])
            hidden = ln_silu(
                hidden, module.ln2_weight[i], module.ln2_bias[i], module.ln_eps
            )
        outputs.append(F.linear(hidden, module.fc3_weight[i], module.fc3_bias[i]))
    return outputs


class _LinearFrozenTN(torch.autograd.Function):
    """
    Apply a frozen linear layer with a pre-transposed backward weight.

    Only the input gradient is produced.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, weight_t):
        """
        Args:
            x: Input, ``(R, in)``.
            weight: ``(out, in)``.
            bias: ``(out,)`` or None.
            weight_t: A contiguous ``(in, out)`` copy of ``weight``.

        Returns:
            ``(R, out)``.
        """
        ctx.save_for_backward(weight_t)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: ``(R, out)``.

        Returns:
            ``dx`` only.
        """
        (weight_t,) = ctx.saved_tensors
        return grad_out.contiguous().mm(weight_t.t()), None, None, None


TN_INNER = os.environ.get("UMA_NITRO_TN_INNER", "1") == "1"


S5_BLK = int(os.environ.get("UMA_NITRO_S5_BLK", "1024"))
S5_WARPS = int(os.environ.get("UMA_NITRO_S5_WARPS", "4"))
AMX_GRID = int(os.environ.get("UMA_NITRO_AMX_GRID", "1024"))
AMX_BLK = int(os.environ.get("UMA_NITRO_AMX_BLK", "2048"))
AMX_WARPS = int(os.environ.get("UMA_NITRO_AMX_WARPS", "8"))


@triton.jit
def _absmax_partial_kernel(x_ptr, out_ptr, n, BLK: tl.constexpr, GRID: tl.constexpr):
    """
    Per-program partial ``max|x|`` into ``out[program]``.
    """
    pid = tl.program_id(0)
    acc = 0.0
    base = pid.to(tl.int64) * BLK
    step = GRID * BLK
    while base < n:
        off = base + tl.arange(0, BLK)
        v = tl.load(x_ptr + off, mask=off < n, other=0.0)
        acc = tl.maximum(acc, tl.max(tl.abs(v)))
        base += step
    tl.store(out_ptr + pid, acc)


@triton.jit
def _pow2_scale_kernel(part_ptr, out_ptr, GRID: tl.constexpr):
    """
    Reduce the partials and emit the exact power-of-two scale putting the max near 2^14.

    """
    v = tl.load(part_ptr + tl.arange(0, GRID))
    m = tl.maximum(tl.max(v), 1e-30)
    tl.store(out_ptr, tl.exp2(tl.floor(tl.log2(FP16_SCALE_TARGET / m))))


@triton_op("uma_nitro::pow2_scale", mutates_args={"partials", "out"})
def _pow2_scale(x: torch.Tensor, partials: torch.Tensor, out: torch.Tensor) -> None:
    """
    Compute an exact power-of-two fp16 scale with a two-stage reduction.
    """
    n = x.numel()
    wrap_triton(_absmax_partial_kernel)[(AMX_GRID,)](
        x,
        partials,
        n,
        BLK=AMX_BLK,
        GRID=AMX_GRID,
        num_warps=AMX_WARPS,
    )
    wrap_triton(_pow2_scale_kernel)[(1,)](partials, out, GRID=AMX_GRID, num_warps=4)


FP16_RADIAL = os.environ.get("UMA_NITRO_FP16_RADIAL", "1") == "1"
FP16_MERGED_BWD = os.environ.get("UMA_NITRO_FP16_MERGED_BWD", "1") == "1"


@triton.jit
def _ln_silu_fwd_fp16_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    rs_ptr,
    num_rows,
    eps,
    x_stride,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    ``_ln_silu_fwd_kernel`` writing an fp16 output with a per-row power-of-two scale.

    The row max is taken over values already held in registers, so the scale costs one
    within-row reduction and one extra ``(R,)`` store. ``rs`` is the *reciprocal*, which
    is exact because the scale is a power of two, and is what ``fp16_mm``'s epilogue
    multiplies the accumulator by.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    w = tl.load(w_ptr + c)[None, :]
    b = tl.load(b_ptr + c)[None, :]
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        off = rows[:, None] * x_stride + c[None, :]
        out_off = rows[:, None] * C + c[None, :]
        x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        y = d * rstd[:, None] * w + b
        o = y * tl.sigmoid(y)
        # exp2(floor(log2(.))) is an exact power of two, so the round trip through
        # fp16 loses only mantissa, never a binade. The largest element of the row
        # lands in [2^14, 2^15), well inside fp16's 65504 max normal.
        mx = tl.max(tl.where(m[:, None], tl.abs(o), 0.0), axis=1)
        sc = tl.exp2(tl.floor(tl.log2(FP16_SCALE_TARGET / tl.maximum(mx, 1e-30))))
        tl.store(out_ptr + out_off, (o * sc[:, None]).to(tl.float16), mask=m[:, None])
        tl.store(mean_ptr + rows, mean, mask=m)
        tl.store(rstd_ptr + rows, rstd, mask=m)
        tl.store(rs_ptr + rows, 1.0 / sc, mask=m)
        blk += GRID_S


@triton_op(
    "uma_nitro::ln_silu_fwd_fp16", mutates_args={"out", "mean", "rstd", "rscale"}
)
def _ln_silu_fwd_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    rscale: torch.Tensor,
    eps: float,
) -> None:
    """
    Launch the fused LayerNorm+SiLU forward with an fp16, per-row-scaled output.

    Args:
        x: Input, ``(R, C)``, possibly a strided column slice.
        weight: LayerNorm weight, ``(C,)``.
        bias: LayerNorm bias, ``(C,)``.
        out: Output, ``(R, C)`` fp16; mutated.
        mean: Saved mean, ``(R,)``; mutated.
        rstd: Saved reciprocal std, ``(R,)``; mutated.
        rscale: Reciprocal per-row scale, ``(R,)``; mutated.
        eps: LayerNorm epsilon.
    """
    rows, channels = x.shape
    _g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_ln_silu_fwd_fp16_kernel)[(_g,)](
        x,
        weight,
        bias,
        out,
        mean,
        rstd,
        rscale,
        rows,
        eps,
        x.stride(0),
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=_g,
        num_warps=LNS_WARPS,
    )


@triton.jit
def _ln_silu_multi_bwd_fp16_kernel(
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    rs_ptr,
    num_rows,
    x_stride,
    dx_stride,
    gscale,
    L: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
    SC_MASK: tl.constexpr,
):
    """
    ``_ln_silu_multi_bwd_kernel`` writing fp16 ``dx`` with one power-of-two scale per row.

    The scale has to cover the whole ``L * C`` row, because ``fp16_mm``'s epilogue
    applies one ``rscale`` per output row of the GEMM that consumes this buffer. That
    forces the two-pass shape below: every block's ``dx`` tile is held in registers
    while the row max is accumulated, then all of them are stored. At ``L = 5``,
    ``C = 128`` and ``BLOCK_R = 8`` that is 20 fp32 per thread at 256 threads, which
    costs no occupancy here.

    Each upstream gradient ``dl`` may arrive fp16 with an exact per-row power-of-two
    scale; bit ``l`` of ``SC_MASK`` says so, and the value is then
    ``dl[r, :] * sl[r] * gscale``. That scale is free: the producing GEMM is linear in
    its A operand, so dropping its ``rscale`` epilogue leaves the accumulator already
    scaled by the A operand's own per-row power of two, and ``gscale`` is the single
    constant power of two that keeps the accumulator inside fp16's range.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        a0 = tl.zeros((BLOCK_R, C), dtype=tl.float32)
        a1 = tl.zeros((BLOCK_R, C), dtype=tl.float32)
        a2 = tl.zeros((BLOCK_R, C), dtype=tl.float32)
        a3 = tl.zeros((BLOCK_R, C), dtype=tl.float32)
        a4 = tl.zeros((BLOCK_R, C), dtype=tl.float32)
        mx = tl.zeros((BLOCK_R,), dtype=tl.float32)
        for l in tl.static_range(L):
            off = rows[:, None] * x_stride + l * C + c[None, :]
            flat = rows[:, None] * C + c[None, :]
            x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + l * num_rows + rows, mask=m, other=0.0)
            rstd = tl.load(rstd_ptr + l * num_rows + rows, mask=m, other=0.0)
            xhat = (x - mean[:, None]) * rstd[:, None]
            w = tl.load(w_ptr + l * C + c)[None, :]
            y = xhat * w + tl.load(b_ptr + l * C + c)[None, :]
            sg = tl.sigmoid(y)
            if l == 0:
                g = tl.load(d0 + flat, mask=m[:, None], other=0.0).to(tl.float32)
                if SC_MASK & 1:
                    g *= (tl.load(s0 + rows, mask=m, other=0.0) * gscale)[:, None]
            elif l == 1:
                g = tl.load(d1 + flat, mask=m[:, None], other=0.0).to(tl.float32)
                if SC_MASK & 2:
                    g *= (tl.load(s1 + rows, mask=m, other=0.0) * gscale)[:, None]
            elif l == 2:
                g = tl.load(d2 + flat, mask=m[:, None], other=0.0).to(tl.float32)
                if SC_MASK & 4:
                    g *= (tl.load(s2 + rows, mask=m, other=0.0) * gscale)[:, None]
            elif l == 3:
                g = tl.load(d3 + flat, mask=m[:, None], other=0.0).to(tl.float32)
                if SC_MASK & 8:
                    g *= (tl.load(s3 + rows, mask=m, other=0.0) * gscale)[:, None]
            else:
                g = tl.load(d4 + flat, mask=m[:, None], other=0.0).to(tl.float32)
                if SC_MASK & 16:
                    g *= (tl.load(s4 + rows, mask=m, other=0.0) * gscale)[:, None]
            dxhat = g * sg * (1.0 + y * (1.0 - sg)) * w
            c1 = tl.sum(dxhat, axis=1) / C
            c2 = tl.sum(dxhat * xhat, axis=1) / C
            dx = rstd[:, None] * (dxhat - c1[:, None] - xhat * c2[:, None])
            mx = tl.maximum(mx, tl.max(tl.where(m[:, None], tl.abs(dx), 0.0), axis=1))
            if l == 0:
                a0 = dx
            elif l == 1:
                a1 = dx
            elif l == 2:
                a2 = dx
            elif l == 3:
                a3 = dx
            else:
                a4 = dx
        sc = tl.exp2(tl.floor(tl.log2(FP16_SCALE_TARGET / tl.maximum(mx, 1e-30))))
        for l in tl.static_range(L):
            v = a0
            if l == 1:
                v = a1
            elif l == 2:
                v = a2
            elif l == 3:
                v = a3
            elif l == 4:
                v = a4
            tl.store(
                dx_ptr + rows[:, None] * dx_stride + l * C + c[None, :],
                (v * sc[:, None]).to(tl.float16),
                mask=m[:, None],
            )
        tl.store(rs_ptr + rows, 1.0 / sc, mask=m)
        blk += GRID_S


@triton_op("uma_nitro::ln_silu_multi_bwd_fp16", mutates_args={"dx", "rscale"})
def _ln_silu_multi_bwd_fp16(
    grads: list[torch.Tensor],
    gscales: list[torch.Tensor],
    sc_mask: int,
    gscale: float,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    rscale: torch.Tensor,
) -> None:
    """
    Launch the fp16, per-row-scaled multi-block SiLU'+LayerNorm backward.

    Args:
        grads: One contiguous ``(R, C)`` upstream gradient per column block, at most 5.
            fp32, or fp16 when the matching ``sc_mask`` bit is set.
        gscales: One ``(R,)`` fp32 per-row scale per block. Entries whose ``sc_mask``
            bit is clear are never read and may be any valid ``(R,)`` tensor.
        sc_mask: Bit ``l`` set means ``grads[l]`` is scaled and must be multiplied by
            ``gscales[l][row] * gscale``.
        gscale: The single constant power of two shared by every scaled block.
        x: Joint pre-norm buffer, ``(R, L * C)``.
        weight: Per-block weights, ``(L, C)``.
        bias: Per-block biases, ``(L, C)``.
        mean: Saved means, ``(L, R)``.
        rstd: Saved reciprocal stds, ``(L, R)``.
        dx: Joint output gradient, ``(R, L * C)`` fp16; mutated.
        rscale: Reciprocal per-row scale, ``(R,)``; mutated.
    """
    rows = x.shape[0]
    blocks, channels = weight.shape
    _assert_lns_blocks(blocks)
    pad = list(grads) + [grads[0]] * (5 - len(grads))
    spad = list(gscales) + [rscale] * (5 - len(gscales))
    g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_ln_silu_multi_bwd_fp16_kernel)[(g,)](
        *pad[:5],
        *spad[:5],
        x,
        weight,
        bias,
        mean,
        rstd,
        dx,
        rscale,
        rows,
        x.stride(0),
        dx.stride(0),
        gscale,
        L=blocks,
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=g,
        num_warps=LNS_WARPS,
        SC_MASK=sc_mask,
    )


@triton.jit
def _ln_silu_bwd_dx_fp16_kernel(
    grad_out_ptr,
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dxs_ptr,
    num_rows,
    x_stride,
    dx_stride,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    Write the LayerNorm-SiLU input gradient in scaled fp16.

    A per-row power-of-two scale preserves range while reduction remains fp32.
    """
    c = tl.arange(0, C)
    r = tl.arange(0, BLOCK_R)
    w = tl.load(w_ptr + c)[None, :]
    bb = tl.load(b_ptr + c)[None, :]
    blk = tl.program_id(0)
    while blk * BLOCK_R < num_rows:
        rows = (blk * BLOCK_R + r).to(tl.int64)
        m = rows < num_rows
        off = rows[:, None] * x_stride + c[None, :]
        flat = rows[:, None] * C + c[None, :]
        dxo = rows[:, None] * dx_stride + c[None, :]
        x = tl.load(x_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=m, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=m, other=0.0)
        xhat = (x - mean[:, None]) * rstd[:, None]
        y = xhat * w + bb
        sg = tl.sigmoid(y)
        g = tl.load(grad_out_ptr + flat, mask=m[:, None], other=0.0).to(tl.float32)
        dxhat = g * sg * (1.0 + y * (1.0 - sg)) * w
        c1 = tl.sum(dxhat, axis=1) / C
        c2 = tl.sum(dxhat * xhat, axis=1) / C
        dx = rstd[:, None] * (dxhat - c1[:, None] - xhat * c2[:, None])
        mx = tl.max(tl.where(m[:, None], tl.abs(dx), 0.0), axis=1)
        sc = tl.exp2(tl.floor(tl.log2(FP16_SCALE_TARGET / tl.maximum(mx, 1e-30))))
        tl.store(dx_ptr + dxo, (dx * sc[:, None]).to(tl.float16), mask=m[:, None])
        tl.store(dxs_ptr + rows, 1.0 / sc, mask=m)
        blk += GRID_S


@triton_op("uma_nitro::ln_silu_bwd_dx_fp16", mutates_args={"dx", "dxs"})
def _ln_silu_bwd_dx_fp16(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    dxs: torch.Tensor,
) -> None:
    """
    Launch the fused SiLU'+LayerNorm backward with an fp16, per-row-scaled ``dx``.

    Args:
        grad_out: Upstream gradient, ``(R, C)``, contiguous; fp16 or fp32.
        x: Forward input, ``(R, C)``, possibly a strided column slice.
        weight: LayerNorm weight, ``(C,)``.
        bias: LayerNorm bias, ``(C,)``.
        mean: Saved mean, ``(R,)``.
        rstd: Saved reciprocal std, ``(R,)``.
        dx: Result, ``(R, C)`` fp16; mutated.
        dxs: Reciprocal per-row scale, ``(R,)`` fp32; mutated.
    """
    rows, channels = x.shape
    _g = min(triton.cdiv(rows, BLOCK_R), LNS_GRID)
    wrap_triton(_ln_silu_bwd_dx_fp16_kernel)[(_g,)](
        grad_out,
        x,
        weight,
        bias,
        mean,
        rstd,
        dx,
        dxs,
        rows,
        x.stride(0),
        dx.stride(0),
        C=channels,
        BLOCK_R=BLOCK_R,
        GRID_S=_g,
        num_warps=LNS_WARPS,
    )


FP16_LN1 = os.environ.get("UMA_NITRO_FP16_LN1", "1") == "1"

FP16_XEGRAD = os.environ.get("UMA_NITRO_FP16_XEGRAD", "1") == "1"

FP16_GRADBLK = os.environ.get("UMA_NITRO_FP16_GRADBLK", "1") == "1"


def _ln1_narrow():
    """
    Whether the merged head may emit fp16 blocks this call.

    Every consumer has to be one of the token-aware nodes, so the whole fp16
    radial-layer stack has to be on.
    """
    return bool(
        FP16_LN1
        and FP16_MERGED_BWD
        and TN_INNER
        and FUSE_RADLAYER
        and FP16_RADIAL
        and FP16_GRAD_RAD
    )


# The constant power of two the ``grad_blocks`` accumulator is divided by before its
# fp16 store, so that ``dx``'s own per-row scale can be inherited instead of a new one
# being computed. ``dx`` is stored with its row max in [8192, 16384], so the
# accumulator is bounded by ``16384 * max_k sum_j |fc2_w[j, k]|``; the live bound is
# 40.0 on uma-s-1p2, giving 6.6e5 without a divisor against fp16's 65504.
# ``_gradblk_bound`` asserts it rather than trusting this comment.
GRADBLK_GSCALE = 64.0


def _gradblk_narrow():
    """
    Whether blocks 1..n-1 may hand their gradient back fp16 with a per-row scale.

    Requires the forward narrowing, because the channel is the same token: a block's
    token dtype is what pins its gradient's dtype.
    """
    return bool(FP16_GRADBLK and _ln1_narrow())


def _gradblk_bound(fc2_weight):
    """
    Assert that ``GRADBLK_GSCALE`` keeps the ``grad_blocks`` accumulator in range.

    ``dblocks = dx @ fc2_weight`` with ``dx`` stored fp16 at a per-row max of at most
    16384, so ``|acc[r, k]| <= 16384 * sum_j |fc2_w[j, k]|`` and the stored value is
    that over ``GRADBLK_GSCALE``. An overflow here would appear end-to-end as ``inf``
    rather than as a deviation, so it is checked once against the real weights.

    Args:
        fc2_weight: ``(layers, out, in)`` fc2 weights.

    Returns:
        The worst-case stored magnitude.
    """
    bound = float(fc2_weight.detach().float().abs().sum(dim=1).max())
    worst = _FP16_SCALE_TARGET * bound / GRADBLK_GSCALE
    if worst > 32768.0:
        raise AssertionError(
            f"fp16 grad_blocks would overflow: bound {bound:.3f} gives {worst:.3e} "
            f"at GRADBLK_GSCALE={GRADBLK_GSCALE}; raise it to a larger power of two"
        )
    return worst


FP16_XEDGE = os.environ.get("UMA_NITRO_FP16_XEDGE", "0") == "1"


def _xedge_store_narrow():
    """
    Whether ``x_edge`` itself may be stored fp16.

    Requires the fp16 gradient token, since an fp16 value with an fp32 gradient
    would need the *other* token direction and there is nothing to gain from it.
    """
    return bool(FP16_XEDGE and _xegrad_narrow())


def _xegrad_narrow():
    """
    Whether ``x_edge``'s gradient may be fp16 this call.

    Only ``_MergedRadialHead`` returns it, so the merged head has to be on.
    """
    return bool(
        FP16_XEGRAD
        and FP16_MERGED_BWD
        and os.environ.get("UMA_NITRO_NO_MERGED_RADIAL") != "1"
    )


class _MergedRadialHead(torch.autograd.Function):
    """
    Combine the merged radial GEMM and its LayerNorm-SiLU split.

    The internal gradient may use scaled fp16 while the boundary remains fp32.
    """

    @staticmethod
    def forward(ctx, x_edge, xe_tok, w, w16, b, wt16, lw, lb, eps, hidden, nblocks):
        """
        Args:
            x_edge: Edge embedding, ``(E, 288)``.
            xe_tok: fp16 ``(E, 288)`` gradient-channel token for ``x_edge``, or
                None when that gradient stays fp32.
            w: Merged weight, ``(640, 288)``.
            w16: fp16 copy of ``w``, the forward GEMM's B operand when ``x_edge``
                arrives fp16.
            b: Merged bias, ``(640,)``.
            wt16: fp16 ``(288, 640)`` copy of ``w.t()``, the backward GEMM's B operand.
            lw: Per-block LayerNorm weights, ``(nblocks, hidden)``.
            lb: Per-block LayerNorm biases, ``(nblocks, hidden)``.
            eps: LayerNorm epsilon.
            hidden: Columns per block.
            nblocks: Number of column blocks.

        Returns:
            ``nblocks`` block tensors, followed by ``nblocks`` gradient tokens when
            the blocks are fp16, followed by ``nblocks`` ``(rows,)`` fp32 scale
            tokens when those gradients are themselves fp16.
        """
        narrow = _ln1_narrow()
        gnarrow = narrow and _gradblk_narrow()
        h_all = (
            fp16_mm(x_edge, w16, bias=b)
            if x_edge.dtype == torch.float16
            else F.linear(x_edge, w, b)
        )
        rows = h_all.shape[0]
        joint = torch.empty(
            (nblocks, rows, hidden),
            device=h_all.device,
            dtype=torch.float16 if narrow else h_all.dtype,
        )
        mean = torch.empty(nblocks, rows, device=h_all.device, dtype=torch.float32)
        rstd = torch.empty(nblocks, rows, device=h_all.device, dtype=torch.float32)
        torch.ops.uma_nitro.ln_silu_multi_fwd(h_all, lw, lb, joint, mean, rstd, eps)
        ctx.save_for_backward(h_all, lw, lb, mean, rstd, wt16)
        ctx.dims = (hidden, nblocks)
        ctx.narrow = narrow
        ctx.gnarrow = gnarrow
        ctx.xe_narrow = xe_tok is not None
        blocks = tuple(joint[i] for i in range(nblocks))
        if not narrow:
            return blocks
        toks = tuple(
            torch.empty(
                1,
                device=h_all.device,
                dtype=torch.float16 if (gnarrow and i > 0) else h_all.dtype,
            ).expand(rows, hidden)
            for i in range(nblocks)
        )
        stoks = ()
        if gnarrow:
            # The second channel: each narrowed consumer returns its per-row
            # power-of-two scale here, since an fp16 gradient is meaningless
            # without it and there is no other way back into this node.
            stoks = tuple(
                torch.empty(1, device=h_all.device, dtype=torch.float32).expand(rows)
                for _ in range(nblocks)
            )
        ctx.mark_non_differentiable(*blocks)
        ctx.set_materialize_grads(False)
        return blocks + toks + stoks

    @staticmethod
    def backward(ctx, *grads):
        """
        Args:
            grads: One upstream gradient per column block, preceded by ``nblocks``
                ``None`` slots for the non-differentiable fp16 blocks when those
                are in use.

        Returns:
            ``dx_edge`` only; every weight here is frozen. It is returned in
            ``xe_tok``'s slot, as fp16, when that token is present.
        """
        hidden, nblocks = ctx.dims
        h_all, lw, lb, mean, rstd, wt16 = ctx.saved_tensors
        rows = h_all.shape[0]
        scales = (None,) * nblocks
        if ctx.narrow:
            if ctx.gnarrow:
                scales = grads[2 * nblocks : 3 * nblocks]
            grads = grads[nblocks : 2 * nblocks]
        grad_h = torch.empty(
            rows, nblocks * hidden, device=h_all.device, dtype=torch.float16
        )
        rscale = torch.empty(rows, device=h_all.device, dtype=torch.float32)
        gl, sl, mask = [], [], 0
        for i, g in enumerate(grads):
            scaled = ctx.gnarrow and scales[i] is not None
            dtype = torch.float16 if scaled else h_all.dtype
            gl.append(
                g.contiguous()
                if g is not None
                else torch.zeros(rows, hidden, device=h_all.device, dtype=dtype)
            )
            if scaled:
                mask |= 1 << i
                sl.append(scales[i])
            else:
                # Never read; any valid (rows,) fp32 pointer will do.
                sl.append(rscale)
        torch.ops.uma_nitro.ln_silu_multi_bwd_fp16(
            gl,
            sl,
            mask,
            GRADBLK_GSCALE,
            h_all,
            lw,
            lb,
            mean,
            rstd,
            grad_h,
            rscale,
        )
        tail = (None,) * 9
        if ctx.xe_narrow:
            gx = torch.empty(
                rows, wt16.shape[0], device=h_all.device, dtype=torch.float16
            )
            fp16_mm(grad_h, wt16, out=gx, rscale=rscale)
            return (None, gx, *tail)
        return (fp16_mm(grad_h, wt16, rscale=rscale), None, *tail)


class _LNSiLUFp16Linear(torch.autograd.Function):
    """
    ``fc3(ln_silu(x))`` as one node, with the LayerNorm+SiLU output stored fp16.

    Splitting this across ``_LNSiLU`` and ``_LinearFrozenTN`` cannot work: autograd
    requires an fp16 output's gradient to be fp16 too, and the gradient here comes
    out of an fp32 cuBLAS GEMM. Owning both halves keeps the node's *boundary* fp32
    -- ``x`` in, fp32 ``out``, fp32 ``dx`` -- while the intermediate that only these
    two kernels ever see is fp16.

    The forward is bit-comparable to, not bit-identical with, ``F.linear(ln_silu(x))``:
    the A operand carries 11 significand bits instead of 24 (TF32's 11 in practice) and
    the accumulation order is CuTeDSL's rather than cuBLAS's. Gated by the four-system
    round-trip probe, not by op-level bit-identity.
    """

    @staticmethod
    def forward(ctx, x, ln_w, ln_b, eps, w16, bias, weight_t):
        """
        Args:
            x: fc2 output, ``(R, C)``.
            ln_w: LayerNorm weight, ``(C,)``.
            ln_b: LayerNorm bias, ``(C,)``.
            eps: LayerNorm epsilon.
            w16: fp16 fc3 weight, ``(N, C)``.
            bias: fp32 fc3 bias, ``(N,)``.
            weight_t: Contiguous ``(C, N)`` fp32 copy of the fc3 weight, for the
                backward's ``tn`` GEMM.

        Returns:
            ``(R, N)`` fp32.
        """
        rows, channels = x.shape
        h16 = torch.empty(rows, channels, device=x.device, dtype=torch.float16)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        rscale = torch.empty(rows, device=x.device, dtype=torch.float32)
        torch.ops.uma_nitro.ln_silu_fwd_fp16(
            x, ln_w, ln_b, h16, mean, rstd, rscale, eps
        )
        ctx.save_for_backward(x, ln_w, ln_b, mean, rstd, weight_t)
        return fp16_mm(h16, w16, bias=bias, rscale=rscale)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: ``(R, N)``.

        Returns:
            ``dx`` only; the LayerNorm and fc3 parameters are frozen.
        """
        x, ln_w, ln_b, mean, rstd, weight_t = ctx.saved_tensors
        dh = grad_out.contiguous().mm(weight_t.t())
        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        torch.ops.uma_nitro.ln_silu_bwd_dx(dh, x, ln_w, ln_b, mean, rstd, dx)
        return dx, None, None, None, None, None, None


FP16_GRAD_RAD = os.environ.get("UMA_NITRO_FP16_GRAD_RAD", "1") == "1"

FP16_DH = os.environ.get("UMA_NITRO_FP16_DH", "1") == "1"


class _LNSiLUFp16LinearGradRad(torch.autograd.Function):
    """
    Combine fc3(ln_silu(x)) with scaled-fp16 gradient tokens.

    Auxiliary tokens carry a narrow gradient and its per-edge scale without
    changing the fp32 forward output.
    """

    @staticmethod
    def forward(ctx, x, ln_w, ln_b, eps, w16, bias, weight_t, w16_t):
        """
        Args:
            x: fc2 output, ``(R, C)``.
            ln_w: LayerNorm weight, ``(C,)``.
            ln_b: LayerNorm bias, ``(C,)``.
            eps: LayerNorm epsilon.
            w16: fp16 fc3 weight, ``(N, C)``.
            bias: fp32 fc3 bias, ``(N,)``.
            weight_t: Unused; kept so the two variants share a call shape.
            w16_t: fp16 ``(C, N)`` copy of the fc3 weight, the backward GEMM's B.

        Returns:
            ``(radial, tok, tok_s)`` -- the fp32 ``(R, N)`` result and the two
            gradient-channel tokens.
        """
        del weight_t
        rows, channels = x.shape
        h16 = torch.empty(rows, channels, device=x.device, dtype=torch.float16)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        rscale = torch.empty(rows, device=x.device, dtype=torch.float32)
        torch.ops.uma_nitro.ln_silu_fwd_fp16(
            x, ln_w, ln_b, h16, mean, rstd, rscale, eps
        )
        out = fp16_mm(h16, w16, bias=bias, rscale=rscale)
        ctx.save_for_backward(x, ln_w, ln_b, mean, rstd, w16_t)
        tok = torch.empty(1, device=x.device, dtype=torch.float16).expand(out.shape)
        tok_s = torch.empty(1, device=x.device, dtype=torch.float32).expand(rows)
        ctx.mark_non_differentiable(out)
        ctx.set_materialize_grads(False)
        return out, tok, tok_s

    @staticmethod
    def backward(ctx, grad_out, grad_tok, grad_tok_s):
        """
        Args:
            grad_out: Always None -- ``out`` is marked non-differentiable.
            grad_tok: The scaled fp16 ``grad_rad``, ``(R, N)``.
            grad_tok_s: Its per-edge reciprocal scale, ``(R,)``.

        Returns:
            ``dx`` only; the LayerNorm and fc3 parameters are frozen.
        """
        del grad_out
        x, ln_w, ln_b, mean, rstd, w16_t = ctx.saved_tensors
        if FP16_DH:
            dh = torch.empty(
                grad_tok.shape[0], x.shape[1], device=x.device, dtype=torch.float16
            )
            fp16_mm(grad_tok, w16_t, out=dh, rscale=grad_tok_s)
        else:
            dh = fp16_mm(grad_tok, w16_t, rscale=grad_tok_s)
        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        torch.ops.uma_nitro.ln_silu_bwd_dx(dh, x, ln_w, ln_b, mean, rstd, dx)
        return dx, None, None, None, None, None, None, None


FUSE_RADLAYER = os.environ.get("UMA_NITRO_FUSE_RADLAYER", "1") == "1"
FP16_DX = os.environ.get("UMA_NITRO_FP16_DX", "1") == "1"


class _RadialLayerGradRad(torch.autograd.Function):
    """
    Run fc3(ln_silu(fc2(x))) as one autograd node.

    Merged boundaries permit a private scaled-fp16 gradient while inputs and
    outputs remain fp32.
    """

    @staticmethod
    def forward(
        ctx,
        blocks_i,
        tok,
        stok,
        gbinv,
        fc2_w,
        fc2_b,
        fc2t,
        fc2t16,
        fc2w16,
        ln_w,
        ln_b,
        eps,
        w16,
        bias,
        w16_t,
    ):
        """
        Args:
            blocks_i: This layer's LayerNorm+SiLU block, ``(R, C)``; fp16 when the
                merged head narrows it, in which case ``tok`` carries the gradient.
            tok: ``(R, C)`` gradient-channel token, or None. fp16 when this node may
                hand its gradient back narrowed, fp32 otherwise.
            stok: ``(R,)`` fp32 scale-channel token, or None. An fp16 gradient is
                meaningless without its per-row scale and this is the only way back.
            gbinv: Cached ``(R,)`` fp32 constant ``1 / GRADBLK_GSCALE``, the backward
                GEMM's ``rscale`` when the gradient is narrowed. A constant vector
                rather than a scalar because the epilogue reads ``rscale`` as a
                vector fragment; a stride-0 ``expand`` faults.
            fc2_w: fc2 weight, ``(C, C)``.
            fc2_b: fc2 bias, ``(C,)``.
            fc2t: Contiguous ``(C, C)`` fp32 copy of ``fc2_w.t()``, for the fp32
                backward GEMM.
            fc2t16: fp16 copy of the same, for the narrowed backward GEMM.
            fc2w16: fp16 ``(C, C)`` copy of ``fc2_w`` itself, the forward GEMM's B
                operand when ``blocks_i`` arrives fp16.
            ln_w: LayerNorm weight, ``(C,)``.
            ln_b: LayerNorm bias, ``(C,)``.
            eps: LayerNorm epsilon.
            w16: fp16 fc3 weight, ``(N, C)``.
            bias: fp32 fc3 bias, ``(N,)``.
            w16_t: fp16 ``(C, N)`` fc3 weight, the backward GEMM's B operand.

        Returns:
            ``(radial, tok, tok_s)``, as ``_LNSiLUFp16LinearGradRad``.
        """
        narrow = blocks_i.dtype == torch.float16
        x = (
            fp16_mm(blocks_i, fc2w16, bias=fc2_b)
            if narrow
            else F.linear(blocks_i, fc2_w, fc2_b)
        )
        rows, channels = x.shape
        h16 = torch.empty(rows, channels, device=x.device, dtype=torch.float16)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        rscale = torch.empty(rows, device=x.device, dtype=torch.float32)
        torch.ops.uma_nitro.ln_silu_fwd_fp16(
            x, ln_w, ln_b, h16, mean, rstd, rscale, eps
        )
        out = fp16_mm(h16, w16, bias=bias, rscale=rscale)
        ctx.narrow = narrow
        ctx.gnarrow = stok is not None
        ctx.save_for_backward(x, ln_w, ln_b, mean, rstd, w16_t, fc2t, fc2t16, gbinv)
        tok = torch.empty(1, device=x.device, dtype=torch.float16).expand(out.shape)
        tok_s = torch.empty(1, device=x.device, dtype=torch.float32).expand(rows)
        ctx.mark_non_differentiable(out)
        ctx.set_materialize_grads(False)
        return out, tok, tok_s

    @staticmethod
    def backward(ctx, grad_out, grad_tok, grad_tok_s):
        """
        Args:
            grad_out: Always None -- ``out`` is marked non-differentiable.
            grad_tok: The scaled fp16 ``grad_rad``, ``(R, N)``.
            grad_tok_s: Its per-edge reciprocal scale, ``(R,)``.

        Returns:
            The gradient of ``blocks_i`` only; every weight here is frozen. It goes
            in ``tok``'s slot when the forward value was narrowed, and fp16 with
            ``dxs`` in ``stok``'s slot when the gradient is narrowed too.
        """
        del grad_out
        x, ln_w, ln_b, mean, rstd, w16_t, fc2t, fc2t16, gbinv = ctx.saved_tensors
        rows, channels = x.shape
        dev = x.device
        if FP16_DH:
            dh = torch.empty(rows, channels, device=dev, dtype=torch.float16)
            fp16_mm(grad_tok, w16_t, out=dh, rscale=grad_tok_s)
        else:
            dh = fp16_mm(grad_tok, w16_t, rscale=grad_tok_s)
        if FP16_DX:
            dx = torch.empty(rows, channels, device=dev, dtype=torch.float16)
            dxs = torch.empty(rows, device=dev, dtype=torch.float32)
            torch.ops.uma_nitro.ln_silu_bwd_dx_fp16(
                dh, x, ln_w, ln_b, mean, rstd, dx, dxs
            )
            # fp16_mm computes ``a @ w.t()`` with ``w`` stored ``(N, K)``; here
            # N is fc2's input width and K its output width, so ``w`` is exactly
            # the ``(in, out)`` transpose the fp32 path already caches.
            if ctx.gnarrow:
                # Drop the ``rscale=dxs`` epilogue and store the accumulator, which
                # a linear GEMM leaves scaled by ``dx``'s own exact per-row power of
                # two. ``gbinv`` divides by one constant power of two so it cannot
                # overflow. The consumer multiplies by ``dxs[row] * GRADBLK_GSCALE``,
                # so the value it sees differs from the fp32 path by exactly one
                # fp16 rounding and nothing else.
                dblocks = torch.empty(
                    rows, fc2t16.shape[0], device=dev, dtype=torch.float16
                )
                fp16_mm(dx, fc2t16, out=dblocks, rscale=gbinv)
                tail = (None,) * 12
                return (None, dblocks, dxs, *tail)
            dblocks = fp16_mm(dx, fc2t16, rscale=dxs)
        else:
            dx = torch.empty(rows, channels, device=dev, dtype=x.dtype)
            torch.ops.uma_nitro.ln_silu_bwd_dx(dh, x, ln_w, ln_b, mean, rstd, dx)
            dblocks = dx.mm(fc2t.t())
        tail = (None,) * 12
        if ctx.narrow:
            return (None, dblocks, None, *tail)
        return (dblocks, None, None, *tail)


class _LinearFp16Tok(torch.autograd.Function):
    """
    Apply a frozen linear layer to fp16 input with an fp32 input gradient.

    A token carries the gradient independently of the narrow forward input.
    """

    @staticmethod
    def forward(ctx, x16, tok, stok, w16, bias, wt):
        """
        Args:
            x16: fp16 activations, ``(R, C)``.
            tok: fp32 ``(R, C)`` gradient-channel token.
            stok: Unused ``(R,)`` scale token. Block 0's ``dx`` comes out of an fp32
                cuBLAS GEMM with no per-row scale to inherit, so this slot always
                takes ``None`` and the block stays fp32.
            w16: fp16 weight, ``(out, C)``.
            bias: fp32 bias, ``(out,)``.
            wt: Contiguous fp32 ``(C, out)`` copy of the weight's transpose.

        Returns:
            ``(R, out)`` fp32.
        """
        del tok, stok
        ctx.save_for_backward(wt)
        return fp16_mm(x16, w16, bias=bias)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: ``(R, out)``.

        Returns:
            ``dx`` in the token's slot; the weight is frozen.
        """
        (wt,) = ctx.saved_tensors
        return None, grad_out.contiguous().mm(wt.t()), None, None, None, None


def _merged_radials(backbone, x_edge, xe_tok=None, after=None, degree_stop=None):
    """
    Run one merged radial projection for its two consumers.

    The output is split into the edge-degree block and radial hidden-state blocks
    inside one autograd node.
    """
    net = backbone.edge_degree_embedding.rad_func.net
    layers = list(net._modules.values())
    mlp = backbone._unified_radial_mlp
    cache = getattr(backbone, "_uma_nitro_merged_radial", None)
    if cache is None:
        # Frozen at inference, so the concatenation happens once.
        w = torch.cat([layers[0].weight, mlp.W1_cat], dim=0)
        b = torch.cat([layers[0].bias, mlp.b1_cat], dim=0)
        lw = torch.cat([layers[1].weight.reshape(1, -1), mlp.ln1_weight], dim=0)
        lb = torch.cat([layers[1].bias.reshape(1, -1), mlp.ln1_bias], dim=0)
        cache = (
            w,
            b,
            lw,
            lb,
            w.t().contiguous(),
            mlp.fc2_weight.transpose(1, 2).contiguous(),
            mlp.fc3_weight.transpose(1, 2).contiguous(),
            mlp.fc3_weight.half().contiguous(),
            # fp16 (288, 640) for the backward GEMM's B operand: dx = grad_h @ w,
            # and fp16_mm computes a @ W.t(), so W is w.t() = wt.
            w.t().contiguous().half(),
            # fp16 (C, N) fc3 weight for the *backward* GEMM's B operand:
            # dh = grad_rad @ fc3_weight, and fp16_mm computes a @ W.t(), so
            # W is fc3_weight.t() = fc3t.
            mlp.fc3_weight.transpose(1, 2).contiguous().half(),
            # fp16 (in, out) fc2 weight for the merged radial layer's backward
            # GEMM: dblocks = dx @ fc2_weight, and fp16_mm computes a @ W.t(),
            # so W is fc2_weight.t() = fc2t.
            mlp.fc2_weight.transpose(1, 2).contiguous().half(),
            # fp16 (out, in) fc2 weight for the *forward* GEMM's B operand, used
            # when `ln1_out` arrives fp16: x = blocks @ fc2_weight.t().
            mlp.fc2_weight.half().contiguous(),
            # fp16 (out, in) copy of the edge-degree net's first post-head Linear,
            # block 0's consumer, for the same reason.
            layers[3].weight.half().contiguous(),
            # fp16 (640, 288) merged weight, for an fp16 `x_edge`.
            w.half().contiguous(),
            # Worst-case magnitude of the fp16 `grad_blocks` store, checked against
            # the live weights rather than assumed. Raises if it would overflow.
            _gradblk_bound(mlp.fc2_weight),
        )
        if not layers[0].weight.requires_grad:
            backbone._uma_nitro_merged_radial = cache
    (
        w,
        b,
        lw,
        lb,
        wt,
        fc2t,
        fc3t,
        fc3h,
        wt16,
        fc3t16,
        fc2t16,
        fc2w16,
        deg_w16,
        w16,
        _gb_worst,
    ) = cache
    hidden = mlp.hidden_features
    nblk = mlp.num_layers + 1
    # Layer 3 is block 0's first consumer and must be a frozen Linear for the fp16
    # hand-off; the token channel has no other consumer to route through.
    narrow = (
        _ln1_narrow()
        and isinstance(layers[3], nn.Linear)
        and not layers[3].weight.requires_grad
    )
    toks = stoks = (None,) * nblk
    gbinv = None
    if FP16_MERGED_BWD:
        out = _MergedRadialHead.apply(
            x_edge, xe_tok, w, w16, b, wt16, lw, lb, mlp.ln_eps, hidden, nblk
        )
        # Slice on the returned arity rather than on a locally recomputed
        # predicate, so the two can never disagree about how many tokens exist.
        if len(out) == 3 * nblk:
            blocks = out[:nblk]
            toks, stoks = out[nblk : 2 * nblk], out[2 * nblk :]
        elif len(out) == 2 * nblk:
            blocks, toks = out[:nblk], out[nblk:]
        else:
            blocks = out
        if narrow and stoks[1] is not None:
            # One constant `1 / GRADBLK_GSCALE` vector, the backward GEMM's rscale.
            # Built here on the calling stream and cached: a buffer first allocated
            # inside a side-stream thunk would be recorded into the capture on that
            # stream and read from the main stream forever after.
            rows = x_edge.shape[0]
            cached = getattr(backbone, "_uma_nitro_gradblk_inv", None)
            if cached is None or cached.shape[0] != rows:
                cached = torch.full(
                    (rows,),
                    1.0 / GRADBLK_GSCALE,
                    device=x_edge.device,
                    dtype=torch.float32,
                )
                backbone._uma_nitro_gradblk_inv = cached
            gbinv = cached
    else:
        h_all = _LinearFrozenTN.apply(x_edge, w, b, wt)
        blocks = _LNSiLUSplit.apply(h_all, lw, lb, mlp.ln_eps, hidden, nblk)
    _sequential_warm(net, start=3)

    def _layer_chain(index):
        """
        Build one radial layer's ``fc2 -> LayerNorm+SiLU -> fc3`` chain as a thunk.

        A factory rather than an inline lambda in the loop: Python closures capture the
        loop *variable*, not its value, so five inline lambdas would every one of them
        read the final index. Binding it as a parameter here freezes it per call.

        Args:
            index: Radial MLP layer index, selecting ``blocks[index + 1]``.

        Returns:
            A zero-argument callable producing that layer's ``(E, C)`` output.
        """

        def _chain():
            if TN_INNER:
                if FUSE_RADLAYER and FP16_RADIAL and FP16_GRAD_RAD:
                    # fc2 + LayerNorm/SiLU + fc3 as one node, so fc2's output
                    # gradient stops being an autograd boundary and can be fp16.
                    return _RadialLayerGradRad.apply(
                        blocks[index + 1],
                        toks[index + 1],
                        stoks[index + 1] if gbinv is not None else None,
                        gbinv,
                        mlp.fc2_weight[index],
                        mlp.fc2_bias[index],
                        fc2t[index],
                        fc2t16[index],
                        fc2w16[index],
                        mlp.ln2_weight[index],
                        mlp.ln2_bias[index],
                        mlp.ln_eps,
                        fc3h[index],
                        mlp.fc3_bias[index],
                        fc3t16[index],
                    )
                h = _LinearFrozenTN.apply(
                    blocks[index + 1],
                    mlp.fc2_weight[index],
                    mlp.fc2_bias[index],
                    fc2t[index],
                )
                if FP16_RADIAL and FP16_GRAD_RAD:
                    # Returns (radial, tok, tok_s); `_block` unpacks the triple and
                    # hands the tokens to the span, whose backward fills them.
                    return _LNSiLUFp16LinearGradRad.apply(
                        h,
                        mlp.ln2_weight[index],
                        mlp.ln2_bias[index],
                        mlp.ln_eps,
                        fc3h[index],
                        mlp.fc3_bias[index],
                        fc3t[index],
                        fc3t16[index],
                    )
                if FP16_RADIAL:
                    return _LNSiLUFp16Linear.apply(
                        h,
                        mlp.ln2_weight[index],
                        mlp.ln2_bias[index],
                        mlp.ln_eps,
                        fc3h[index],
                        mlp.fc3_bias[index],
                        fc3t[index],
                    )
                h = ln_silu(h, mlp.ln2_weight[index], mlp.ln2_bias[index], mlp.ln_eps)
                return _LinearFrozenTN.apply(
                    h, mlp.fc3_weight[index], mlp.fc3_bias[index], fc3t[index]
                )
            h = F.linear(blocks[index + 1], mlp.fc2_weight[index], mlp.fc2_bias[index])
            h = ln_silu(h, mlp.ln2_weight[index], mlp.ln2_bias[index], mlp.ln_eps)
            return F.linear(h, mlp.fc3_weight[index], mlp.fc3_bias[index])

        return _chain

    # Block 0 rejoins the edge-degree net after its LayerNorm+SiLU (layers 0..2). It
    # leads the list so it runs on the calling stream, since it is the longest chain
    # and the other four then overlap behind it rather than after it.
    def _degree_chain():
        if narrow:
            # Block 0's first consumer takes the fp16 block directly; the rest of
            # the net continues from layer 4 in fp32.
            head = _LinearFp16Tok.apply(
                blocks[0],
                toks[0],
                stoks[0],
                deg_w16,
                layers[3].bias,
                layers[3]._uma_nitro_wt,
            )
            out = _sequential(net, head, start=4, stop=degree_stop)
        else:
            out = _sequential(net, blocks[0], start=3, stop=degree_stop)
        return out if after is None else after(out)

    results = _parallel(
        [_degree_chain] + [_layer_chain(i) for i in range(mlp.num_layers)]
    )
    return results[0], results[1:]


FUSE_M0LIN = os.environ.get("UMA_NITRO_FUSE_M0LIN", "1") == "1"
FP16_M0RAD = os.environ.get("UMA_NITRO_FP16_M0RAD", "1") == "1"


class _DegreeReduceFused(torch.autograd.Function):
    """
    Combine the edge-degree output linear and m=0 node reduction.

    Their private intermediate gradient may use fp16 without changing the fp32
    autograd boundary.
    """

    @staticmethod
    def forward(
        ctx,
        h,
        weight,
        bias,
        wt,
        wt16,
        wig,
        offsets,
        residual,
        target,
        rescale,
        coeffs,
        channels,
    ):
        """
        Args:
            h: The net's penultimate activation, ``(E, C)``.
            weight: Final Linear weight, ``(3C, C)``.
            bias: Final Linear bias, ``(3C,)``.
            wt: Contiguous ``(C, 3C)`` fp32 copy of ``weight.t()``.
            wt16: fp16 copy of the same, the narrowed backward GEMM's B operand.
            wig: Inverse Wigner with the envelope, ``(E, 9, 9)``.
            offsets: Target-side CSR row pointers.
            residual: Node residual, ``(N, 9, C)``.
            target: Target node per edge.
            rescale: Edge-degree rescale factor.
            coeffs: m=0 coefficient count, 3.
            channels: Sphere channels.

        Returns:
            Initial node features, ``(N, 9, C)``.
        """
        radial = F.linear(h, weight, bias).reshape(-1, coeffs, channels)
        out = residual.clone() if FED_ESPLIT > 1 else torch.empty_like(residual)
        torch.ops.uma_nitro.fused_m0_segreduce(
            wig.contiguous(), radial.contiguous(), offsets, residual, out, rescale
        )
        ctx.save_for_backward(wig, radial, target, wt, wt16)
        ctx.rescale = rescale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: ``(N, 9, C)``.

        Returns:
            The gradient of ``h``, of ``wig`` and of ``residual``; the Linear's
            parameters are frozen.
        """
        wig, radial, target, wt, wt16 = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_wig = torch.zeros_like(wig)
        grad_rad = torch.empty(
            radial.shape,
            device=radial.device,
            dtype=torch.float16 if FP16_M0RAD else radial.dtype,
        )
        torch.ops.uma_nitro.fused_m0_bwd(
            grad_out, target, wig, radial, grad_wig, grad_rad, ctx.rescale
        )
        flat = grad_rad.reshape(grad_rad.shape[0], -1)
        # fp16_mm computes ``a @ w.t()`` with ``w`` stored ``(N, K)``; here N is the
        # Linear's input width and K its output width, so ``w`` is the same
        # ``(in, out)`` transpose the fp32 path uses.
        dh = fp16_mm(flat, wt16) if FP16_M0RAD else flat.mm(wt.t())
        tail = (None,) * 8
        return dh, None, None, None, None, grad_wig, None, grad_out, *tail


def _backbone_forward(backbone, tensors, dataset_name, graph):
    atomic_numbers = tensors["atomic_numbers"].long()
    pos = tensors["pos"]
    cell = tensors["cell"]
    batch = tensors["batch"]
    natoms = tensors["natoms"]
    charge = tensors["charge"]
    spin = tensors["spin"]

    # NOTE: dynamo drops record_function inside compiled regions, so these
    # inner spans show up only in an eager trace. The spans in
    # `forward` wraps the compiled call and always appears.
    dataset_key = backbone.dataset_embedding.dataset_mapping[dataset_name]
    dataset_module = backbone.dataset_embedding.dataset_emb_dict._modules[dataset_key]
    charge_w = backbone.charge_embedding.rand_emb.weight
    spin_w = backbone.spin_embedding.rand_emb.weight
    # One kernel for the whole per-system embedding chain: two index offsets, two
    # `.long()` casts, three table lookups, a `zeros(())` for the dataset index and
    # the `cat` that placed the three blocks side by side -- nine launches on a
    # single row of a few hundred floats. All of it is index arithmetic and gathers,
    # so the result is bit-identical.
    csd_cat = torch.empty(
        1,
        charge_w.shape[1] + spin_w.shape[1] + dataset_module.weight.shape[1],
        device=pos.device,
        dtype=charge_w.dtype,
    )
    torch.ops.uma_nitro.csd_cat(
        charge,
        spin,
        charge_w,
        spin_w,
        dataset_module.weight[0],
        csd_cat,
        int(backbone.charge_embedding._index_offset),
        int(backbone.spin_embedding._index_offset),
    )
    # The SiLU is *not* applied here: it is folded into the per-node gather below,
    # which is the only consumer.
    csd_mixed_emb = _linear(backbone.mix_csd, csd_cat)

    # requires_grad_ is set in _move_inputs, outside the compiled region:
    # dynamo cannot trace a change to a tensor's requires_grad.
    edge_index, edge_distance, edge_distance_vec, node_offsets = graph
    # One (2, E) int32 cast for every layer's backward; see _SPAN_I32.
    _edge_index_i32(edge_index)

    # One kernel for `silu(csd)[batch]` and the whole `(N, P, C)` node init. The
    # eager form was silu, index_select, zeros, embedding and two slice-assign
    # `copy_` -- seven launches, of which two allocated and then immediately
    # overwrote the same rows. Nothing here carries a gradient: the embedding
    # tables and `mix_csd` are frozen and `charge`/`spin` are integer inputs.
    sys_node_embedding = torch.empty(
        atomic_numbers.shape[0],
        backbone.sphere_channels,
        device=pos.device,
        dtype=pos.dtype,
    )
    x_message = torch.empty(
        atomic_numbers.shape[0],
        backbone.sph_feature_size,
        backbone.sphere_channels,
        device=pos.device,
        dtype=pos.dtype,
    )
    torch.ops.uma_nitro.node_init(
        csd_mixed_emb,
        batch,
        backbone.sphere_embedding.weight,
        atomic_numbers,
        sys_node_embedding,
        x_message,
    )

    envelope = backbone.envelope
    _assert_envelope_exponent(envelope)
    _assert_lmax2(backbone)
    _assert_m_to_l_table()
    distance_expansion = backbone.distance_expansion
    _xe_narrow = _xegrad_narrow()
    _emb = edge_embedding(
        edge_distance,
        distance_expansion.offset,
        atomic_numbers,
        edge_index,
        backbone.source_embedding.weight,
        backbone.target_embedding.weight,
        float(distance_expansion.coeff),
        float(backbone.cutoff),
        float(envelope.a),
        float(envelope.b),
        float(envelope.c),
        _xe_narrow,
        _xedge_store_narrow(),
    )
    x_edge, edge_env = _emb[0], _emb[1]
    xe_tok = _emb[2] if _xe_narrow else None
    # Deliberately sequenced *after* edge_embedding, which is the only reason this
    # is not next to the other per-edge geometry: the envelope is folded into
    # wigner_inv inside the expansion kernel, so it has to exist first. The two
    # computations are otherwise independent and neither consumes RNG the other
    # needs, so the gamma draw is unaffected by the move.
    with record_function("wigner"):
        wigner, wigner_inv_envelope = _quaternion_wigner(
            edge_distance_vec, backbone.wigner_c_l2, edge_env
        )

    edge_degree = backbone.edge_degree_embedding
    # Hoisted above the radial fork so the edge-degree node reduction can join it.
    # `_Fanout` is pure aliasing in the forward and depends on nothing here.
    nlayers = backbone.num_layers
    if os.environ.get("UMA_NITRO_NO_FANOUT") == "1":
        w_fan = [wigner] * nlayers
        wi_fan = [wigner_inv_envelope] * (nlayers + 1)
    else:
        w_fan = _Fanout.apply(wigner, nlayers)
        wi_fan = _Fanout.apply(wigner_inv_envelope, nlayers + 1)

    _node_residual = x_message

    def _edge_degree_reduce(radial):
        """
        Fold the edge-degree matrix product into its node reduction.
        """
        return fused_m0_edge_to_node(
            wi_fan[nlayers],
            radial.reshape(
                -1, edge_degree.m_0_num_coefficients, edge_degree.sphere_channels
            ),
            node_offsets,
            _node_residual,
            edge_index[1],
            edge_degree.rescale_factor,
        )

    # Warmed here, on the calling stream. This branch runs inside ``_parallel``,
    # and a cache populated inside a thunk is populated on a *side* stream -- under
    # capture the first call is the capture, so the buffer would be recorded there
    # and read from the main stream forever after.
    _deg_last = list(edge_degree.rad_func.net._modules.values())[-1]
    if getattr(_deg_last, "_uma_nitro_wt", None) is None:
        _deg_last._uma_nitro_wt = _deg_last.weight.t().contiguous()
    if getattr(_deg_last, "_uma_nitro_wt16", None) is None:
        _deg_last._uma_nitro_wt16 = _deg_last._uma_nitro_wt.half()

    def _edge_degree_reduce_fused(h):
        """
        The same reduction, with the net's final Linear pulled into its node.

        Args:
            h: The net's penultimate activation, ``(E, C)``.

        Returns:
            Initial node features, ``(N, 9, C)``.
        """
        return _DegreeReduceFused.apply(
            h,
            _deg_last.weight,
            _deg_last.bias,
            _deg_last._uma_nitro_wt,
            _deg_last._uma_nitro_wt16,
            wi_fan[nlayers],
            node_offsets,
            _node_residual,
            edge_index[1],
            edge_degree.rescale_factor,
            edge_degree.m_0_num_coefficients,
            edge_degree.sphere_channels,
        )

    if os.environ.get("UMA_NITRO_NO_MERGED_RADIAL") == "1":
        radial = _sequential(edge_degree.rad_func.net, x_edge)
        _merged_per_layer = None
        x_message = _edge_degree_reduce(radial)
    elif M0_FORK:
        # `_fused_m0_segreduce` is one program per node -- 0.76 waves at 100 atoms --
        # and standing alone between the radial fork's join and the block loop it had
        # nothing beside it. Its only input is the edge-degree branch of that same
        # fork, so appending it to that branch lets the four other radial chains fill
        # the machine while it runs. Same kernels in the same per-stream order.
        _fuse_m0lin = FUSE_M0LIN
        x_message, _merged_per_layer = _merged_radials(
            backbone,
            x_edge,
            xe_tok=xe_tok,
            after=_edge_degree_reduce_fused if _fuse_m0lin else _edge_degree_reduce,
            degree_stop=(
                len(edge_degree.rad_func.net._modules) - 1 if _fuse_m0lin else None
            ),
        )
    else:
        radial, _merged_per_layer = _merged_radials(backbone, x_edge, xe_tok)
        x_message = _edge_degree_reduce(radial)

    bc_scratch = _bc_scratch(backbone, x_message, natoms, backbone.num_layers)
    with record_function("radial_mlp"):
        x_edge_per_layer = (
            _unified_radials(backbone._unified_radial_mlp, x_edge)
            if _merged_per_layer is None
            else _merged_per_layer
        )
    for i in range(backbone.num_layers):
        with record_function(f"block_{i}"):
            x_message = _block(
                backbone.blocks[i],
                x_message,
                x_edge_per_layer[i],
                edge_index,
                w_fan[i],
                wi_fan[i],
                sys_node_embedding,
                node_offsets,
            )
            x_message = _balance_charge(
                backbone, x_message, charge, natoms, batch, bc_scratch, i
            )
    with record_function("final_norm"):
        out = _equivariant_rms_norm(backbone.norm, x_message)
    return out, pos, cell


@triton.jit
def _row0_bwd_kernel(
    g_ptr,
    out_ptr,
    num_nodes,
    ROW: tl.constexpr,
    C: tl.constexpr,
    GRID_S: tl.constexpr,
):
    """
    Adjoint of ``x[:, 0, :]``: scatter ``g`` into row 0 and zero the rest, in one pass.
    """
    c = tl.arange(0, C)
    node = tl.program_id(0)
    while node < num_nodes:
        base = node * ROW
        tl.store(out_ptr + base + c, tl.load(g_ptr + node * C + c))
        for i in tl.static_range(1, ROW // C):
            tl.store(out_ptr + base + i * C + c, 0.0)
        node += GRID_S


@triton_op("uma_nitro::row0_bwd", mutates_args={"out"})
def _row0_bwd_op(g: torch.Tensor, out: torch.Tensor) -> None:
    """
    Launch the row-0 adjoint.

    Args:
        g: Gradient of the ``(N, C)`` output, contiguous.
        out: ``(N, P, C)`` destination; mutated in full.
    """
    nodes, coeffs, channels = out.shape
    grid = min(nodes, 4096)
    wrap_triton(_row0_bwd_kernel)[(grid,)](
        g,
        out,
        nodes,
        ROW=coeffs * channels,
        C=channels,
        GRID_S=grid,
        num_warps=4,
    )


class _Row0(torch.autograd.Function):
    """
    Return coefficient row zero with a custom one-kernel adjoint.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Args:
            x: ``(N, P, C)``.

        Returns:
            The ``(N, C)`` ``l=0`` row, as a strided view.
        """
        ctx.shape = x.shape
        return x.select(1, 0)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: ``(N, C)``.

        Returns:
            ``(N, P, C)``, zero outside row 0.
        """
        out = torch.empty(ctx.shape, device=grad_out.device, dtype=grad_out.dtype)
        torch.ops.uma_nitro.row0_bwd(grad_out.contiguous(), out)
        return out


@triton.jit
def _energy_sum_kernel(ne_ptr, batch_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    ``energy.index_add_(0, batch, node_energy.to(float64))`` as one launch.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(ne_ptr + i, mask=m, other=0.0).to(tl.float64)
    b = tl.load(batch_ptr + i, mask=m, other=0)
    tl.atomic_add(out_ptr + b, v, mask=m)


@triton.jit
def _energy_sum_bwd_kernel(g_ptr, batch_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    Adjoint of the per-system energy sum: gather and narrow in one pass.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    b = tl.load(batch_ptr + i, mask=m, other=0)
    g = tl.load(g_ptr + b, mask=m, other=0.0)
    tl.store(out_ptr + i, g.to(out_ptr.dtype.element_ty), mask=m)


@triton_op("uma_nitro::energy_sum", mutates_args={"out"})
def _energy_sum(
    node_energy: torch.Tensor, batch: torch.Tensor, out: torch.Tensor
) -> None:
    """
    Accumulate per-node energies into per-system totals in float64.

    Args:
        node_energy: Per-node energy, ``(N,)``.
        batch: System index per atom, ``(N,)``.
        out: Output ``(B,)`` float64, pre-zeroed; mutated.
    """
    n = node_energy.shape[0]
    wrap_triton(_energy_sum_kernel)[(triton.cdiv(n, 512),)](
        node_energy, batch, out, n, BLOCK=512, num_warps=4
    )


@triton_op("uma_nitro::energy_sum_bwd", mutates_args={"out"})
def _energy_sum_bwd(grad: torch.Tensor, batch: torch.Tensor, out: torch.Tensor) -> None:
    """
    Scatter the per-system energy gradient back to nodes.

    Args:
        grad: Gradient of the per-system energy, ``(B,)`` float64.
        batch: System index per atom, ``(N,)``.
        out: Output ``(N,)``; mutated.
    """
    n = out.shape[0]
    wrap_triton(_energy_sum_bwd_kernel)[(triton.cdiv(n, 512),)](
        grad, batch, out, n, BLOCK=512, num_warps=4
    )


class _EnergySum(torch.autograd.Function):
    """
    Per-system energy totals, without the float64 upcast as its own launch.

    The eager form was ``zeros`` + ``.to(float64)`` + ``index_add_`` forward and
    ``index_select`` + ``.to(float32)`` backward -- five launches for a reduction
    onto one scalar. The upcast happens in the accumulator here, so it is free, and
    the atomic ordering is the one ``index_add_`` already had.
    """

    @staticmethod
    def forward(ctx, node_energy, batch, num_systems):
        """
        Args:
            node_energy: Per-node energy, ``(N,)``.
            batch: System index per atom, ``(N,)``.
            num_systems: Number of systems.

        Returns:
            Per-system energy, ``(B,)`` float64.
        """
        out = torch.zeros(num_systems, device=node_energy.device, dtype=torch.float64)
        torch.ops.uma_nitro.energy_sum(node_energy, batch, out)
        ctx.save_for_backward(batch)
        ctx.dtype = node_energy.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: Gradient of the per-system energy, ``(B,)``.

        Returns:
            The per-node gradient; ``None`` for the index and the count.
        """
        (batch,) = ctx.saved_tensors
        out = torch.empty(batch.shape[0], device=grad_out.device, dtype=ctx.dtype)
        torch.ops.uma_nitro.energy_sum_bwd(grad_out.contiguous(), batch, out)
        return out, None, None


def _head_energy(head, node_embedding, batch, natoms):
    """
    Compute per-system energy. Traceable; no differentiation.

    Args:
        head: The energy head.
        node_embedding: Backbone output, shape ``(N, 9, C)``.
        batch: System index per atom, shape ``(N,)``.
        natoms: Atoms per system, shape ``(B,)``.

    Returns:
        Per-system energy in float64, shape ``(B,)``.
    """
    scalar_embedding = _Row0.apply(node_embedding)
    node_energy = _sequential(head.energy_block, scalar_embedding).view(-1)
    return _EnergySum.apply(node_energy, batch, len(natoms))


@triton.jit
def _det3_abs_kernel(c_ptr, out_ptr, n):
    """
    Compute abs(det(cell)) for each 3-by-3 cell.

    The cofactor expansion is evaluated in float64 and rounded once.
    """
    b = tl.program_id(0)
    if b < n:
        p = c_ptr + b * 9
        a0 = tl.load(p + 0).to(tl.float64)
        a1 = tl.load(p + 1).to(tl.float64)
        a2 = tl.load(p + 2).to(tl.float64)
        a3 = tl.load(p + 3).to(tl.float64)
        a4 = tl.load(p + 4).to(tl.float64)
        a5 = tl.load(p + 5).to(tl.float64)
        a6 = tl.load(p + 6).to(tl.float64)
        a7 = tl.load(p + 7).to(tl.float64)
        a8 = tl.load(p + 8).to(tl.float64)
        det = (
            a0 * (a4 * a8 - a5 * a7)
            - a1 * (a3 * a8 - a5 * a6)
            + a2 * (a3 * a7 - a4 * a6)
        )
        tl.store(out_ptr + b, tl.abs(det).to(out_ptr.dtype.element_ty))


@triton_op("uma_nitro::det3_abs", mutates_args={"out"})
def _det3_abs_op(cell: torch.Tensor, out: torch.Tensor) -> None:
    """
    Launch the batched 3x3 ``|det|``.

    Args:
        cell: ``(B, 3, 3)``, contiguous.
        out: ``(B, 1)`` output; mutated.
    """
    wrap_triton(_det3_abs_kernel)[(cell.shape[0],)](
        cell, out, cell.shape[0], num_warps=1
    )


def _cell_volume(cell):
    """
    ``|det(cell)|`` as ``(B, 1)``, in one launch.

    Args:
        cell: ``(B, 3, 3)``.

    Returns:
        ``(B, 1)`` cell volumes.
    """
    out = torch.empty((cell.shape[0], 1), device=cell.device, dtype=cell.dtype)
    torch.ops.uma_nitro.det3_abs(cell.contiguous(), out)
    return out


@triton.jit
def _pos_virial_kernel(
    g_ptr, p_ptr, batch_ptr, out_ptr, num_atoms, GRID_S: tl.constexpr
):
    """
    Scatter the per-atom virial outer product into per-system ``(B, 9)`` fp64.

    The product itself is formed in fp32 and widened before the accumulation, exactly as
    ``(grad.unsqueeze(2) * pos.unsqueeze(1)).to(float64)`` did; only the materialized
    ``(N, 3, 3)`` fp32 intermediate and its fp64 copy disappear. The accumulation is
    ``tl.atomic_add``, the same non-deterministic order ``index_add_`` already had.
    """
    n = tl.program_id(0)
    while n < num_atoms:
        b = tl.load(batch_ptr + n)
        base = b * 9
        for i in tl.static_range(3):
            gi = tl.load(g_ptr + n * 3 + i)
            for j in tl.static_range(3):
                v = gi * tl.load(p_ptr + n * 3 + j)
                tl.atomic_add(out_ptr + base + i * 3 + j, v.to(tl.float64))
        n += GRID_S


@triton_op("uma_nitro::pos_virial", mutates_args={"out"})
def _pos_virial_op(
    grad: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Launch the position-virial scatter.

    Args:
        grad: ``dE/dpos``, ``(N, 3)``.
        pos: Positions, ``(N, 3)``.
        batch: System index per atom, ``(N,)``.
        out: ``(B, 9)`` fp64, pre-zeroed; mutated.
    """
    atoms = pos.shape[0]
    grid = min(atoms, 4096)
    wrap_triton(_pos_virial_kernel)[(grid,)](
        grad, pos, batch, out, atoms, GRID_S=grid, num_warps=1
    )


@triton.jit
def _stress_kernel(pv_ptr, cv_ptr, vol_ptr, out_ptr, n):
    """
    ``stress = ((pv + pv.T + cv + cv.T) / 2) / volume`` for one system per program.

    Five elementwise launches on ``(1, 3, 3)`` -- three adds and two divides -- for 9
    values. The term order is left-associative and the fp32 ``cv``/``volume`` are widened
    before use, matching the torch expression's promotion exactly.
    """
    b = tl.program_id(0)
    if b < n:
        pv = pv_ptr + b * 9
        cv = cv_ptr + b * 9
        vol = tl.load(vol_ptr + b).to(tl.float64)
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                a = tl.load(pv + i * 3 + j)
                bt = tl.load(pv + j * 3 + i)
                c = tl.load(cv + i * 3 + j).to(tl.float64)
                d = tl.load(cv + j * 3 + i).to(tl.float64)
                tl.store(out_ptr + b * 9 + i * 3 + j, (((a + bt) + c) + d) / 2.0 / vol)


@triton_op("uma_nitro::stress_combine", mutates_args={"out"})
def _stress_combine_op(
    pos_virial: torch.Tensor,
    cell_virial: torch.Tensor,
    volume: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Launch the virial symmetrization and volume divide.

    Args:
        pos_virial: ``(B, 9)`` fp64.
        cell_virial: ``(B, 3, 3)``, ``cell.mT @ dE/dcell``.
        volume: ``(B, 1)`` cell volumes.
        out: ``(B, 9)`` fp64 stress; mutated.
    """
    wrap_triton(_stress_kernel)[(volume.shape[0],)](
        pos_virial, cell_virial, volume, out, volume.shape[0], num_warps=1
    )


def _head_derivatives(energy_part, pos, cell, batch):
    """
    Differentiate energy to get forces and stress. Runs outside the graph.

    The model predicts only energy; forces are ``-dE/dpos`` by definition, and
    stress needs both ``dE/dpos`` and ``dE/dcell``. That is what makes this a
    conservative model, and why ``pos`` and ``cell`` carry ``requires_grad``.

    Args:
        energy_part: Per-system energy from ``_head_energy``, shape ``(B,)``.
        pos: Positions, shape ``(N, 3)``, requiring grad.
        cell: Unit cells, shape ``(B, 3, 3)``, requiring grad.
        batch: System index per atom, shape ``(N,)``.

    Returns:
        A dict with normalized energy, forces, and stress.
    """
    grads = torch.autograd.grad(
        [energy_part.sum()],
        [pos, cell],
        create_graph=False,
    )

    def _pos_virial():
        """
        Scatter the per-atom virial outer product into per-system ``(B, 3, 3)`` fp64.

        The ``zeros`` stays inside this thunk with the ``index_add_`` that consumes it.
        They are one unit: the allocation is only ever touched by this branch, and
        splitting them across the fork would put the fill on the calling stream and the
        accumulation on a side one with nothing ordering the two.

        Returns:
            ``(B, 9)`` fp64 position virial.
        """
        out = torch.zeros((cell.shape[0], 9), device=pos.device, dtype=torch.float64)
        torch.ops.uma_nitro.pos_virial(grads[0], pos.detach(), batch, out)
        return out

    pos_virial, forces, cell_virial, volume = _parallel(
        [
            _pos_virial,
            lambda: torch.neg(grads[0]),
            lambda: cell.mT @ grads[1],
            lambda: _cell_volume(cell.detach()),
        ]
    )
    stress = torch.empty_like(pos_virial)
    torch.ops.uma_nitro.stress_combine(
        pos_virial, cell_virial.contiguous(), volume, stress
    )
    return {"energy": energy_part, "forces": forces, "stress": stress}


@functools.cache
def _device_buffer(name, shape, dtype, device):
    """
    Return a cached device buffer for one named input.

    Keyed by ``name`` as well as shape: ``natoms``, ``charge`` and ``spin`` are all
    ``(1,)``, so a shape-only key aliases them onto one buffer and whichever field
    is staged last wins for all three -- which produced NaN energies.

    Persistent so a captured CUDA graph can be replayed: the recorded kernels
    hold the addresses seen at capture time.

    Args:
        name: Input field name, part of the cache key.
        shape: Buffer shape.
        dtype: Buffer dtype.
        device: Device string (hashable, for the cache key).

    Returns:
        A device tensor.
    """
    return torch.empty(shape, dtype=dtype, device=device)


@functools.cache
def _pinned_buffer(name, slot, shape, dtype):
    """
    Return a field-specific cached pinned staging buffer.

    Buffers are keyed by field and slot to prevent aliasing between asynchronous
    host-to-device copies.
    """
    return torch.empty(shape, dtype=dtype).pin_memory()


_STAGE_SLOTS = 2
_stage_slot = 0


@functools.cache
def _staging_ready(slot):
    """
    Return the event guarding reuse of one staging slot.
    """
    return torch.cuda.Event()


def _move_inputs(data, device, dtype):
    """
    Stage input fields asynchronously and enable required derivatives.
    """
    tensors = {}
    global _stage_slot
    slot = _stage_slot
    _stage_slot = (_stage_slot + 1) % _STAGE_SLOTS
    ready = _staging_ready(slot)
    ready.synchronize()
    for name in (
        "atomic_numbers",
        "pos",
        "cell",
        "natoms",
        "batch",
        "pbc",
        "charge",
        "spin",
    ):
        source = getattr(data, name)
        if source.is_floating_point():
            source = source.to(dtype)
        if source.device.type == "cpu":
            staging = _pinned_buffer(name, slot, tuple(source.shape), source.dtype)
            staging.copy_(source)
            # Persistent device buffer, not a fresh .to(): a replayed CUDA graph
            # re-executes recorded kernels against the pointers captured at
            # capture time, so every input must live at a stable address and be
            # refreshed by copy_ rather than reallocated.
            dev = _device_buffer(name, tuple(source.shape), source.dtype, str(device))
            # no_grad: the persistent pos/cell buffers keep requires_grad=True
            # across calls, and copy_ into a leaf that requires grad is illegal.
            with torch.no_grad():
                dev.copy_(staging, non_blocking=True)
            tensors[name] = dev
        else:
            dev = _device_buffer(name, tuple(source.shape), source.dtype, str(device))
            with torch.no_grad():
                dev.copy_(source, non_blocking=True)
            tensors[name] = dev

    ready.record()
    tensors["pos"].requires_grad_(True)
    tensors["cell"].requires_grad_(True)
    return tensors


@contextmanager
def _tf32_context():
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    precision = "high" if _USE_TF32 else "highest"
    try:
        torch.backends.cuda.matmul.allow_tf32 = _USE_TF32
        torch.backends.cudnn.allow_tf32 = _USE_TF32
        torch.set_float32_matmul_precision(precision)
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
        torch.set_float32_matmul_precision(original_precision)


# ``_compiled_graph_builder`` is removed for the same reason: it called
# ``torch.compile`` and nothing reached it. ``_build_graph`` keeps its
# ``graph_core`` parameter so a caller can still inject a compiled builder.


def _build_graph(backbone, tensors, graph_core=None, caps=None):
    """
    Build the padded radius graph with explicit capacities.
    """
    if caps is None:
        caps = plan_capacities(tensors["pos"].shape[0], edges_per_atom=EDGES_PER_ATOM)
    builder = _graph_padded if graph_core is None else graph_core
    edge_index, distances, vectors, n_edges, overflow, node_offsets = builder(
        tensors["pos"],
        tensors["cell"],
        tensors["natoms"],
        tensors["batch"],
        tensors["pbc"],
        backbone.cutoff,
        caps,
    )
    del n_edges
    return edge_index, distances, vectors, node_offsets, overflow


def _forward_tensors(backbone, head, tensors, dataset_name, graph):
    """
    The forward-only tensor compute, with nothing untraceable in it.

    This is the function to hand to ``torch.compile``. Two things are
    deliberately *not* in here:

    * ``_tf32_context``. Compiling ``forward`` instead would put
      ``torch.get_float32_matmul_precision()`` inside the graph, and it returns
      a non-Tensor -- an unresumable break, so dynamo compiles up to it and
      abandons the rest of the frame to eager.
    * ``torch.autograd.grad``. ``pos`` and ``cell`` are leaves created outside
      this region, so differentiating with respect to them inside it makes the
      autograd graph cross the compile boundary, which dynamo reports as
      "the autograd graph extends beyond the compiled region boundary". Keeping
      the derivative in the caller sidesteps it entirely rather than fighting
      it, and the backward still runs this compiled graph's backward -- so the
      custom Triton backward kernels are unaffected.

    Args:
        backbone: The eSCN-MD backbone.
        head: The energy head.
        tensors: Device-resident inputs from ``_move_inputs``.
        dataset_name: UMA task name.

    Returns:
        Per-system energy, still normalized, shape ``(B,)``.
    """
    node_embedding, _pos, _cell = _backbone_forward(
        backbone, tensors, dataset_name, graph
    )
    return _head_energy(head, node_embedding, tensors["batch"], tensors["natoms"])


# The sole capture state, keyed by (shapes, dataset, capacities, model).
# Independent CUDA graphs may share a pool only when replayed in capture order;
# capacity keys can be revisited in any order, so a key change replaces the graph.
_GRAPH_STATE = {}

# Monotonically increasing, never reused. `id()` would be wrong here in the one way
# that matters: a freed model's graphs stay in `_GRAPH_STATE` holding its weights'
# addresses, and CPython happily hands the same `id()` to the *next* object at that
# address -- so a rebuilt predictor could match a dead entry, which is precisely the
# bug the token exists to prevent.
_NEXT_MODEL_TOKEN = [0]


def _model_token(model):
    """
    A stable identity for the model a graph is captured against.

    Stamped on the module itself, next to the ``_uma_nitro_*`` weight caches, which
    have the same lifetime and the same reason for existing: both are derived from
    one model's parameters and must not survive it.

    A stale entry is only a dead dict slot -- it cannot be *matched*. On the next
    call, the key mismatch flushes that graph before capturing against the new model.

    Args:
        model: The ``HydraModel`` whose backbone and heads the graph reads.

    Returns:
        An int, distinct for every model this process has captured against.
    """
    token = getattr(model, "_uma_nitro_graph_token", None)
    if token is None:
        _NEXT_MODEL_TOKEN[0] += 1
        token = _NEXT_MODEL_TOKEN[0]
        model._uma_nitro_graph_token = token
    return token


_USE_GRAPH = os.environ.get("UMA_NITRO_NO_CUDAGRAPH") != "1"


# The pool of the sole live graph. A differently keyed graph is captured only after
# this pool and its graph have been released.
_GRAPH_POOL = None


def _flush_graphs():
    """
    Drop the captured graph and release its pool.

    Replays and the output clones queued after them are asynchronous, so all device
    work must finish before graph executables or pool tensors are released.
    """
    global _GRAPH_POOL
    torch.cuda.synchronize()
    for graph, _ in _GRAPH_STATE.values():
        graph.reset()
    _GRAPH_STATE.clear()
    _GRAPH_POOL = None
    torch.cuda.empty_cache()


_WARMUP_STREAM: list = []


def _warmup_stream():
    """
    Return the process-wide capture-warmup stream, creating it on first use.
    """
    if not _WARMUP_STREAM:
        _WARMUP_STREAM.append(torch.cuda.Stream())
    return _WARMUP_STREAM[0]


WARM_SEQ = os.environ.get("UMA_NITRO_WARM_SEQ", "1") == "1"
WARM_SEQ_PROBE = os.environ.get("UMA_NITRO_WARM_SEQ_PROBE", "0") == "1"

CAP_STREAM = os.environ.get("UMA_NITRO_CAP_STREAM", "1") == "1"


def _cap_stream(side):
    """
    Return the stream to capture on, or ``None`` for torch's internal one.
    """
    return side if CAP_STREAM else None


_CAPTURE_POOL: list = []


def _capture(build):
    """
    Warm up on a side stream, then capture ``build`` into a CUDA graph.

    Args:
        build: Zero-argument callable running the whole capturable region and
            returning the output dict.

    Returns:
        ``(graph, outputs)``; ``outputs`` are the graph-pool tensors that each
        replay overwrites in place.
    """
    # Warmup must run on a non-default stream: capture reuses the allocator's
    # state, and warming up on the default stream leaves blocks that the graph
    # pool cannot claim. Cached rather than fresh -- see `_WARMUP_STREAM`.
    side = _warmup_stream()
    side.wait_stream(torch.cuda.current_stream())
    global _GRAPH_POOL, _SEQ_FORK
    dev = torch.cuda.current_device()
    with torch.cuda.stream(side):
        _SEQ_FORK = WARM_SEQ
        try:
            for _ in range(3):
                build()
        finally:
            _SEQ_FORK = False
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    if WARM_SEQ and _MAX_FORK[0]:
        _side_streams(_MAX_FORK[0])
        if WARM_SEQ_PROBE:
            probe = torch.zeros((8, 8), device=f"cuda:{dev}")
            for stream in _side_streams(_MAX_FORK[0]):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    probe.mm(probe)
                torch.cuda.current_stream().wait_stream(stream)
            del probe
        torch.cuda.synchronize()

    # Only one graph is retained. A new key flushes this graph and its pool before
    # capture, so independently replayed graphs never share allocator state.
    graph = torch.cuda.CUDAGraph()
    if _CAPTURE_POOL:
        # Same stream as the warmup: a pool's blocks are stream-bound.
        with torch.cuda.graph(graph, pool=_CAPTURE_POOL[0], stream=side):
            outputs = build()
        _GRAPH_POOL = _CAPTURE_POOL[0]
    else:
        assert _GRAPH_POOL is None
        with torch.cuda.graph(graph, stream=_cap_stream(side)):
            outputs = build()
        _GRAPH_POOL = graph.pool()
    torch.cuda.synchronize()
    return graph, outputs


def _get_or_capture_graph(key, build):
    """Reuse the current graph, or replace it when its cache key changes."""
    state = _GRAPH_STATE.get(key)
    if state is None:
        if _GRAPH_STATE:
            _flush_graphs()
        state = _capture(build)
        _GRAPH_STATE[key] = state
    return state


def _forward_once(model, data, core):
    """
    Run one UMA Nitro inference attempt.

    Static padded shapes make the region capturable. The overflow flag is checked
    before outputs are returned.

    Args:
        model: The prepared ``UMANitroModel``.
        data: A single-system ``AtomicData``.
        core: The tensor compute, normally ``_forward_tensors``.

    Returns:
        Normalized ``energy`` ``(1,)``, ``forces`` ``(N, 3)`` and ``stress``
        ``(1, 9)``.
    """
    backbone = model.backbone
    head = model.head
    dataset_name = data.dataset[0]
    device = backbone.sphere_embedding.weight.device
    dtype = backbone.sphere_embedding.weight.dtype
    with record_function("move_inputs"):
        tensors = _move_inputs(data, device, dtype)

    geometry_key = _geometry_key(data)
    caps = _measured_caps(backbone, tensors, dataset_name, geometry_key)
    capacity_cache_key = _capacity_cache_key(
        tensors["pos"].shape[0], dataset_name, geometry_key
    )
    _apply_regime(tensors["pos"].shape[0], caps["edge_capacity"])

    def _build():
        """
        Run the whole capturable region against the persistent input buffers.
        """
        # Precision getters cannot be traced, so the tf32 context stays here;
        # under capture it is entered once, at capture time, which is correct
        # because it only sets global precision state.
        with _tf32_context():
            with record_function("graph_build"):
                # Fully compile-free: no _compiled_graph_builder.
                graph_all = _build_graph(backbone, tensors, graph_core=None, caps=caps)
                graph, overflow = graph_all[:4], graph_all[4]
            with record_function("forward_core"):
                energy_part = core(backbone, head, tensors, dataset_name, graph)
            with record_function("derivatives"):
                raw = _head_derivatives(
                    energy_part, tensors["pos"], tensors["cell"], tensors["batch"]
                )
            # Outputs stay normalized: the predict unit applies each task's
            # normalizer and element references, as for every other model.
            out = raw
            if FINITE_GUARD:
                # Into the *same* flag, so the assert below covers it and no
                # second launch is needed. Reading `overflow` afterwards sees
                # the update: it is a view of the flag's storage.
                torch.ops.uma_nitro.finite_guard(
                    out["forces"], out["energy"], out["stress"], overflow
                )
            # Carried out of the captured region so it can be inspected after a
            # replay. The capacity is tight now, so silent overflow would mean
            # dropped edges with a plausible-looking energy.
            out["_overflow"] = overflow
            return out

    if not _USE_GRAPH:
        built = _build()
        _check_graph_flag(
            built["_overflow"], capacity_cache_key, tensors["pos"].shape[0]
        )
        del built["_overflow"]
        return built

    key = (
        tuple(tensors["pos"].shape),
        tuple(tensors["natoms"].shape),
        dataset_name,
        tuple(sorted(caps.items())),
        _model_token(model),
    )
    state = _get_or_capture_graph(key, _build)
    graph, outputs = state
    # Shape guard. Unlike the device assert -- which catches a geometry outgrowing a
    # correct capacity -- this catches a *plumbing* mistake: a capacity that never
    # reached the builder, so the graph was captured at some other size than the one
    # keyed here. Cheap, host-side, no sync.
    assert outputs["forces"].shape[0] == tensors["pos"].shape[0], (
        f"captured graph is for {outputs['forces'].shape[0]} atoms, called with "
        f"{tensors['pos'].shape[0]}"
    )
    with record_function("graph_replay"):
        graph.replay()
    _check_graph_flag(outputs["_overflow"], capacity_cache_key, tensors["pos"].shape[0])
    # Clone: every replay overwrites these graph-pool tensors in place, so handing
    # them out directly would make successive calls alias one another -- the
    # correctness harness compares several samples and would silently see them as
    # identical. The outputs are per-system energy/forces/stress, so this is cheap.
    # detach(), not just clone(). Cloning a grad-requiring tensor keeps its autograd
    # graph alive, so a caller retaining predictions across calls retains one full
    # forward graph each -- the leak upstream #2136 fixes at the predict() boundary for
    # conservative models. Nothing here differentiates the outputs again.
    return {k: v.detach().clone() for k, v in outputs.items() if k != "_overflow"}


def _nitro_forward(model, data, core=_forward_tensors):
    """
    Run UMA Nitro inference, rebuilding if the radius graph outgrows its capacity.
    """
    retries = 0
    while True:
        try:
            return _forward_once(model, data, core)
        except _GraphCapacityOverflow as error:
            _LIVE_EDGES.pop(error.cache_key, None)
            if _GRAPH_STATE:
                _flush_graphs()
            if retries >= _CAPACITY_RECOVERY_RETRIES:
                raise UMANitroNotSupportedError(
                    "UMA Nitro: radius-graph capacity still overflowing after "
                    f"{retries} remeasurement(s) at {error.natoms} atoms.\n"
                    f"{_OVERFLOW_HELP}"
                ) from None
            retries += 1


_FLAG_COUPLINGS = (
    (
        "FP16_LN1",
        (
            "FP16_MERGED_BWD",
            "TN_INNER",
            "FUSE_RADLAYER",
            "FP16_RADIAL",
            "FP16_GRAD_RAD",
        ),
    ),
    (
        "FP16_GRADBLK",
        (
            "FP16_LN1",
            "FP16_MERGED_BWD",
            "TN_INNER",
            "FUSE_RADLAYER",
            "FP16_RADIAL",
            "FP16_GRAD_RAD",
        ),
    ),
)


def _flag_coupling_warnings(values=None):
    """
    Report every flag that is requested but made inert by another being off.

    Args:
        values: Mapping of flag name to value; defaults to this module's globals,
            which is what the import-time call uses.

    Returns:
        List of message strings, empty when the configuration is consistent.
    """
    values = globals() if values is None else values
    out = []
    for flag, needs in _FLAG_COUPLINGS:
        if not values.get(flag):
            continue
        missing = [name for name in needs if not values.get(name)]
        if missing:
            out.append(
                f"UMA_NITRO_{flag}=1 is inert because "
                + ", ".join(f"UMA_NITRO_{name}=0" for name in missing)
                + f": {flag}'s fp16 value can only travel through the token-aware "
                "radial-layer nodes those flags provide, so the fp32 path is used "
                "and this narrowing contributes nothing."
            )
    return out


for _coupling_msg in _flag_coupling_warnings():
    warnings.warn(_coupling_msg, RuntimeWarning, stacklevel=2)


# Force and stress precision must be selected before CUDA graph capture.
_FORCE_PATH_FP16_FLAGS = (
    "FP16_D",
    "FP16_DM",
    "FP16_DH",
    "FP16_DX",
    "FP16_MERGED_BWD",
    "FP16_XEGRAD",
    "FP16_GRADBLK",
    "FP16_GRAD_RAD",
    "FP16_M0RAD",
    "FP16_RADIAL",
    "FP16_LN1",
)


def set_force_precision(force_fp32: bool) -> None:
    """
    Select the force and stress numerical precision for UMA Nitro.

    Args:
        force_fp32: Whether force and stress intermediates use fp32. Captured
            graphs are flushed so a change takes effect.
    """
    g = globals()
    for _name in _FORCE_PATH_FP16_FLAGS:
        g[_name] = not force_fp32
    if _GRAPH_STATE:
        _flush_graphs()


# ============================================================================
# Standalone UMA-S model
# ============================================================================
# Everything below turns a UMA-S inference checkpoint into the module tree the
# kernels above read. The containers carry parameters and layout constants only;
# they have no forward of their own. Their attribute names are the ones the
# kernels address (``blocks[i].edge_wise.so2_conv_1.fc_m0`` and so on).

# The checkpoints these kernels were written and qualified for. The hash gate in
# verify_checkpoint_file stands in for checking their fixed architecture.
_SUPPORTED_CHECKPOINTS = {
    "3497615fd30a24c5b35cd3b41a682e6e": NITRO_CHECKPOINT,
    "eb6fed512b3f0801b27dfdd6e8c1f9ee": "uma-s-1p2p1-ft-d4rev",
    "fc041fe67c6e1501bf8b71cc490826f7": "uma-s-1p2p1-ft-r2scan",
}

# Architecture of those checkpoints, baked into the kernels above.
_LMAX_MMAX = 2
_SPHERE_CHANNELS = 128
_HIDDEN_CHANNELS = 128
_EDGE_CHANNELS = 128
_NUM_LAYERS = 4
_CUTOFF = 6.0
_NUM_DISTANCE_BASIS = 32
_MAX_NUM_ELEMENTS = 100
_CHARGE_CHANNELS = (0, 3)  # charge_balanced_channels [0, 1, 2]
# UMA-S 1.2.1 averages the composition embedding without the zero self value.
_COMPOSITION_INCLUDE_SELF = False

# What the kernels compute: conservative energy, forces and stress; no Hessian.
_REGRESS_CONFIG = GradRegressConfig(
    direct_forces=False,
    direct_stress=False,
    forces=True,
    stress=True,
    hessian=False,
    hessian_vmap=False,
)
_RMS_NORM_EPS = 1e-5
_EDGE_DEGREE_RESCALE = 5.0
_GAUSSIAN_WIDTH = 2.0
_CHARGE_INDEX_OFFSET = 100
_SPIN_INDEX_OFFSET = 0
_SUPPORTED_PROPERTIES = frozenset({"energy", "forces", "stress"})


def _l_expand_index(lmax: int) -> torch.Tensor:
    """
    Map each ``(l, m)`` coefficient to its degree ``l``.

    Args:
        lmax: Maximum degree.

    Returns:
        Long tensor of shape ``((lmax + 1) ** 2,)``.
    """
    return torch.cat(
        [
            torch.full((2 * lval + 1,), lval, dtype=torch.long)
            for lval in range(lmax + 1)
        ]
    )


def _gate_expand_index(lmax: int, mmax: int, m_prime: bool) -> torch.Tensor:
    """
    Gate index for every non-scalar coefficient, as ``GateActivation`` builds it.

    Args:
        lmax: Maximum degree.
        mmax: Maximum order.
        m_prime: Whether coefficients are in m-major (rotated edge) order.

    Returns:
        Long tensor indexing the ``lmax`` gating scalars.
    """
    if m_prime:
        parts = [torch.arange(lmax)]
        for mval in range(1, mmax + 1):
            parts += [torch.arange(mval - 1, lmax), torch.arange(mval - 1, lmax)]
        return torch.cat(parts).long()
    return torch.cat(
        [
            torch.full((min(2 * lval + 1, 2 * mmax + 1),), lval - 1, dtype=torch.long)
            for lval in range(1, lmax + 1)
        ]
    )


def _frozen_linear(in_features: int, out_features: int, bias: bool = True):
    """
    Return an uninitialized ``nn.Linear``; its values come from the checkpoint.
    """
    return nn.Linear(in_features, out_features, bias=bias, device="meta").to_empty(
        device="cpu"
    )


def _radial_net(channels: list[int]) -> nn.Sequential:
    """
    ``Linear -> LayerNorm -> SiLU`` stack ending in a ``Linear``, as in ``RadialMLP``.
    """
    layers = []
    for i in range(1, len(channels)):
        layers.append(_frozen_linear(channels[i - 1], channels[i]))
        if i < len(channels) - 1:
            layers += [nn.LayerNorm(channels[i]), nn.SiLU()]
    return nn.Sequential(*layers)


class _RadialMLP(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.net = _radial_net(channels)


class _EquivariantRMSNorm(nn.Module):
    def __init__(self, lmax: int, num_channels: int) -> None:
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = _RMS_NORM_EPS
        self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer("expand_index", _l_expand_index(lmax), persistent=False)
        weight = torch.cat(
            [
                torch.full((2 * lval + 1, 1), 1.0 / (2 * lval + 1))
                for lval in range(lmax + 1)
            ]
        )
        self.register_buffer(
            "balance_degree_weight", weight / (lmax + 1), persistent=False
        )


class _SO3Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(lmax + 1, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("expand_index", _l_expand_index(lmax), persistent=False)


class _GateActivation(nn.Module):
    def __init__(self, lmax: int, mmax: int, num_channels: int, m_prime: bool):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.register_buffer(
            "expand_index", _gate_expand_index(lmax, mmax, m_prime), persistent=False
        )


class _SO2mConv(nn.Module):
    """
    One ``m > 0`` SO(2) linear in block-diagonal form, ``[[W1, -W2], [W2, W1]]``.
    """

    def __init__(self, m: int, channels: int, out_channels: int, lmax: int):
        super().__init__()
        num_l = lmax - m + 1
        self.out_channels_half = out_channels * num_l
        self.fc = _frozen_linear(num_l * channels, 2 * self.out_channels_half, False)
        self.register_buffer("_w_block", None, persistent=False)

    @torch.no_grad()
    def _build_w_block(self) -> None:
        w1, w2 = self.fc.weight.split(self.out_channels_half, dim=0)
        self._w_block = torch.cat(
            [torch.cat([w1, -w2], dim=1), torch.cat([w2, w1], dim=1)], dim=0
        ).contiguous()


class _SO2Convolution(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        lmax: int,
        mmax: int,
        extra_m0_output_channels: int,
    ) -> None:
        super().__init__()
        self.extra_m0_output_channels = extra_m0_output_channels
        in_m0 = (lmax + 1) * channels
        self.fc_m0 = _frozen_linear(
            in_m0, out_channels * (lmax + 1) + extra_m0_output_channels
        )
        self.so2_m_conv = nn.ModuleList(
            _SO2mConv(m, channels, out_channels, lmax) for m in range(1, mmax + 1)
        )
        # Coefficients per m-block of a rotated edge: m = 0, then (+m, -m) pairs.
        self.m_split_sizes = [lmax + 1] + [
            2 * (lmax - m + 1) for m in range(1, mmax + 1)
        ]

    def radial_features(self) -> int:
        return self.fc_m0.in_features + sum(c.fc.in_features for c in self.so2_m_conv)


class _Edgewise(nn.Module):
    def __init__(self, channels: int, hidden: int, lmax: int, mmax: int) -> None:
        super().__init__()
        self.act = _GateActivation(lmax, mmax, hidden, m_prime=True)
        self.so2_conv_1 = _SO2Convolution(
            2 * channels, hidden, lmax, mmax, lmax * hidden
        )
        self.so2_conv_2 = _SO2Convolution(hidden, channels, lmax, mmax, 0)


class _SpectralAtomwise(nn.Module):
    def __init__(self, channels: int, hidden: int, lmax: int) -> None:
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            _frozen_linear(channels, lmax * hidden), nn.SiLU()
        )
        self.so3_linear_1 = _SO3Linear(channels, hidden, lmax)
        self.act = _GateActivation(lmax, lmax, hidden, m_prime=False)
        self.so3_linear_2 = _SO3Linear(hidden, channels, lmax)


class _Block(nn.Module):
    def __init__(self, channels: int, hidden: int, lmax: int, mmax: int) -> None:
        super().__init__()
        self.norm_1 = _EquivariantRMSNorm(lmax, channels)
        self.edge_wise = _Edgewise(channels, hidden, lmax, mmax)
        self.norm_2 = _EquivariantRMSNorm(lmax, channels)
        self.atom_wise = _SpectralAtomwise(channels, hidden, lmax)


class _ChgSpinEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, channels: int, index_offset: int):
        super().__init__()
        self._index_offset = index_offset
        self.rand_emb = nn.Embedding(num_embeddings, channels)


class _DatasetEmbedding(nn.Module):
    def __init__(self, channels: int, dataset_mapping: dict[str, str]) -> None:
        super().__init__()
        self.dataset_mapping = dict(dataset_mapping)
        self.dataset_emb_dict = nn.ModuleDict(
            {name: nn.Embedding(1, channels) for name in dataset_mapping}
        )


class _EdgeDegreeEmbedding(nn.Module):
    def __init__(self, channels: int, lmax: int, edge_channels: list[int]) -> None:
        super().__init__()
        self.sphere_channels = channels
        self.m_0_num_coefficients = lmax + 1
        self.rescale_factor = _EDGE_DEGREE_RESCALE
        self.rad_func = _RadialMLP([*edge_channels, (lmax + 1) * channels])


class _GaussianSmearing(nn.Module):
    def __init__(self, stop: float, num_gaussians: int) -> None:
        super().__init__()
        offset = torch.linspace(0.0, stop, num_gaussians, dtype=torch.float32)
        self.coeff = -0.5 / (_GAUSSIAN_WIDTH * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)


class _PolynomialEnvelope:
    def __init__(self, exponent: float) -> None:
        self.p = float(exponent)
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2


class _UnifiedRadialMLP(nn.Module):
    """
    Every block's conv1 radial MLP in one layout: the first layers concatenated
    for a single GEMM, the rest stacked per block.
    """

    def __init__(self, nets: list[nn.Sequential]) -> None:
        super().__init__()
        self.num_layers = len(nets)
        self.hidden_features = nets[0][0].out_features
        self.ln_eps = nets[0][1].eps
        for name, index, combine in (
            ("W1_cat", (0, "weight"), torch.cat),
            ("b1_cat", (0, "bias"), torch.cat),
            ("ln1_weight", (1, "weight"), torch.stack),
            ("ln1_bias", (1, "bias"), torch.stack),
            ("fc2_weight", (3, "weight"), torch.stack),
            ("fc2_bias", (3, "bias"), torch.stack),
            ("ln2_weight", (4, "weight"), torch.stack),
            ("ln2_bias", (4, "bias"), torch.stack),
            ("fc3_weight", (6, "weight"), torch.stack),
            ("fc3_bias", (6, "bias"), torch.stack),
        ):
            layer, attr = index
            self.register_buffer(
                name,
                combine([getattr(net[layer], attr).detach() for net in nets], dim=0),
                persistent=False,
            )


class _Backbone(nn.Module):
    def __init__(self, dataset_mapping: dict[str, str]) -> None:
        super().__init__()
        lmax = mmax = _LMAX_MMAX
        channels, hidden, edge = _SPHERE_CHANNELS, _HIDDEN_CHANNELS, _EDGE_CHANNELS
        num_elements, num_basis = _MAX_NUM_ELEMENTS, _NUM_DISTANCE_BASIS
        # Read by the predict unit like eSCNMDBackbone.regress_config; a copy so
        # no caller can change the shared constant.
        self.regress_config = dataclasses.replace(_REGRESS_CONFIG)
        self.lmax = lmax
        self.mmax = mmax
        self.sph_feature_size = (lmax + 1) ** 2
        self.sphere_channels = channels
        self.num_layers = _NUM_LAYERS
        self.cutoff = _CUTOFF
        self.charge_channel_start, self.charge_channel_end = _CHARGE_CHANNELS

        self.sphere_embedding = nn.Embedding(num_elements, channels)
        self.charge_embedding = _ChgSpinEmbedding(201, channels, _CHARGE_INDEX_OFFSET)
        self.spin_embedding = _ChgSpinEmbedding(101, channels, _SPIN_INDEX_OFFSET)
        self.dataset_embedding = _DatasetEmbedding(channels, dataset_mapping)
        self.mix_csd = _frozen_linear(3 * channels, channels)
        self.distance_expansion = _GaussianSmearing(self.cutoff, num_basis)
        self.source_embedding = nn.Embedding(num_elements, edge)
        self.target_embedding = nn.Embedding(num_elements, edge)
        edge_channels = [num_basis + 2 * edge, edge, edge]
        self.edge_degree_embedding = _EdgeDegreeEmbedding(channels, lmax, edge_channels)
        self.envelope = _PolynomialEnvelope(_ENV_EXPONENT)
        self.blocks = nn.ModuleList(
            _Block(channels, hidden, lmax, mmax) for _ in range(self.num_layers)
        )
        for block in self.blocks:
            conv1 = block.edge_wise.so2_conv_1
            conv1.rad_func = _RadialMLP([*edge_channels, conv1.radial_features()])
        self.norm = _EquivariantRMSNorm(lmax, channels)
        self.register_buffer("wigner_c_l2", _wigner_l2_coefficients(), persistent=False)
        self._unified_radial_mlp = None

    def build_unified_radial_mlp(self) -> None:
        """
        Gather the per-block radial MLPs into the unified layout the kernels read.
        """
        self._unified_radial_mlp = _UnifiedRadialMLP(
            [block.edge_wise.so2_conv_1.rad_func.net for block in self.blocks]
        )
        for block in self.blocks:
            del block.edge_wise.so2_conv_1.rad_func

    def get_default_untrained_tasks(self, checkpoint_tasks: dict, settings) -> list:
        """
        Stress tasks for every energy dataset without one, as the UMA backbone adds.
        """
        from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task

        energy_tasks = {}
        stress_datasets = set(settings.predict_untrained_stress)
        for task in checkpoint_tasks.values():
            if task.property == "energy":
                energy_tasks.update(dict.fromkeys(task.datasets, task))
            elif task.property == "stress":
                stress_datasets.update(task.datasets)
        tasks = []
        for dataset in set(energy_tasks) - stress_datasets:
            energy_task = energy_tasks[dataset]
            prefix = "" if energy_task.name == "energy" else f"{dataset}_"
            tasks.append(
                Task(
                    name=f"{prefix}stress",
                    level="system",
                    property="stress",
                    out_spec=OutputSpec(
                        dim=[1, 9], dtype=settings.base_precision_dtype
                    ),
                    normalizer=energy_task.normalizer,
                    datasets=[dataset],
                    loss_fn=None,
                    element_references=None,
                    metrics=[],
                    train_on_free_atoms=True,
                    eval_on_free_atoms=True,
                    inference_only=True,
                )
            )
        return tasks


class _EnergyHead(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.energy_block = nn.Sequential(
            _frozen_linear(channels, hidden),
            nn.SiLU(),
            _frozen_linear(hidden, hidden),
            nn.SiLU(),
            _frozen_linear(hidden, 1),
        )


def _wigner_l2_coefficients() -> torch.Tensor:
    """
    Load the precomputed degree-2 Wigner polynomial coefficients.
    """
    from fairchem.core.models.uma.common.quaternion.wigner_d_custom_kernels import (
        _load_and_decompress_coefficients,
    )

    return _load_and_decompress_coefficients()["C_l2"]


def verify_checkpoint_file(path) -> None:
    """
    Check that ``path`` is one of the checkpoints UMA Nitro supports.

    The MD5 sum is cached in a ``<path>.md5`` sidecar (``cached_file_md5``).

    Args:
        path: Checkpoint file.

    Raises:
        UMANitroNotSupportedError: If the file's MD5 is not on the allowlist.
    """
    digest = cached_file_md5(path)
    if digest not in _SUPPORTED_CHECKPOINTS:
        supported = ", ".join(
            f"{name} (md5 {checksum})"
            for checksum, name in _SUPPORTED_CHECKPOINTS.items()
        )
        raise UMANitroNotSupportedError(
            f"{path} is not supported in nitro mode (md5 {digest}); supported "
            f"checkpoints are {supported}"
        )


def _validate_device(device) -> None:
    """
    Check that ``device`` supports Nitro and matches the selected kernel policy.

    Raises:
        UMANitroNotSupportedError: Off CUDA, outside compute capability 8.x/9.x/10.x,
            or on a GPU other than the one the kernels were chosen for.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise UMANitroNotSupportedError("nitro mode requires a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (8, 9, 10):
        raise UMANitroNotSupportedError(
            "nitro mode supports only CUDA compute capability 8.x, 9.x, or 10.x, "
            f"not {capability[0]}.{capability[1]}"
        )
    if capability != _CUDA_CAPABILITY:
        # The kernel policy is chosen once, for the GPU current at import.
        raise UMANitroNotSupportedError(
            f"{device} has compute capability {capability[0]}.{capability[1]}, but "
            "UMA Nitro chose its kernels for compute capability "
            f"{_CUDA_CAPABILITY[0]}.{_CUDA_CAPABILITY[1]}"
        )


def _validate_settings(settings) -> None:
    """
    Check that Nitro can honor the inference settings.

    Raises:
        UMANitroNotSupportedError: For a setting outside Nitro's contract.
    """
    reasons = {
        "execution_mode is set (it selects HydraModel kernels)": (
            settings.execution_mode is not None
        ),
        "merge_mole=False": not settings.merge_mole,
        "activation checkpointing": settings.activation_checkpointing,
        "external graph generation": settings.external_graph_gen,
        "a precision other than float32": (
            settings.base_precision_dtype != torch.float32
        ),
        "tf32=False (Nitro computes in TF32)": not settings.tf32,
        "Hessian prediction": bool(settings.predict_untrained_hessian),
    }
    unsupported = [reason for reason, applies in reasons.items() if applies]
    if unsupported:
        raise UMANitroNotSupportedError(", ".join(unsupported))


def _load_tensor(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """
    Copy one checkpoint tensor into a container parameter, checking its shape.
    """
    target = module.get_parameter(name)
    if target.shape != value.shape:
        raise UMANitroNotSupportedError(
            f"UMA Nitro: checkpoint tensor {name} has shape {tuple(value.shape)}, "
            f"expected {tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(value)


class UMANitroModel(nn.Module):
    """
    Standalone UMA-S model built from checkpoint weights.

    Implements the parts of the ``HydraModel`` interface that ``MLIPPredictUnit``
    uses: task bookkeeping, ``prepare_for_inference`` (the MOLE merge),
    ``on_predict_check`` (the merged-input contract), atoms validation, and a
    forward returning normalized outputs keyed by task. One system is evaluated
    per call with a fixed composition, charge, spin and task; any other input
    raises ``MergeMoleConsistencyError`` so the predict unit falls back exactly
    as it does for the multigpu preset's fast path.
    """

    def __init__(self, model_config: dict, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        # The architecture is fixed (see verify_checkpoint_file); the config only
        # supplies data: the task datasets and single-atom support.
        dataset_mapping = model_config["backbone"]["dataset_mapping"]
        ((head_name, head_config),) = model_config["heads"].items()
        self.supports_single_atoms = bool(model_config.get("supports_single_atoms"))
        self.backbone = _Backbone(dataset_mapping)
        self.head = _EnergyHead(_SPHERE_CHANNELS, _HIDDEN_CHANNELS)
        wrapped_head = "head_cls" in head_config
        self._head_prefix = (
            f"output_heads.{head_name}.head."
            if wrapped_head
            else f"output_heads.{head_name}."
        )
        head_mapping = head_config.get("dataset_mapping") or dataset_mapping
        targets = sorted(set(head_mapping.values()))
        self._head_expert_index = {
            name: targets.index(target) for name, target in head_mapping.items()
        }
        self._tasks: dict = {}
        self._dataset_to_tasks: dict = {}
        self._merged_key = None
        # Unmerged MOLE tensors stay on the CPU, outside the module tree, until
        # prepare_for_inference merges them for the first system.
        self._experts = self._load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(cls, checkpoint) -> UMANitroModel:
        """
        Build the model from a loaded UMA inference checkpoint.

        Args:
            checkpoint: An ``MLIPInferenceCheckpoint`` (``torch.load`` result).

        Returns:
            The unprepared model, with its checkpoint tasks set up.
        """
        model = cls(checkpoint.model_config, checkpoint.ema_state_dict)
        model.setup_tasks(checkpoint.tasks_config)
        return model

    def _load_state_dict(self, state_dict: dict) -> dict[str, torch.Tensor]:
        """
        Load plain weights into the containers and return the MOLE tensors.

        Args:
            state_dict: EMA state dict; keys may carry ``AveragedModel``'s
                ``module.`` prefix.

        Returns:
            The expert weights and shared biases, the routing MLP and the
            composition embedding, keyed by checkpoint name without the prefix.
        """
        tensors = {
            key.removeprefix("module."): value.detach()
            for key, value in state_dict.items()
            if key != "n_averaged"
        }
        mole_layers = {
            key.removesuffix(".weights") for key in tensors if key.endswith(".weights")
        }
        experts = {}
        parameters = dict(self.named_parameters())
        loaded = set()
        for key, value in tensors.items():
            if key.startswith("backbone.Jd_"):
                continue  # Euler-path rotation tables; Nitro uses quaternions.
            if key.rsplit(".", 1)[0] in mole_layers or key.startswith(
                ("backbone.routing_mlp.", "backbone.composition_embedding.")
            ):
                experts[key] = value.to("cpu", torch.float32)
                continue
            name = key.replace(self._head_prefix, "head.", 1)
            if name not in parameters:
                raise UMANitroNotSupportedError(
                    f"UMA Nitro: unexpected checkpoint tensor {key}"
                )
            _load_tensor(self, name, value)
            loaded.add(name)
        merged_later = {
            name
            for name in parameters
            if _expert_key(name, self._head_prefix) in mole_layers
        }
        missing = set(parameters) - loaded - merged_later
        if missing:
            raise UMANitroNotSupportedError(
                f"UMA Nitro: checkpoint is missing {sorted(missing)[:5]}"
            )
        self.backbone.build_unified_radial_mlp()
        return experts

    # ------------------------------------------------------------------ tasks
    @property
    def tasks(self) -> dict:
        return self._tasks

    @property
    def dataset_to_tasks(self) -> dict:
        return self._dataset_to_tasks

    def setup_tasks(self, tasks_config: list) -> None:
        """
        Instantiate the checkpoint tasks.
        """
        import hydra

        self._tasks = {}
        self.add_tasks([hydra.utils.instantiate(config) for config in tasks_config])

    def add_tasks(self, tasks) -> None:
        """
        Register tasks; Nitro always computes energy, forces and stress.
        """
        from fairchem.core.models.base import _get_dataset_to_tasks_map

        for task in tasks:
            if task.property not in _SUPPORTED_PROPERTIES:
                raise ValueError(
                    f"uma_nitro predicts energy, forces and stress only; task "
                    f"{task.name!r} requests {task.property!r}"
                )
            unknown = set(task.datasets) - set(self._head_expert_index)
            if unknown:
                raise ValueError(f"task {task.name!r} uses unknown datasets {unknown}")
            self._tasks.setdefault(task.name, task)
        self._dataset_to_tasks = _get_dataset_to_tasks_map(self._tasks.values())

    def validate_atoms_data(self, atoms, task_name: str) -> None:
        from fairchem.core.units.mlip_unit.api.inference import validate_uma_atoms_data

        validate_uma_atoms_data(atoms, task_name)

    # ------------------------------------------------------------ inference
    @staticmethod
    def validate(settings, path, device) -> None:
        """
        Check that Nitro can serve this request: the GPU, the settings, then the
        checkpoint file (the only check that reads it).

        Args:
            settings: The predictor's inference settings.
            path: The checkpoint file.
            device: The predictor's device.

        Raises:
            UMANitroNotSupportedError: For a request outside Nitro's contract.
        """
        _validate_device(device)
        _validate_settings(settings)
        verify_checkpoint_file(path)

    def _merge_key(self, data) -> tuple:
        """
        The inputs the merged weights depend on; one system per call.
        """
        from fairchem.core.units.mlip_unit.api.inference import (
            MergeMoleConsistencyError,
        )

        if data.natoms.numel() != 1:
            raise MergeMoleConsistencyError("UMA Nitro evaluates one system per call")
        counts = torch.bincount(
            data.atomic_numbers.long().cpu(), minlength=_MAX_NUM_ELEMENTS
        )
        reduced = counts.double() / counts.sum()
        return (
            reduced,
            int(data.charge.reshape(-1)[0]),
            int(data.spin.reshape(-1)[0]),
            data.dataset[0],
        )

    def on_predict_check(self, data) -> None:
        """
        Raise ``MergeMoleConsistencyError`` if ``data`` breaks the merged contract.
        """
        if self._merged_key is None:
            return
        from fairchem.core.units.mlip_unit.api.inference import (
            MergeMoleConsistencyError,
        )

        reduced, charge, spin, dataset = self._merge_key(data)
        merged = self._merged_key
        if not torch.allclose(reduced, merged[0], rtol=1e-5, atol=0.0):
            raise MergeMoleConsistencyError("Compositions differ from merged model")
        if charge != merged[1]:
            raise MergeMoleConsistencyError("Charge differs from merged model")
        if spin != merged[2]:
            raise MergeMoleConsistencyError("Spin differs from merged model")
        if dataset != merged[3]:
            raise MergeMoleConsistencyError("Task differs from merged model")

    def prepare_for_inference(self, data, settings) -> UMANitroModel:
        """
        Merge the MOLE experts for ``data``'s system and select fp32 forces.

        Args:
            data: The first system to be evaluated.
            settings: Inference settings, validated against Nitro's contract.

        Returns:
            ``self``.
        """
        _validate_settings(settings)
        if self._merged_key is not None:
            return self
        key = self._merge_key(data)
        dataset = key[3]
        if dataset not in self._head_expert_index:
            raise ValueError(f"unknown task {dataset!r} for this checkpoint")
        coefficients = self._expert_coefficients(data)
        for name, _ in list(self.named_parameters()):
            layer = _expert_key(name, self._head_prefix)
            weights = self._experts.get(f"{layer}.weights")
            if weights is None:
                continue
            if name.endswith(".bias"):
                # Experts share one bias; only the weights are mixed.
                _load_tensor(self, name, self._experts[f"{layer}.bias"])
                continue
            if name.startswith("head."):
                # The head selects the task's expert with a one-hot mixture.
                mix = torch.zeros(1, weights.shape[0], dtype=weights.dtype)
                mix[0, self._head_expert_index[dataset]] = 1.0
            else:
                mix = coefficients
            _load_tensor(self, name, torch.einsum("eoi, be->boi", weights, mix)[0])
        self._experts = {}
        self._merged_key = key
        set_force_precision(True)
        return self

    @torch.no_grad()
    def _expert_coefficients(self, data) -> torch.Tensor:
        """
        Expert mixing coefficients for one system, as the UMA routing MLP computes.
        """
        backbone = self.backbone
        charge = data.charge.reshape(-1)[:1].long().cpu()
        spin = data.spin.reshape(-1)[:1].long().cpu()
        dataset_key = backbone.dataset_embedding.dataset_mapping[data.dataset[0]]
        csd = torch.cat(
            (
                backbone.charge_embedding.rand_emb.weight[
                    charge + _CHARGE_INDEX_OFFSET
                ],
                backbone.spin_embedding.rand_emb.weight[spin + _SPIN_INDEX_OFFSET],
                backbone.dataset_embedding.dataset_emb_dict[dataset_key].weight,
            ),
            dim=1,
        )
        csd = F.silu(backbone.mix_csd(csd))
        experts = self._experts
        atomic_numbers = data.atomic_numbers.long().cpu()
        by_atom = F.embedding(
            atomic_numbers, experts["backbone.composition_embedding.weight"]
        )
        composition = by_atom.new_zeros(1, by_atom.shape[1]).index_reduce_(
            0,
            torch.zeros_like(atomic_numbers),
            by_atom,
            reduce="mean",
            include_self=_COMPOSITION_INCLUDE_SELF,
        )
        x = torch.cat((composition, csd), dim=1)
        for i in (0, 2, 4):
            x = F.silu(
                F.linear(
                    x,
                    experts[f"backbone.routing_mlp.{i}.weight"],
                    experts[f"backbone.routing_mlp.{i}.bias"],
                )
            )
        # The checkpoint's routing uses the default softmax coefficient norm.
        return torch.softmax(x, dim=1) + 0.005

    def forward(self, data) -> dict[str, dict[str, torch.Tensor]]:
        """
        Predict normalized energy, forces and stress for one system.

        Returns:
            ``{task_name: {property: tensor}}`` for the system's task.
        """
        raw = _nitro_forward(self, data)
        # Element references in the predict unit read the full-system fields the
        # composable backbone records on ``data``; Nitro never partitions a graph.
        data["atomic_numbers_full"] = data["atomic_numbers"]
        data["batch_full"] = data["batch"]
        return {
            task.name: {task.property: raw[task.property]}
            for task in self._dataset_to_tasks[data.dataset[0]]
        }

    def release_inference_resources(self) -> None:
        """
        Free captured CUDA graphs before this model is discarded.
        """
        if _GRAPH_STATE:
            _flush_graphs()


def _expert_key(parameter_name: str, head_prefix: str) -> str:
    """
    Checkpoint name of the layer that produces a container parameter.
    """
    module = parameter_name.rsplit(".", 1)[0]
    if module.startswith("head."):
        return module.replace("head.", head_prefix, 1)
    return module
