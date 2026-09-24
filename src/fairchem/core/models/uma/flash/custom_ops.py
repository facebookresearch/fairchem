"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Autograd-aware wrappers around the umas_flash Triton kernels.

Every launch site lives here so three cross-cutting concerns are handled in
one place:

* **Device selection.** Triton resolves the target device from the *ambient*
  current device (``driver.active.get_current_device()``), not from the tensors
  it is handed. ATen inserts a device guard for native ops; ``custom_op`` does
  not. Without ``_device_of`` below, running on ``cuda:1`` while ``cuda:0`` is
  current launches into the wrong context with foreign pointers.
* **Empty partitions.** A graph parallel rank (or a padded batch) can own zero
  edges. Autotuned kernels cannot be benchmarked on an empty grid, so the ops
  short-circuit instead.
* **torch.compile visibility.** ``triton_op`` plus ``wrap_triton`` leaves the
  kernel launches visible to the compiler, so it can plan around them, and
  ``register_autograd`` gives the backward pass the same treatment. Output
  shapes are inferred from the traced body, which is why there are no
  ``register_fake`` implementations here to keep in sync by hand.
"""

from __future__ import annotations

import contextlib

import torch
import triton
from torch import (
    Tensor,  # - needed at runtime for triton_op schema inference
)
from torch.library import triton_op, wrap_triton

from fairchem.core.models.uma.flash.constants import (
    NUM_WIGNER_ENTRIES,
    X_M0_MULT,
    X_M1_MULT,
    X_M2_MULT,
    Z_M0_MULT,
    Z_M1_MULT,
    Z_M2_MULT,
)
from fairchem.core.models.uma.flash.kernels.gather import (
    _gather_split_bwd_l1_kernel,
    _gather_split_bwd_l2_kernel,
    _gather_split_fwd_kernel,
)
from fairchem.core.models.uma.flash.kernels.gating import (
    _gating_split_bwd_kernel,
    _gating_split_fwd_kernel,
)
from fairchem.core.models.uma.flash.kernels.geom import (
    _geom_bwd_kernel,
    _geom_fwd_kernel,
)
from fairchem.core.models.uma.flash.kernels.ln_silu import (
    _ln_silu_bwd_dx_kernel,
    _ln_silu_fwd_kernel,
)
from fairchem.core.models.uma.flash.kernels.radial import (
    _radial_stage1_bwd_kernel,
    _radial_stage1_fwd_kernel,
)
from fairchem.core.models.uma.flash.kernels.scatter import (
    _init_node_l0_add_kernel,
    _init_scatter_bwd_kernel,
    _init_scatter_fwd_kernel,
    _scatter_split_bwd_l1_kernel,
    _scatter_split_bwd_l2_kernel,
    _scatter_split_fwd_kernel,
)

X_M0, X_M1, X_M2 = X_M0_MULT.value, X_M1_MULT.value, X_M2_MULT.value
Z_M0, Z_M1, Z_M2 = Z_M0_MULT.value, Z_M1_MULT.value, Z_M2_MULT.value


def _device_of(tensor: Tensor):
    """
    Make ``tensor``'s device current for the duration of a kernel launch.

    Returns a null context when it already is, which is the single-GPU case and
    must stay free.
    """
    index = tensor.device.index
    if index is None or index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(index)


# =============================================================================
# 1. Geometry: Gaussian basis, cutoff envelope and the 34 Wigner entries
# =============================================================================


@triton_op("fairchem::flash_geom_fwd", mutates_args=())
def flash_geom_fwd(
    edge_vec: Tensor, offset_g: Tensor, coeff_g: float, cutoff: float
) -> tuple[Tensor, Tensor, Tensor]:
    E, B = edge_vec.shape[0], offset_g.shape[0]
    gauss = torch.empty((E, B), device=edge_vec.device, dtype=edge_vec.dtype)
    # Component-major so every consumer reads one Wigner entry as a
    # contiguous vector over edges instead of a stride-34 gather.
    wigner = torch.empty(
        (NUM_WIGNER_ENTRIES, E), device=edge_vec.device, dtype=edge_vec.dtype
    )
    env = torch.empty((E,), device=edge_vec.device, dtype=edge_vec.dtype)
    if E == 0:
        return gauss, wigner, env

    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(edge_vec):
        wrap_triton(_geom_fwd_kernel)[grid](
            edge_vec,
            offset_g,
            gauss,
            wigner,
            env,
            E,
            B,
            float(cutoff),
            float(coeff_g),
            edge_vec.stride(0),
            edge_vec.stride(1),
            gauss.stride(0),
            gauss.stride(1),
            wigner.stride(1),
            wigner.stride(0),
            BLOCK_B=triton.next_power_of_2(B),
        )
    return gauss, wigner, env


@triton_op("fairchem::flash_geom_bwd", mutates_args=())
def flash_geom_bwd(
    g_gauss: Tensor,
    g_wigner: Tensor,
    g_env: Tensor,
    edge_vec: Tensor,
    offset_g: Tensor,
    coeff_g: float,
    cutoff: float,
) -> Tensor:
    E, B = edge_vec.shape[0], offset_g.shape[0]
    g_edge_vec = torch.empty_like(edge_vec)
    if E == 0:
        return g_edge_vec

    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(edge_vec):
        wrap_triton(_geom_bwd_kernel)[grid](
            edge_vec,
            offset_g,
            g_gauss,
            g_wigner,
            g_env,
            g_edge_vec,
            E,
            B,
            float(cutoff),
            float(coeff_g),
            edge_vec.stride(0),
            edge_vec.stride(1),
            g_gauss.stride(0),
            g_gauss.stride(1),
            g_wigner.stride(1),
            g_wigner.stride(0),
            BLOCK_B=triton.next_power_of_2(B),
        )
    return g_edge_vec


def _setup_geom(ctx, inputs, output):
    edge_vec, offset_g, coeff_g, cutoff = inputs
    ctx.save_for_backward(edge_vec, offset_g)
    ctx.coeff_g, ctx.cutoff = coeff_g, cutoff


def _bwd_geom(ctx, g_gauss, g_wigner, g_env):
    edge_vec, offset_g = ctx.saved_tensors
    g_edge_vec = torch.ops.fairchem.flash_geom_bwd(
        g_gauss.contiguous(),
        g_wigner.contiguous(),
        g_env.contiguous(),
        edge_vec,
        offset_g,
        ctx.coeff_g,
        ctx.cutoff,
    )
    return g_edge_vec, None, None, None


flash_geom_fwd.register_autograd(_bwd_geom, setup_context=_setup_geom)


# =============================================================================
# 2. Fused LayerNorm + SiLU
# =============================================================================


@triton_op("fairchem::flash_ln_silu_fwd", mutates_args=())
def flash_ln_silu_fwd(
    x: Tensor, gamma: Tensor, beta: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    M, D = x.shape
    out = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    if M == 0:
        return out, mean, rstd

    with _device_of(x):
        wrap_triton(_ln_silu_fwd_kernel)[(M,)](
            x,
            gamma,
            beta,
            out,
            mean,
            rstd,
            M,
            D,
            float(eps),
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_D=triton.next_power_of_2(D),
        )
    return out, mean, rstd


@triton_op("fairchem::flash_ln_silu_bwd", mutates_args=())
def flash_ln_silu_bwd(
    g_out: Tensor,
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    mean: Tensor,
    rstd: Tensor,
) -> Tensor:
    M, D = x.shape
    g_x = torch.empty_like(x)
    if M == 0:
        return g_x

    with _device_of(x):
        wrap_triton(_ln_silu_bwd_dx_kernel)[(M,)](
            x,
            gamma,
            beta,
            mean,
            rstd,
            g_out,
            g_x,
            M,
            D,
            x.stride(0),
            x.stride(1),
            g_out.stride(0),
            g_out.stride(1),
            g_x.stride(0),
            g_x.stride(1),
            BLOCK_D=triton.next_power_of_2(D),
        )
    return g_x


def _setup_ln(ctx, inputs, output):
    x, gamma, beta, _eps = inputs
    _out, mean, rstd = output
    ctx.save_for_backward(x, gamma, beta, mean, rstd)


def _bwd_ln(ctx, g_out, g_mean, g_rstd):
    x, gamma, beta, mean, rstd = ctx.saved_tensors
    g_x = torch.ops.fairchem.flash_ln_silu_bwd(
        g_out.contiguous(), x, gamma, beta, mean, rstd
    )
    return g_x, None, None, None


flash_ln_silu_fwd.register_autograd(_bwd_ln, setup_context=_setup_ln)


# =============================================================================
# 3. Fused radial stage 1: GEMM + element tables + LayerNorm + SiLU
# =============================================================================


def _stage1_blocks(B: int, H: int):
    # tl.dot needs at least 16 in every dimension, and powers of two throughout.
    return max(16, triton.next_power_of_2(B)), max(16, triton.next_power_of_2(H))


@triton_op("fairchem::flash_radial_stage1_fwd", mutates_args=())
def flash_radial_stage1_fwd(
    gauss: Tensor,
    W_gauss: Tensor,
    bias: Tensor,
    table_src: Tensor,
    table_tgt: Tensor,
    z_i: Tensor,
    z_j: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    E, B = gauss.shape
    H = W_gauss.shape[0]
    out = torch.empty((E, H), device=gauss.device, dtype=gauss.dtype)
    mean = torch.empty(E, device=gauss.device, dtype=torch.float32)
    rstd = torch.empty(E, device=gauss.device, dtype=torch.float32)
    if E == 0:
        return out, mean, rstd

    block_b, block_h = _stage1_blocks(B, H)
    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(gauss):
        wrap_triton(_radial_stage1_fwd_kernel)[grid](
            gauss,
            W_gauss,
            bias,
            table_src,
            table_tgt,
            z_i,
            z_j,
            gamma,
            beta,
            out,
            mean,
            rstd,
            E,
            B,
            H,
            float(eps),
            gauss.stride(0),
            out.stride(0),
            BLOCK_B=block_b,
            BLOCK_H=block_h,
        )
    return out, mean, rstd


@triton_op("fairchem::flash_radial_stage1_bwd", mutates_args=())
def flash_radial_stage1_bwd(
    g_out: Tensor,
    gauss: Tensor,
    W_gauss: Tensor,
    bias: Tensor,
    table_src: Tensor,
    table_tgt: Tensor,
    z_i: Tensor,
    z_j: Tensor,
    gamma: Tensor,
    beta: Tensor,
    mean: Tensor,
    rstd: Tensor,
) -> Tensor:
    E, B = gauss.shape
    H = W_gauss.shape[0]
    g_gauss = torch.empty_like(gauss)
    if E == 0:
        return g_gauss

    block_b, block_h = _stage1_blocks(B, H)
    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(gauss):
        wrap_triton(_radial_stage1_bwd_kernel)[grid](
            g_out,
            gauss,
            W_gauss,
            bias,
            table_src,
            table_tgt,
            z_i,
            z_j,
            gamma,
            beta,
            mean,
            rstd,
            g_gauss,
            E,
            B,
            H,
            gauss.stride(0),
            g_out.stride(0),
            BLOCK_B=block_b,
            BLOCK_H=block_h,
        )
    return g_gauss


def _setup_stage1(ctx, inputs, output):
    gauss, W_gauss, bias, table_src, table_tgt, z_i, z_j, gamma, beta, _eps = inputs
    _out, mean, rstd = output
    ctx.save_for_backward(
        gauss, W_gauss, bias, table_src, table_tgt, z_i, z_j, gamma, beta, mean, rstd
    )


def _bwd_stage1(ctx, g_out, g_mean, g_rstd):
    saved = ctx.saved_tensors
    g_gauss = torch.ops.fairchem.flash_radial_stage1_bwd(g_out.contiguous(), *saved)
    # Everything after gauss is an inference buffer or an integer index, the
    # same contract as flash_ln_silu_bwd.
    return (g_gauss,) + (None,) * 9


flash_radial_stage1_fwd.register_autograd(_bwd_stage1, setup_context=_setup_stage1)


# =============================================================================
# 4. Fused SO(2) gating
# =============================================================================


@triton_op("fairchem::flash_gating_fwd", mutates_args=())
def flash_gating_fwd(
    Y_m0: Tensor, Y_m1: Tensor, Y_m2: Tensor, C: int
) -> tuple[Tensor, Tensor, Tensor]:
    E = Y_m0.shape[0]
    kwargs = {"device": Y_m0.device, "dtype": Y_m0.dtype}
    Z_m0 = torch.empty((E, Z_M0 * C), **kwargs)
    Z_m1 = torch.empty((E, Z_M1 * C), **kwargs)
    Z_m2 = torch.empty((E, Z_M2 * C), **kwargs)
    if E == 0:
        return Z_m0, Z_m1, Z_m2

    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(Y_m0):
        wrap_triton(_gating_split_fwd_kernel)[grid](
            Y_m0, Y_m1, Y_m2, Z_m0, Z_m1, Z_m2, E, C
        )
    return Z_m0, Z_m1, Z_m2


@triton_op("fairchem::flash_gating_bwd", mutates_args=())
def flash_gating_bwd(
    g_z_m0: Tensor,
    g_z_m1: Tensor,
    g_z_m2: Tensor,
    Y_m0: Tensor,
    Y_m1: Tensor,
    Y_m2: Tensor,
    C: int,
) -> tuple[Tensor, Tensor, Tensor]:
    E = Y_m0.shape[0]
    g_y_m0 = torch.empty_like(Y_m0)
    g_y_m1 = torch.empty_like(Y_m1)
    g_y_m2 = torch.empty_like(Y_m2)
    if E == 0:
        return g_y_m0, g_y_m1, g_y_m2

    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(Y_m0):
        wrap_triton(_gating_split_bwd_kernel)[grid](
            g_z_m0,
            g_z_m1,
            g_z_m2,
            Y_m0,
            Y_m1,
            Y_m2,
            g_y_m0,
            g_y_m1,
            g_y_m2,
            E,
            C,
        )
    return g_y_m0, g_y_m1, g_y_m2


def _setup_gating(ctx, inputs, output):
    Y_m0, Y_m1, Y_m2, C = inputs
    ctx.save_for_backward(Y_m0, Y_m1, Y_m2)
    ctx.C = C


def _bwd_gating(ctx, g_z_m0, g_z_m1, g_z_m2):
    Y_m0, Y_m1, Y_m2 = ctx.saved_tensors
    gy0, gy1, gy2 = torch.ops.fairchem.flash_gating_bwd(
        g_z_m0.contiguous(),
        g_z_m1.contiguous(),
        g_z_m2.contiguous(),
        Y_m0,
        Y_m1,
        Y_m2,
        ctx.C,
    )
    return gy0, gy1, gy2, None


flash_gating_fwd.register_autograd(_bwd_gating, setup_context=_setup_gating)


# =============================================================================
# 5. M->L rotation fused with the edge-to-node scatter
# =============================================================================


@triton_op("fairchem::flash_scatter_fwd", mutates_args=())
def flash_scatter_fwd(
    Z_m0: Tensor,
    Z_m1: Tensor,
    Z_m2: Tensor,
    wigner: Tensor,
    env: Tensor,
    x_res: Tensor,
    scatter_target: Tensor,
) -> Tensor:
    E = Z_m0.shape[0]
    C = Z_m0.shape[1] // Z_M0
    if E == 0:
        return x_res.clone()

    # Seed the accumulator with the block's residual and let the kernel add on
    # top. The autotuner declares this buffer `restore_value`, so trials leave
    # it as they found it -- which is what makes seeding safe, and saves a
    # zeros_like plus a full-tensor add on the way out.
    x_acc = x_res.clone()
    grid = lambda META: (  # noqa: E731
        triton.cdiv(E, META["BLOCK_E"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    with _device_of(Z_m0):
        wrap_triton(_scatter_split_fwd_kernel)[grid](
            scatter_target,
            Z_m0,
            Z_m1,
            Z_m2,
            wigner,
            env,
            x_acc,
            E,
            C,
            x_acc.stride(0),
            x_acc.stride(1),
            x_acc.stride(2),
            wigner.stride(1),
            wigner.stride(0),
        )
    return x_acc


@triton_op("fairchem::flash_scatter_bwd", mutates_args=())
def flash_scatter_bwd(
    g_out: Tensor,
    Z_m0: Tensor,
    Z_m1: Tensor,
    Z_m2: Tensor,
    wigner: Tensor,
    env: Tensor,
    scatter_target: Tensor,
    C: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    E = Z_m0.shape[0]
    g_Z_m0 = torch.empty_like(Z_m0)
    g_Z_m1 = torch.empty_like(Z_m1)
    g_Z_m2 = torch.empty_like(Z_m2)
    g_wigner = torch.zeros_like(wigner)
    g_env = torch.zeros_like(env)
    if E == 0:
        return g_Z_m0, g_Z_m1, g_Z_m2, g_wigner, g_env

    g_env_l1 = torch.empty_like(env)
    g_env_l2 = torch.empty_like(env)
    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(g_out):
        for kernel, g_env_part in (
            (_scatter_split_bwd_l1_kernel, g_env_l1),
            (_scatter_split_bwd_l2_kernel, g_env_l2),
        ):
            wrap_triton(kernel)[grid](
                g_out,
                scatter_target,
                Z_m0,
                Z_m1,
                Z_m2,
                wigner,
                env,
                g_Z_m0,
                g_Z_m1,
                g_Z_m2,
                g_wigner,
                g_env_part,
                E,
                C,
                g_out.stride(0),
                g_out.stride(1),
                g_out.stride(2),
                wigner.stride(1),
                wigner.stride(0),
            )
    return g_Z_m0, g_Z_m1, g_Z_m2, g_wigner, g_env_l1 + g_env_l2


def _setup_scatter(ctx, inputs, output):
    Z_m0, Z_m1, Z_m2, wigner, env, _x_res, scatter_target = inputs
    ctx.save_for_backward(Z_m0, Z_m1, Z_m2, wigner, env, scatter_target)
    ctx.C = Z_m0.shape[1] // Z_M0


def _bwd_scatter(ctx, g_out):
    Z_m0, Z_m1, Z_m2, wigner, env, scatter_target = ctx.saved_tensors
    g_m0, g_m1, g_m2, g_wig, g_env = torch.ops.fairchem.flash_scatter_bwd(
        g_out.contiguous(),
        Z_m0,
        Z_m1,
        Z_m2,
        wigner,
        env,
        scatter_target,
        ctx.C,
    )
    # d x_out / d x_res is the identity.
    return g_m0, g_m1, g_m2, g_wig, g_env, g_out, None, None


flash_scatter_fwd.register_autograd(_bwd_scatter, setup_context=_setup_scatter)


# =============================================================================
# 6. Edge-degree initialisation scatter
# =============================================================================


def _init_node_l0_add(x_out: Tensor, Z: Tensor, batch: Tensor, W_sphere, csd_emb):
    """
    Add the l=0 self-embedding in one node-parallel launch, in place.

    Deliberately not autotuned: it is a non-idempotent read-modify-write, so
    every autotune trial would add the embedding again, and there is nothing
    worth tuning in an N*C pointwise pass.
    """
    N, _, C = x_out.shape
    if N == 0:
        return x_out
    BLOCK_N = 32
    wrap_triton(_init_node_l0_add_kernel)[(triton.cdiv(N, BLOCK_N),)](
        Z,
        batch,
        W_sphere,
        csd_emb,
        x_out,
        N,
        C,
        W_sphere.stride(0),
        W_sphere.stride(1),
        csd_emb.stride(0),
        csd_emb.stride(1),
        x_out.stride(0),
        x_out.stride(1),
        x_out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_C=triton.next_power_of_2(C),
        num_warps=4,
    )
    return x_out


@triton_op("fairchem::flash_init_scatter_fwd", mutates_args=())
def flash_init_scatter_fwd(
    rad_out: Tensor,
    wigner: Tensor,
    env: Tensor,
    Z: Tensor,
    batch: Tensor,
    W_sphere: Tensor,
    csd_emb: Tensor,
    scatter_target: Tensor,
    num_nodes: int,
    inv_rescale: float,
) -> Tensor:
    E = rad_out.shape[0]
    C = rad_out.shape[1] // 3
    x_out = torch.zeros((num_nodes, 9, C), device=rad_out.device, dtype=rad_out.dtype)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(E, META["BLOCK_E"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    with _device_of(rad_out):
        if E > 0:
            wrap_triton(_init_scatter_fwd_kernel)[grid](
                rad_out,
                wigner,
                env,
                scatter_target,
                x_out,
                E,
                C,
                float(inv_rescale),
                rad_out.stride(0),
                1,
                wigner.stride(1),
                wigner.stride(0),
                x_out.stride(0),
                x_out.stride(1),
                x_out.stride(2),
            )
        # Runs after the scatter: its target is `reset_to_zero`.
        _init_node_l0_add(x_out, Z, batch, W_sphere, csd_emb)
    return x_out


@triton_op("fairchem::flash_init_scatter_bwd", mutates_args=())
def flash_init_scatter_bwd(
    g_out: Tensor,
    rad_out: Tensor,
    wigner: Tensor,
    env: Tensor,
    scatter_target: Tensor,
    Z: Tensor,
    batch: Tensor,
    num_z: int,
    num_batches: int,
    C: int,
    inv_rescale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    E = rad_out.shape[0]
    g_rad = torch.zeros_like(rad_out)
    g_wig = torch.zeros_like(wigner)
    g_env = torch.zeros_like(env)

    with _device_of(g_out):
        if E > 0:
            grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
            wrap_triton(_init_scatter_bwd_kernel)[grid](
                g_out,
                scatter_target,
                rad_out,
                wigner,
                env,
                g_rad,
                g_wig,
                g_env,
                E,
                C,
                float(inv_rescale),
                g_out.stride(0),
                g_out.stride(1),
                g_out.stride(2),
                g_rad.stride(0),
                1,
                wigner.stride(1),
                wigner.stride(0),
            )

    # Pointwise l=0 embeddings, gathered by atomic number and by system.
    g_out_l0 = g_out[:, 0, :].contiguous()
    g_W_sphere = torch.zeros((num_z, C), device=g_out.device, dtype=g_out.dtype)
    g_W_sphere.index_add_(0, Z, g_out_l0)
    g_csd_emb = torch.zeros((num_batches, C), device=g_out.device, dtype=g_out.dtype)
    g_csd_emb.index_add_(0, batch, g_out_l0)

    return g_rad, g_wig, g_env, g_W_sphere, g_csd_emb


def _setup_init_scatter(ctx, inputs, output):
    (
        rad_out,
        wigner,
        env,
        Z,
        batch,
        W_sphere,
        csd_emb,
        scatter_target,
        _num_nodes,
        inv_rescale,
    ) = inputs
    ctx.save_for_backward(rad_out, wigner, env, scatter_target, Z, batch)
    ctx.num_z = W_sphere.shape[0]
    ctx.num_batches = csd_emb.shape[0]
    ctx.C = rad_out.shape[1] // 3
    ctx.inv_rescale = inv_rescale


def _bwd_init_scatter(ctx, g_out):
    rad_out, wigner, env, scatter_target, Z, batch = ctx.saved_tensors
    (
        g_rad,
        g_wig,
        g_env,
        g_W_sphere,
        g_csd_emb,
    ) = torch.ops.fairchem.flash_init_scatter_bwd(
        g_out.contiguous(),
        rad_out,
        wigner,
        env,
        scatter_target,
        Z,
        batch,
        ctx.num_z,
        ctx.num_batches,
        ctx.C,
        ctx.inv_rescale,
    )
    return (
        g_rad,
        g_wig,
        g_env,
        None,
        None,
        g_W_sphere,
        g_csd_emb,
        None,
        None,
        None,
        None,
    )


flash_init_scatter_fwd.register_autograd(
    _bwd_init_scatter, setup_context=_setup_init_scatter
)


# =============================================================================
# 7. Node-to-edge gather fused with the L->M rotation
# =============================================================================


@triton_op("fairchem::flash_gather_fwd", mutates_args=())
def flash_gather_fwd(
    x_in: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    wigner: Tensor,
    h: Tensor,
    W3: Tensor,
    b3: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Gather, rotate L->M, and scale by the radial weights.

    Takes the radial MLP's hidden state and its last linear rather than the
    applied result. The result is still built -- cuBLAS does that GEMM far
    better than a Triton epilogue can, see the note in _setup_gather -- but it
    is local to this call, so what autograd holds until the backward is the
    [E, H] hidden state instead of the [E, 12C] output.
    """
    N, _, C = x_in.shape
    E = idx_i.shape[0]
    kwargs = {"device": x_in.device, "dtype": x_in.dtype}
    X_m0 = torch.empty((E, X_M0 * C), **kwargs)
    X_m1 = torch.empty((E, X_M1 * C), **kwargs)
    X_m2 = torch.empty((E, X_M2 * C), **kwargs)
    if E == 0:
        return X_m0, X_m1, X_m2

    rad_out = torch.nn.functional.linear(h, W3, b3)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(E, META["BLOCK_E"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    with _device_of(x_in):
        wrap_triton(_gather_split_fwd_kernel)[grid](
            x_in,
            idx_i,
            idx_j,
            wigner,
            rad_out,
            X_m0,
            X_m1,
            X_m2,
            E,
            N,
            C,
            x_in.stride(0),
            x_in.stride(1),
            x_in.stride(2),
            wigner.stride(1),
            wigner.stride(0),
            rad_out.stride(0),
            rad_out.stride(1),
        )
    return X_m0, X_m1, X_m2


@triton_op("fairchem::flash_gather_bwd", mutates_args=())
def flash_gather_bwd(
    g_x_m0: Tensor,
    g_x_m1: Tensor,
    g_x_m2: Tensor,
    x_in: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    wigner: Tensor,
    rad_out: Tensor,
    N: int,
    C: int,
) -> tuple[Tensor, Tensor, Tensor]:
    E = idx_i.shape[0]
    g_wigner = torch.zeros_like(wigner)
    g_rad_out = torch.zeros_like(rad_out)
    if E == 0:
        return torch.zeros_like(x_in), g_wigner, g_rad_out

    # One node-gradient accumulator per kernel: `reset_to_zero` would clear the
    # other kernel's contribution while autotuning. L1 writes l=0..3 and L2
    # writes l=4..8, so summing them is exact.
    g_x_l1 = torch.zeros_like(x_in)
    g_x_l2 = torch.zeros_like(x_in)
    grid = lambda META: (triton.cdiv(E, META["BLOCK_E"]),)  # noqa: E731
    with _device_of(x_in):
        for kernel, g_x_part in (
            (_gather_split_bwd_l1_kernel, g_x_l1),
            (_gather_split_bwd_l2_kernel, g_x_l2),
        ):
            wrap_triton(kernel)[grid](
                g_x_m0,
                g_x_m1,
                g_x_m2,
                x_in,
                idx_i,
                idx_j,
                wigner,
                rad_out,
                g_x_part,
                g_wigner,
                g_rad_out,
                E,
                N,
                C,
                x_in.stride(0),
                x_in.stride(1),
                x_in.stride(2),
                wigner.stride(1),
                wigner.stride(0),
                rad_out.stride(0),
                rad_out.stride(1),
            )
    return g_x_l1 + g_x_l2, g_wigner, g_rad_out


def _setup_gather(ctx, inputs, output):
    x_in, idx_i, idx_j, wigner, h, W3, b3 = inputs
    # h, not the radial output: that is the whole point of folding the last
    # linear into the kernel. At C=128 this saves 1408 floats per edge per
    # layer of activation that would otherwise stay resident until the
    # backward runs.
    ctx.save_for_backward(x_in, idx_i, idx_j, wigner, h, W3, b3)
    # Folding the last linear into the kernel was tried and reverted: pinned to
    # ieee it loses the tensor cores cuBLAS uses, and measured 3.6x slower for
    # a 40% memory saving. Saving h and rebuilding the output keeps the memory
    # win -- 512 bytes an edge held instead of 6 KB -- at no cost in the
    # forward and one extra GEMM in the backward.
    ctx.N = x_in.shape[0]
    ctx.C = x_in.shape[2]


def _bwd_gather(ctx, g_x_m0, g_x_m1, g_x_m2):
    x_in, idx_i, idx_j, wigner, h, W3, b3 = ctx.saved_tensors
    # Rebuild the radial output for the duration of the backward. The two
    # backward kernels walk the channel dimension in a loop and would have to
    # accumulate g_h across it; recomputing keeps them untouched, and the
    # tensor is transient here rather than held from the forward.
    rad_out = torch.nn.functional.linear(h, W3, b3)
    g_x_in, g_wigner, g_rad_out = torch.ops.fairchem.flash_gather_bwd(
        g_x_m0.contiguous(),
        g_x_m1.contiguous(),
        g_x_m2.contiguous(),
        x_in,
        idx_i,
        idx_j,
        wigner,
        rad_out,
        ctx.N,
        ctx.C,
    )
    # W3 and b3 are inference buffers, as everywhere else in this file.
    return g_x_in, None, None, g_wigner, g_rad_out @ W3, None, None


flash_gather_fwd.register_autograd(_bwd_gather, setup_context=_setup_gather)


# =============================================================================
# Public helpers used by the flash feature path
# =============================================================================


def precompute_geometry(edge_vec, offset_g, coeff_g, cutoff):
    return torch.ops.fairchem.flash_geom_fwd(edge_vec, offset_g, coeff_g, cutoff)


def layer_norm_silu(x, gamma, beta, eps=1e-7):
    return torch.ops.fairchem.flash_ln_silu_fwd(
        x.contiguous(), gamma.contiguous(), beta.contiguous(), eps
    )[0]


def fused_gating_split(Y_m0, Y_m1, Y_m2, C):
    return torch.ops.fairchem.flash_gating_fwd(Y_m0, Y_m1, Y_m2, C)


def rotate_scatter_split(Z_m0, Z_m1, Z_m2, wigner, env, x_res, scatter_target):
    return torch.ops.fairchem.flash_scatter_fwd(
        Z_m0, Z_m1, Z_m2, wigner, env, x_res, scatter_target
    )


def init_node_scatter(
    rad_out,
    wigner,
    env,
    Z,
    batch,
    W_sphere,
    csd_emb,
    scatter_target,
    num_nodes,
    inv_rescale=1.0,
):
    return torch.ops.fairchem.flash_init_scatter_fwd(
        rad_out,
        wigner,
        env,
        Z,
        batch,
        W_sphere,
        csd_emb,
        scatter_target,
        num_nodes,
        inv_rescale,
    )


def radial_stage1(
    gauss, W_gauss, bias, table_src, table_tgt, z_i, z_j, gamma, beta, eps
):
    out, _mean, _rstd = torch.ops.fairchem.flash_radial_stage1_fwd(
        gauss, W_gauss, bias, table_src, table_tgt, z_i, z_j, gamma, beta, eps
    )
    return out


def gather_wigner_split(x_in, idx_i, idx_j, wigner, h, W3, b3):
    return torch.ops.fairchem.flash_gather_fwd(x_in, idx_i, idx_j, wigner, h, W3, b3)
