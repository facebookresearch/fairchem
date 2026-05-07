"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

JIT-compiled C++ kernels for UMASFastCPUBackend.

The C++ source is built lazily on first use via
torch.utils.cpp_extension.load_inline (ninja-backed). Output is cached
under ~/.cache/torch_extensions, so subsequent process starts hit cache
instantly. The first build pays ~15s of compile time which lands in the
backend's lazy_init / first prepare_for_inference call.

Only forward kernels live in C++. Backward stays in PyTorch eager so
autograd correctness mirrors the default ExecutionBackend exactly; the
forward path is what dominates per-iteration wall time, and the backward
of an inverse-rotation + scatter is already well-served by ATen ops.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

_LOCK = threading.Lock()
# Module handle once built. Stored inside a single-element list so we
# don't need a `global` statement in `_build` (Ruff PLW0603).
_KERNELS_BOX: list = [None]

# Fused (gather x_full[edge_index[k]]) + (cat src/tgt) + (bmm wigner @ ...)
# implemented as two per-edge GEMMs into the same [M, 2C] output buffer.
# Skips the intermediate [E, L, 2C] allocation that the default
# ExecutionBackend materializes via torch.cat. Per-edge work fits in L1
# (L*C floats = ~5KB for L=9, C=128) so we want one OMP thread per edge.
_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>
#include <cstdint>

at::Tensor fused_node_to_edge_wigner_permute(
    at::Tensor x_full,        // [N, L, C]   float32 contiguous
    at::Tensor edge_index,    // [2, E]      int64    contiguous
    at::Tensor wigner)        // [E, M, L]   float32 contiguous
{
    TORCH_CHECK(x_full.dim() == 3, "x_full must be 3D");
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                "edge_index must be [2, E]");
    TORCH_CHECK(wigner.dim() == 3, "wigner must be 3D");
    TORCH_CHECK(x_full.is_contiguous(), "x_full must be contiguous");
    TORCH_CHECK(edge_index.is_contiguous(), "edge_index must be contiguous");
    TORCH_CHECK(wigner.is_contiguous(), "wigner must be contiguous");
    TORCH_CHECK(x_full.scalar_type() == at::kFloat,
                "x_full must be float32");
    TORCH_CHECK(wigner.scalar_type() == at::kFloat,
                "wigner must be float32");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong,
                "edge_index must be int64");

    const int64_t L = x_full.size(1);
    const int64_t C = x_full.size(2);
    const int64_t E = wigner.size(0);
    const int64_t M = wigner.size(1);
    TORCH_CHECK(wigner.size(2) == L,
                "wigner last dim must equal x_full L");
    TORCH_CHECK(edge_index.size(1) == E,
                "edge_index E must equal wigner E");

    auto out = at::empty({E, M, 2 * C}, x_full.options());
    if (E == 0) return out;

    const float* X = x_full.data_ptr<float>();
    const float* W = wigner.data_ptr<float>();
    const int64_t* EI = edge_index.data_ptr<int64_t>();
    float* Y = out.data_ptr<float>();

    const int64_t x_n_stride = L * C;
    const int64_t y_e_stride = M * 2 * C;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; ++e) {
        const int64_t src = EI[e];
        const int64_t tgt = EI[E + e];
        const float* Xs = X + src * x_n_stride;     // [L, C]
        const float* Xt = X + tgt * x_n_stride;     // [L, C]
        const float* We = W + e * (M * L);          // [M, L]
        float* Ye = Y + e * y_e_stride;             // [M, 2C]

        // For each output (m, c), accumulate sum_l W[m,l] * X[*, l, c].
        // Inner C loop is auto-vectorized by GCC with -O3 -march=native:
        // float* + uniform stride + scalar broadcast (wml) + fmadd is
        // the textbook AVX-512 fmadd_ps pattern.
        for (int64_t m = 0; m < M; ++m) {
            const float* Wm = We + m * L;
            float* Ys = Ye + m * (2 * C);   // [C]   (source half)
            float* Yt = Ys + C;             // [C]   (target half)

            // zero-init the two halves
            for (int64_t c = 0; c < C; ++c) {
                Ys[c] = 0.0f;
                Yt[c] = 0.0f;
            }

            for (int64_t l = 0; l < L; ++l) {
                const float wml = Wm[l];
                const float* Xs_l = Xs + l * C;
                const float* Xt_l = Xt + l * C;
                #pragma GCC ivdep
                for (int64_t c = 0; c < C; ++c) {
                    Ys[c] += wml * Xs_l[c];
                    Yt[c] += wml * Xt_l[c];
                }
            }
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_node_to_edge_wigner_permute",
          &fused_node_to_edge_wigner_permute,
          "Fused gather + cat-free + bmm for node->edge Wigner rotation");
}
"""


def _build() -> object:
    """Lazy-build (and cache) the C++ extension module."""
    if _KERNELS_BOX[0] is not None:
        return _KERNELS_BOX[0]
    with _LOCK:
        if _KERNELS_BOX[0] is not None:
            return _KERNELS_BOX[0]
        from torch.utils import cpp_extension

        _KERNELS_BOX[0] = cpp_extension.load_inline(
            name="fairchem_uma_cpu_kernels",
            cpp_sources=[_CPP_SRC],
            extra_cflags=[
                "-O3",
                "-march=native",
                "-fopenmp",
                "-ffast-math",
                "-funroll-loops",
            ],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    return _KERNELS_BOX[0]


class _FusedNodeToEdgeWignerPermute(torch.autograd.Function):
    """
    autograd.Function: C++ forward + PyTorch eager backward.

    The default ExecutionBackend.node_to_edge_wigner_permute computes::

        x_message = cat(x_full[edge_index[0]], x_full[edge_index[1]], dim=2)
        return bmm(wigner, x_message)        # [E, M, 2C]

    Our forward fuses gather+cat+bmm into one OMP-parallel kernel with
    no [E, L, 2C] intermediate. The backward reproduces the eager
    formula exactly so gradients (and therefore forces/stress) match.
    """

    @staticmethod
    def forward(
        ctx,
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        kernels = _build()
        # Ensure contiguity once at the boundary; re-using the same
        # tensors across iterations means this is essentially a no-op.
        x_full_c = x_full.contiguous()
        edge_index_c = edge_index.contiguous()
        wigner_c = wigner.contiguous()
        ctx.save_for_backward(x_full_c, edge_index_c, wigner_c)
        return kernels.fused_node_to_edge_wigner_permute(
            x_full_c, edge_index_c, wigner_c
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_full, edge_index, wigner = ctx.saved_tensors
        C = x_full.size(2)

        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[2]

        grad_x_full: Optional[torch.Tensor] = None
        grad_wigner: Optional[torch.Tensor] = None

        # Both branches deliberately avoid re-materializing the
        # [E, L, 2C] cat that the eager forward built. The bmm splits
        # below have identical FLOP counts to the all-at-once bmms but
        # skip the [E, L, 2C] / [E, M, 2C] intermediate allocations,
        # which is the whole point of the C++ forward kernel — we don't
        # want to give those savings back in backward.
        if need_x:
            wigner_T = wigner.transpose(1, 2)  # [E, L, M]
            # grad_x_src = wigner^T @ grad_out[..., :C]   -> [E, L, C]
            # grad_x_tgt = wigner^T @ grad_out[..., C:]   -> [E, L, C]
            grad_src = torch.bmm(wigner_T, grad_out[..., :C].contiguous())
            grad_tgt = torch.bmm(wigner_T, grad_out[..., C:].contiguous())
            grad_x_full = torch.zeros_like(x_full)
            grad_x_full.index_add_(0, edge_index[0], grad_src)
            grad_x_full.index_add_(0, edge_index[1], grad_tgt)

        if need_w:
            # grad_wigner = grad_out_src @ x_src^T + grad_out_tgt @ x_tgt^T
            x_src = x_full.index_select(0, edge_index[0])  # [E, L, C]
            x_tgt = x_full.index_select(0, edge_index[1])  # [E, L, C]
            grad_wigner = torch.bmm(
                grad_out[..., :C].contiguous(), x_src.transpose(1, 2)
            )
            grad_wigner = grad_wigner.add_(
                torch.bmm(grad_out[..., C:].contiguous(), x_tgt.transpose(1, 2))
            )

        return grad_x_full, None, grad_wigner


def fused_node_to_edge_wigner_permute(
    x_full: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    return _FusedNodeToEdgeWignerPermute.apply(x_full, edge_index, wigner)
