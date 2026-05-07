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

// Backward of fused_node_to_edge_wigner_permute, grad_x_full path.
//
// Forward was: out = bmm(wigner, cat(x_full[src], x_full[tgt], 2))
// so grad_x_full[src[e], l, c] += sum_m wigner[e, m, l] * grad_out[e, m, c]
//    grad_x_full[tgt[e], l, c] += sum_m wigner[e, m, l] * grad_out[e, m, C+c]
//
// Per-edge: two M->L rotations into [L, C] each, then scatter-add to
// dst nodes. Same scatter conflict story as exp9, so per-thread scratch.
at::Tensor fused_node_to_edge_grad_x_full(
    at::Tensor edge_index,    // [2, E]      int64
    at::Tensor wigner,        // [E, M, L]   float32
    at::Tensor grad_out,      // [E, M, 2C]  float32
    int64_t num_nodes)
{
    TORCH_CHECK(wigner.dim() == 3 && grad_out.dim() == 3,
                "wigner and grad_out must be 3D");
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                "edge_index must be [2, E]");
    TORCH_CHECK(wigner.is_contiguous() && grad_out.is_contiguous()
                && edge_index.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(wigner.scalar_type() == at::kFloat
                && grad_out.scalar_type() == at::kFloat,
                "wigner and grad_out must be float32");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong,
                "edge_index must be int64");

    const int64_t E = wigner.size(0);
    const int64_t M = wigner.size(1);
    const int64_t L = wigner.size(2);
    TORCH_CHECK(grad_out.size(0) == E && grad_out.size(1) == M,
                "grad_out must be [E, M, 2C]");
    const int64_t twoC = grad_out.size(2);
    TORCH_CHECK(twoC % 2 == 0, "grad_out last dim must be even (=2C)");
    const int64_t C = twoC / 2;

    auto out = at::zeros({num_nodes, L, C}, wigner.options());
    if (E == 0) return out;

    const float* W = wigner.data_ptr<float>();
    const float* G = grad_out.data_ptr<float>();
    const int64_t* EI = edge_index.data_ptr<int64_t>();
    float* Y = out.data_ptr<float>();

    const int64_t w_e_stride = M * L;
    const int64_t g_e_stride = M * twoC;
    const int64_t y_n_stride = L * C;

    const int max_threads = omp_get_max_threads();
    auto scratch = at::zeros({(int64_t)max_threads, num_nodes, L, C},
                             wigner.options());
    float* SB = scratch.data_ptr<float>();
    const int64_t s_t_stride = num_nodes * L * C;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* myout = SB + (int64_t)tid * s_t_stride;

        #pragma omp for schedule(static)
        for (int64_t e = 0; e < E; ++e) {
            const int64_t src = EI[e];
            const int64_t tgt = EI[E + e];
            const float* We = W + e * w_e_stride;        // [M, L]
            const float* Ge = G + e * g_e_stride;        // [M, 2C]
            float* dst_src = myout + src * y_n_stride;   // [L, C]
            float* dst_tgt = myout + tgt * y_n_stride;   // [L, C]

            // For each output (l, c), accumulate:
            //   src += sum_m W[m, l] * G[m, c]
            //   tgt += sum_m W[m, l] * G[m, C+c]
            for (int64_t l = 0; l < L; ++l) {
                float* drow_s = dst_src + l * C;
                float* drow_t = dst_tgt + l * C;
                for (int64_t m = 0; m < M; ++m) {
                    const float wml = We[m * L + l];
                    const float* gs = Ge + m * twoC;          // [2C]
                    const float* gt = gs + C;                 // tgt half
                    #pragma GCC ivdep
                    for (int64_t c = 0; c < C; ++c) {
                        drow_s[c] += wml * gs[c];
                        drow_t[c] += wml * gt[c];
                    }
                }
            }
        }
    }

    // Reduce scratch slabs into out.
    const int64_t total = num_nodes * L * C;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total; ++i) {
        float s = 0.0f;
        for (int t = 0; t < max_threads; ++t) {
            s += SB[(int64_t)t * s_t_stride + i];
        }
        Y[i] = s;
    }
    return out;
}

// Backward of fused_node_to_edge_wigner_permute, grad_wigner path.
//
// grad_wigner[e, m, l] = sum_c grad_out[e, m, c]   * x_full[src[e], l, c]
//                       + sum_c grad_out[e, m, C+c] * x_full[tgt[e], l, c]
//
// Per-edge GEMMs of (M x C) x (C x L) for src and tgt halves; output is
// per-edge so no scatter conflict.
at::Tensor fused_node_to_edge_grad_wigner(
    at::Tensor x_full,        // [N, L, C]
    at::Tensor edge_index,    // [2, E]
    at::Tensor grad_out)      // [E, M, 2C]
{
    TORCH_CHECK(x_full.dim() == 3 && grad_out.dim() == 3,
                "x_full and grad_out must be 3D");
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                "edge_index must be [2, E]");
    TORCH_CHECK(x_full.is_contiguous() && grad_out.is_contiguous()
                && edge_index.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(x_full.scalar_type() == at::kFloat
                && grad_out.scalar_type() == at::kFloat,
                "x_full and grad_out must be float32");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong,
                "edge_index must be int64");

    const int64_t L = x_full.size(1);
    const int64_t C = x_full.size(2);
    const int64_t E = grad_out.size(0);
    const int64_t M = grad_out.size(1);
    const int64_t twoC = grad_out.size(2);
    TORCH_CHECK(twoC == 2 * C, "grad_out last dim must equal 2 * C");

    auto out = at::empty({E, M, L}, x_full.options());
    if (E == 0) return out;

    const float* X = x_full.data_ptr<float>();
    const float* G = grad_out.data_ptr<float>();
    const int64_t* EI = edge_index.data_ptr<int64_t>();
    float* Y = out.data_ptr<float>();

    const int64_t x_n_stride = L * C;
    const int64_t g_e_stride = M * twoC;
    const int64_t y_e_stride = M * L;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; ++e) {
        const int64_t src = EI[e];
        const int64_t tgt = EI[E + e];
        const float* Xs = X + src * x_n_stride;     // [L, C]
        const float* Xt = X + tgt * x_n_stride;     // [L, C]
        const float* Ge = G + e * g_e_stride;        // [M, 2C]
        float* Ye = Y + e * y_e_stride;              // [M, L]

        // For each (m, l), accumulate src + tgt contributions over c.
        for (int64_t m = 0; m < M; ++m) {
            const float* gs = Ge + m * twoC;        // [2C]
            const float* gt = gs + C;               // tgt half
            float* yrow = Ye + m * L;               // [L]

            for (int64_t l = 0; l < L; ++l) {
                const float* xs = Xs + l * C;
                const float* xt = Xt + l * C;
                float acc = 0.0f;
                #pragma GCC ivdep
                for (int64_t c = 0; c < C; ++c) {
                    acc += gs[c] * xs[c] + gt[c] * xt[c];
                }
                yrow[l] = acc;
            }
        }
    }
    return out;
}


// permute_wigner_inv_edge_to_node:
//   rotated[e, l, c] = sum_m wigner_inv[e, l, m] * x_message[e, m, c]
//   output[dst[e] - node_offset, l, c] += rotated[e, l, c]
//
// Concurrent scatter into the same output node would race, so each
// thread accumulates into its own [num_nodes, L, C] scratch slab.
at::Tensor fused_permute_wigner_inv_edge_to_node(
    at::Tensor x_message,     // [E, M, C]
    at::Tensor wigner_inv,    // [E, L, M]
    at::Tensor edge_index,    // [2, E]
    int64_t num_nodes,
    int64_t node_offset)
{
    TORCH_CHECK(x_message.dim() == 3 && wigner_inv.dim() == 3,
                "x_message and wigner_inv must be 3D");
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                "edge_index must be [2, E]");
    TORCH_CHECK(x_message.is_contiguous() && wigner_inv.is_contiguous()
                && edge_index.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(x_message.scalar_type() == at::kFloat
                && wigner_inv.scalar_type() == at::kFloat,
                "x_message and wigner_inv must be float32");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong,
                "edge_index must be int64");

    const int64_t E = x_message.size(0);
    const int64_t M = x_message.size(1);
    const int64_t C = x_message.size(2);
    const int64_t L = wigner_inv.size(1);
    TORCH_CHECK(wigner_inv.size(0) == E && wigner_inv.size(2) == M,
                "wigner_inv must be [E, L, M] matching x_message");
    TORCH_CHECK(edge_index.size(1) == E,
                "edge_index E must equal x_message E");

    auto out = at::zeros({num_nodes, L, C}, x_message.options());
    if (E == 0) return out;

    const float* X = x_message.data_ptr<float>();
    const float* W = wigner_inv.data_ptr<float>();
    const int64_t* EI = edge_index.data_ptr<int64_t>();
    float* Y = out.data_ptr<float>();

    const int64_t x_e_stride = M * C;
    const int64_t w_e_stride = L * M;
    const int64_t y_n_stride = L * C;

    const int max_threads = omp_get_max_threads();
    auto scratch = at::zeros({(int64_t)max_threads, num_nodes, L, C},
                             x_message.options());
    float* SB = scratch.data_ptr<float>();
    const int64_t s_t_stride = num_nodes * L * C;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* myout = SB + (int64_t)tid * s_t_stride;

        #pragma omp for schedule(static)
        for (int64_t e = 0; e < E; ++e) {
            const int64_t dst = EI[E + e] - node_offset;
            const float* Xe = X + e * x_e_stride;       // [M, C]
            const float* We = W + e * w_e_stride;       // [L, M]
            float* dst_row = myout + dst * y_n_stride;  // [L, C]

            for (int64_t l = 0; l < L; ++l) {
                const float* Wl = We + l * M;
                float* drow = dst_row + l * C;
                for (int64_t m = 0; m < M; ++m) {
                    const float wlm = Wl[m];
                    const float* Xm = Xe + m * C;
                    #pragma GCC ivdep
                    for (int64_t c = 0; c < C; ++c) {
                        drow[c] += wlm * Xm[c];
                    }
                }
            }
        }
    }

    const int64_t total = num_nodes * L * C;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total; ++i) {
        float s = 0.0f;
        for (int t = 0; t < max_threads; ++t) {
            s += SB[(int64_t)t * s_t_stride + i];
        }
        Y[i] = s;
    }
    return out;
}

// Backward of fused_permute_wigner_inv_edge_to_node, grad_x_message path.
// grad_x_message[e, m, c] = sum_l wigner_inv[e, l, m] * grad_out[dst[e], l, c]
// Per-edge GEMM, no scatter contention.
at::Tensor fused_permute_wigner_inv_grad_x_message(
    at::Tensor wigner_inv,    // [E, L, M]
    at::Tensor edge_index,    // [2, E]
    at::Tensor grad_out,      // [N, L, C]
    int64_t node_offset)
{
    TORCH_CHECK(wigner_inv.dim() == 3 && grad_out.dim() == 3,
                "wigner_inv and grad_out must be 3D");
    TORCH_CHECK(wigner_inv.is_contiguous() && grad_out.is_contiguous()
                && edge_index.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(wigner_inv.scalar_type() == at::kFloat
                && grad_out.scalar_type() == at::kFloat,
                "wigner_inv and grad_out must be float32");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong,
                "edge_index must be int64");

    const int64_t E = wigner_inv.size(0);
    const int64_t L = wigner_inv.size(1);
    const int64_t M = wigner_inv.size(2);
    const int64_t C = grad_out.size(2);
    TORCH_CHECK(grad_out.size(1) == L,
                "grad_out L must match wigner_inv L");

    auto out = at::empty({E, M, C}, wigner_inv.options());
    if (E == 0) return out;

    const float* G = grad_out.data_ptr<float>();
    const float* W = wigner_inv.data_ptr<float>();
    const int64_t* EI = edge_index.data_ptr<int64_t>();
    float* Y = out.data_ptr<float>();

    const int64_t g_n_stride = L * C;
    const int64_t w_e_stride = L * M;
    const int64_t y_e_stride = M * C;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; ++e) {
        const int64_t dst = EI[E + e] - node_offset;
        const float* Gd = G + dst * g_n_stride;     // [L, C]
        const float* We = W + e * w_e_stride;        // [L, M]
        float* Ye = Y + e * y_e_stride;              // [M, C]

        // grad_x_message[m, c] = sum_l W[l, m] * Gd[l, c]
        for (int64_t m = 0; m < M; ++m) {
            float* yrow = Ye + m * C;
            // zero-init
            for (int64_t c = 0; c < C; ++c) yrow[c] = 0.0f;
            for (int64_t l = 0; l < L; ++l) {
                const float wlm = We[l * M + m];
                const float* gd_l = Gd + l * C;
                #pragma GCC ivdep
                for (int64_t c = 0; c < C; ++c) {
                    yrow[c] += wlm * gd_l[c];
                }
            }
        }
    }
    return out;
}

// Backward of fused_permute_wigner_inv_edge_to_node, grad_wigner_inv path.
// grad_wigner_inv[e, l, m] = sum_c grad_out[dst[e], l, c] * x_message[e, m, c]
// Per-edge GEMM, no scatter contention.
at::Tensor fused_permute_wigner_inv_grad_wigner(
    at::Tensor x_message,     // [E, M, C]
    at::Tensor edge_index,    // [2, E]
    at::Tensor grad_out,      // [N, L, C]
    int64_t node_offset)
{
    TORCH_CHECK(x_message.dim() == 3 && grad_out.dim() == 3,
                "x_message and grad_out must be 3D");
    TORCH_CHECK(x_message.is_contiguous() && grad_out.is_contiguous()
                && edge_index.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(x_message.scalar_type() == at::kFloat
                && grad_out.scalar_type() == at::kFloat,
                "x_message and grad_out must be float32");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong,
                "edge_index must be int64");

    const int64_t E = x_message.size(0);
    const int64_t M = x_message.size(1);
    const int64_t C = x_message.size(2);
    const int64_t L = grad_out.size(1);

    auto out = at::empty({E, L, M}, x_message.options());
    if (E == 0) return out;

    const float* X = x_message.data_ptr<float>();
    const float* G = grad_out.data_ptr<float>();
    const int64_t* EI = edge_index.data_ptr<int64_t>();
    float* Y = out.data_ptr<float>();

    const int64_t x_e_stride = M * C;
    const int64_t g_n_stride = L * C;
    const int64_t y_e_stride = L * M;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; ++e) {
        const int64_t dst = EI[E + e] - node_offset;
        const float* Xe = X + e * x_e_stride;      // [M, C]
        const float* Gd = G + dst * g_n_stride;    // [L, C]
        float* Ye = Y + e * y_e_stride;             // [L, M]

        for (int64_t l = 0; l < L; ++l) {
            const float* gd_l = Gd + l * C;
            float* yrow = Ye + l * M;               // [M]
            for (int64_t m = 0; m < M; ++m) {
                const float* xm = Xe + m * C;
                float acc = 0.0f;
                #pragma GCC ivdep
                for (int64_t c = 0; c < C; ++c) {
                    acc += gd_l[c] * xm[c];
                }
                yrow[m] = acc;
            }
        }
    }
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_node_to_edge_wigner_permute",
          &fused_node_to_edge_wigner_permute,
          "Fused gather + cat-free + bmm for node->edge Wigner rotation");
    m.def("fused_node_to_edge_grad_x_full",
          &fused_node_to_edge_grad_x_full,
          "Backward: scatter wigner^T @ grad_out into x_full grad");
    m.def("fused_node_to_edge_grad_wigner",
          &fused_node_to_edge_grad_wigner,
          "Backward: per-edge bmm(grad_out, x_message^T) split into halves");
    m.def("fused_permute_wigner_inv_edge_to_node",
          &fused_permute_wigner_inv_edge_to_node,
          "Fused per-edge bmm + scatter for edge->node Wigner inverse rotation");
    m.def("fused_permute_wigner_inv_grad_x_message",
          &fused_permute_wigner_inv_grad_x_message,
          "Backward: per-edge wigner_inv^T @ grad_out[dst]");
    m.def("fused_permute_wigner_inv_grad_wigner",
          &fused_permute_wigner_inv_grad_wigner,
          "Backward: per-edge grad_out[dst] @ x_message^T");
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
        kernels = _build()

        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[2]

        grad_x_full: Optional[torch.Tensor] = None
        grad_wigner: Optional[torch.Tensor] = None

        # grad_out comes from upstream as [E, M, 2C]. The contiguous()
        # call here is essentially a no-op on the typical hot-path
        # (autograd allocates contiguous grads) and protects the C++
        # kernel from rare non-contiguous callers.
        grad_out_c = grad_out.contiguous()

        if need_x:
            grad_x_full = kernels.fused_node_to_edge_grad_x_full(
                edge_index, wigner, grad_out_c, x_full.size(0)
            )

        if need_w:
            grad_wigner = kernels.fused_node_to_edge_grad_wigner(
                x_full, edge_index, grad_out_c
            )

        return grad_x_full, None, grad_wigner


def fused_node_to_edge_wigner_permute(
    x_full: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    return _FusedNodeToEdgeWignerPermute.apply(x_full, edge_index, wigner)


class _FusedPermuteWignerInvEdgeToNode(torch.autograd.Function):
    """
    autograd.Function: C++ forward + C++ backward for the symmetric
    edge->node Wigner inverse rotation + scatter.
    """

    @staticmethod
    def forward(
        ctx,
        x_message: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_offset: int,
    ) -> torch.Tensor:
        kernels = _build()
        x_message_c = x_message.contiguous()
        wigner_inv_c = wigner_inv.contiguous()
        edge_index_c = edge_index.contiguous()
        ctx.save_for_backward(x_message_c, wigner_inv_c, edge_index_c)
        ctx.node_offset = int(node_offset)
        return kernels.fused_permute_wigner_inv_edge_to_node(
            x_message_c,
            wigner_inv_c,
            edge_index_c,
            int(num_nodes),
            int(node_offset),
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_message, wigner_inv, edge_index = ctx.saved_tensors
        node_offset = ctx.node_offset
        kernels = _build()

        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]

        grad_x_message: Optional[torch.Tensor] = None
        grad_wigner_inv: Optional[torch.Tensor] = None

        grad_out_c = grad_out.contiguous()
        if need_x:
            grad_x_message = kernels.fused_permute_wigner_inv_grad_x_message(
                wigner_inv, edge_index, grad_out_c, node_offset
            )
        if need_w:
            grad_wigner_inv = kernels.fused_permute_wigner_inv_grad_wigner(
                x_message, edge_index, grad_out_c, node_offset
            )

        return grad_x_message, grad_wigner_inv, None, None, None


def fused_permute_wigner_inv_edge_to_node(
    x_message: torch.Tensor,
    wigner_inv: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_offset: int = 0,
) -> torch.Tensor:
    return _FusedPermuteWignerInvEdgeToNode.apply(
        x_message, wigner_inv, edge_index, num_nodes, node_offset
    )
