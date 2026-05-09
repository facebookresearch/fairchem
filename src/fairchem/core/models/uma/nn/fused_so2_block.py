"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fused SO2 conv1 + GateActivation(m_prime=True) + conv2 block for
UMA-S 1.2 inference (lmax=mmax=2). Replaces the eager Python sequence

    x_message, x_0_gating = so2_conv_1(x_message, x_edge)
    x_message = act(x_0_gating, x_message)
    x_message = so2_conv_2(x_message)

with a single C++ op exposed via torch.autograd.Function. The win is
eliminating ~140 internal ATen op dispatches per iter on small-E
systems where the dispatch tax dominates.

Build is phased (see scripts/exp42_phase3_plan.md):
- Phase 0: stub kernel + binding spike
- Phase 1: C++ forward, Python autograd backward via reference
- Phase 2: C++ backward                                      (this file)
- Phase 3: integration into Edgewise call site
- Phase 4: perf_check + size sweep validation
"""

from __future__ import annotations

import threading

import torch

_LOCK = threading.Lock()
_KERNELS_BOX: list = [None]


# Phase 2 kernels:
#  - fused_so2_block_forward        : returns (out, conv1_main, gating_post_sig)
#                                     conv1_main and gating_post_sig saved for backward.
#  - fused_so2_block_backward       : returns (grad_x, grad_x_edge, grad_w_m0_1,
#                                     grad_b_m0_1, grad_W_b1_m1, grad_W_b1_m2,
#                                     grad_w_m0_2, grad_b_m0_2, grad_W_b2_m1,
#                                     grad_W_b2_m2). Each weight grad is computed
#                                     only when its `need_*` flag is true; otherwise
#                                     a zero-element placeholder is returned.
_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>

// =====================================================================
// Forward
// =====================================================================
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_so2_block_forward(
    at::Tensor x,                          // [E, 9, S]
    at::Tensor x_edge,                     // [E, R_total]
    at::Tensor w_m0_1, at::Tensor b_m0_1,
    at::Tensor W_b1_m1, at::Tensor W_b1_m2,
    at::Tensor w_m0_2, at::Tensor b_m0_2,
    at::Tensor W_b2_m1, at::Tensor W_b2_m2,
    int64_t m_split_0, int64_t m_split_1, int64_t m_split_2,
    int64_t edge_split_0, int64_t edge_split_1, int64_t edge_split_2,
    int64_t extra_m0, int64_t lmax,
    int64_t m_out_1, int64_t m_out_2)
{
    TORCH_CHECK(lmax == 2, "fused_so2_block_forward only supports lmax=2");
    TORCH_CHECK(x.dim() == 3, "x must be [E, coeffs, sphere]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");

    const int64_t E = x.size(0);
    const int64_t S = x.size(2);
    const int64_t total_coeffs = (lmax + 1) * (lmax + 1);
    const int64_t num_l_1 = lmax;
    const int64_t num_l_2 = lmax - 1;
    const int64_t offset_1 = m_split_0;
    const int64_t offset_2 = offset_1 + m_split_1;

    auto opts = x.options();

    // ===== Conv1 =====
    auto x_edge_0 = x_edge.narrow(1, 0, edge_split_0);
    auto x_0_flat = x.narrow(1, 0, m_split_0).reshape({E, m_split_0 * S});
    auto x_0_radial = x_0_flat * x_edge_0;
    auto z_0 = at::addmm(b_m0_1, x_0_radial, w_m0_1.t());
    auto gating_pre_sig = z_0.narrow(1, 0, extra_m0);

    auto conv1_main = at::empty({E, total_coeffs, m_out_1}, opts);
    conv1_main.narrow(1, 0, m_split_0).copy_(
        z_0.narrow(1, extra_m0, m_split_0 * m_out_1)
            .reshape({E, m_split_0, m_out_1})
    );

    auto x_edge_1 = x_edge.narrow(1, edge_split_0, edge_split_1);
    auto x_1_view = x.narrow(1, offset_1, m_split_1).reshape({E, 2, -1});
    auto x_1_scaled = x_1_view * x_edge_1.unsqueeze(1);
    auto x_1_cat = x_1_scaled.reshape({E, -1}).contiguous();
    auto out_cat_1 = at::matmul(x_1_cat, W_b1_m1.t());
    conv1_main.narrow(1, offset_1, num_l_1).copy_(
        out_cat_1.narrow(1, 0, num_l_1 * m_out_1).reshape({E, num_l_1, m_out_1})
    );
    conv1_main.narrow(1, offset_1 + num_l_1, num_l_1).copy_(
        out_cat_1.narrow(1, num_l_1 * m_out_1, num_l_1 * m_out_1)
            .reshape({E, num_l_1, m_out_1})
    );

    auto x_edge_2 = x_edge.narrow(
        1, edge_split_0 + edge_split_1, edge_split_2);
    auto x_2_view = x.narrow(1, offset_2, m_split_2).reshape({E, 2, -1});
    auto x_2_scaled = x_2_view * x_edge_2.unsqueeze(1);
    auto x_2_cat = x_2_scaled.reshape({E, -1}).contiguous();
    auto out_cat_2 = at::matmul(x_2_cat, W_b1_m2.t());
    conv1_main.narrow(1, offset_2, num_l_2).copy_(
        out_cat_2.narrow(1, 0, num_l_2 * m_out_1).reshape({E, num_l_2, m_out_1})
    );
    conv1_main.narrow(1, offset_2 + num_l_2, num_l_2).copy_(
        out_cat_2.narrow(1, num_l_2 * m_out_1, num_l_2 * m_out_1)
            .reshape({E, num_l_2, m_out_1})
    );

    // ===== GateActivation (m_prime=True) =====
    auto gating_post_sig = at::sigmoid(gating_pre_sig);  // (E, 256)
    auto gating = gating_post_sig.view({E, lmax, m_out_1});

    auto act_out = at::empty({E, total_coeffs, m_out_1}, opts);
    act_out.narrow(1, 0, 1).copy_(at::silu(conv1_main.narrow(1, 0, 1)));
    {
        auto v_first = conv1_main.narrow(1, 1, 6).reshape({E, 3, 2, m_out_1});
        auto v_first_out = v_first * gating.unsqueeze(1);
        act_out.narrow(1, 1, 6).copy_(v_first_out.reshape({E, 6, m_out_1}));
    }
    {
        auto v_last = conv1_main.narrow(1, 7, 2) * gating.narrow(1, 1, 1);
        act_out.narrow(1, 7, 2).copy_(v_last);
    }

    // ===== Conv2 =====
    auto out = at::empty({E, total_coeffs, m_out_2}, opts);

    auto x0_2_flat = act_out.narrow(1, 0, m_split_0)
        .reshape({E, m_split_0 * m_out_1});
    auto z_0_2 = at::addmm(b_m0_2, x0_2_flat, w_m0_2.t());
    out.narrow(1, 0, m_split_0).copy_(
        z_0_2.reshape({E, m_split_0, m_out_2})
    );

    auto x1_2_view = act_out.narrow(1, offset_1, m_split_1).reshape({E, 2, -1});
    auto x1_2_cat = x1_2_view.reshape({E, -1}).contiguous();
    auto out_cat_1_2 = at::matmul(x1_2_cat, W_b2_m1.t());
    out.narrow(1, offset_1, num_l_1).copy_(
        out_cat_1_2.narrow(1, 0, num_l_1 * m_out_2).reshape({E, num_l_1, m_out_2})
    );
    out.narrow(1, offset_1 + num_l_1, num_l_1).copy_(
        out_cat_1_2.narrow(1, num_l_1 * m_out_2, num_l_1 * m_out_2)
            .reshape({E, num_l_1, m_out_2})
    );

    auto x2_2_view = act_out.narrow(1, offset_2, m_split_2).reshape({E, 2, -1});
    auto x2_2_cat = x2_2_view.reshape({E, -1}).contiguous();
    auto out_cat_2_2 = at::matmul(x2_2_cat, W_b2_m2.t());
    out.narrow(1, offset_2, num_l_2).copy_(
        out_cat_2_2.narrow(1, 0, num_l_2 * m_out_2).reshape({E, num_l_2, m_out_2})
    );
    out.narrow(1, offset_2 + num_l_2, num_l_2).copy_(
        out_cat_2_2.narrow(1, num_l_2 * m_out_2, num_l_2 * m_out_2)
            .reshape({E, num_l_2, m_out_2})
    );

    return std::make_tuple(out, conv1_main, gating_post_sig);
}


// =====================================================================
// Backward
// =====================================================================
//
// Returns: (grad_x, grad_x_edge,
//           grad_w_m0_1, grad_b_m0_1, grad_W_b1_m1, grad_W_b1_m2,
//           grad_w_m0_2, grad_b_m0_2, grad_W_b2_m1, grad_W_b2_m2)
//
// Each weight grad is returned only if the corresponding need_* flag
// is true; else an empty tensor is returned (caller treats as None).
std::vector<at::Tensor> fused_so2_block_backward(
    at::Tensor grad_out,                   // [E, 9, M2]
    at::Tensor x,                          // [E, 9, S]   saved
    at::Tensor x_edge,                     // [E, R]      saved
    at::Tensor w_m0_1, at::Tensor b_m0_1,
    at::Tensor W_b1_m1, at::Tensor W_b1_m2,
    at::Tensor w_m0_2, at::Tensor b_m0_2,
    at::Tensor W_b2_m1, at::Tensor W_b2_m2,
    at::Tensor conv1_main,                 // [E, 9, M1]  saved from forward
    at::Tensor gating_post_sig,            // [E, extra_m0]  saved from forward
    int64_t m_split_0, int64_t m_split_1, int64_t m_split_2,
    int64_t edge_split_0, int64_t edge_split_1, int64_t edge_split_2,
    int64_t extra_m0, int64_t lmax,
    int64_t m_out_1, int64_t m_out_2,
    bool need_x, bool need_x_edge,
    bool need_w_m0_1, bool need_b_m0_1,
    bool need_W_b1_m1, bool need_W_b1_m2,
    bool need_w_m0_2, bool need_b_m0_2,
    bool need_W_b2_m1, bool need_W_b2_m2)
{
    const int64_t E = x.size(0);
    const int64_t S = x.size(2);
    const int64_t total_coeffs = (lmax + 1) * (lmax + 1);
    const int64_t num_l_1 = lmax;
    const int64_t num_l_2 = lmax - 1;
    const int64_t offset_1 = m_split_0;
    const int64_t offset_2 = offset_1 + m_split_1;

    auto opts = x.options();

    // -------------------------------------------------------------------
    // Reconstruct cheap intermediates from x / x_edge / conv1_main /
    // gating_post_sig.  All reshapes/narrows are zero-cost views.
    // -------------------------------------------------------------------
    auto x_edge_0 = x_edge.narrow(1, 0, edge_split_0);
    auto x_edge_1 = x_edge.narrow(1, edge_split_0, edge_split_1);
    auto x_edge_2 = x_edge.narrow(
        1, edge_split_0 + edge_split_1, edge_split_2);

    auto x_0_flat = x.narrow(1, 0, m_split_0).reshape({E, m_split_0 * S});
    auto x_1_view = x.narrow(1, offset_1, m_split_1).reshape({E, 2, -1});
    auto x_2_view = x.narrow(1, offset_2, m_split_2).reshape({E, 2, -1});

    auto gating = gating_post_sig.view({E, lmax, m_out_1});

    // act_out reconstruction (needed for conv2 weight grads only).  We
    // build it lazily below on demand.

    // -------------------------------------------------------------------
    // ===== Conv2 backward =====
    //
    //   conv2 takes act_out (E, 9, M1) -> out (E, 9, M2)
    //
    //   grad_act_out_m0 = grad_z_0_2 @ w_m0_2,   shape (E, 3, M1)
    //   grad_act_out_m1 = grad_o1 @ W_b2_m1,     shape (E, 4, M1)
    //   grad_act_out_m2 = grad_o2 @ W_b2_m2,     shape (E, 2, M1)
    // -------------------------------------------------------------------
    auto grad_act_out = at::empty({E, total_coeffs, m_out_1}, opts);

    // m=0 of conv2
    auto grad_z_0_2 = grad_out.narrow(1, 0, m_split_0).reshape({E, m_split_0 * m_out_2});
    if (need_x || need_x_edge || true /* need_act_for_act_bwd */) {
        auto g = at::matmul(grad_z_0_2, w_m0_2);  // (E, m_split_0 * M1)
        grad_act_out.narrow(1, 0, m_split_0).copy_(
            g.reshape({E, m_split_0, m_out_1})
        );
    }

    // m=1 of conv2  (positions [3..6] are contiguous along dim 1, so the
    // narrow+reshape is already what cat(real, imag) would produce -- no
    // explicit cat needed.)
    {
        auto grad_o1_2 = grad_out.narrow(1, offset_1, m_split_1)
            .reshape({E, m_split_1 * m_out_2})
            .contiguous();
        auto g = at::matmul(grad_o1_2, W_b2_m1);  // (E, 4*M1)
        grad_act_out.narrow(1, offset_1, m_split_1).copy_(
            g.reshape({E, m_split_1, m_out_1})
        );
    }

    // m=2 of conv2  (same trick)
    {
        auto grad_o2_2 = grad_out.narrow(1, offset_2, m_split_2)
            .reshape({E, m_split_2 * m_out_2})
            .contiguous();
        auto g = at::matmul(grad_o2_2, W_b2_m2);
        grad_act_out.narrow(1, offset_2, m_split_2).copy_(
            g.reshape({E, m_split_2, m_out_1})
        );
    }

    // -------------------------------------------------------------------
    // ===== Activation backward (m_prime=True, lmax=mmax=2) =====
    //
    //  forward (recap):
    //    gating_pre_sig = ...
    //    gating = sigmoid(gating_pre_sig).view(E, 2, M1)
    //    act_out[:, 0:1, :] = silu(conv1_main[:, 0:1, :])
    //    act_out[:, 1:7, :] = (conv1_main[:, 1:7, :].view(E,3,2,M1) * gating[:,None,:,:]).reshape(E,6,M1)
    //    act_out[:, 7:9, :] = conv1_main[:, 7:9, :] * gating[:, 1:2, :]
    //
    //  backward gives grad_act_out -> grad_conv1_main, grad_gating_post_sig
    // -------------------------------------------------------------------
    auto grad_conv1_main = at::empty({E, total_coeffs, m_out_1}, opts);
    auto grad_gating = at::zeros({E, lmax, m_out_1}, opts);

    // -- silu backward at position 0 --
    {
        auto u = conv1_main.narrow(1, 0, 1);             // (E, 1, M1)
        auto g = grad_act_out.narrow(1, 0, 1);
        auto sig_u = at::sigmoid(u);
        // d/du silu(u) = sig(u) * (1 + u*(1 - sig(u)))
        auto deriv = sig_u * (1.0 + u * (1.0 - sig_u));
        grad_conv1_main.narrow(1, 0, 1).copy_(g * deriv);
    }

    // -- positions 1..6 (interleaved m=(0,1,2) x l=(1,2)) --
    {
        auto v_first = conv1_main.narrow(1, 1, 6).reshape({E, 3, 2, m_out_1});
        auto grad_v_first = grad_act_out.narrow(1, 1, 6).reshape({E, 3, 2, m_out_1});
        // grad to the input copy of conv1_main: grad_v * gating
        auto gating_b = gating.unsqueeze(1);            // (E, 1, 2, M1)
        grad_conv1_main.narrow(1, 1, 6).copy_(
            (grad_v_first * gating_b).reshape({E, 6, m_out_1})
        );
        // grad to gating: sum over the m-dimension of (grad_v * v_first_input)
        grad_gating += (grad_v_first * v_first).sum(1);  // (E, 2, M1)
    }

    // -- positions 7..8 (both use gating[:,1,:]) --
    {
        auto v_last = conv1_main.narrow(1, 7, 2);                 // (E, 2, M1)
        auto grad_v_last = grad_act_out.narrow(1, 7, 2);
        auto gating_l2 = gating.narrow(1, 1, 1);                  // (E, 1, M1)
        grad_conv1_main.narrow(1, 7, 2).copy_(grad_v_last * gating_l2);
        // sum over the 2 positions for the gating l=2 contribution
        auto extra = (grad_v_last * v_last).sum(1, /*keepdim=*/true);  // (E,1,M1)
        grad_gating.narrow(1, 1, 1) += extra;
    }

    // -- sigmoid backward: d/dx sigmoid(x) = gating * (1 - gating) --
    auto sig_deriv = gating * (1.0 - gating);  // (E, 2, M1)
    auto grad_gating_pre_sig =
        (grad_gating * sig_deriv).reshape({E, extra_m0});  // (E, extra_m0)

    // -------------------------------------------------------------------
    // ===== Conv1 backward (with radial mul) =====
    // -------------------------------------------------------------------
    at::Tensor grad_x;
    at::Tensor grad_x_edge;
    if (need_x) {
        grad_x = at::empty({E, total_coeffs, S}, opts);
    }
    if (need_x_edge) {
        grad_x_edge = at::empty_like(x_edge);
    }

    // m=0 of conv1
    {
        auto grad_z_0_main =
            grad_conv1_main.narrow(1, 0, m_split_0).reshape({E, m_split_0 * m_out_1});
        // grad_z_0 = cat(grad_gating_pre_sig, grad_z_0_main) along dim 1
        auto grad_z_0 = at::cat({grad_gating_pre_sig, grad_z_0_main}, 1);  // (E, 640)

        if (need_x || need_x_edge) {
            auto grad_x_0_radial = at::matmul(grad_z_0, w_m0_1);  // (E, 384)
            if (need_x) {
                auto grad_x_0_flat = grad_x_0_radial * x_edge_0;
                grad_x.narrow(1, 0, m_split_0).copy_(
                    grad_x_0_flat.reshape({E, m_split_0, S})
                );
            }
            if (need_x_edge) {
                grad_x_edge.narrow(1, 0, edge_split_0)
                    .copy_(grad_x_0_radial * x_0_flat);
            }
        }
    }

    // m=1 of conv1  (positions [3..6] contiguous in dim 1; cat-free)
    {
        auto grad_out_cat_1 = grad_conv1_main.narrow(1, offset_1, m_split_1)
            .reshape({E, m_split_1 * m_out_1})
            .contiguous();
        if (need_x || need_x_edge) {
            auto grad_x_1_cat = at::matmul(grad_out_cat_1, W_b1_m1);  // (E, 4*S)
            auto grad_x_1_scaled = grad_x_1_cat.view({E, 2, -1});
            if (need_x) {
                auto grad_x_1_view = grad_x_1_scaled * x_edge_1.unsqueeze(1);
                grad_x.narrow(1, offset_1, m_split_1).copy_(
                    grad_x_1_view.reshape({E, m_split_1, S})
                );
            }
            if (need_x_edge) {
                grad_x_edge.narrow(1, edge_split_0, edge_split_1)
                    .copy_((grad_x_1_scaled * x_1_view).sum(1));
            }
        }
    }

    // m=2 of conv1  (positions [7..8] contiguous in dim 1; cat-free)
    {
        auto grad_out_cat_2 = grad_conv1_main.narrow(1, offset_2, m_split_2)
            .reshape({E, m_split_2 * m_out_1})
            .contiguous();
        if (need_x || need_x_edge) {
            auto grad_x_2_cat = at::matmul(grad_out_cat_2, W_b1_m2);
            auto grad_x_2_scaled = grad_x_2_cat.view({E, 2, -1});
            if (need_x) {
                auto grad_x_2_view = grad_x_2_scaled * x_edge_2.unsqueeze(1);
                grad_x.narrow(1, offset_2, m_split_2).copy_(
                    grad_x_2_view.reshape({E, m_split_2, S})
                );
            }
            if (need_x_edge) {
                grad_x_edge.narrow(1, edge_split_0 + edge_split_1, edge_split_2)
                    .copy_((grad_x_2_scaled * x_2_view).sum(1));
            }
        }
    }

    // -------------------------------------------------------------------
    // Weight grads (only computed when requested -- inference skips these)
    //
    // Reconstruction notes:
    //   x_0_radial = x_0_flat * x_edge_0          (recompute for grad_w_m0_1)
    //   x_1_cat    = (x_1_view * x_edge_1.unsqueeze(1)).reshape(E,-1)  for grad_W_b1_m1
    //   x_2_cat    = (x_2_view * x_edge_2.unsqueeze(1)).reshape(E,-1)  for grad_W_b1_m2
    //   act_out    = full activation output                            for conv2 weight grads
    // Recompute these only if needed.
    // -------------------------------------------------------------------
    auto empty_t = at::empty({0}, opts);
    at::Tensor grad_w_m0_1 = empty_t;
    at::Tensor grad_b_m0_1 = empty_t;
    at::Tensor grad_W_b1_m1 = empty_t;
    at::Tensor grad_W_b1_m2 = empty_t;
    at::Tensor grad_w_m0_2 = empty_t;
    at::Tensor grad_b_m0_2 = empty_t;
    at::Tensor grad_W_b2_m1 = empty_t;
    at::Tensor grad_W_b2_m2 = empty_t;

    bool any_conv1_w_grad = need_w_m0_1 || need_b_m0_1 || need_W_b1_m1 || need_W_b1_m2;
    bool any_conv2_w_grad = need_w_m0_2 || need_b_m0_2 || need_W_b2_m1 || need_W_b2_m2;

    if (any_conv1_w_grad) {
        auto grad_z_0_main =
            grad_conv1_main.narrow(1, 0, m_split_0).reshape({E, m_split_0 * m_out_1});
        auto grad_z_0 = at::cat({grad_gating_pre_sig, grad_z_0_main}, 1);
        if (need_w_m0_1) {
            auto x_0_radial = x_0_flat * x_edge_0;
            grad_w_m0_1 = at::matmul(grad_z_0.t(), x_0_radial);
        }
        if (need_b_m0_1) {
            grad_b_m0_1 = grad_z_0.sum(0);
        }
        if (need_W_b1_m1) {
            auto grad_real =
                grad_conv1_main.narrow(1, offset_1, num_l_1).reshape({E, num_l_1 * m_out_1});
            auto grad_imag =
                grad_conv1_main.narrow(1, offset_1 + num_l_1, num_l_1)
                    .reshape({E, num_l_1 * m_out_1});
            auto grad_out_cat = at::cat({grad_real, grad_imag}, 1);
            auto x_1_cat = (x_1_view * x_edge_1.unsqueeze(1)).reshape({E, -1}).contiguous();
            grad_W_b1_m1 = at::matmul(grad_out_cat.t(), x_1_cat);
        }
        if (need_W_b1_m2) {
            auto grad_real =
                grad_conv1_main.narrow(1, offset_2, num_l_2).reshape({E, num_l_2 * m_out_1});
            auto grad_imag =
                grad_conv1_main.narrow(1, offset_2 + num_l_2, num_l_2)
                    .reshape({E, num_l_2 * m_out_1});
            auto grad_out_cat = at::cat({grad_real, grad_imag}, 1);
            auto x_2_cat = (x_2_view * x_edge_2.unsqueeze(1)).reshape({E, -1}).contiguous();
            grad_W_b1_m2 = at::matmul(grad_out_cat.t(), x_2_cat);
        }
    }

    if (any_conv2_w_grad) {
        // act_out reconstruction
        auto act_out = at::empty({E, total_coeffs, m_out_1}, opts);
        act_out.narrow(1, 0, 1).copy_(at::silu(conv1_main.narrow(1, 0, 1)));
        {
            auto v_first = conv1_main.narrow(1, 1, 6).reshape({E, 3, 2, m_out_1});
            auto v_first_out = v_first * gating.unsqueeze(1);
            act_out.narrow(1, 1, 6).copy_(v_first_out.reshape({E, 6, m_out_1}));
        }
        {
            auto v_last = conv1_main.narrow(1, 7, 2) * gating.narrow(1, 1, 1);
            act_out.narrow(1, 7, 2).copy_(v_last);
        }

        if (need_w_m0_2) {
            auto x0_2_flat = act_out.narrow(1, 0, m_split_0)
                .reshape({E, m_split_0 * m_out_1}).contiguous();
            auto grad_z = grad_out.narrow(1, 0, m_split_0)
                .reshape({E, m_split_0 * m_out_2}).contiguous();
            grad_w_m0_2 = at::matmul(grad_z.t(), x0_2_flat);
        }
        if (need_b_m0_2) {
            auto grad_z = grad_out.narrow(1, 0, m_split_0)
                .reshape({E, m_split_0 * m_out_2});
            grad_b_m0_2 = grad_z.sum(0);
        }
        if (need_W_b2_m1) {
            auto x_cat = act_out.narrow(1, offset_1, m_split_1)
                .reshape({E, -1}).contiguous();
            auto grad_real = grad_out.narrow(1, offset_1, num_l_1)
                .reshape({E, num_l_1 * m_out_2});
            auto grad_imag = grad_out.narrow(1, offset_1 + num_l_1, num_l_1)
                .reshape({E, num_l_1 * m_out_2});
            auto grad_o = at::cat({grad_real, grad_imag}, 1);
            grad_W_b2_m1 = at::matmul(grad_o.t(), x_cat);
        }
        if (need_W_b2_m2) {
            auto x_cat = act_out.narrow(1, offset_2, m_split_2)
                .reshape({E, -1}).contiguous();
            auto grad_real = grad_out.narrow(1, offset_2, num_l_2)
                .reshape({E, num_l_2 * m_out_2});
            auto grad_imag = grad_out.narrow(1, offset_2 + num_l_2, num_l_2)
                .reshape({E, num_l_2 * m_out_2});
            auto grad_o = at::cat({grad_real, grad_imag}, 1);
            grad_W_b2_m2 = at::matmul(grad_o.t(), x_cat);
        }
    }

    if (!need_x) grad_x = empty_t;
    if (!need_x_edge) grad_x_edge = empty_t;

    return std::vector<at::Tensor>{
        grad_x, grad_x_edge,
        grad_w_m0_1, grad_b_m0_1, grad_W_b1_m1, grad_W_b1_m2,
        grad_w_m0_2, grad_b_m0_2, grad_W_b2_m1, grad_W_b2_m2,
    };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_so2_block_forward",
          &fused_so2_block_forward,
          "Fused SO2 block forward (returns (out, conv1_main, gating_post_sig))");
    m.def("fused_so2_block_backward",
          &fused_so2_block_backward,
          "Fused SO2 block backward (returns 10-element list of grads)");
}
"""


def _build() -> object:
    """Lazy-build the fused SO2 block C++ extension (cached after 1st call)."""
    if _KERNELS_BOX[0] is not None:
        return _KERNELS_BOX[0]
    with _LOCK:
        if _KERNELS_BOX[0] is not None:
            return _KERNELS_BOX[0]
        from torch.utils import cpp_extension

        _KERNELS_BOX[0] = cpp_extension.load_inline(
            name="fairchem_uma_fused_so2_block_v3",
            cpp_sources=[_CPP_SRC],
            extra_cflags=[
                "-O3",
                "-march=native",
                "-mprefer-vector-width=512",
                "-fopenmp",
                "-ffast-math",
                "-funroll-loops",
            ],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    return _KERNELS_BOX[0]


def _eager_so2_block_forward(
    x: torch.Tensor,
    x_edge: torch.Tensor,
    w_m0_1: torch.Tensor,
    b_m0_1: torch.Tensor,
    W_b1_m1: torch.Tensor,
    W_b1_m2: torch.Tensor,
    w_m0_2: torch.Tensor,
    b_m0_2: torch.Tensor,
    W_b2_m1: torch.Tensor,
    W_b2_m2: torch.Tensor,
    m_split_sizes: tuple[int, int, int],
    edge_split_sizes: tuple[int, int, int],
    extra_m0: int,
    lmax: int,
    m_out_1: int,
    m_out_2: int,
) -> torch.Tensor:
    """
    Pure-Python reference of the fused SO2 block. Used:
    - in tests, to validate the C++ kernel bit-exactly,
    - independent verification against the live module pipeline.

    Mirrors `_FusedConv1Func.forward` math + `GateActivation.forward`
    (m_prime=True branch) + `_FusedConv2Func.forward` math.
    """
    E = x.shape[0]
    num_l_1 = lmax
    num_l_2 = lmax - 1
    offset_1 = m_split_sizes[0]
    offset_2 = offset_1 + m_split_sizes[1]

    x_edge_0 = x_edge.narrow(1, 0, edge_split_sizes[0])
    x_edge_1 = x_edge.narrow(1, edge_split_sizes[0], edge_split_sizes[1])
    x_edge_2 = x_edge.narrow(
        1, edge_split_sizes[0] + edge_split_sizes[1], edge_split_sizes[2]
    )

    # m=0
    x_0_flat = x.narrow(1, 0, m_split_sizes[0]).reshape(E, -1)
    x_0_radial = x_0_flat * x_edge_0
    z_0 = torch.addmm(b_m0_1, x_0_radial, w_m0_1.t())
    gating_pre_sig = z_0[:, :extra_m0]
    main_m0 = z_0[:, extra_m0:].view(E, m_split_sizes[0], m_out_1)

    # m=1
    x_1_view = x.narrow(1, offset_1, m_split_sizes[1]).reshape(E, 2, -1)
    x_1_scaled = x_1_view * x_edge_1.unsqueeze(1)
    x_1_cat = x_1_scaled.reshape(E, -1).contiguous()
    out_cat_1 = x_1_cat @ W_b1_m1.t()
    main_m1_real = out_cat_1[:, : num_l_1 * m_out_1].view(E, num_l_1, m_out_1)
    main_m1_imag = out_cat_1[:, num_l_1 * m_out_1 :].view(E, num_l_1, m_out_1)

    # m=2
    x_2_view = x.narrow(1, offset_2, m_split_sizes[2]).reshape(E, 2, -1)
    x_2_scaled = x_2_view * x_edge_2.unsqueeze(1)
    x_2_cat = x_2_scaled.reshape(E, -1).contiguous()
    out_cat_2 = x_2_cat @ W_b1_m2.t()
    main_m2_real = out_cat_2[:, : num_l_2 * m_out_1].view(E, num_l_2, m_out_1)
    main_m2_imag = out_cat_2[:, num_l_2 * m_out_1 :].view(E, num_l_2, m_out_1)

    conv1_main = torch.cat(
        [main_m0, main_m1_real, main_m1_imag, main_m2_real, main_m2_imag], dim=1
    )

    gating = torch.sigmoid(gating_pre_sig).view(E, lmax, m_out_1)
    scalars = torch.nn.functional.silu(conv1_main[:, 0:1, :])
    v_first = (
        conv1_main[:, 1:7, :].reshape(E, 3, 2, m_out_1) * gating.unsqueeze(1)
    ).reshape(E, 6, m_out_1)
    v_last = conv1_main[:, 7:9, :] * gating[:, 1:2, :]
    act_out = torch.cat([scalars, v_first, v_last], dim=1)

    # m=0 conv2
    x0_2 = act_out.narrow(1, 0, m_split_sizes[0]).reshape(E, -1)
    z_0_2 = torch.addmm(b_m0_2, x0_2, w_m0_2.t())
    out_m0 = z_0_2.view(E, m_split_sizes[0], m_out_2)

    # m=1 conv2
    x1_2 = act_out.narrow(1, offset_1, m_split_sizes[1]).reshape(E, -1).contiguous()
    o1 = x1_2 @ W_b2_m1.t()
    out_m1_real = o1[:, : num_l_1 * m_out_2].view(E, num_l_1, m_out_2)
    out_m1_imag = o1[:, num_l_1 * m_out_2 :].view(E, num_l_1, m_out_2)

    # m=2 conv2
    x2_2 = act_out.narrow(1, offset_2, m_split_sizes[2]).reshape(E, -1).contiguous()
    o2 = x2_2 @ W_b2_m2.t()
    out_m2_real = o2[:, : num_l_2 * m_out_2].view(E, num_l_2, m_out_2)
    out_m2_imag = o2[:, num_l_2 * m_out_2 :].view(E, num_l_2, m_out_2)

    return torch.cat(
        [out_m0, out_m1_real, out_m1_imag, out_m2_real, out_m2_imag], dim=1
    )


class _FusedSO2BlockFunc(torch.autograd.Function):
    """
    autograd.Function wrapping the fused SO2 block.

    Phase 2: forward calls C++ kernel returning (out, conv1_main,
    gating_post_sig). Backward calls C++ kernel using saved tensors
    and returns 10 input gradients (or empty tensors -> None for
    inputs whose `requires_grad=False`).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        w_m0_1: torch.Tensor,
        b_m0_1: torch.Tensor,
        W_b1_m1: torch.Tensor,
        W_b1_m2: torch.Tensor,
        w_m0_2: torch.Tensor,
        b_m0_2: torch.Tensor,
        W_b2_m1: torch.Tensor,
        W_b2_m2: torch.Tensor,
        m_split_sizes: tuple[int, int, int],
        edge_split_sizes: tuple[int, int, int],
        extra_m0: int,
        lmax: int,
        m_out_1: int,
        m_out_2: int,
    ) -> torch.Tensor:
        kernels = _build()
        out, conv1_main, gating_post_sig = kernels.fused_so2_block_forward(
            x.contiguous(),
            x_edge.contiguous(),
            w_m0_1.contiguous(),
            b_m0_1.contiguous(),
            W_b1_m1.contiguous(),
            W_b1_m2.contiguous(),
            w_m0_2.contiguous(),
            b_m0_2.contiguous(),
            W_b2_m1.contiguous(),
            W_b2_m2.contiguous(),
            int(m_split_sizes[0]),
            int(m_split_sizes[1]),
            int(m_split_sizes[2]),
            int(edge_split_sizes[0]),
            int(edge_split_sizes[1]),
            int(edge_split_sizes[2]),
            int(extra_m0),
            int(lmax),
            int(m_out_1),
            int(m_out_2),
        )
        ctx.save_for_backward(
            x,
            x_edge,
            w_m0_1,
            b_m0_1,
            W_b1_m1,
            W_b1_m2,
            w_m0_2,
            b_m0_2,
            W_b2_m1,
            W_b2_m2,
            conv1_main,
            gating_post_sig,
        )
        ctx.m_split_sizes = m_split_sizes
        ctx.edge_split_sizes = edge_split_sizes
        ctx.extra_m0 = int(extra_m0)
        ctx.lmax = int(lmax)
        ctx.m_out_1 = int(m_out_1)
        ctx.m_out_2 = int(m_out_2)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        kernels = _build()
        (
            x,
            x_edge,
            w_m0_1,
            b_m0_1,
            W_b1_m1,
            W_b1_m2,
            w_m0_2,
            b_m0_2,
            W_b2_m1,
            W_b2_m2,
            conv1_main,
            gating_post_sig,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad

        grads = kernels.fused_so2_block_backward(
            grad_out.contiguous(),
            x,
            x_edge,
            w_m0_1,
            b_m0_1,
            W_b1_m1,
            W_b1_m2,
            w_m0_2,
            b_m0_2,
            W_b2_m1,
            W_b2_m2,
            conv1_main,
            gating_post_sig,
            int(ctx.m_split_sizes[0]),
            int(ctx.m_split_sizes[1]),
            int(ctx.m_split_sizes[2]),
            int(ctx.edge_split_sizes[0]),
            int(ctx.edge_split_sizes[1]),
            int(ctx.edge_split_sizes[2]),
            int(ctx.extra_m0),
            int(ctx.lmax),
            int(ctx.m_out_1),
            int(ctx.m_out_2),
            bool(needs[0]),  # x
            bool(needs[1]),  # x_edge
            bool(needs[2]),  # w_m0_1
            bool(needs[3]),  # b_m0_1
            bool(needs[4]),  # W_b1_m1
            bool(needs[5]),  # W_b1_m2
            bool(needs[6]),  # w_m0_2
            bool(needs[7]),  # b_m0_2
            bool(needs[8]),  # W_b2_m1
            bool(needs[9]),  # W_b2_m2
        )

        # Empty tensor returned from C++ for grads we don't need;
        # convert to None for autograd.
        out_grads = []
        for i, g in enumerate(grads):
            if needs[i] and g.numel() > 0:
                out_grads.append(g)
            else:
                out_grads.append(None)

        return (
            *out_grads,
            None,  # m_split_sizes
            None,  # edge_split_sizes
            None,  # extra_m0
            None,  # lmax
            None,  # m_out_1
            None,  # m_out_2
        )


def fused_so2_block(
    x: torch.Tensor,
    x_edge: torch.Tensor,
    w_m0_1: torch.Tensor,
    b_m0_1: torch.Tensor,
    W_b1_m1: torch.Tensor,
    W_b1_m2: torch.Tensor,
    w_m0_2: torch.Tensor,
    b_m0_2: torch.Tensor,
    W_b2_m1: torch.Tensor,
    W_b2_m2: torch.Tensor,
    m_split_sizes: tuple[int, int, int],
    edge_split_sizes: tuple[int, int, int],
    extra_m0: int,
    lmax: int,
    m_out_1: int,
    m_out_2: int,
) -> torch.Tensor:
    """
    Fused SO2 conv1 + GateActivation(m_prime=True) + conv2 block.

    Args:
        x: [E, (lmax+1)**2, sphere_channels] post-Wigner edge features.
        x_edge: [E, total_radial] radial features for conv1.
        w_m0_1, b_m0_1: conv1 m=0 Linear weight/bias.
        W_b1_m1, W_b1_m2: conv1 m=1, m=2 prebuilt block-diagonal weights.
        w_m0_2, b_m0_2: conv2 m=0 Linear weight/bias.
        W_b2_m1, W_b2_m2: conv2 m=1, m=2 prebuilt block-diagonal weights.
        m_split_sizes: per-m coefficient counts (e.g. (3, 4, 2) for lmax=mmax=2).
        edge_split_sizes: per-m radial-feature counts.
        extra_m0: number of gating channels at the start of conv1's m=0 output.
        lmax: equal to mmax in this kernel.
        m_out_1, m_out_2: output channel counts for conv1 and conv2.

    Returns:
        [E, (lmax+1)**2, m_out_2]
    """
    return _FusedSO2BlockFunc.apply(
        x,
        x_edge,
        w_m0_1,
        b_m0_1,
        W_b1_m1,
        W_b1_m2,
        w_m0_2,
        b_m0_2,
        W_b2_m1,
        W_b2_m2,
        m_split_sizes,
        edge_split_sizes,
        extra_m0,
        lmax,
        m_out_1,
        m_out_2,
    )
