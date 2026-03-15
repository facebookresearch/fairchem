/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * CPU-optimized kernels for UMA inference.
 * Fused gather + block-diagonal Wigner rotation + L↔M permutation.
 *
 * Compiled via torch.utils.cpp_extension.load().
 */

#include <torch/extension.h>
#include <omp.h>
#include <cstring>

// L_TO_M_GATHER_IDX: out[m] = rotated[L_TO_M[m]]
// For output M-position m, read from L-position L_TO_M[m]
static constexpr int L_TO_M[9] = {0, 2, 6, 3, 7, 1, 5, 8, 4};
// M_TO_L_GATHER_IDX: out[l] = input[M_TO_L[l]]
// For output L-position l, read from M-position M_TO_L[l]
static constexpr int M_TO_L[9] = {0, 5, 1, 3, 8, 6, 2, 4, 7};

// Block structure: L=0 is index 0, L=1 is indices 1-3, L=2 is indices 4-8
// Given an L-index, which block does it belong to, and what's the block start?
static constexpr int BLOCK_START[9] = {0, 1, 1, 1, 4, 4, 4, 4, 4};
static constexpr int BLOCK_SIZE[9]  = {1, 3, 3, 3, 5, 5, 5, 5, 5};

/*
 * Forward: node_to_edge_wigner_permute
 *
 * Fuses: gather x[src], x[tgt] -> concat -> block-diag Wigner rotate -> L→M permute
 * Input:  x [N, 9, C], edge_index [2, E], wigner [E, 9, 9]
 * Output: out [E, 9, 2C] in M-major order
 *
 * Python equivalent:
 *   x_cat = cat(x[src], x[tgt], dim=2)  # [E, 9, 2C]
 *   rotated = block_diag_bmm(W, x_cat)  # [E, 9, 2C] L-major
 *   out = rotated[:, L_TO_M, :]          # gather: out[m] = rotated[L_TO_M[m]]
 */
torch::Tensor node_to_edge_wigner_permute_fwd(
    const torch::Tensor& x,          // [N, 9, C]
    const torch::Tensor& edge_index,  // [2, E]
    const torch::Tensor& wigner       // [E, 9, 9]
) {
    const int64_t N = x.size(0);
    const int64_t C = x.size(2);
    const int64_t E = edge_index.size(1);
    const int64_t C2 = C * 2;

    auto out = torch::empty({E, 9, C2}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const int64_t* ei_ptr = edge_index.data_ptr<int64_t>();
    const float* w_ptr = wigner.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const int64_t x_stride_n = 9 * C;
    const int64_t out_stride_e = 9 * C2;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; e++) {
        const int64_t src = ei_ptr[e];
        const int64_t tgt = ei_ptr[E + e];

        const float* xs = x_ptr + src * x_stride_n;
        const float* xt = x_ptr + tgt * x_stride_n;
        const float* W = w_ptr + e * 81;
        float* O = out_ptr + e * out_stride_e;

        // For each output M-position m (0..8):
        //   l = L_TO_M[m]  (which L-index to read from)
        //   out[m, :] = sum_j W[l, j] * x_cat[j, :]
        //   where j ranges over the block containing l
        for (int m = 0; m < 9; m++) {
            const int l = L_TO_M[m];  // L-index for this output row
            const int bs = BLOCK_START[l];
            const int bn = BLOCK_SIZE[l];
            float* dst = O + m * C2;

            for (int64_t c = 0; c < C; c++) {
                float vs = 0.0f, vt = 0.0f;
                for (int j = 0; j < bn; j++) {
                    const int lj = bs + j;
                    vs += W[l * 9 + lj] * xs[lj * C + c];
                    vt += W[l * 9 + lj] * xt[lj * C + c];
                }
                dst[c]     = vs;
                dst[C + c] = vt;
            }
        }
    }

    return out;
}

/*
 * Backward dx: node_to_edge_wigner_permute
 *
 * grad is in M-major. We need:
 *   grad_l[l] = grad[m] where L_TO_M[m] == l, i.e. m = M_TO_L_inv[l]
 *   Actually: grad_l = grad[:, M_TO_L, :] means grad_l[l] = grad[M_TO_L[l]]
 *   grad_cat[j] = sum_l W^T[j,l] * grad_l[l] = sum_l W[l,j] * grad_l[l]
 *   Then scatter grad_cat[:, :C] to src, grad_cat[:, C:] to tgt
 */
torch::Tensor node_to_edge_wigner_permute_bwd_dx(
    const torch::Tensor& grad_out,    // [E, 9, 2C] M-major
    const torch::Tensor& wigner,      // [E, 9, 9]
    const torch::Tensor& edge_index,  // [2, E]
    int64_t num_nodes
) {
    const int64_t E = edge_index.size(1);
    const int64_t C2 = grad_out.size(2);
    const int64_t C = C2 / 2;

    const int num_threads = omp_get_max_threads();
    auto grad_x = torch::zeros({num_nodes, 9, C}, grad_out.options());

    const float* g_ptr = grad_out.data_ptr<float>();
    const float* w_ptr = wigner.data_ptr<float>();
    const int64_t* ei_ptr = edge_index.data_ptr<int64_t>();
    float* gx_ptr = grad_x.data_ptr<float>();

    const int64_t g_stride_e = 9 * C2;
    const int64_t gx_stride_n = 9 * C;

    // Thread-local buffers for scatter
    std::vector<std::vector<float>> local_bufs(num_threads);
    for (auto& buf : local_bufs) {
        buf.resize(num_nodes * 9 * C, 0.0f);
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* local = local_bufs[tid].data();

        #pragma omp for schedule(static)
        for (int64_t e = 0; e < E; e++) {
            const int64_t src_node = ei_ptr[e];
            const int64_t tgt_node = ei_ptr[E + e];
            const float* G = g_ptr + e * g_stride_e;
            const float* W = w_ptr + e * 81;

            // Precompute M→L permuted grad pointers
            const float* gl[9];
            for (int li = 0; li < 9; li++) {
                gl[li] = G + M_TO_L[li] * C2;
            }

            float* ds_base = local + src_node * gx_stride_n;
            float* dt_base = local + tgt_node * gx_stride_n;

            // L=0: j=0, block has only l=0
            {
                const float w_val = W[0];
                float* ds = ds_base;
                float* dt = dt_base;
                const float* g0 = gl[0];
                for (int64_t c = 0; c < C; c++) {
                    ds[c] += w_val * g0[c];
                    dt[c] += w_val * g0[C + c];
                }
            }

            // L=1: j=1,2,3; block rows l=1,2,3
            for (int j = 0; j < 3; j++) {
                float* ds = ds_base + (1 + j) * C;
                float* dt = dt_base + (1 + j) * C;
                const float w0 = W[1 * 9 + (1 + j)];
                const float w1 = W[2 * 9 + (1 + j)];
                const float w2 = W[3 * 9 + (1 + j)];
                const float* g1 = gl[1];
                const float* g2 = gl[2];
                const float* g3 = gl[3];
                for (int64_t c = 0; c < C; c++) {
                    const float vs = w0 * g1[c] + w1 * g2[c] + w2 * g3[c];
                    const float vt = w0 * g1[C+c] + w1 * g2[C+c] + w2 * g3[C+c];
                    ds[c] += vs;
                    dt[c] += vt;
                }
            }

            // L=2: j=4..8; block rows l=4..8
            for (int j = 0; j < 5; j++) {
                float* ds = ds_base + (4 + j) * C;
                float* dt = dt_base + (4 + j) * C;
                float ww[5];
                for (int i = 0; i < 5; i++) {
                    ww[i] = W[(4 + i) * 9 + (4 + j)];
                }
                for (int64_t c = 0; c < C; c++) {
                    float vs = 0.0f, vt = 0.0f;
                    for (int i = 0; i < 5; i++) {
                        vs += ww[i] * gl[4 + i][c];
                        vt += ww[i] * gl[4 + i][C + c];
                    }
                    ds[c] += vs;
                    dt[c] += vt;
                }
            }
        }
    }

    // Reduce thread-local buffers
    const int64_t total = num_nodes * 9 * C;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total; i++) {
        float sum = 0.0f;
        for (int t = 0; t < num_threads; t++) {
            sum += local_bufs[t][i];
        }
        gx_ptr[i] = sum;
    }

    return grad_x;
}

/*
 * Backward dw: node_to_edge_wigner_permute
 *
 * grad_wigner[l, j] = sum_c grad_l[l, c] * x_cat[j, c]
 * where grad_l[l] = grad[M_TO_L[l]]
 * Only block-diagonal entries.
 */
torch::Tensor node_to_edge_wigner_permute_bwd_dw(
    const torch::Tensor& grad_out,    // [E, 9, 2C] M-major
    const torch::Tensor& x,           // [N, 9, C]
    const torch::Tensor& edge_index   // [2, E]
) {
    const int64_t E = edge_index.size(1);
    const int64_t C2 = grad_out.size(2);
    const int64_t C = C2 / 2;

    auto grad_wigner = torch::zeros({E, 9, 9}, grad_out.options());

    const float* g_ptr = grad_out.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const int64_t* ei_ptr = edge_index.data_ptr<int64_t>();
    float* gw_ptr = grad_wigner.data_ptr<float>();

    const int64_t g_stride = 9 * C2;
    const int64_t x_stride = 9 * C;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; e++) {
        const int64_t src_node = ei_ptr[e];
        const int64_t tgt_node = ei_ptr[E + e];
        const float* G = g_ptr + e * g_stride;
        const float* xs = x_ptr + src_node * x_stride;
        const float* xt = x_ptr + tgt_node * x_stride;
        float* GW = gw_ptr + e * 81;

        // dW[l, j] = sum_c grad_l[l, c] * x_cat[j, c]
        // grad_l[l] = G[M_TO_L[l]]
        // x_cat[j, c] = [xs[j,c], xt[j,c]]

        // L=0 block
        {
            const float* gl = G + M_TO_L[0] * C2;
            float dot = 0.0f;
            for (int64_t c = 0; c < C; c++) {
                dot += gl[c] * xs[0 * C + c] + gl[C + c] * xt[0 * C + c];
            }
            GW[0] = dot;
        }

        // L=1 block (3x3)
        for (int i = 0; i < 3; i++) {
            const float* gl = G + M_TO_L[1 + i] * C2;
            for (int j = 0; j < 3; j++) {
                float dot = 0.0f;
                const float* xsj = xs + (1 + j) * C;
                const float* xtj = xt + (1 + j) * C;
                for (int64_t c = 0; c < C; c++) {
                    dot += gl[c] * xsj[c] + gl[C + c] * xtj[c];
                }
                GW[(1 + i) * 9 + (1 + j)] = dot;
            }
        }

        // L=2 block (5x5)
        for (int i = 0; i < 5; i++) {
            const float* gl = G + M_TO_L[4 + i] * C2;
            for (int j = 0; j < 5; j++) {
                float dot = 0.0f;
                const float* xsj = xs + (4 + j) * C;
                const float* xtj = xt + (4 + j) * C;
                for (int64_t c = 0; c < C; c++) {
                    dot += gl[c] * xsj[c] + gl[C + c] * xtj[c];
                }
                GW[(4 + i) * 9 + (4 + j)] = dot;
            }
        }
    }

    return grad_wigner;
}

/*
 * Forward: permute_wigner_inv_edge_to_node
 *
 * Fuses: M→L permute + block-diagonal Wigner inverse rotation
 * Input:  x [E, 9, C] M-major, wigner_inv [E, 9, 9]
 * Output: out [E, 9, C] L-major
 *
 * Python equivalent:
 *   x_l = x[:, M_TO_L, :]  (gather: x_l[l] = x[M_TO_L[l]])
 *   out = block_diag_bmm(W_inv, x_l)
 */
torch::Tensor permute_wigner_inv_fwd(
    const torch::Tensor& x,          // [E, 9, C] M-major
    const torch::Tensor& wigner_inv  // [E, 9, 9]
) {
    const int64_t E = x.size(0);
    const int64_t C = x.size(2);

    auto out = torch::empty({E, 9, C}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = wigner_inv.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const int64_t stride = 9 * C;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; e++) {
        const float* X = x_ptr + e * stride;
        const float* W = w_ptr + e * 81;
        float* O = out_ptr + e * stride;

        // out[i] = sum_j W_inv[i,j] * x_l[j]
        // where x_l[j] = X[M_TO_L[j]] (M→L gather)
        for (int i = 0; i < 9; i++) {
            const int bs = BLOCK_START[i];
            const int bn = BLOCK_SIZE[i];
            float* dst = O + i * C;

            for (int64_t c = 0; c < C; c++) {
                float v = 0.0f;
                for (int j = 0; j < bn; j++) {
                    const int lj = bs + j;
                    // x_l[lj] = X[M_TO_L[lj]]
                    v += W[i * 9 + lj] * X[M_TO_L[lj] * C + c];
                }
                dst[c] = v;
            }
        }
    }

    return out;
}

/*
 * Backward dx: permute_wigner_inv_edge_to_node
 *
 * grad_x_l[j] = sum_i W_inv^T[j,i] * grad_out[i] = sum_i W_inv[i,j] * grad_out[i]
 * Then L→M permute: grad_x[M_TO_L[l]] = grad_x_l[l]
 * Equivalently: grad_x[m] = grad_x_l[L_TO_M[m]]  (gather with L_TO_M)
 */
torch::Tensor permute_wigner_inv_bwd_dx(
    const torch::Tensor& grad_out,   // [E, 9, C] L-major
    const torch::Tensor& wigner_inv  // [E, 9, 9]
) {
    const int64_t E = grad_out.size(0);
    const int64_t C = grad_out.size(2);

    auto grad_x = torch::empty({E, 9, C}, grad_out.options());

    const float* g_ptr = grad_out.data_ptr<float>();
    const float* w_ptr = wigner_inv.data_ptr<float>();
    float* gx_ptr = grad_x.data_ptr<float>();

    const int64_t stride = 9 * C;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; e++) {
        const float* G = g_ptr + e * stride;
        const float* W = w_ptr + e * 81;
        float* GX = gx_ptr + e * stride;

        // grad_x_l[j] = sum_i_in_block(j) W[i,j] * G[i]
        // Then output in M-major: GX[m] = grad_x_l[L_TO_M[m]]
        // Compute all 9 grad_x_l values first, then permute

        float grad_x_l[9];  // per-channel would be too large, do channel loop outside

        for (int64_t c = 0; c < C; c++) {
            // Compute grad_x_l for this channel
            for (int j = 0; j < 9; j++) {
                const int bs = BLOCK_START[j];
                const int bn = BLOCK_SIZE[j];
                float v = 0.0f;
                for (int i = 0; i < bn; i++) {
                    v += W[(bs + i) * 9 + j] * G[(bs + i) * C + c];
                }
                grad_x_l[j] = v;
            }
            // L→M permute: GX[m] = grad_x_l[L_TO_M[m]]
            for (int m = 0; m < 9; m++) {
                GX[m * C + c] = grad_x_l[L_TO_M[m]];
            }
        }
    }

    return grad_x;
}

/*
 * Backward dw: permute_wigner_inv_edge_to_node
 *
 * grad_wigner[i, j] = sum_c grad_out[i, c] * x_l[j, c]
 * where x_l[j] = x_m[M_TO_L[j]]
 */
torch::Tensor permute_wigner_inv_bwd_dw(
    const torch::Tensor& grad_out,  // [E, 9, C] L-major
    const torch::Tensor& x_m        // [E, 9, C] M-major
) {
    const int64_t E = grad_out.size(0);
    const int64_t C = grad_out.size(2);

    auto grad_wigner = torch::zeros({E, 9, 9}, grad_out.options());

    const float* g_ptr = grad_out.data_ptr<float>();
    const float* x_ptr = x_m.data_ptr<float>();
    float* gw_ptr = grad_wigner.data_ptr<float>();

    const int64_t stride = 9 * C;

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; e++) {
        const float* G = g_ptr + e * stride;
        const float* X = x_ptr + e * stride;
        float* GW = gw_ptr + e * 81;

        // L=0
        {
            const float* xl0 = X + M_TO_L[0] * C;
            float dot = 0.0f;
            for (int64_t c = 0; c < C; c++) {
                dot += G[0 * C + c] * xl0[c];
            }
            GW[0] = dot;
        }

        // L=1
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const float* xl = X + M_TO_L[1 + j] * C;
                float dot = 0.0f;
                for (int64_t c = 0; c < C; c++) {
                    dot += G[(1 + i) * C + c] * xl[c];
                }
                GW[(1 + i) * 9 + (1 + j)] = dot;
            }
        }

        // L=2
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                const float* xl = X + M_TO_L[4 + j] * C;
                float dot = 0.0f;
                for (int64_t c = 0; c < C; c++) {
                    dot += G[(4 + i) * C + c] * xl[c];
                }
                GW[(4 + i) * 9 + (4 + j)] = dot;
            }
        }
    }

    return grad_wigner;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("node_to_edge_wigner_permute_fwd", &node_to_edge_wigner_permute_fwd,
          "Fused gather + Wigner rotate + L->M permute (forward)");
    m.def("node_to_edge_wigner_permute_bwd_dx", &node_to_edge_wigner_permute_bwd_dx,
          "Backward dx for node_to_edge_wigner_permute");
    m.def("node_to_edge_wigner_permute_bwd_dw", &node_to_edge_wigner_permute_bwd_dw,
          "Backward dw for node_to_edge_wigner_permute");
    m.def("permute_wigner_inv_fwd", &permute_wigner_inv_fwd,
          "Fused M->L permute + Wigner inverse (forward)");
    m.def("permute_wigner_inv_bwd_dx", &permute_wigner_inv_bwd_dx,
          "Backward dx for permute_wigner_inv");
    m.def("permute_wigner_inv_bwd_dw", &permute_wigner_inv_bwd_dw,
          "Backward dw for permute_wigner_inv");
}
