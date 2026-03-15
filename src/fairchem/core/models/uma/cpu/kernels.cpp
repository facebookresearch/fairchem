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
        //   l = L_TO_M[m]  (gather: out[m] reads from rotated[L_TO_M[m]])
        //   out[m,:] = sum_j W[l,j] * x_cat[j,:]
        // Unroll by block, use precomputed l = L_TO_M[m]

        // m=0: l=L_TO_M[0]=0, L=0 block
        {
            const int l = L_TO_M[0];  // =0
            const float w00 = W[0];
            float* dst = O;
            for (int64_t c = 0; c < C; c++) {
                dst[c]     = w00 * xs[c];
                dst[C + c] = w00 * xt[c];
            }
        }

        // Output positions where L_TO_M[m] falls in L=1 block (l=1,2,3)
        for (int m = 0; m < 9; m++) {
            const int l = L_TO_M[m];
            if (l < 1 || l > 3) continue;
            float* dst = O + m * C2;
            const float w0 = W[l * 9 + 1];
            const float w1 = W[l * 9 + 2];
            const float w2 = W[l * 9 + 3];
            const float* xs1 = xs + 1 * C;
            const float* xs2 = xs + 2 * C;
            const float* xs3 = xs + 3 * C;
            const float* xt1 = xt + 1 * C;
            const float* xt2 = xt + 2 * C;
            const float* xt3 = xt + 3 * C;
            for (int64_t c = 0; c < C; c++) {
                dst[c]     = w0 * xs1[c] + w1 * xs2[c] + w2 * xs3[c];
                dst[C + c] = w0 * xt1[c] + w1 * xt2[c] + w2 * xt3[c];
            }
        }

        // Output positions where L_TO_M[m] falls in L=2 block (l=4..8)
        for (int m = 0; m < 9; m++) {
            const int l = L_TO_M[m];
            if (l < 4) continue;
            float* dst = O + m * C2;
            float ww[5];
            for (int j = 0; j < 5; j++) {
                ww[j] = W[l * 9 + (4 + j)];
            }
            for (int64_t c = 0; c < C; c++) {
                float vs = 0.0f, vt = 0.0f;
                for (int j = 0; j < 5; j++) {
                    vs += ww[j] * xs[(4 + j) * C + c];
                    vt += ww[j] * xt[(4 + j) * C + c];
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
        // Precompute x_l pointers for better vectorization
        const float* xl[9];
        for (int j = 0; j < 9; j++) {
            xl[j] = X + M_TO_L[j] * C;
        }

        // L=0
        {
            const float w00 = W[0];
            float* dst = O;
            for (int64_t c = 0; c < C; c++) {
                dst[c] = w00 * xl[0][c];
            }
        }

        // L=1
        for (int i = 0; i < 3; i++) {
            float* dst = O + (1 + i) * C;
            const float w0 = W[(1 + i) * 9 + 1];
            const float w1 = W[(1 + i) * 9 + 2];
            const float w2 = W[(1 + i) * 9 + 3];
            for (int64_t c = 0; c < C; c++) {
                dst[c] = w0 * xl[1][c] + w1 * xl[2][c] + w2 * xl[3][c];
            }
        }

        // L=2
        for (int i = 0; i < 5; i++) {
            float* dst = O + (4 + i) * C;
            float ww[5];
            for (int j = 0; j < 5; j++) {
                ww[j] = W[(4 + i) * 9 + (4 + j)];
            }
            for (int64_t c = 0; c < C; c++) {
                float v = 0.0f;
                for (int j = 0; j < 5; j++) {
                    v += ww[j] * xl[4 + j][c];
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

        // grad_x_l[j] = sum_i W^T[j,i] * G[i] = sum_i W[i,j] * G[i]
        // Then L→M permute: GX[m] = grad_x_l[L_TO_M[m]]
        // Iterate over output M-positions, gather from grad_x_l

        // For each output m, l = L_TO_M[m], compute grad_x_l[l] and write to GX[m]
        for (int m = 0; m < 9; m++) {
            const int l = L_TO_M[m];
            const int bs = BLOCK_START[l];
            const int bn = BLOCK_SIZE[l];
            float* dst = GX + m * C;

            // Preload W^T column weights: W[bs+i, l] for i in 0..bn-1
            float wt[5];
            for (int i = 0; i < bn; i++) {
                wt[i] = W[(bs + i) * 9 + l];
            }

            // Vectorizable inner loop over channels
            for (int64_t c = 0; c < C; c++) {
                float v = 0.0f;
                for (int i = 0; i < bn; i++) {
                    v += wt[i] * G[(bs + i) * C + c];
                }
                dst[c] = v;
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

/*
 * Fused: permute_wigner_inv + scatter to nodes
 *
 * Combines M→L permute + Wigner inverse rotation + index_add scatter
 * into a single pass, avoiding materializing the [E,9,C] intermediate.
 * Uses thread-local accumulators for the scatter.
 */
torch::Tensor permute_wigner_inv_scatter(
    const torch::Tensor& x,           // [E, 9, C] M-major
    const torch::Tensor& wigner_inv,   // [E, 9, 9]
    const torch::Tensor& edge_index,   // [2, E] (we use row 1 = tgt)
    int64_t num_nodes,
    int64_t node_offset
) {
    const int64_t E = x.size(0);
    const int64_t C = x.size(2);
    const int num_threads = omp_get_max_threads();

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = wigner_inv.data_ptr<float>();
    const int64_t* ei_ptr = edge_index.data_ptr<int64_t>();

    const int64_t stride = 9 * C;
    const int64_t out_size = num_nodes * 9 * C;

    // Thread-local accumulators
    std::vector<std::vector<float>> local_bufs(num_threads);
    for (auto& buf : local_bufs) {
        buf.resize(out_size, 0.0f);
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* local = local_bufs[tid].data();

        #pragma omp for schedule(static)
        for (int64_t e = 0; e < E; e++) {
            const float* X = x_ptr + e * stride;
            const float* W = w_ptr + e * 81;
            const int64_t tgt = ei_ptr[E + e] - node_offset;
            float* dst_base = local + tgt * stride;

            // Precompute x_l pointers
            const float* xl[9];
            for (int j = 0; j < 9; j++) {
                xl[j] = X + M_TO_L[j] * C;
            }

            // L=0
            {
                const float w00 = W[0];
                float* dst = dst_base;
                for (int64_t c = 0; c < C; c++) {
                    dst[c] += w00 * xl[0][c];
                }
            }

            // L=1
            for (int i = 0; i < 3; i++) {
                float* dst = dst_base + (1 + i) * C;
                const float w0 = W[(1+i)*9+1], w1 = W[(1+i)*9+2], w2 = W[(1+i)*9+3];
                for (int64_t c = 0; c < C; c++) {
                    dst[c] += w0 * xl[1][c] + w1 * xl[2][c] + w2 * xl[3][c];
                }
            }

            // L=2
            for (int i = 0; i < 5; i++) {
                float* dst = dst_base + (4 + i) * C;
                float ww[5];
                for (int j = 0; j < 5; j++) ww[j] = W[(4+i)*9+(4+j)];
                for (int64_t c = 0; c < C; c++) {
                    float v = 0.0f;
                    for (int j = 0; j < 5; j++) v += ww[j] * xl[4+j][c];
                    dst[c] += v;
                }
            }
        }
    }

    // Reduce
    auto result = torch::zeros({num_nodes, 9, C}, x.options());
    float* r_ptr = result.data_ptr<float>();
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < out_size; i++) {
        float sum = 0.0f;
        for (int t = 0; t < num_threads; t++) sum += local_bufs[t][i];
        r_ptr[i] = sum;
    }

    return result;
}

/*
 * Combined backward: node_to_edge_wigner_permute bwd_dx + bwd_dw
 *
 * Single pass over edges computing both grad_x (with scatter) and grad_wigner.
 * Avoids redundant loading of grad_out and Wigner matrices.
 */
std::vector<torch::Tensor> node_to_edge_wigner_permute_bwd_combined(
    const torch::Tensor& grad_out,    // [E, 9, 2C] M-major
    const torch::Tensor& wigner,      // [E, 9, 9]
    const torch::Tensor& x,           // [N, 9, C]
    const torch::Tensor& edge_index,  // [2, E]
    int64_t num_nodes
) {
    const int64_t E = edge_index.size(1);
    const int64_t C2 = grad_out.size(2);
    const int64_t C = C2 / 2;
    const int num_threads = omp_get_max_threads();

    auto grad_wigner = torch::zeros({E, 9, 9}, grad_out.options());

    const float* g_ptr = grad_out.data_ptr<float>();
    const float* w_ptr = wigner.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const int64_t* ei_ptr = edge_index.data_ptr<int64_t>();
    float* gw_ptr = grad_wigner.data_ptr<float>();

    const int64_t g_stride_e = 9 * C2;
    const int64_t x_stride = 9 * C;
    const int64_t gx_stride_n = 9 * C;

    // Thread-local buffers for grad_x scatter
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
            const float* xs = x_ptr + src_node * x_stride;
            const float* xt = x_ptr + tgt_node * x_stride;
            float* GW = gw_ptr + e * 81;

            // Precompute M→L permuted grad pointers (shared by dx and dw)
            const float* gl[9];
            for (int li = 0; li < 9; li++) {
                gl[li] = G + M_TO_L[li] * C2;
            }

            float* ds_base = local + src_node * gx_stride_n;
            float* dt_base = local + tgt_node * gx_stride_n;

            // === L=0 block ===
            {
                const float w_val = W[0];
                // bwd_dx
                float* ds = ds_base;
                float* dt = dt_base;
                for (int64_t c = 0; c < C; c++) {
                    ds[c] += w_val * gl[0][c];
                    dt[c] += w_val * gl[0][C + c];
                }
                // bwd_dw
                float dot = 0.0f;
                for (int64_t c = 0; c < C; c++) {
                    dot += gl[0][c] * xs[c] + gl[0][C + c] * xt[c];
                }
                GW[0] = dot;
            }

            // === L=1 block ===
            // bwd_dx
            for (int j = 0; j < 3; j++) {
                float* ds = ds_base + (1 + j) * C;
                float* dt = dt_base + (1 + j) * C;
                const float w0 = W[1 * 9 + (1 + j)];
                const float w1 = W[2 * 9 + (1 + j)];
                const float w2 = W[3 * 9 + (1 + j)];
                for (int64_t c = 0; c < C; c++) {
                    const float vs = w0 * gl[1][c] + w1 * gl[2][c] + w2 * gl[3][c];
                    const float vt = w0 * gl[1][C+c] + w1 * gl[2][C+c] + w2 * gl[3][C+c];
                    ds[c] += vs;
                    dt[c] += vt;
                }
            }
            // bwd_dw
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    float dot = 0.0f;
                    const float* xsj = xs + (1 + j) * C;
                    const float* xtj = xt + (1 + j) * C;
                    for (int64_t c = 0; c < C; c++) {
                        dot += gl[1 + i][c] * xsj[c] + gl[1 + i][C + c] * xtj[c];
                    }
                    GW[(1 + i) * 9 + (1 + j)] = dot;
                }
            }

            // === L=2 block ===
            // bwd_dx
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
            // bwd_dw
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    float dot = 0.0f;
                    const float* xsj = xs + (4 + j) * C;
                    const float* xtj = xt + (4 + j) * C;
                    for (int64_t c = 0; c < C; c++) {
                        dot += gl[4 + i][c] * xsj[c] + gl[4 + i][C + c] * xtj[c];
                    }
                    GW[(4 + i) * 9 + (4 + j)] = dot;
                }
            }
        }
    }

    // Reduce thread-local buffers for grad_x
    auto grad_x = torch::zeros({num_nodes, 9, C}, grad_out.options());
    float* gx_ptr = grad_x.data_ptr<float>();
    const int64_t total = num_nodes * 9 * C;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total; i++) {
        float sum = 0.0f;
        for (int t = 0; t < num_threads; t++) {
            sum += local_bufs[t][i];
        }
        gx_ptr[i] = sum;
    }

    return {grad_x, grad_wigner};
}

// Forward declaration
torch::Tensor fused_edgewise_inner(
    const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&,
    int64_t, const std::vector<int64_t>&, const std::vector<int64_t>&);
std::vector<torch::Tensor> fused_so2_conv1_forward(
    const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    int64_t, const std::vector<int64_t>&, const std::vector<int64_t>&);

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
    m.def("permute_wigner_inv_scatter", &permute_wigner_inv_scatter,
          "Fused M->L permute + Wigner inverse + scatter to nodes");
    m.def("node_to_edge_wigner_permute_bwd_combined", &node_to_edge_wigner_permute_bwd_combined,
          "Combined bwd_dx + bwd_dw for node_to_edge_wigner_permute");
    m.def("fused_edgewise_inner", &fused_edgewise_inner, "Fused edgewise inner loop");
    m.def("fused_so2_conv1_forward", &fused_so2_conv1_forward,
          "Fused SO2 conv1 forward");
}

/*
 * Fused SO2 conv1 forward: radial multiply + GEMM + output assembly
 * for all m-orders in a single call.
 *
 * Eliminates: x.split(), x.view(), x*radial.unsqueeze(), x.flatten(),
 *             torch.cat(out), and the Python dispatch for each.
 *
 * Input: x [E, 9, 2C], x_edge [E, total_radial_features],
 *        fc_m0_weight [m0_out, m0_in], fc_m0_bias [m0_out],
 *        w_block_1 [2*out1, 2*in1], w_block_2 [2*out2, 2*in2]
 * Output: out [E, 9, C_out], gating [E, extra_m0]
 */
std::vector<torch::Tensor> fused_so2_conv1_forward(
    const torch::Tensor& x,           // [E, 9, 2C]
    const torch::Tensor& x_edge,      // [E, total_radial]
    const torch::Tensor& fc_m0_weight, // [m0_out, m0_in]
    const torch::Tensor& fc_m0_bias,   // [m0_out]
    const torch::Tensor& w_block_1,    // [2*out_half1, 2*in_size1]
    const torch::Tensor& w_block_2,    // [2*out_half2, 2*in_size2]
    int64_t extra_m0_channels,
    const std::vector<int64_t>& m_split_sizes,    // [3, 4, 2]
    const std::vector<int64_t>& edge_split_sizes   // [768, 512, 256]
) {
    const int64_t E = x.size(0);
    const int64_t C_in = x.size(2);

    // m=0: radial * flatten → addmm
    int64_t m0_coeffs = m_split_sizes[0];  // 3
    auto x_m0 = x.narrow(1, 0, m0_coeffs).reshape({E, -1});  // [E, 3*2C]
    auto edge_m0 = x_edge.narrow(1, 0, edge_split_sizes[0]);
    auto x_m0_scaled = x_m0 * edge_m0;
    auto m0_out_full = torch::addmm(fc_m0_bias, x_m0_scaled, fc_m0_weight.t());

    int64_t m0_main_size = fc_m0_weight.size(0) - extra_m0_channels;
    int64_t C_out = m0_main_size / m0_coeffs;
    auto gating = m0_out_full.narrow(1, 0, extra_m0_channels);
    auto m0_out = m0_out_full.narrow(1, extra_m0_channels, m0_main_size);

    // Pre-allocate output [E, 9, C_out]
    auto out = torch::empty({E, 9, C_out}, x.options());

    // Write m0 output
    out.narrow(1, 0, m0_coeffs).copy_(m0_out.view({E, m0_coeffs, C_out}));

    // m=1: radial * x → mm(w_block_1) → write real/imag
    int64_t m1_coeffs = m_split_sizes[1];  // 4
    int64_t m1_offset = m0_coeffs;
    auto x_m1 = x.narrow(1, m1_offset, m1_coeffs).reshape({E, 2, -1});
    int64_t edge_m1_offset = edge_split_sizes[0];
    auto edge_m1 = x_edge.narrow(1, edge_m1_offset, edge_split_sizes[1]);
    auto x_m1_scaled = (x_m1 * edge_m1.unsqueeze(1)).reshape({E, -1});
    auto m1_result = torch::mm(x_m1_scaled, w_block_1.t());
    int64_t num_l_1 = m1_result.size(1) / (2 * C_out);
    auto m1_view = m1_result.view({E, 2, num_l_1, C_out});
    int64_t out_offset = m0_coeffs;
    out.narrow(1, out_offset, num_l_1).copy_(m1_view.select(1, 0));
    out_offset += num_l_1;
    out.narrow(1, out_offset, num_l_1).copy_(m1_view.select(1, 1));
    out_offset += num_l_1;

    // m=2: radial * x → mm(w_block_2) → write real/imag
    int64_t m2_coeffs = m_split_sizes[2];  // 2
    int64_t m2_input_offset = m1_offset + m1_coeffs;
    auto x_m2 = x.narrow(1, m2_input_offset, m2_coeffs).reshape({E, 2, -1});
    int64_t edge_m2_offset = edge_m1_offset + edge_split_sizes[1];
    auto edge_m2 = x_edge.narrow(1, edge_m2_offset, edge_split_sizes[2]);
    auto x_m2_scaled = (x_m2 * edge_m2.unsqueeze(1)).reshape({E, -1});
    auto m2_result = torch::mm(x_m2_scaled, w_block_2.t());
    int64_t num_l_2 = m2_result.size(1) / (2 * C_out);
    auto m2_view = m2_result.view({E, 2, num_l_2, C_out});
    out.narrow(1, out_offset, num_l_2).copy_(m2_view.select(1, 0));
    out_offset += num_l_2;
    out.narrow(1, out_offset, num_l_2).copy_(m2_view.select(1, 1));

    return {out, gating};
}

/*
 * Fused SO2 conv1 + GateActivation + SO2 conv2 forward
 *
 * Eliminates ALL Python dispatch overhead for the edgewise inner loop.
 * Uses ATen C++ ops directly (at::mm, at::addmm, at::silu, etc.)
 * which still support autograd but avoid Python interpreter overhead.
 *
 * Pipeline: x_message → conv1 → gate_act → conv2 → out
 */
torch::Tensor fused_edgewise_inner(
    const torch::Tensor& x_message,     // [E, 9, 2C] from node_to_edge
    const torch::Tensor& x_edge,        // [E, total_radial]
    // Conv1 weights
    const torch::Tensor& c1_fc_m0_w,    // [m0_out, m0_in]
    const torch::Tensor& c1_fc_m0_b,    // [m0_out]
    const torch::Tensor& c1_w_block_1,  // [2*oh1, 2*is1]
    const torch::Tensor& c1_w_block_2,  // [2*oh2, 2*is2]
    // Conv2 weights
    const torch::Tensor& c2_fc_m0_w,    // [m0_out2, m0_in2]
    const torch::Tensor& c2_fc_m0_b,    // [m0_out2]
    const torch::Tensor& c2_w_block_1,  // [2*oh1, 2*is1]
    const torch::Tensor& c2_w_block_2,  // [2*oh2, 2*is2]
    // Activation
    const torch::Tensor& act_expand_index, // [8] index for gate expansion
    // Sizes
    int64_t extra_m0_channels,
    const std::vector<int64_t>& m_split_sizes,    // [3, 4, 2]
    const std::vector<int64_t>& edge_split_sizes  // [768, 512, 256]
) {
    const int64_t E = x_message.size(0);
    const int64_t C_in = x_message.size(2);

    // ========== SO2 Conv1 ==========
    // Split input by m-order
    auto x_m0 = x_message.narrow(1, 0, m_split_sizes[0]).reshape({E, -1});
    int64_t m1_off = m_split_sizes[0];
    auto x_m1 = x_message.narrow(1, m1_off, m_split_sizes[1]);
    int64_t m2_off = m1_off + m_split_sizes[1];
    auto x_m2 = x_message.narrow(1, m2_off, m_split_sizes[2]);

    // Split edge features
    auto e_m0 = x_edge.narrow(1, 0, edge_split_sizes[0]);
    auto e_m1 = x_edge.narrow(1, edge_split_sizes[0], edge_split_sizes[1]);
    auto e_m2 = x_edge.narrow(1, edge_split_sizes[0] + edge_split_sizes[1], edge_split_sizes[2]);

    // m=0: radial * flatten → addmm
    auto c1_m0_out = at::addmm(c1_fc_m0_b, x_m0 * e_m0, c1_fc_m0_w.t());
    auto gating = c1_m0_out.narrow(1, 0, extra_m0_channels);
    auto c1_m0_main = c1_m0_out.narrow(1, extra_m0_channels,
                                         c1_fc_m0_w.size(0) - extra_m0_channels);
    int64_t C_hidden = c1_m0_main.size(1) / m_split_sizes[0];

    // m=1: radial * x → mm(w_block)
    auto x_m1_v = x_m1.reshape({E, 2, -1});
    auto c1_m1_out = at::mm((x_m1_v * e_m1.unsqueeze(1)).reshape({E, -1}), c1_w_block_1.t());
    int64_t nl1 = c1_m1_out.size(1) / (2 * C_hidden);
    auto c1_m1_v = c1_m1_out.reshape({E, 2, nl1, C_hidden});

    // m=2: radial * x → mm(w_block)
    auto x_m2_v = x_m2.reshape({E, 2, -1});
    auto c1_m2_out = at::mm((x_m2_v * e_m2.unsqueeze(1)).reshape({E, -1}), c1_w_block_2.t());
    int64_t nl2 = c1_m2_out.size(1) / (2 * C_hidden);
    auto c1_m2_v = c1_m2_out.reshape({E, 2, nl2, C_hidden});

    // Assemble conv1 output [E, 9, C_hidden]
    auto conv1_out = at::cat({
        c1_m0_main.reshape({E, m_split_sizes[0], C_hidden}),
        c1_m1_v.select(1, 0),  // real
        c1_m1_v.select(1, 1),  // imag
        c1_m2_v.select(1, 0),  // real
        c1_m2_v.select(1, 1),  // imag
    }, 1);

    // ========== GateActivation ==========
    // gate_act(gating): sigmoid
    auto gate_scalars = at::sigmoid(gating).reshape({E, -1, C_hidden});
    // Expand gating via index_select
    gate_scalars = at::index_select(gate_scalars, 1, act_expand_index);
    // Split scalar (L=0) and vector parts
    auto act_scalar = conv1_out.narrow(1, 0, 1);
    auto act_vector = conv1_out.narrow(1, 1, conv1_out.size(1) - 1);
    // Apply activations
    act_scalar = at::silu(act_scalar);
    act_vector = act_vector * gate_scalars;
    auto activated = at::cat({act_scalar, act_vector}, 1);

    // ========== SO2 Conv2 ==========
    // Split activated by m-order (same m_split_sizes)
    auto a_m0 = activated.narrow(1, 0, m_split_sizes[0]).reshape({E, -1});
    auto a_m1 = activated.narrow(1, m1_off, m_split_sizes[1]);
    auto a_m2 = activated.narrow(1, m2_off, m_split_sizes[2]);

    // m=0: linear
    auto c2_m0_out = at::addmm(c2_fc_m0_b, a_m0, c2_fc_m0_w.t());
    int64_t C_out = c2_m0_out.size(1) / m_split_sizes[0];

    // m=1: block GEMM
    auto c2_m1_out = at::mm(a_m1.reshape({E, -1}), c2_w_block_1.t());
    int64_t c2_nl1 = c2_m1_out.size(1) / (2 * C_out);
    auto c2_m1_v = c2_m1_out.reshape({E, 2, c2_nl1, C_out});

    // m=2: block GEMM
    auto c2_m2_out = at::mm(a_m2.reshape({E, -1}), c2_w_block_2.t());
    int64_t c2_nl2 = c2_m2_out.size(1) / (2 * C_out);
    auto c2_m2_v = c2_m2_out.reshape({E, 2, c2_nl2, C_out});

    // Assemble output [E, 9, C_out]
    return at::cat({
        c2_m0_out.reshape({E, m_split_sizes[0], C_out}),
        c2_m1_v.select(1, 0),
        c2_m1_v.select(1, 1),
        c2_m2_v.select(1, 0),
        c2_m2_v.select(1, 1),
    }, 1);
}

/*
 * Fused radial-scaled GEMM: y = (x * radial) @ W.T + bias
 *
 * Fuses the elementwise multiply (x * radial) with the GEMM,
 * eliminating the intermediate tensor allocation.
 * Uses cblas_sgemm for the GEMM with radial scaling applied inline.
 *
 * For small-to-medium matrices, this saves one memory pass over the data.
 */
torch::Tensor fused_radial_addmm(
    const torch::Tensor& bias,    // [N_out]
    const torch::Tensor& x,       // [E, K]
    const torch::Tensor& radial,  // [E, K]
    const torch::Tensor& weight   // [N_out, K]
) {
    const int64_t E = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);

    // Fused: compute x * radial on-the-fly during GEMM
    // Unfortunately, BLAS doesn't support fused multiply-GEMM,
    // so we do the multiply into a pre-allocated buffer and then GEMM.
    // But we can reuse a thread-local buffer to avoid allocation.
    
    // For now, just do the multiply + addmm with minimal allocation
    auto scaled = x * radial;  // [E, K]
    return at::addmm(bias, scaled, weight.t());
}
