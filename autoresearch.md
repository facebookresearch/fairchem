# FAIRChem UMA-S Inference Speed Optimization Report

## Overview

Autonomous research sessions to maximize MD QPS (molecular dynamics queries per second) for the `umas_fast_gpu` execution backend on UMA-S-1p2 model, 2000-atom aperiodic carbon FCC system, A100-SXM4-40GB GPU.

**Results: 5.18 → 6.93 QPS (+33.8%), 21.4 → 17.96 GB memory (-16.1%)**

## Branches

| Branch | Purpose | QPS | Memory |
|--------|---------|-----|--------|
| `fix/permute-wigner-inv-bwd-dw-channels` | Bug fix: bwd_dw kernel dropping channels > 128 | 5.23 | 21.4 GB |
| `autoresearch/mar14` | All speed optimizations (rebased on fix) | 6.93 | 17.96 GB |
| `autoresearch/mar14b` | Earlier experiment branch (superseded by mar14) | 6.11 | 18.0 GB |

## Bug Fix: `permute_wigner_inv_edge_to_node_bwd_dw_kernel`

**File:** `src/fairchem/core/models/uma/triton/kernels.py`

The kernel had `tl.arange(0, 128)` hardcoded but `C` (sphere_channels) was 512 for UMA-S. This meant only 128 of 512 channels were processed in the gradient computation, producing ~48% of the correct `grad_wigner`. Forces still passed the 5e-3 tolerance because this gradient path is one of many contributing to forces.

**Fix:** Change `tl.arange(0, 128)` to `tl.arange(0, C)` (1-line fix). Added power-of-2 asserts on sphere_channels in kernel launchers, and a parametrized regression test at C=128, 256, 512.

**Test tolerances:** Tightened E2E force tolerances from rtol/atol=5e-2 to 2e-3, and added C=512 coverage to all kernel unit tests.

## Optimization Categories

### 1. Memory Elimination (largest impact: +18% QPS, -16% memory)

**Eliminate x_edge side buffer** (`fe8224adc`): The forward kernel stored `x_edge` [E, 9, 2C] (~1.8GB per layer) for backward grad_wigner computation. Instead, save `x` (node features, ~36MB) and recompute x_edge in backward via gather + cat. Removes 18 stores per channel block per edge from the forward kernel.

**Fused bwd_dw kernel** (`b3ce391d1`): Wrote a new Triton kernel that gathers from node features + permutes M→L + computes block-diagonal outer product in one pass. Eliminates the 1.8GB x_edge recomputation (cat) in backward entirely.

**Eliminate x_l side buffer** (`279279a81`): Modified `permute_wigner_inv_edge_to_node_bwd_dw_kernel` to load from M-major positions (using M_TO_L_GATHER_IDX offsets) instead of requiring a pre-permuted x_l buffer. Removes 9 stores per channel block per edge from the forward kernel.

**Avoid M→L permutation copy** (`7ebad3dcc`): In NodeToEdgeWignerPermute backward, replaced `_permute_m_to_l(grad_out)` (full [E,9,2C] copy) with direct fancy indexing of the needed M-major positions for each BMM block.

**Split bwd_dx output** (`38c3b6a11`): Changed the bwd_dx kernel to output src/tgt gradients as separate contiguous [E,9*C] buffers instead of interleaved [E,9,2C]. Eliminates 2x 900MB non-contiguous slice+reshape copies.

### 2. CUDA Allocator Configuration (+7% QPS)

**expandable_segments** (`0ae966d3a`): Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at module import time. Eliminates periodic 500-800ms GC stalls that occurred every 5-8 MD steps due to memory fragmentation. Average QPS jumped from 6.08 to 6.51 (now equals instantaneous QPS).

### 3. torch.compile Inductor Configuration (+5% QPS)

**coordinate_descent_tuning** (`16e9d3040`): Enables coordinate descent tuning for inductor-generated Triton kernels (the `triton_poi_fused_cat`, `triton_poi_fused_mul_view`, etc.). These auto-generated kernels take ~40% of CUDA time. Tuning their block sizes improved throughput by ~5%.

**aggressive_fusion** (`3d75fb33c`): Merges more operations into single fused kernels, reducing kernel launch count and memory traffic.

**size_asserts=False** (`df98e3a81`): Removes runtime shape assertions from generated code.

**automatic_dynamic_shapes=False** (`4dcef5732`): Disables dynamic shape guards, generating tighter code for static shapes.

### 4. Kernel Tuning (+0.5% QPS)

**BLOCK_C=256** (`d76ea2719`): Changed channel block size from 128 to 256, reducing grid size by 2x for forward/backward kernels.

**GRID_E_STRIDE=4096 for backward kernels** (`c6fd15fa8`): Increased thread blocks from 2048 to 4096 for all backward kernels. More parallelism better saturates the GPU during the backward pass.

### 5. Code Structure Improvements (neutral QPS, cleaner code)

**Custom autograd for edge_degree_scatter** (`cc5b4185f`): Replaced the chain of bmm→to→div→index_add with a single autograd.Function. Custom backward computes grad_radial and grad_wigner directly.

**Combined bwd dispatch** (`1e2db135a`): Merged bwd_dx and bwd_dw kernel launches into a single triton_op wrapper, reducing Python dispatch overhead.

**Reorder backward** (`ca6087bd6`): Launch bwd_dw before scatter to allow potential GPU overlap.

## Experiments That Did NOT Work

| Experiment | QPS | Why |
|-----------|-----|-----|
| num_warps=4 (all kernels) | 3.86 | Too many warps reduces per-warp work |
| GRID_E_STRIDE=4096 (all kernels) | 3.93 | Forward kernels need fewer threads |
| GRID_E_STRIDE=1024 | 3.94 | Too few threads, under-utilizes GPU |
| Broadcast multiply for edge_degree_scatter | 3.98 | More kernel launches than bmm |
| Atomic scatter fusion (bwd_dx) | 5.73 | Atomic contention at ~25 edges/node |
| Fused autograd for permute_wigner_inv+scatter | 6.10 | Autograd overhead is negligible |
| 3x rank-1 BMM in edge_degree_scatter | 6.01 | More kernel launches |
| Buffer caching (Python global) | 5.90 | Caused 5 more graph breaks |
| einsum instead of bmm | 6.09 | Same kernel under compile |
| Combined index_add scatter | 6.05 | Cat creates extra allocation |
| torch.empty for grad_wigner_flat | FAIL | Off-diagonal elements ARE read upstream |
| Fused rotation+scatter via atomic_add | 6.11 | Non-deterministic, fails gradcheck |
| num_stages=2 or 3 | 6.07 | Grid-stride loops don't benefit from pipelining |
| Triton autotune | 6.06 | Manual configs already optimal |
| max_autotune (Triton GEMM) | FAIL | Breaks energy precision |
| realize_opcount_threshold < 30 | FAIL | Too-aggressive fusion breaks precision |
| cudnn.benchmark | 6.46 | No cuDNN ops in this model |
| gc.disable() | 6.48 | Interferes with benchmark teardown |
| CUDA graph trees | CRASH | Incompatible with dynamic shapes (MD) |

## System Information

- GPU: NVIDIA A100-SXM4-40GB (SM 8.0)
- Model: UMA-S-1p2 (sphere_channels=512, lmax=2, mmax=2, 4 layers)
- System: 2000-atom aperiodic carbon FCC, Langevin dynamics at 400K
- PyTorch: ~2.8.0, Triton included
- Benchmark: 15 warmup + 200 measured MD steps
