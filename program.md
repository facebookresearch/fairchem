Autonomous research loop for improving fairchem UMA-S inference throughput on CPU.

## What is fairchem

FAIRChem is Meta FAIR Chemistry's ML framework for atomistic simulations. The core model is **UMA** (Universal Model for Atoms) — a graph neural network that takes atomic positions as input and predicts energy and forces. It uses an SO(3)-equivariant backbone (eSCN-MD) with Wigner rotation matrices for message passing on a neighbor graph.

**How inference works**: The model predicts total energy E in the forward pass. Forces are computed as **forces = -dE/dpos** via `torch.autograd` backward pass — there is no separate force prediction head. This means BOTH forward AND backward kernels are on the critical path for every single inference call. Any optimization must preserve the correctness of both passes.

The model has multiple **execution backends**. The default `general` backend uses standard PyTorch ops. The `umas_fast_gpu` backend replaces key operations with custom Triton CUDA kernels for GPU throughput. **We are building and optimizing the `umas_fast_cpu` backend** — a new backend that replaces key operations with custom C/C++ kernels for CPU throughput. The model weights, the backbone architecture, the graph generation, and all other framework code are frozen. Only the kernel implementations and backend dispatch code can be modified.

## Goal

**Maximize MD QPS** (molecular dynamics queries per second) for the `umas_fast_cpu` execution backend on a 100-atom aperiodic (no PBC) carbon FCC system running Langevin dynamics at 400K.

Forces must match the gold-standard reference (generated once from the `general` backend without compile) within tolerance — correctness is a hard gate.

## System hardware

- **CPU**: Intel Xeon Platinum 8358 @ 2.60GHz
- **Cores**: 16 physical cores (no hyperthreading)
- **SIMD**: AVX-512 (avx512f, avx512dq, avx512cd, avx512bw, avx512vl, avx512_vnni)
- **Cache**: L1d 512 KiB (16 instances), L2 64 MiB (16 instances)
- **All 16 cores are available for use** — use OpenMP, torch threading, or manual parallelism

## Environment setup

Every command must be run with the fairchem venv activated and PYTHONPATH set:

```bash
source ~/fairchem_venv/bin/activate
export PYTHONPATH=/home/ubuntu/fairchem/src:$PYTHONPATH
export HF_TOKEN=<your_hf_token>
cd /home/ubuntu/fairchem/configs/uma/speed
```

Package management uses `uv pip install` (inside fairchem_venv). The fairchem source tree at `/home/ubuntu/fairchem/src` is on PYTHONPATH and takes precedence over site-packages.

## Branch

All work happens on the **`cpu_backend_autoresearch`** branch in `/home/ubuntu/fairchem`.

## Gold standard reference

The gold standard is a pickle file `configs/uma/speed/gold_forces.pkl` containing forces and energy from the `general` backend (no compile) on the canonical 100-atom system. It already exists. To regenerate:

```bash
python compare_forces.py --generate --device cpu
```

This file is read-only after generation. All backends are compared against it.

## Current baseline

| Metric | Value |
|--------|-------|
| Force correctness | PASS (energy err: 1.3e-6 eV, forces max err: 9.5e-7) |
| Static QPS | 2.68 |
| MD QPS | 2.07 |
| System | 100 atoms, non-PBC FCC carbon |
| Backend | umas_fast_cpu (pure PyTorch, no C++ kernels yet) |

## Architecture overview

The UMA model uses an eSCN-MD backbone that processes atomic systems through:

1. **Graph generation**: Build neighbor list from atomic positions (internal_graph_gen_version=2)
2. **Edge embeddings**: Compute radial basis functions and Wigner rotation matrices
3. **Message passing layers** (N blocks): Each block does:
   - `node_to_edge_wigner_permute`: Gather node features, rotate L-order -> M-order via Wigner matrices
   - `so2_conv_1`: SO(2) convolution with radial function (block-diagonal GEMM in fast backends)
   - `so2_conv_2`: Second SO(2) convolution
   - `permute_wigner_inv_edge_to_node`: Rotate M-order -> L-order and scatter back to nodes
   - `edge_degree_scatter`: Radial embedding scattered to nodes via Wigner inverse
4. **Heads**: Energy prediction from node features, forces via autograd backward

### Backend class hierarchy

```
ExecutionBackend (general — pure PyTorch reference)
  └── UMASFastPytorchBackend (block-diagonal SO2 conv + unified radial MLP)
        ├── UMASFastGPUBackend (Triton CUDA kernels)
        └── UMASFastCPUBackend (C/C++ CPU kernels)  ← THIS IS WHAT WE OPTIMIZE
```

The `umas_fast_cpu` backend inherits from `UMASFastPytorchBackend` (getting SO2 block-diagonal conversion and unified radial MLP for free), then overrides the three hot operations:

- `node_to_edge_wigner_permute`: Gather + block-diagonal Wigner rotation + L→M permute
- `permute_wigner_inv_edge_to_node`: M→L permute + Wigner inverse rotation
- `edge_degree_scatter`: m=0 column select + BMM + scatter
- `prepare_wigner`: Passthrough (kernels handle L-to-M internally, same as GPU backend)

### The math in detail

The Wigner rotation matrix is **block-diagonal** for lmax=2:
- L=0: 1×1 block (scalar multiply)
- L=1: 3×3 block
- L=2: 5×5 block
- Only 35 of 81 elements are nonzero

L-major to M-major permutation indices: `[0, 2, 6, 3, 7, 1, 5, 8, 4]`
M-major to L-major permutation indices: `[0, 5, 1, 3, 8, 6, 2, 4, 7]`

For `node_to_edge_wigner_permute` forward:
1. Gather x[src] and x[tgt] from node features [N, 9, C] using edge_index
2. Concatenate to [E, 9, 2C]
3. Block-diagonal Wigner multiply in L-major order
4. Permute L→M to get output [E, 9, 2C]

For `permute_wigner_inv_edge_to_node` forward:
1. Permute M→L on input [E, 9, C]
2. Block-diagonal Wigner inverse multiply
3. Output [E, 9, C] in L-major order (then scatter to nodes via index_add)

## In-scope files (YOU MODIFY THESE)

All paths relative to `/home/ubuntu/fairchem/src/fairchem/core/models/uma/`:

1. `nn/execution_backends.py` — Backend dispatch: `UMASFastCPUBackend` + `UMAS_FAST_CPU` enum
2. `cpu/__init__.py` — CPU kernel module exports
3. `cpu/ops.py` — Python wrappers + `torch.autograd.Function` for CPU kernels
4. `cpu/kernels.cpp` — C/C++ CPU kernels (to be created, compiled via `torch.utils.cpp_extension`)

## C/C++ kernel compilation

You can write C/C++ code and compile it as a PyTorch extension. Use JIT compilation:

```python
from torch.utils.cpp_extension import load

cpu_kernels = load(
    name="uma_cpu_kernels",
    sources=["path/to/kernels.cpp"],
    extra_cflags=["-O3", "-fopenmp", "-mavx512f", "-mavx512dq"],
    extra_ldflags=["-lgomp"],
    verbose=True,
)
```

Or use `CppExtension` in setup for ahead-of-time compilation. The C++ code can:
- Use `#include <torch/extension.h>` for ATen tensor access
- Use `#pragma omp parallel for` for multi-threaded loops over edges/nodes (16 cores available!)
- Use AVX-512 intrinsics (`#include <immintrin.h>`) for SIMD vectorization
- Access tensor data directly via `.data_ptr<float>()` for zero-overhead raw pointer math
- Fuse multiple operations into a single pass over the data (e.g., gather + Wigner rotate + permute in one loop)
- Use thread-local accumulators for scatter operations to avoid contention

**Key fusion opportunities in C++:**
- Fuse gather + Wigner rotate + L→M permute into a single kernel (avoids materializing [E,9,C] intermediate)
- Fuse M→L permute + Wigner inverse into a single kernel
- Fuse Wigner inverse + scatter with thread-local node accumulators + reduction
- Hardcode the block-diagonal structure (skip zero multiply for 46 of 81 elements)
- For backward: fuse grad scatter + W^T multiply, fuse outer product for dW

After changing C++ code, clear the JIT cache: `rm -rf /tmp/torch_extensions/*`

## Read-only context (DO NOT MODIFY)

- `nn/so2_layers.py` — SO2 convolution layers (converted by fast backends)
- `nn/unified_radial.py` — Unified radial MLP for batched computation
- `/home/ubuntu/fairchem/src/fairchem/core/models/uma/escn_md.py` — The backbone that calls the backend methods
- `/home/ubuntu/fairchem/configs/uma/speed/bench_common.py` — Benchmark utilities
- `/home/ubuntu/fairchem/configs/uma/speed/compare_forces.py` — Force correctness validation against gold PKL
- `/home/ubuntu/fairchem/configs/uma/speed/check_static_qps.py` — Static QPS benchmark (predict unit directly)
- `/home/ubuntu/fairchem/configs/uma/speed/check_md_qps.py` — MD QPS benchmark (Langevin dynamics)
- `/home/ubuntu/fairchem/configs/uma/speed/gold_forces.pkl` — Gold-standard forces/energy from general backend
- `triton/` — GPU-specific Triton kernels (reference for the math, do not use on CPU)

## Experimentation

Each experiment modifies one or more in-scope files, then evaluates. The evaluation has two phases:

### Phase 1: Correctness gate

```bash
python compare_forces.py --backend umas_fast_cpu --device cpu 2>&1 | tail -10
```

This MUST print `PASS`. If it prints `FAIL`, the modification broke numerical correctness and must be fixed or discarded. The tolerances are: forces atol/rtol=5e-3, energy atol=50meV rtol=1e-4.

### Phase 2: MD QPS measurement

```bash
python check_md_qps.py --backend umas_fast_cpu --device cpu --warmup 5 --steps 20 2>&1 | grep "INFO"
```

Use 5 warmup + 20 measured steps for experiments (faster iteration). The key metric is `MD QPS` from the output.

For final measurements, use more steps:
```bash
python check_md_qps.py --backend umas_fast_cpu --device cpu --warmup 5 --steps 50 2>&1 | grep "INFO"
```

### Phase 3 (optional): Static QPS for reference

```bash
python check_static_qps.py --backend umas_fast_cpu --device cpu --warmup 5 --iters 20 2>&1 | grep "INFO"
```

## Important notes

- After modifying source files, clear `__pycache__`: `rm -rf /home/ubuntu/fairchem/src/fairchem/core/models/uma/cpu/__pycache__ /home/ubuntu/fairchem/src/fairchem/core/models/uma/nn/__pycache__`
- After changing C++ code, clear JIT extension cache: `rm -rf /tmp/torch_extensions/*`
- The model has lmax=2, mmax=2, sphere_channels=512 (for UMA-S-1p2).
- C++ kernels should handle any sphere_channels value.
- Always use `merge_mole=True` and `activation_checkpointing=False`.
- Use `PYTHONPATH=/home/ubuntu/fairchem/src:$PYTHONPATH` for all python commands.

## What you CAN do

- Modify any in-scope file. Write C/C++ kernels, optimize memory layout, add OpenMP parallelism.
- Write C/C++ extension code compiled via `torch.utils.cpp_extension.load()` or `CppExtension`.
- **Fuse kernels in C/C++**: combine gather + rotate + permute into a single loop, combine permute + inverse rotate + scatter, etc. Fusion eliminates intermediate allocations and improves cache utilization.
- Use all 16 CPU cores via OpenMP (`#pragma omp parallel for`) or torch threading.
- Use AVX-512 SIMD intrinsics for vectorized inner loops.
- Use BLAS (OpenBLAS, MKL via PyTorch's ATen) for matrix operations.
- Exploit the block-diagonal Wigner structure (only 35 of 81 elements nonzero).
- Pre-allocate and reuse buffers across inference calls.
- Use `torch.compile` with CPU backend (AOTInductor).
- Add cache-friendly memory access patterns (SoA vs AoS, tiling).
- Install packages via `uv pip install` in fairchem_venv if needed.

## What you CANNOT do

- Modify read-only context files or the benchmark scripts.
- Change the model weights, architecture, or training.
- Change the benchmark system (100 atoms, no PBC, carbon FCC).
- Change merge_mole (must be True) or activation_checkpointing (must be False).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and columns:

```
commit	md_qps	static_qps	forces_pass	status	description
```

1. git commit hash (short, 7 chars)
2. MD QPS (e.g. 2.07) — use 0.0 for crashes
3. Static QPS (e.g. 2.68) — use 0.0 if not measured
4. forces_pass: `yes` or `no`
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	md_qps	static_qps	forces_pass	status	description
5c488c9	2.07	2.68	yes	keep	baseline (pure PyTorch CPU ops)
b2c3d4e	3.50	4.10	yes	keep	C++ fused gather+wigner with OpenMP 16 threads
c3d4e5f	5.20	6.00	yes	keep	AVX-512 inner loop + fused scatter
d4e5f6g	0.00	0.00	no	crash	bad kernel config
```

## The experiment loop

The experiment runs on the **`cpu_backend_autoresearch`** branch in `/home/ubuntu/fairchem`.

LOOP FOREVER:

1. Look at the current state: git log, current QPS baseline, which ideas have been tried.
2. Pick an optimization idea. Modify one or more in-scope files.
3. Clear `__pycache__` and JIT extension cache.
4. `cd /home/ubuntu/fairchem && git add` the changed source files and `git commit` with a descriptive message.
5. Run Phase 1 (force check). If FAIL → fix or revert.
6. Run Phase 2 (MD QPS). Record the result.
7. Optionally run Phase 3 (static QPS).
8. Log results to `configs/uma/speed/results.tsv` (do NOT commit results.tsv).
9. If MD QPS improved → keep the commit (advance the branch).
10. If MD QPS is same or worse → `cd /home/ubuntu/fairchem && git reset --hard HEAD~1` to discard.

**Crashes**: If a run crashes (bad kernel config, compilation error, etc.), use judgment: fix if trivial, otherwise discard and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the kernel code, look for fusion opportunities, try different threading configs, reduce memory allocations. The loop runs until the human interrupts you.

## Optimization ideas to consider

These are starting points, not an exhaustive list:

### Phase A: C/C++ kernel implementation (biggest wins)

1. **Fused gather+Wigner+permute C++ kernel**: Write `cpu/kernels.cpp` that fuses index gather + block-diagonal 9×9 matrix multiply + L→M permute into one OpenMP-parallel loop over edges. Each thread processes a chunk of edges, reading node features directly via pointer arithmetic, applying the 35 nonzero Wigner elements, and writing the permuted output. Avoids materializing the [E,9,C] gather intermediate entirely.

2. **Fused permute+inverse Wigner C++ kernel**: Single-pass M→L permute + block-diagonal W_inv multiply. For the full `permute_wigner_inv_edge_to_node`, fuse with scatter using thread-local accumulators: each thread accumulates into its own [N,9,C] buffer, then reduce across threads at the end.

3. **Fused edge_degree_scatter C++ kernel**: m=0 column select + small BMM (9×3 @ 3×C) + scatter with rescale, all in one loop.

4. **Custom backward C++ kernels**: The backward pass is equally important (forces = autograd backward). Write C++ backward kernels that fuse W^T @ grad + permute + scatter for grad_x, and fuse outer-product + block-diagonal mask for grad_wigner.

### Phase B: Vectorization and threading

5. **AVX-512 inner loops**: The inner loop multiplies a [9×9] block-diagonal matrix by C channels. With sphere_channels=512 and AVX-512 (16 floats/register), that's 32 vector ops per coefficient. Hand-write the inner loop with `_mm512_fmadd_ps`.

6. **OpenMP scheduling**: Try `schedule(static)`, `schedule(dynamic,64)`, `schedule(guided)`. For scatter operations, try `schedule(static)` with thread-local accumulators.

7. **Thread count tuning**: Set `OMP_NUM_THREADS=16` (all cores), or try 8 (fewer threads, less contention). Set `torch.set_num_threads()` for PyTorch ops.

### Phase C: Memory and allocation

8. **Pre-allocate output buffers**: Reuse [E,9,2C] and [E,9,C] output tensors across calls instead of `torch.empty` each time.

9. **Memory layout**: Experiment with [9,E,C] layout (coefficient-major) vs [E,9,C] (edge-major). Coefficient-major may allow better vectorization of the Wigner multiply.

10. **torch.compile on CPU**: Try AOTInductor for the non-kernel parts of the model. May auto-fuse some PyTorch ops.

### Phase D: Profiling

11. **Profile first**: Use `torch.profiler` or `py-spy` to identify the actual bottleneck before optimizing. The graph generation or SO2 conv may dominate, not the Wigner ops.

12. **torch_num_threads tuning**: The InferenceSettings has a `torch_num_threads` field. Experiment with different values for the PyTorch-internal ops (linear layers, etc.).
