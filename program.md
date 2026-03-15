Autonomous research loop for improving fairchem UMA-S inference throughput on CPU.

## What is fairchem

FAIRChem is Meta FAIR Chemistry's ML framework for atomistic simulations. The core model is **UMA** (Universal Model for Atoms) — a graph neural network that takes atomic positions as input and predicts energy and forces. It uses an SO(3)-equivariant backbone (eSCN-MD) with Wigner rotation matrices for message passing on a neighbor graph.

**How inference works**: The model predicts total energy E in the forward pass. Forces are computed as **forces = -dE/dpos** via `torch.autograd` backward pass — there is no separate force prediction head. This means BOTH forward AND backward kernels are on the critical path for every single inference call. Any optimization must preserve the correctness of both passes.

The model has multiple **execution backends**. The default `general` backend uses standard PyTorch ops. The `umas_fast_gpu` backend replaces key operations with custom Triton CUDA kernels for GPU throughput. **We are building and optimizing the `umas_fast_cpu` backend** — a new backend that replaces key operations with custom C/C++ kernels (compiled via `torch.utils.cpp_extension`) for CPU throughput. The model weights, the backbone architecture, the graph generation, and all other framework code are frozen. Only the kernel implementations and backend dispatch code can be modified.

## Goal

**Maximize MD QPS** (molecular dynamics queries per second) for the `umas_fast_cpu` execution backend on a 2000-atom aperiodic (no PBC) carbon FCC system running Langevin dynamics at 400K.

Forces must match the gold-standard reference (generated once from the `general` backend without compile) within tolerance — correctness is a hard gate.

## Gold standard reference

The gold standard is a pickle file `configs/uma/speed/gold_forces.pkl` containing forces and energy from the `general` backend (no compile) on the canonical 2000-atom system. Generate it once:

```bash
cd /home/ubuntu/fairchem/configs/uma/speed
source ~/fairchem_venv/bin/activate
export HF_TOKEN=hf_REDACTED
python compare_forces.py --generate --device cpu
```

This file is read-only after generation. All backends are compared against it.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15_cpu`). The branch `autoresearch/<tag>` must not already exist in the fairchem repo.
2. **Create the branch**: `cd /home/ubuntu/fairchem && git checkout -b autoresearch/<tag>` — all work happens in this repo since the in-scope source files are symlinked into site-packages.
3. **Read the in-scope files**: Read ALL files listed in the "In-scope files" section below.
4. **Read the context files**: Read ALL files listed in the "Read-only context" section.
5. **Generate gold standard** (if not already present):
   ```bash
   cd /home/ubuntu/fairchem/configs/uma/speed
   source ~/fairchem_venv/bin/activate
   export HF_TOKEN=hf_REDACTED
   python compare_forces.py --generate --device cpu
   ```
6. **Run baseline**: Run the evaluation WITHOUT any modifications to establish baseline numbers:
   ```bash
   python compare_forces.py --backend general --device cpu            # must print PASS
   python check_md_qps.py --backend general --device cpu --warmup 5 --steps 50 2>&1 | grep "INFO"
   ```
   The MD QPS from the `general` backend is the **baseline to beat**. Tolerances: forces atol/rtol=5e-3, energy atol=50meV rtol=1e-4.
7. **Initialize results.tsv**: Create `results.tsv` in `configs/uma/speed/` with the header row and the baseline entry.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

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
        └── UMASFastCPUBackend (C/C++ CPU kernels)  ← NEW
```

The `umas_fast_cpu` backend inherits from `UMASFastPytorchBackend` (getting SO2 block-diagonal conversion and unified radial MLP for free), then overrides the three hot operations with C/C++ kernels:

- `node_to_edge_wigner_permute`: Fused gather + block-diagonal Wigner rotation + L→M permute (OpenMP parallel over edges, BLAS for 9×9 BMM)
- `permute_wigner_inv_edge_to_node`: Fused M→L permute + Wigner inverse rotation (OpenMP, thread-local accumulators for scatter)
- `edge_degree_scatter`: m=0 column select + BMM + scatter (BLAS, OpenMP)
- `prepare_wigner`: Passthrough (C++ kernels handle L-to-M internally, same as GPU backend)

The C/C++ extension is JIT-compiled via `torch.utils.cpp_extension.load()` — no new packages required.

## In-scope files (YOU MODIFY THESE)

All paths relative to `/home/ubuntu/fairchem/src/fairchem/core/models/uma/`:

1. `nn/execution_backends.py` — Backend dispatch: add `UMASFastCPUBackend` + `UMAS_FAST_CPU` enum
2. `cpu/__init__.py` — CPU kernel module exports
3. `cpu/kernels.cpp` — C/C++ CPU kernels (forward + backward, compiled via torch cpp_extension)
4. `cpu/ops.py` — Python wrappers + `torch.autograd.Function` for CPU kernels

These files are symlinked from the source tree into site-packages, so edits take effect immediately (clear `__pycache__` if needed).

## Read-only context (DO NOT MODIFY)

- `nn/so2_layers.py` — SO2 convolution layers (converted by fast backends)
- `nn/unified_radial.py` — Unified radial MLP for batched computation
- `/home/ubuntu/fairchem/src/fairchem/core/models/uma/escn_md.py` — The backbone that calls the backend methods
- `/home/ubuntu/fairchem/configs/uma/speed/bench_common.py` — Benchmark utilities
- `/home/ubuntu/fairchem/configs/uma/speed/compare_forces.py` — Force correctness validation against gold PKL
- `/home/ubuntu/fairchem/configs/uma/speed/check_static_qps.py` — Static QPS benchmark (same system in a loop)
- `/home/ubuntu/fairchem/configs/uma/speed/check_md_qps.py` — MD QPS benchmark (Langevin dynamics)
- `/home/ubuntu/fairchem/configs/uma/speed/gold_forces.pkl` — Gold-standard forces/energy from general backend
- `/home/ubuntu/fairchem/tests/core/models/uma/uma_fast/test_execution_backends.py` — Existing tests
- `triton/` — GPU-specific Triton kernels (reference for the math, do not use on CPU)

## Experimentation

Each experiment modifies one or more in-scope files, then evaluates. The evaluation has two phases:

### Phase 1: Correctness gate

```bash
cd /home/ubuntu/fairchem/configs/uma/speed
source ~/fairchem_venv/bin/activate
export HF_TOKEN=hf_REDACTED
python compare_forces.py --backend umas_fast_cpu --device cpu 2>&1 | tail -10
```

This MUST print `PASS`. If it prints `FAIL`, the modification broke numerical correctness and must be fixed or discarded. The tolerances are: forces atol/rtol=5e-3, energy atol=50meV rtol=1e-4.

### Phase 2: MD QPS measurement

```bash
python check_md_qps.py --backend umas_fast_cpu --device cpu --warmup 5 --steps 50 2>&1 | grep "INFO"
```

Use 5 warmup + 50 measured steps for experiments (faster iteration). The key metric is `MD QPS` from the output.

For the baseline and final measurements, use the full 200 steps:
```bash
python check_md_qps.py --backend umas_fast_cpu --device cpu --warmup 5 --steps 200 2>&1 | grep "INFO"
```

### Phase 3 (optional): Static QPS for reference

```bash
python check_static_qps.py --backend umas_fast_cpu --device cpu --warmup 3 --iters 50 2>&1 | grep "INFO"
```

## Important notes

- After modifying source files, clear `__pycache__`: `rm -rf /home/ubuntu/fairchem/src/fairchem/core/models/uma/cpu/__pycache__ /home/ubuntu/fairchem/src/fairchem/core/models/uma/nn/__pycache__`
- Clear any JIT-compiled extension cache if C++ code changes: `rm -rf /tmp/torch_extensions/*`
- The model has lmax=2, mmax=2, sphere_channels=128 (for small models) or 512 (for UMA-S).
- The benchmark uses UMA-S-1p2 which has sphere_channels=512.
- C++ kernels should handle any sphere_channels value.
- Always use `merge_mole=True` and `activation_checkpointing=False`.

## What you CAN do

- Modify any in-scope file. Write C/C++ kernels, optimize memory layout, add OpenMP parallelism.
- Add new C/C++ kernels or fused operations.
- Change how the backend dispatches or prepares the model.
- Use BLAS (OpenBLAS, MKL) for matrix operations via PyTorch's ATen or direct calls.
- Use OpenMP for edge-parallel and node-parallel loops.
- Exploit the block-diagonal Wigner structure (1×1 + 3×3 + 5×5 for lmax=2).
- Pre-allocate and reuse buffers across inference calls.
- Use `torch.compile` with CPU backend (AOTInductor).
- Add cache-friendly memory access patterns (SoA vs AoS, tiling).

## What you CANNOT do

- Modify read-only context files or the benchmark scripts.
- Change the model weights, architecture, or training.
- Change the benchmark system (2000 atoms, no PBC, carbon FCC).
- Install new packages (torch.utils.cpp_extension is already available).
- Change merge_mole (must be True) or activation_checkpointing (must be False).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and columns:

```
commit	md_qps	static_qps	forces_pass	status	description
```

1. git commit hash (short, 7 chars)
2. MD QPS (e.g. 0.45) — use 0.0 for crashes
3. Static QPS (e.g. 0.52) — use 0.0 if not measured
4. forces_pass: `yes` or `no`
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	md_qps	static_qps	forces_pass	status	description
a1b2c3d	0.35	0.40	yes	keep	baseline (general backend on CPU)
b2c3d4e	0.42	0.48	yes	keep	umas_fast_cpu with pure PyTorch (inherited)
c3d4e5f	0.58	0.65	yes	keep	C++ fused gather+wigner with OpenMP
d4e5f6g	0.00	0.00	no	crash	bad kernel config
```

## The experiment loop

The experiment runs on a dedicated branch in `/home/ubuntu/fairchem` (e.g. `autoresearch/mar15_cpu`).

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

**Timeouts**:
- `compare_forces.py`: should complete in under 10 minutes on CPU. Kill after 30 minutes.
- `check_md_qps.py` (50 steps): should complete in under 30 minutes on CPU. Kill after 60 minutes.
- `check_md_qps.py` (200 steps): should complete in under 2 hours on CPU. Kill after 3 hours.
- `check_static_qps.py` (50 iters): should complete in under 30 minutes on CPU. Kill after 60 minutes.

**Crashes**: If a run crashes (bad kernel config, compilation error, etc.), use judgment: fix if trivial, otherwise discard and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the kernel code, look for fusion opportunities, try different threading configs, reduce memory allocations. The loop runs until the human interrupts you.

## Optimization ideas to consider

These are starting points, not an exhaustive list:

1. **Baseline: inherit UMASFastPytorchBackend**: First get the CPU backend working by simply inheriting all ops from the PyTorch backend. This gives SO2 block-diagonal + unified radial MLP but uses PyTorch's default BMM/index_add. Measure this as baseline.
2. **Fused gather+Wigner C++ kernel**: Replace `node_to_edge_wigner_permute` with a C++ kernel that fuses index gather + block-diagonal 9×9 matrix multiply. Avoids materializing the full [E,9,C] gather result. Use OpenMP for edge parallelism.
3. **Fused scatter+inverse Wigner C++ kernel**: Replace `permute_wigner_inv_edge_to_node` with a C++ kernel. Use thread-local accumulators + final reduction for the scatter to avoid atomic operations.
4. **Exploit block-diagonal structure**: The 9×9 Wigner is block-diagonal (1×1 + 3×3 + 5×5). Only 35 of 81 elements are nonzero. Hardcode the block structure in C++ to skip zero multiplies.
5. **OpenMP tuning**: Experiment with `OMP_NUM_THREADS`, `OMP_SCHEDULE`, thread affinity. CPU inference is often memory-bandwidth-bound — fewer threads with better cache locality may win.
6. **BLAS for the BMM**: Use `cblas_sgemm` directly for the block-diagonal matmuls instead of going through PyTorch's ATen. Batch small matmuls or use a single large GEMM with stride tricks.
7. **Memory layout optimization**: The default [N, 9, C] layout means the channel dimension is contiguous. For CPU SIMD, this is good. But for the gather operation, [9, N, C] might give better cache behavior. Experiment.
8. **torch.compile on CPU**: Try AOTInductor for CPU. May auto-fuse some ops. Test with and without.
9. **Pre-allocate buffers**: Reuse output tensors across inference calls to avoid repeated allocation.
10. **SIMD intrinsics**: For the inner loops (9 coefficients × C channels), use AVX2/AVX-512 intrinsics in the C++ kernel for maximum throughput.
11. **Custom backward**: Write C++ backward kernels (grad_x scatter, grad_wigner outer product) that are cache-friendly and OpenMP-parallel.
12. **torch_num_threads tuning**: The InferenceSettings has a `torch_num_threads` field. Experiment with different values.
