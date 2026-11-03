# UMA Nitro source and license provenance

This document records the source and dependency provenance of
[`_inference.py`](src/fairchem/core/models/uma/uma_nitro/_inference.py). It is not
legal advice or a complete operating-system or container SBOM.

## License summary

| Component | Upstream | License |
| --- | --- | --- |
| FairChem and Meta-authored UMA Nitro code | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) | MIT |
| Embedded dense-GEMM example | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | BSD-3-Clause |
| NumPy | [numpy/numpy](https://github.com/numpy/numpy) | BSD-3-Clause |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause for the main project; binary distributions include additional notices |
| Triton | [triton-lang/triton](https://github.com/triton-lang/triton) | MIT |
| CUTLASS/CuTe DSL runtime | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | NVIDIA Software License Agreement |
| CUDA Python bindings | [NVIDIA/cuda-python](https://github.com/NVIDIA/cuda-python) | NVIDIA Software License |
| UMA checkpoint | [facebook/UMA](https://huggingface.co/facebook/UMA) | Gated model terms, separate from FairChem's MIT license |

The repository's MIT license does not relicense the embedded BSD-licensed
CUTLASS source, NVIDIA runtime packages, CUDA, or the UMA checkpoint.

## FairChem source

The implementation includes copied or adapted helpers from FairChem's UMA
Triton kernels:

- `M_TO_L_GATHER_IDX` from
  [`constants.py`](src/fairchem/core/models/uma/triton/constants.py).
- `_wig_rot9`, `_wig_rotT9`, `_wig_dw_store1`, `_wig_dw_store2`, and the fused
  Wigner/conv1 kernels from
  [`kernels.py`](src/fairchem/core/models/uma/triton/kernels.py).

These portions remain covered by FairChem's MIT license and copyright notice.

## Embedded NVIDIA source

The dense-GEMM section is adapted from:

- Repository: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- Upstream release: CUTLASS
  [`v4.7.0`](https://github.com/NVIDIA/cutlass/tree/v4.7.0), commit
  `dcf215af68a2d08d305076c152a06f201728cd53`
- Original file:
  [`examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py`](https://github.com/NVIDIA/cutlass/blob/v4.7.0/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py)
- Original file SHA-256:
  `bb7b76d893219757e2f3701abf1a7c8c819b9966e38aaed30a768596a55aca9f`
- Copied/adapted symbol: the complete dense-GEMM class, named
  `HopperWgmmaGemmKernel` in FairChem. This includes `__init__`,
  `_setup_attributes`, `__call__`, `kernel`, `_compute_stages`,
  `_make_smem_layouts`, `_compute_grid`, `_make_tma_store_atoms_and_tensors`,
  `_make_tma_atoms_and_tensors`, `is_valid_dtypes`, and
  `is_valid_tensor_alignment`.
- License: BSD-3-Clause

The copied region is explicitly delimited by `BEGIN COPIED/ADAPTED NVIDIA
CUTLASS SOURCE` and `END COPIED/ADAPTED NVIDIA CUTLASS SOURCE` comments in
`_inference.py`. FairChem made these changes:

1. Renamed the class for the local namespace.
2. Removed the upstream argument parser, example runner, benchmark harness, and
   `__main__` entry point.
3. Added optional bias and per-row rescaling to the output epilogue.
4. Added an optional accumulator `epilogue_op` before output conversion.
5. Connected the class to FairChem's compile cache and custom-operator wrapper;
   those surrounding integration functions are Meta-authored and outside the
   copied-source boundary.

The original NVIDIA copyright, SPDX identifier, complete BSD conditions, and
disclaimer are retained immediately above the copied class in `_inference.py`.
The upstream notice is:

> Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The embedded BSD-licensed example is distinct from the separately installed
`nvidia-cutlass-dsl` runtime, which is distributed under NVIDIA's software
license.

No source from QuACK or FlashAttention is copied or adapted into UMA Nitro.

## Runtime dependencies

UMA Nitro directly uses:

- Python standard-library modules;
- NumPy;
- PyTorch and its autograd and custom-operator APIs;
- Triton;
- CUTLASS/CuTe DSL on supported NVIDIA architectures; and
- CUDA Python driver bindings.

The provenance audit used `nvidia-cutlass-dsl==4.7.1`, which `fairchem-core`
pins as a dependency on Linux x86_64 and aarch64, the platforms NVIDIA publishes
wheels for.

The authoritative dependency declarations live in
[`packages/fairchem-core/pyproject.toml`](packages/fairchem-core/pyproject.toml).
Release builds should use compatible pinned dependencies or a lock file and
retain all notices supplied by binary distributions.

## Model terms

UMA Nitro does not contain or distribute a checkpoint. It operates on supported
UMA checkpoints loaded by FairChem. Checkpoint terms are governed by the gated
license accepted from the model repository and are not covered by FairChem's MIT
license.

## Distribution requirements

When redistributing this source or a binary that contains it:

1. Preserve the FairChem MIT notice.
2. Preserve the embedded CUTLASS BSD-3-Clause notice and reproduce it with binary
   distributions as required.
3. Comply separately with the licenses for CUDA, CUDA Python, CUTLASS/CuTe DSL,
   and bundled binary dependencies.
4. Preserve and follow the applicable UMA checkpoint terms.
