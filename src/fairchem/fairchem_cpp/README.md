# fairchem_cpp

PyTorch C++/CUDA extensions for FAIRChem, providing GPU-accelerated
kernels for segmented matrix multiplication (`segment_mm`) used by
the MOLEFairchemCpp mixture-of-experts layer.

## Prerequisites

- Python 3.9+
- PyTorch 2.8+ (must be installed first)
- CUDA toolkit (matching your PyTorch CUDA version)
- A C++17 compiler

## Installing with fairchem-core

Install `fairchem-core` first, then build `fairchem_cpp` separately:

```bash
pip install -e packages/fairchem-core[dev]
pip install --no-build-isolation -e src/fairchem/fairchem_cpp
```

## Installing standalone

If you already have a working fairchem environment and want to
(re)build just the C++/CUDA extensions:

```bash
cd src/fairchem/fairchem_cpp
pip install --no-build-isolation -e .
```

Or using the provided Makefile (sets `TORCH_CUDA_ARCH_LIST` for you):

```bash
cd src/fairchem/fairchem_cpp
make install        # build and install in editable mode
make rebuild        # clean + install
make clean          # remove build artifacts
```

### Specifying GPU architectures

By default the Makefile targets Ampere/Ada/Hopper (`8.6;8.9;9.0`).
Override for your hardware:

```bash
TORCH_CUDA_ARCH_LIST="8.0;9.0" make install
```

### CPU-only build

If CUDA is not available, the package builds with CPU-only ops
(pure PyTorch fallbacks):

```bash
USE_CUDA=0 pip install --no-build-isolation -e .
```

## Verifying the installation

```python
import fairchem_cpp

print(fairchem_cpp)  # should print module path without errors
```

Or run the segment_mm tests:

```bash
pytest tests/core/models/uma/nn/test_segment_mm.py \
    -c packages/fairchem-core/pyproject.toml -vv
```

## What's inside

- `csrc/binding.cpp` — op registration (`segment_mm`, `segment_mm_backward`)
- `csrc/cuda/segmentmm.cu` — cuBLAS-based CUDA kernels
- `ops.py` — Python autograd wrappers with three-level structure for
  double backward support (forward, first backward, and second backward
  all route through the same fast CUDA kernels)

## Using fairchem_cpp with MoLE

By default, MoLE (Mixture of Linear Experts) layers use a pure-PyTorch
implementation (`moe_layer_type: pytorch`). To switch to the GPU-accelerated
`fairchem_cpp` kernels, set `moe_layer_type` to `fairchem_cpp` in your
training config:

```yaml
# before
moe_layer_type: pytorch

# after — uses fairchem_cpp segment_mm kernels
moe_layer_type: fairchem_cpp
```

Or override from the CLI:

```bash
fairchem -c configs/uma/training_release/uma_sm_direct_pretrain.yaml \
    moe_layer_type=fairchem_cpp
```
