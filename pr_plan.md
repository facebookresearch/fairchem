# PR Plan: `umas_fast_gpu` Execution Backend

## Development Environment

| Item | Path |
|------|------|
| **Target repo (main)** | `/home/misko/env/feb17_speed_refactor/fairchem_feb22_gpu_speed` |
| **Reference repo (source)** | `/home/misko/env/feb17_speed_refactor/fairchem_pr4_triton_kernels_repo` |
| **Python venv** | `fairchem_feb22_gpu_speed_venv` |

- All PRs land on the **target repo** (`fairchem_feb22_gpu_speed`).
- The **reference repo** (`fairchem_pr4_triton_kernels_repo`) is read-only — used as the source to pull Triton kernel code from. Nothing is committed there.
- Activate the venv before any development or testing: `source fairchem_feb22_gpu_speed_venv/bin/activate`

## Overview

Add a single new execution backend `umas_fast_gpu` to the existing 2-backend system (`general`, `umas_fast_pytorch`). The new backend combines SO2 block conversion (from `umas_fast_pytorch`) with Triton-accelerated gather/rotate/scatter kernels and unified radial MLP precomputation.

Delivered in **2 PRs** landing sequentially on the target repo.

```
fairchem_feb22_gpu_speed (main: general, umas_fast_pytorch)
  └──> PR 1: backend refactoring + triton kernels + umas_fast_gpu (without unified radial)
         └──> PR 2: add unified radial to umas_fast_gpu
```

After PR 2, the system has exactly **3 execution backends**:

| Mode | Class | Description |
|------|-------|-------------|
| `"general"` | `ExecutionBackend` | Default PyTorch (unchanged) |
| `"umas_fast_pytorch"` | `UMASFastPytorchBackend` | SO2 block conversion, PyTorch rotations (unchanged) |
| `"umas_fast_gpu"` | `UMASFastGPUBackend` | SO2 block conversion + Triton kernels + unified radial |

---

## `UMASFastGPUBackend` Design

### Inheritance

```
UMASFastGPUBackend(UMASFastPytorchBackend)
  └── UMASFastPytorchBackend(ExecutionBackend)
        └── ExecutionBackend
```

Extends `UMASFastPytorchBackend` — inherits SO2 block conversion, overrides rotation/scatter with Triton kernels.

### Method Resolution

| Method                        | Source                      | Implementation                                                                   |
|-------------------------------|-----------------------------|----------------------------------------------------------------------------------|
| `validate`                    | `UMASFastGPUBackend`        | Super validation + checks HAS_TRITON, lmax==2, mmax==2, sphere_channels%128==0   |
| `prepare_wigner`              | `UMASFastGPUBackend`        | Passthrough (no einsum — Triton kernels work on raw L-ordered wigner)             |
| `gather_rotate`               | `UMASFastGPUBackend`        | `FusedEdgeGatherWignerL2MTritonBwdEmitFunction` (fused gather + Wigner L→M + emit x_edge) |
| `rotate_back`                 | `UMASFastGPUBackend`        | `FusedMToLThenWignerLmax2Function` (fused M→L permutation + Wigner in one kernel) |
| `edge_degree_scatter`         | `UMASFastGPUBackend`        | Uses `_M0_COL_INDICES_L_ORDER = [0, 2, 6]` for L-ordered wigner_inv              |
| `prepare_model_for_inference` | `UMASFastGPUBackend`        | `super()` (SO2 block conversion) + `_initialize_unified_radial()` (PR 2)         |

Gate activation is **not** part of the backend interface — it stays as a direct `self.act()` call in `escn_md_block.py`, unchanged from main.

### Triton Kernels Used (for UMA-S, direct_forces=False)

| Operation          | Kernel                                       | File                               |
|--------------------|----------------------------------------------|-------------------------------------|
| Fwd gather+rotate  | `fused_edge_gather_wigner_l2m_emit_kernel`   | `triton/edge_gather_wigner_fwd.py`  |
| Bwd grad_x         | `wigner_transform_bwd_kernel` (V2 two-phase) | `triton/edge_gather_wigner_bwd.py`  |
| Bwd grad_wigner    | PyTorch `bmm` (uses saved x_edge from emit)  | (no triton kernel)                  |
| Rotate-back fwd    | `fused_m_to_l_wigner_lmax2_kernel`           | `triton/wigner_ops.py`              |
| Rotate-back bwd dx | `fused_wigner_bwd_dx_l_to_m_kernel`          | `triton/wigner_ops.py`              |
| Rotate-back bwd dw | `wigner_lmax2_bwd_dw_kernel`                 | `triton/wigner_ops.py`              |

---

## PR 1: Backend Refactoring + Triton Kernels + `umas_fast_gpu`

### Theme

Refactors the `ExecutionBackend` base class to support pluggable wigner preparation, adds the Triton kernel package, and introduces the `umas_fast_gpu` backend (without unified radial — that lands in PR 2). After this PR, `execution_mode="umas_fast_gpu"` works end-to-end with Triton-accelerated rotations.

### New Files (4)

All paths relative to `src/fairchem/core/`.

| File                                          | Contents                                                                                      |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------|
| `models/uma/triton/__init__.py`               | `HAS_TRITON` flag. Minimal exports: `HAS_TRITON`, `FusedEdgeGatherWignerL2MTritonBwdEmitFunction`, `FusedMToLThenWignerLmax2Function`. Also export `FusedEdgeGatherWignerL2MTritonV2BwdFunction` (used as test reference). |
| `models/uma/triton/edge_gather_wigner_fwd.py` | Forward Triton kernels: base + emit. Fuses gather + Wigner L→M + L-to-M permutation          |
| `models/uma/triton/edge_gather_wigner_bwd.py` | Backward autograd Functions (emit variant: `FusedEdgeGatherWignerL2MTritonBwdEmitFunction`)   |
| `models/uma/triton/wigner_ops.py`             | M→L rotation kernels: `FusedMToLThenWignerLmax2Function`                                     |

**Note on triton kernel files:** These files are copied wholesale from the reference repo rather than trimmed to only the functions used by `umas_fast_gpu`. The files contain additional autograd Function classes (e.g., `FusedEdgeGatherWignerL2MAllFusedBwdFunction`, `FusedEdgeGatherWignerL2MNodeCentricFunction`) and kernel variants that are not wired up to any backend in this PR. Copying wholesale is simpler and avoids breaking internal helper dependencies between the functions. The `__init__.py` controls what is publicly exported.

### Modified Files (3)

| File                                  | Changes                                                                                     |
|---------------------------------------|---------------------------------------------------------------------------------------------|
| `models/uma/nn/execution_backends.py` | **Base class changes:** `validate()` signature `(settings)` → `(model, settings=None)`. New `prepare_wigner()` method (default: einsum fusion). Add `_M0_COL_INDICES_L_ORDER` constant. Update `UMASFastPytorchBackend.validate` to match new signature. **New backend:** Add `UMAS_FAST_GPU` enum value. Add `UMASFastGPUBackend` class (extends `UMASFastPytorchBackend`). Register in `_EXECUTION_BACKENDS`. |
| `models/uma/escn_md.py`              | Delegate wigner prep to `self.backend.prepare_wigner()`. Change `self.backend.validate(settings)` → `self.backend.validate(self, settings)`. Rename `wigner_and_M_mapping` → `wigner` throughout. Simplify envelope pre-fusion to single variable (see note below). |
| `models/uma/escn_md_block.py`        | Rename `wigner_and_M_mapping` → `wigner` and `wigner_and_M_mapping_inv_envelope` → `wigner_inv_envelope` in parameter names and internal usage (lines 118-119, 145, 148, 189-190 and throughout `Edgewise.forward`, `Edgewise.forward_chunk`, `eSCNMD_Block.forward`). Pure rename — no behavioral change. |

**Note on envelope pre-fusion:** The base repo computes `wigner_inv * edge_envelope` twice into two separate variables (`wigner_and_M_mapping_inv_envelope_for_edge_degree` and `wigner_and_M_mapping_inv_envelope`). These are mathematically identical — the same formula applied to the same tensors. The PR collapses them into a single variable `wigner_inv_envelope` computed once and reused for both edge degree embedding and block message passing. This is behavior-preserving and numerically identical.

### `UMASFastGPUBackend` in PR 1 (without unified radial)

```python
class UMASFastGPUBackend(UMASFastPytorchBackend):
    """
    GPU-optimized backend: SO2 block conversion + Triton kernels.

    Extends UMASFastPytorchBackend with Triton-accelerated
    gather_rotate, rotate_back, and edge_degree_scatter.
    Requires lmax==2, mmax==2, sphere_channels divisible by 128.
    """

    @staticmethod
    def validate(model, settings=None):
        UMASFastPytorchBackend.validate(model, settings)
        from fairchem.core.models.uma.triton import HAS_TRITON
        if not HAS_TRITON:
            raise ValueError("umas_fast_gpu requires Triton")
        if model.lmax != 2 or model.mmax != 2:
            raise ValueError("umas_fast_gpu requires lmax==2 and mmax==2")
        if model.sphere_channels % 128 != 0:
            raise ValueError("sphere_channels must be divisible by 128")

    @staticmethod
    def prepare_wigner(wigner, wigner_inv, mappingReduced, coefficient_index):
        # Passthrough — Triton kernels handle L-to-M internally
        return wigner, wigner_inv

    @staticmethod
    def gather_rotate(x_full, edge_index, wigner):
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
        )
        return FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(
            x_full, edge_index, wigner
        )

    @staticmethod
    def rotate_back(x, wigner_inv):
        from fairchem.core.models.uma.triton.wigner_ops import (
            FusedMToLThenWignerLmax2Function,
        )
        return FusedMToLThenWignerLmax2Function.apply(x, wigner_inv)

    @staticmethod
    def edge_degree_scatter(x, radial_output, wigner_inv, edge_index,
                            m_0_num_coefficients, sphere_channels,
                            rescale_factor, node_offset=0):
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)
        wigner_inv_m0 = wigner_inv[:, :, [0, 2, 6]]  # L-ordered m=0 columns
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)
        x_edge_embedding = x_edge_embedding.to(x.dtype)
        return x.index_add(
            0, edge_index[1] - node_offset,
            x_edge_embedding / rescale_factor,
        )
```

### Key Properties

- `"general"` and `"umas_fast_pytorch"` produce identical results to before (refactoring is behavior-preserving).
- `"umas_fast_gpu"` works end-to-end for UMA-S inference with Triton-accelerated rotations.
- Forces match `"general"` baseline within `atol=1e-3`.

### Verification Scripts (not committed)

Two scripts are provided for manual verification of speed, memory, and correctness. These are **not committed** to the repo but live at the repo root for developer use.

**`compare_forces.py`** — Runs inference in all 3 modes (`general`, `umas_fast_pytorch`, `umas_fast_gpu`) on a deterministic FCC crystal, compares energy and forces against the `general` baseline, reports MAE / max error / peak memory, and prints a pass/fail summary with tolerances (`energy atol=1e-4`, `forces atol=1e-3`).

```
python compare_forces.py --checkpoint /path/to/checkpoint.pt [--natoms 2000] [--modes general umas_fast_gpu]
```

**`run_benchmarks.sh`** — Orchestrates both speed benchmarks (via `fairchem` CLI + `InferenceBenchRunner`) and force correctness comparison. Reports QPS, ns/day, and peak memory for each mode.

```
bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt [--natoms 2000] [--speed-only] [--forces-only]
```

---

## PR 2: Add Unified Radial to `umas_fast_gpu`

### Theme

Adds the `UnifiedRadialMLP` inference optimization to `umas_fast_gpu`. The unified radial batches the first linear layer of all 8 per-layer `RadialMLP` instances into a single GEMM (they all share the same input `x_edge`), reducing kernel launch overhead.

### New Files (1)

| File                              | Contents                                                                                  |
|-----------------------------------|-------------------------------------------------------------------------------------------|
| `models/uma/nn/unified_radial.py` | `UnifiedRadialMLP` class + `create_unified_radial_mlp()` factory. Inference-only module. |

### Modified Files (4)

| File                                  | Changes                                                                                  |
|---------------------------------------|------------------------------------------------------------------------------------------|
| `models/uma/nn/execution_backends.py` | Add `_initialize_unified_radial()` helper. Update `UMASFastGPUBackend.prepare_model_for_inference` to call `super()` then `_initialize_unified_radial(model)`. |
| `models/uma/escn_md.py`              | Add unified radial precomputation: check `hasattr(self, "_unified_radial_mlp")`, if present call once before block loop, pass `radial_per_layer[i]` to each block. |
| `models/uma/escn_md_block.py`        | Thread `precomputed_radial: torch.Tensor | None = None` through `eSCNMD_Block.forward` → `Edgewise.forward` → `Edgewise.forward_chunk` → `self.so2_conv_1()`. Defaults to `None`. |
| `models/uma/nn/so2_layers.py`        | Add `precomputed_radial: torch.Tensor | None = None` parameter to `SO2_Conv1_WithRadialBlock.forward()` and `SO2_Convolution.forward()`. When provided, skip `self.rad_func(x_edge)` and use precomputed values directly. |

### Changes to `UMASFastGPUBackend`

Only `prepare_model_for_inference` changes:

```python
@staticmethod
def prepare_model_for_inference(model):
    UMASFastPytorchBackend.prepare_model_for_inference(model)  # SO2 conversion
    _initialize_unified_radial(model)  # NEW in PR 2
```

### Key Properties

- `so2_layers.py` is modified in PR 2 to accept the `precomputed_radial` parameter (it does **not** exist in the base repo).
- `"umas_fast_gpu"` now includes unified radial. Same mode name, enhanced behavior.
- Forces still match `"general"` baseline.

---

## What's Excluded

These changes from `fairchem_pr4_triton_kernels_repo` are **not** included (unrelated to `umas_fast_gpu`):

- `gate_activation` as a backend method — `umas_fast_gpu` uses PyTorch gating; no benefit to making this pluggable
- `triton/gate_activation.py` — Triton gate activation kernel (not used by `umas_fast_gpu`)
- `layer_norm.py` — autocast decorator removal
- `inference.py` — `base_precision_dtype` field removal
- `radius_graph_pbc.py` / `radius_graph_pbc_nvidia.py` — graph generation changes
- `atomic_data.py`, `ase_datasets.py`, `common_structures.py` — data pipeline changes
- `predict.py`, `ase_calculator.py`, `data_parallel.py` — infrastructure changes
- `docs/plans/` — planning documents
- `tests/core/preprocessing/` — relocated test files
- All intermediate triton backend classes (TritonBackendEdgeDegree, TritonSO2AndRotate, etc.) — collapsed into single `UMASFastGPUBackend`

---

## Testing Plan

### Testing Philosophy

1. **Reuse existing infrastructure** — checkpoint fixtures, `AtomicData.from_ase`, `MLIPPredictUnit`.
2. **Test at two levels** — low-level kernel unit tests (fwd/bwd pairs) and end-to-end model tests (energy + forces).
3. **Baseline is `"general"`** — all comparisons against the default PyTorch backend.
4. **GPU + Triton gating** — all triton tests marked `pytest.mark.gpu` and `skipif(not HAS_TRITON)`.

### Existing Test Infrastructure (Reused)

| Infrastructure                 | Location                  | What It Provides                                            |
|--------------------------------|---------------------------|-------------------------------------------------------------|
| `direct_checkpoint` fixture    | `tests/core/conftest.py`  | Session-scoped trained UMA-S checkpoint (non-MOLE, 2 steps) |
| `direct_mole_checkpoint`       | `tests/core/conftest.py`  | Session-scoped trained UMA-MOLE checkpoint (2 steps)        |
| `fake_uma_dataset` fixture     | `tests/core/conftest.py`  | Session-scoped fake dataset for training checkpoints        |
| `seed_fixture`                 | `tests/conftest.py`       | Seeds random/numpy/torch/cuda to 42                         |
| `AtomicData.from_ase` pattern  | `test_escn_md.py` et al.  | Create graph data from ASE Atoms                            |
| `MLIPPredictUnit`              | `test_equivariance.py`    | Full model instantiation with InferenceSettings             |
| `get_fcc_crystal_by_num_atoms` | `common_structures.py`    | Deterministic test crystal structures                       |

---

### PR 1 Tests

#### 1A. Regression: Existing Tests Pass Unchanged

**What:** Run the full existing test suite to verify refactoring is behavior-preserving.

```
pytest tests/core/models/uma/test_escn_md.py
pytest tests/core/units/mlip_unit/test_equivariance.py
pytest tests/core/units/mlip_unit/test_mlip_unit.py
```

**Why:** The `validate()` signature change and `prepare_wigner()` delegation must not alter behavior for `"general"` and `"umas_fast_pytorch"`.

**Expected:** All existing tests pass with zero tolerance change.

#### 1B. Triton Kernel Unit Tests: Forward Kernels

**File:** `tests/core/models/uma/triton/test_triton_kernels.py` (new)

**Markers:** `pytestmark = pytest.mark.gpu`, classes gated with `@pytest.mark.skipif(not HAS_TRITON)`.

##### 1B.1 `TestEdgeGatherWignerForward`

Tests the base forward kernel against pure PyTorch reference.

**Fixture:**
```python
@pytest.fixture()
def graph_data(self):
    torch.manual_seed(42)
    N, E, C = 200, 5000, 128
    x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
    edge_index = torch.stack([
        torch.randint(0, N, (E,), device="cuda"),
        torch.randint(0, N, (E,), device="cuda"),
    ])
    wigner = torch.randn(E, 9, 9, device="cuda", dtype=torch.float32)
    return x, edge_index, wigner
```

| Test | What | Reference | Tolerance |
|------|------|-----------|-----------|
| `test_fwd_matches_pytorch` | `fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)` matches PyTorch gather + cat + bmm + L-to-M permutation | PyTorch implementation | `atol=1e-5, rtol=1e-5` |

##### 1B.2 `TestEmitForwardKernel`

| Test | What | Reference | Tolerance |
|------|------|-----------|-----------|
| `test_main_output_matches_base_kernel` | Emit kernel main output == base kernel output | Base forward kernel | `atol=0, rtol=0` (exact) |
| `test_side_outputs_match_gather` | `x_edge[:,:,:C] == x[edge_index[0]]` and `x_edge[:,:,C:] == x[edge_index[1]]` | PyTorch indexing | `atol=0, rtol=0` (exact) |

##### 1B.3 `TestWignerOpsForward`

| Test | What | Reference | Tolerance |
|------|------|-----------|-----------|
| `test_m_to_l_then_wigner_matches_pytorch` | `MToLThenWignerLmax2Function(x, wigner)` matches PyTorch permute M→L then bmm | PyTorch permute + bmm | `atol=1e-5, rtol=1e-5` |
| `test_fused_matches_non_fused` | `FusedMToLThenWignerLmax2Function` == `MToLThenWignerLmax2Function` | Non-fused variant | `atol=0, rtol=0` (exact) |

#### 1C. Triton Kernel Unit Tests: Backward Kernels

##### 1C.1 `TestEdgeGatherWignerBackward`

Tests backward passes via `.apply()` forward → `.sum().backward()` → compare gradients.

**Setup:** N=100, E=3000, C=128. Both `x` and `wigner` require grad.

| Test | Function Tested | Reference | Fwd Tol | grad_x Tol | grad_wigner Tol |
|------|-----------------|-----------|---------|------------|-----------------|
| `test_v2_bwd_matches_pytorch` | `FusedEdgeGatherWignerL2MTritonV2BwdFunction` | PyTorch autograd (manual gather+bmm fwd) | exact | `atol=1e-3` | `atol=1e-3` per L-block |
| `test_emit_bwd_matches_v2` | `FusedEdgeGatherWignerL2MTritonBwdEmitFunction` | V2 function (above) | exact | `atol=1e-3` | `atol=1e-3` per L-block |

##### 1C.2 `TestWignerOpsBackward`

| Test | Function Tested | Reference | Fwd Tol | grad_x Tol | grad_wigner Tol |
|------|-----------------|-----------|---------|------------|-----------------|
| `test_fused_m2l_bwd_matches_pytorch` | `FusedMToLThenWignerLmax2Function` | PyTorch autograd | exact | `atol=1e-3` | `atol=1e-3` |

##### 1C.3 `TestFusedGradWigner`

| Test | What | Reference | Tolerance |
|------|------|-----------|-----------|
| `test_matches_reference` | `fused_grad_wigner` vs PyTorch per-block bmm | PyTorch reference | `atol=1e-3` per L-block |
| `test_off_diagonal_zeros` | Off-block-diagonal entries are exactly 0 | Structural invariant | Exact zero |
| `test_channels_256` | C=256 multi-tile case | PyTorch reference | `atol=1e-2` |
| `test_small_graph` | N=5, E=10 tiny graph | PyTorch reference | `atol=1e-3` |

#### 1D. End-to-End Model Tests: Energy and Forces

**File:** `tests/core/models/uma/test_execution_backends.py` (new)

**Markers:** `pytestmark = pytest.mark.gpu`, gated with `@pytest.mark.skipif(not HAS_TRITON)`.

Verify that `umas_fast_gpu` produces numerically matching energy and forces vs `"general"` baseline, using the full model with `direct_forces=False`. Since forces are computed via `torch.autograd.grad(-energy, pos)`, force correctness implicitly validates the full backward pass through all Triton kernels.

##### Shared Infrastructure

```python
@pytest.fixture(scope="module")
def test_graph():
    """Deterministic test graph on GPU."""
    atoms = get_fcc_crystal_by_num_atoms(100)
    data_obj = AtomicData.from_ase(
        atoms, max_neigh=20, radius=6.0,
        r_edges=True, task_name="omat",
    )
    data_obj.natoms = torch.tensor(len(atoms))
    data_obj.charge = torch.LongTensor([0])
    data_obj.spin = torch.LongTensor([0])
    return data_obj


def _predict_energy_forces(checkpoint_path, execution_mode, test_graph):
    """Run prediction with a given execution mode, return (energy, forces)."""
    settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=True,
        execution_mode=execution_mode,
    )
    predictor = MLIPPredictUnit(
        checkpoint_path, "cuda", inference_settings=settings
    )
    data = test_graph.clone().to("cuda")
    data.pos.requires_grad = True
    batch = atomicdata_list_to_batch([data])
    output = predictor.predict(batch)
    return output["energy"], output["forces"]
```

##### Tests

| Test | Mode Tested | Baseline | Energy Tol | Forces Tol |
|------|-------------|----------|------------|------------|
| `test_general_is_deterministic` | `"general"` | Self (2 runs) | `atol=0` | `atol=0` |
| `test_umas_fast_pytorch_matches_general` | `"umas_fast_pytorch"` | `"general"` | `atol=1e-5` | `atol=1e-5` |
| `test_umas_fast_gpu_matches_general` | `"umas_fast_gpu"` | `"general"` | `atol=1e-4` | `atol=1e-3` |

**Note on tolerances:** Triton kernels use different FP accumulation order than PyTorch, hence looser tolerances. The `1e-3` force tolerance is consistent with kernel-level backward tests.

##### Training Gradient Flow

| Test | What | How |
|------|------|-----|
| `test_umas_fast_gpu_training_grad_flows` | Verify `loss.backward()` produces non-None, non-zero gradients on all parameters | Create model in `"umas_fast_gpu"` mode, compute energy loss, call `loss.backward()`, assert `param.grad is not None and param.grad.abs().sum() > 0` for all parameters |

#### 1E. Manual Verification: Speed + Memory + Correctness

Run the verification scripts (not committed to the repo):

```bash
# Full benchmark: speed + force comparison
bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt --natoms 2000

# Force comparison only (quick correctness check)
python compare_forces.py --checkpoint /path/to/checkpoint.pt

# Speed benchmark only
bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt --speed-only --natoms 6000
```

Expected outcomes:
- `umas_fast_gpu` shows measurable speedup over `general` and `umas_fast_pytorch`
- `umas_fast_gpu` shows reduced peak memory vs `general`
- Force MAE vs `general` baseline < `1e-3`, energy diff < `1e-4`

---

### PR 2 Tests

#### 2A. Regression: All PR 1 Tests Still Pass

```
pytest tests/core/models/uma/
pytest tests/core/units/mlip_unit/
```

The unified radial changes are additive. `precomputed_radial` defaults to `None`, so existing code paths are unaffected.

#### 2B. `UnifiedRadialMLP` Unit Tests

**File:** `tests/core/models/uma/nn/test_unified_radial.py` (new)

These tests run on **CPU** (no triton/GPU dependency).

| Test | What | Reference | Tolerance |
|------|------|-----------|-----------|
| `test_unified_matches_per_layer` | Create 8 `RadialMLP` with random weights, create `UnifiedRadialMLP` from them, run both on same `x_edge`. Each `unified[i]` must match `rad_funcs[i](x_edge)`. | Per-layer `RadialMLP.forward()` | `atol=1e-6, rtol=1e-6` |
| `test_unified_output_shapes` | Output list length == N, each tensor shape `[E, out_features]` | Expected shapes | Exact |
| `test_unified_is_inference_only` | `list(unified.parameters()) == []` (all weights are buffers) | Structural check | Exact |

#### 2C. End-to-End: `umas_fast_gpu` with Unified Radial

**File:** `tests/core/models/uma/test_execution_backends.py` (extend from PR 1)

The existing `test_umas_fast_gpu_matches_general` test continues to pass — the mode name is the same, but now `prepare_model_for_inference` also initializes the unified radial. Energy and forces should match with the same tolerances (or tighter, since unified radial only changes GEMM batching order).

| Test | Mode | Baseline | Energy Tol | Forces Tol |
|------|------|----------|------------|------------|
| `test_umas_fast_gpu_matches_general` | `"umas_fast_gpu"` (now with unified radial) | `"general"` | `atol=1e-4` | `atol=1e-3` |

#### 2D. Precomputed Radial Threading

| Test | What | How |
|------|------|-----|
| `test_precomputed_radial_none_is_noop` | `precomputed_radial=None` produces identical output to omitting it | Run same block forward with and without the kwarg, compare. Bitwise identical. |

#### 2E. Manual Verification: Speed + Memory with Unified Radial

Re-run the benchmark scripts to verify the unified radial adds further speedup:

```bash
bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt --natoms 2000
```

Expected: `umas_fast_gpu` (now with unified radial) shows additional speedup from reduced kernel launch overhead for radial MLPs.

---

### Test Summary Matrix

| Test Category                    | PR | GPU | Triton | Level  | Fwd | Bwd | Energy | Forces |
|----------------------------------|----|----|--------|--------|-----|-----|--------|--------|
| Existing regression tests        | 1  | No  | No     | Model  | Yes | Yes | Yes    | Yes    |
| Forward kernel vs PyTorch        | 1  | Yes | Yes    | Kernel | Yes | No  | No     | No     |
| Emit kernel side outputs         | 1  | Yes | Yes    | Kernel | Yes | No  | No     | No     |
| Wigner ops forward               | 1  | Yes | Yes    | Kernel | Yes | No  | No     | No     |
| Backward grad_x vs PyTorch       | 1  | Yes | Yes    | Kernel | No  | Yes | No     | No     |
| Backward emit vs V2              | 1  | Yes | Yes    | Kernel | Yes | Yes | No     | No     |
| Wigner ops backward              | 1  | Yes | Yes    | Kernel | No  | Yes | No     | No     |
| Fused grad_wigner kernel         | 1  | Yes | Yes    | Kernel | No  | Yes | No     | No     |
| E2E umas_fast_gpu vs general     | 1  | Yes | Yes    | Model  | Yes | Yes | Yes    | Yes    |
| Training gradient flow           | 1  | Yes | Yes    | Model  | No  | Yes | No     | No     |
| Manual speed/memory/correctness  | 1  | Yes | Yes    | System | Yes | Yes | Yes    | Yes    |
| UnifiedRadialMLP unit test       | 2  | No  | No     | Module | Yes | No  | No     | No     |
| E2E umas_fast_gpu+unified vs gen | 2  | Yes | Yes    | Model  | Yes | Yes | Yes    | Yes    |
| Precomputed radial no-op         | 2  | No  | No     | Module | Yes | No  | No     | No     |
| Manual speed/memory w/ unified   | 2  | Yes | Yes    | System | Yes | Yes | Yes    | Yes    |

### Tolerance Summary

| Comparison                           | Forward | grad_x | grad_wigner         | Energy | Forces |
|--------------------------------------|---------|--------|---------------------|--------|--------|
| Triton fwd kernel vs PyTorch         | 1e-5    | --     | --                  | --     | --     |
| Emit fwd vs base fwd                 | exact   | --     | --                  | --     | --     |
| Triton bwd vs PyTorch bwd            | --      | 1e-3   | 1e-3 per L-block    | --     | --     |
| Fused M2L vs PyTorch                 | 1e-5    | 1e-3   | 1e-3                | --     | --     |
| umas_fast_gpu vs general (E2E)       | --      | --     | --                  | 1e-4   | 1e-3   |
| Unified radial vs per-layer          | 1e-6    | --     | --                  | --     | --     |
| umas_fast_gpu+unified vs general     | --      | --     | --                  | 1e-4   | 1e-3   |
