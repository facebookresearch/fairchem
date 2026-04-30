# CLAUDE.md
FAIRChem is Meta FAIR Chemistry's ML framework for atomistic simulations. Core abstractions: foundation models (UMA) with backbone+heads architecture, ASE calculator integration, Hydra-based config, and multi-task training via TorchTNT.

## Scale-Down Debugging Protocol (MANDATORY)

**ALWAYS validate changes bottom-up: 1 GPU → 2 GPUs → 8 GPUs → multi-node.**

1. **Start at 1 GPU**: Run all tests and sanity checks against the baseline (main branch). Compare compile time, inference speed, and numerical outputs. If anything differs from main at 1 GPU, fix it before scaling up.
2. **Scale incrementally**: Only after 1 GPU passes, move to 2 GPUs, then 8 GPUs (1 node), then multi-node.
3. **If something breaks or doesn't make sense at large scale**: ALWAYS scale DOWN until you find the bug. NEVER debug at large scale — it wastes GPU-hours, has long queue times, and makes root-cause analysis impossible.
4. **Baseline comparison at every scale**: At each GPU count, compare your branch against main with identical settings. Any regression in compile time, throughput, or correctness must be investigated at the smallest scale where it reproduces.

This applies to: compile time, inference throughput, numerical accuracy, memory usage, and any other observable behavior. No exceptions.

## Development Commands

```bash
# Install
pip install -e packages/fairchem-core[dev]

# Tests (always pass -c flag)
pytest tests -c packages/fairchem-core/pyproject.toml
pytest tests/core/models/test_uma.py -vv
pytest tests/core -m "not gpu"

# Lint & format — REQUIRED for every modified file before committing
pre-commit run --files path/to/modified_file.py

# CLI
fairchem -c config.yaml [overrides...]
```

## Code Style

**IMPORTANT: You MUST run `pre-commit run --files /path/to/modified_file.py` on every file you modify, before considering the task complete. No exceptions.**

**Every file must start with:**
```python
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations
```

**Line length**: 88 characters. **Linter**: Ruff (config in `ruff.toml`).

**Docstrings** use Google convention. No text on opening/closing quote lines:
```python
# WRONG
"""This is wrong."""

# RIGHT
"""
Short description.
"""

# RIGHT (with args)
"""
Short description.

Args:
    x: The input tensor.

Returns:
    The processed tensor.
"""
```

**Imports**: isort enforced via Ruff. `fairchem.core` is `known-first-party`. Use `TYPE_CHECKING` for type-only imports:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
```

## Testing Conventions

All tests go in `tests/` (mirrors `src/fairchem/` structure). Always run with:
```bash
pytest tests -c packages/fairchem-core/pyproject.toml
```

### Test Markers
- `@pytest.mark.gpu`: GPU-only (auto-skipped when CUDA unavailable)
- `@pytest.mark.cpu_and_gpu`: Runs on both CPU and GPU
- `@pytest.mark.dgl`: Requires `fairchem_cpp`
- `@pytest.mark.inference_check`: Inference validation (skipped by default)

### Key Fixtures

**Root conftest (`tests/conftest.py`):**
- `seed_fixture` (function): Seeds all RNGs to 42
- `water_xyz_file` (session): Path to a minimal 3-atom water XYZ file
- `compile_reset_state` (function): Resets `torch.compiler` before/after test
- `setup_before_each_test` (autouse): Cleans up Ray, GPU memory, distributed state

**Core conftest (`tests/core/conftest.py`):**
- `dummy_binary_dataset` (session, parametrized): ASE dataset in both LMDB and CIF formats
- `fake_uma_dataset` (session): Full UMA training dataset path + config
- `direct_checkpoint` (session): Trained model checkpoint (inference + resume)
- `direct_mole_checkpoint` (session): Trained MOLE checkpoint
- `torch_deterministic` (function): Enables deterministic algorithms
- `snapshot` (function): Syrupy snapshot with approximate numpy comparison (`Approx`)

**Dataset conftest (`tests/core/datasets/conftest.py`):**
- `structures` (module): List of test atoms [H2O molecule, Cu bulk, Pt slab]

### Test Patterns

GPU/CPU dual tests:
```python
@pytest.mark.gpu
def test_something_gpu():
    _test_something("cuda")


def test_something_cpu():
    _test_something("cpu")


def _test_something(device):
    # shared implementation
    ...
```

Snapshot testing with approximate comparison:
```python
def test_values(snapshot):
    result = compute_something()
    assert pytest.approx(result.numpy(), abs=1e-3) == snapshot
```

Integration tests using the CLI:
```python
from tests.core.testing_utils import launch_main


def test_training(fake_uma_dataset):
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        f"datasets.data_root_dir={fake_uma_dataset}",
        "job.device_type=CPU",
        "max_steps=2",
    ]
    launch_main(sys_args)
```

## Architecture

### Model System (Backbone + Heads)

Models use `HydraModel` (registered as `"hydra"`): one backbone extracts features, multiple heads predict properties.

```
BackboneInterface.forward(data: AtomicData) -> dict[str, Tensor]   # features
HeadInterface.forward(data: AtomicData, emb: dict) -> dict[str, Tensor]  # predictions
```

Primary backbone: `escnmd_backbone` (SO(3)-equivariant eSCN with MD modifications).
Heads: `MLP_Energy_Head`, `Linear_Force_Head`, `DatasetSpecificSingleHeadWrapper`.

### Registry Pattern

Components are registered for dynamic Hydra instantiation:
```python
@registry.register_model("my_backbone")
class MyBackbone(nn.Module, BackboneInterface): ...
```

Available decorators: `register_model`, `register_dataset`, `register_loss`, `register_task`, `register_logger`, `register_trainer`.

Lookup: `registry.get_model_class("my_backbone")` or by full import path `"fairchem.core.models.my_module.MyBackbone"`.

### Data Flow

```
ASE Atoms -> AtomicData.from_ase() -> graph generation -> backbone -> heads -> predictions
```

`AtomicData` required fields: `pos, atomic_numbers, cell, pbc, natoms, edge_index, cell_offsets, nedges, charge, spin, fixed, tags`.
Optional targets: `energy, forces, stress`.

Batching via `atomicdata_list_to_batch()`. Multi-task collation via `MTCollater` (fills missing targets with `inf` for loss masking).

### Configuration (Hydra)

YAML configs use `_target_` keys for component instantiation:
```yaml
runner:
  _target_: fairchem.core.components.train.train_runner.TrainEvalRunner
  train_eval_unit:
    _target_: fairchem.core.units.mlip_unit.mlip_unit.MLIPTrainEvalUnit
    model:
      _target_: fairchem.core.models.base.HydraModel
      backbone: ${backbone}
```

Config sections: `job`, `runner`, `datasets`, `tasks`, `backbone`, `optimizer`.
Default configs in `configs/`. Overrides via CLI: `fairchem -c config.yaml key=value`.

### Task Names
- `oc20`: Catalysis (Open Catalyst)
- `omat`: Inorganic materials
- `omol`: Molecules
- `odac`: Metal-organic frameworks
- `omc`: Molecular crystals

### Training Flow

`TrainEvalRunner` orchestrates training via TorchTNT's `fit()`. Core unit: `MLIPTrainEvalUnit` (handles forward pass, loss, metrics, EMA, gradient clipping). Checkpoints use DCP (Distributed Checkpoint Protocol) with `dcp_to_torch_save()` for inference export.

### Model Loading and Inference

```python
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")
atoms.calc = calc
```

## Repository Structure

```
src/fairchem/core/
├── models/              # Backbones and heads (UMA, eSCN-MD, GemNet)
├── datasets/            # Data loading (LMDB, ASE), collaters, samplers
├── components/          # Runner components (train, evaluate, calculate)
├── units/               # TorchTNT train/eval/predict units
├── modules/             # Loss, schedulers, normalizers, evaluators
├── launchers/           # Local, Ray, SLURM job launchers
├── common/              # Registry, distributed utils, logging
├── graph/               # Graph generation, neighbor finding with PBC
└── _cli.py              # CLI entry point

tests/                   # All tests (mirrors src structure)
packages/                # Installable packages (fairchem-core, fairchem-data-*)
configs/                 # Hydra YAML configs (datasets, tasks, backbone, optimizer)
```

## Key Dependencies

- `torch~=2.8.0`, `e3nn>=0.5` - PyTorch + equivariant neural networks
- `ase>=3.26.0` - Atomic Simulation Environment
- `torchtnt` - PyTorch training framework (TrainUnit/EvalUnit)
- `hydra-core` + `omegaconf` - Configuration management
- `lmdb` - Dataset storage format
- `ray[serve]>=2.53.0` - Distributed computing

Anytime we learn something that could be beneficial in future coding sessions, automatically add it to CLAUDE.md.

This includes:
- Gotchas that are not obvious
- Subtle bugs that manifest under specific conditions
- Repeat corrections I make to the output of coding agents

## Learned Gotchas

### Gloo backend does not support `dist.all_to_all`
The Gloo distributed backend (used in CPU-only tests) does not support the `all_to_all` collective operation. When writing code that uses `all_to_all`, always provide a Gloo-compatible fallback using pairwise `isend`/`irecv` via `dist.batch_isend_irecv`. See `graph_parallel._safe_all_to_all()` for the pattern.

### pre-commit cache location
On the FAIR-SC cluster, the researcher agent's HOME is `agents/researcher/`, but pre-commit stores its cache under the real user HOME (`/storage/home/rgao/.cache/pre-commit/`). To run pre-commit successfully with the cached hooks, use `HOME=/storage/home/rgao` before the command: `HOME=/storage/home/rgao .venv/bin/pre-commit run --files <path>`. The `ruff` hooks are pre-cached there.

### Ruff PD011 false positive with PyTorch `.values`
Ruff's PD011 rule flags `.values` as a pandas anti-pattern, but PyTorch's `torch.return_types` NamedTuples also have `.values`. Use `[0]` indexing instead of `.values` to avoid the lint error: `result.max(dim=0)[0]` instead of `result.max(dim=0).values`.

### libgomp.so.1 path on H200 nodes
On FAIR-SC H200 nodes, `libgomp.so.1` is at `/usr/lib/x86_64-linux-gnu/libgomp.so.1`, NOT `/usr/lib64/libgomp.so.1`. When setting `LD_PRELOAD` for SLURM jobs, always check both paths.

### profai-cli launch-experiment venv
Use `--python /path/to/.venv/bin/python` flag with `profai-cli launch-experiment` to ensure SLURM jobs use the correct virtual environment. Without it, the system Python is used and fairchem won't be importable.

### Spatial partitioning does NOT handle PBC wrapping
The `partition_atoms_spatial()` k-means uses raw Cartesian positions without periodic boundary condition (PBC) minimum-image convention. This means atoms near opposite sides of a periodic box may be assigned to different clusters even if they're neighbors through the periodic image. This is intentional — for large systems (64k+ atoms) the fraction of PBC-crossing edges is small, and the graph generation handles PBC edges correctly regardless of how atoms are partitioned. For small periodic systems, the boundary fraction may be higher than the theoretical surface-to-volume prediction.

### build_gp_context expects pre-filtered edge_index
The `build_gp_context()` function expects the edge_index to be already filtered to only include edges whose targets are in this rank's partition. Passing the full unfiltered edge_index will cause index-out-of-bounds errors in the local edge_index. The filtering step is: `target_in_partition = (rank_assignments == rank)[edge_index[1]]; filtered = edge_index[:, target_in_partition]`.

### Overlap path is eval-mode only
The `_forward_overlap()` path in `Edgewise` uses `start_all_to_all_collect`/`finish_all_to_all_collect` which don't participate in autograd. This means gradients won't flow through the all-to-all communication in the overlap path. It's gated on `not self.training` to prevent use during training. For training, the sync `all_to_all_collect` (which wraps `AllToAllCollect` autograd function) is always used.

### NEVER use torchrun directly — always use `fairchem` CLI
Multi-GPU and multi-node jobs MUST be launched via the `fairchem` CLI: `fairchem -c config.yaml [overrides...]`. The CLI handles all distributed setup (torchrun, SLURM submission via submitit, etc.). Using `torchrun` directly bypasses the fairchem launcher infrastructure and may cause subtle configuration issues.

### Hydra overrides for SLURM config
The SLURM job configs (`configs/uma/speed/job/slurm.yaml`) use structured configs via OmegaConf. To override fields that exist in the YAML, use regular Hydra syntax: `job.scheduler.slurm.qos=h200_lowest`. To add fields not in the YAML but in the SlurmConfig dataclass (like `timeout_hr`), use `+` prefix: `+job.scheduler.slurm.timeout_hr=1`. This applies to any structured config field.

### Backbone config overrides via runner.overrides
To pass backbone config overrides (e.g., enabling all-to-all GP) through InferenceBenchRunner, use Hydra `+` prefix since `runner.overrides` is not in the default YAML: `'+runner.overrides={backbone: {use_all_to_all_gp: true, gp_partition_strategy: spatial}}'`. MLIPPredictUnit's `_build_overrides_from_settings()` merges these with checkpoint defaults, with user overrides taking precedence.

### fairchem CLI cannot submit from within SLURM
The `_cli.py` explicitly blocks SLURM submission from within an active SLURM job (`assert os.getenv("SLURM_SUBMIT_HOST") is None`). Always run `fairchem -c ... job=slurm` from a login node, not from an `srun` session. This also means profai-cli's `launch-experiment --qos` (which creates its own SLURM job) cannot be used to wrap `fairchem -c ... job=slurm` — it would create a double submission.

### AllToAllCollect backward must match forward arg count
The `AllToAllCollect.forward()` uses `torch.autograd.Function.apply()`. Every argument passed to `apply()` must have a corresponding `None` gradient returned in `backward()`. If you add new arguments to `forward()`, you MUST also add corresponding `None`s to the backward return tuple, or autograd will error: "returned an incorrect number of gradients".

### Prefer all_to_all_single over all_to_all for packed tensors
For NCCL, `dist.all_to_all_single(output, input, output_split_sizes, input_split_sizes, group)` operates on packed contiguous tensors directly, avoiding the Python list creation overhead from `tensor.split()` + `list()` that `dist.all_to_all(output_list, input_list, group)` requires. When send/recv data is already contiguous (as in AllToAllCollect), `all_to_all_single` is more efficient. However, it's not supported on Gloo — always provide a fallback. Also note that `split()` creates views into the original tensor, so after `all_to_all` fills the views, the original buffer already contains the data — a subsequent `torch.cat()` on the views is a redundant copy.

### Morton Z-order normalization must use global scale
When computing Morton Z-order codes for spatial partitioning, normalize positions using a SINGLE global scale factor (the largest bounding-box extent), NOT per-dimension normalization. Per-dimension normalization amplifies noise in short dimensions — e.g., a 100-unit x-gap becomes indistinguishable from a 2-unit y-gap after independent rescaling, destroying spatial locality in the Morton curve. Use `extent = (pos.max(0)[0] - min_pos).max()` instead of per-dimension `extent = pos.max(0)[0] - min_pos`.

### All-gather baseline also has per-layer collectives
The all-gather GP baseline has one all-gather per `Edgewise.forward()` layer (not just one upfront). UMA-S has 4 layers, so the baseline does 4 all-gathers per forward pass. The A2A path does 4 all-to-all_collect calls per forward (sending less data) plus a setup all_to_all for index exchange. The performance comparison is about setup overhead, not per-layer communication volume.

### Morton partition must use balanced splitting, not ceil-based chunking
The `partition_atoms_spatial()` function must use `arange(N) * P // N` (balanced splitting) instead of `arange(N) // ceil(N/P)` (ceil-based chunking). The ceil-based formula leaves trailing ranks empty when N is not a multiple of P — e.g., 1000 atoms / 64 ranks gives rank 63 zero atoms because `ceil(1000/64) = 16` and `63 * 16 = 1008 > 1000`. This causes NCCL hangs because the rank crashes at the empty-partition assertion while other ranks block on the collective. The balanced formula `i * P // N` distributes atoms evenly: first N%P ranks get `ceil(N/P)`, rest get `floor(N/P)`.

### Communication is <1% of UMA-S forward pass time
At 8 GPUs (intra-node NVLink), the all-gather per-layer communication takes ~0.015ms vs ~65-98ms total forward pass (0.02%). At 64 GPUs across 8 nodes, it's still only ~1ms vs ~65-91ms total (1.1%). The performance numbers in benchmark results are in **ns/day** (not QPS) — convert with `QPS = ns_per_day × 1e6 / 86400`. This means reducing communication volume alone cannot produce meaningful speedups; A2A must also reduce overhead in other areas (memory, synchronization, etc.) to be competitive.

### UMASFastPytorchBackend requires activation_checkpointing=False
When using `execution_mode: "umas_fast_gpu"` (the default speed benchmark mode), `activation_checkpointing` must be `False`. Setting it to `True` raises `ValueError: UMASFastPytorchBackend requires activation_checkpointing=False`. For benchmarks with `compile=False`, just omit the activation_checkpointing override entirely (defaults to False).

### Always use inference presets (turbo/default), not manual settings
Use the standard presets from `inference.py` — don't manually mix and match settings like `compile=True` with `tf32=False`. The presets are:
- **turbo**: tf32=True, activation_checkpointing=False, merge_mole=True, compile=True, execution_mode=None (auto → umas_fast_gpu)
- **default**: tf32=False, activation_checkpointing=True, merge_mole=False, compile=False, execution_mode=None (auto → general, since act_ckpt=True fails umas_fast_gpu validation)
- **traineval**: tf32=False, activation_checkpointing=False, merge_mole=False, compile=False, internal_graph_gen_version=1

Via Hydra CLI, override individual fields (not by preset name): `runner.inference_settings.tf32=false runner.inference_settings.compile=false` etc. To set execution_mode to auto-detect: `runner.inference_settings.execution_mode=null`.

### BL torch.compile overhead — UNDER INVESTIGATION
At 64 GPUs with all-gather GP on our branch, torch.compile took ~23 min per atom count (92 min total for 4 sizes). This is likely a BUG introduced by our code changes (e.g., `@torch.compiler.disable` on `_generate_graph`, additional branches in `Edgewise.forward`), NOT an inherent property of all-gather GP. Normal GP=64 compile on main takes <10 minutes. The 1-GPU comparison test will confirm whether our branch regressed compile time. Do NOT claim this as an A2A advantage until the root cause is confirmed.
