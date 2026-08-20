# CLAUDE.md
FAIRChem is Meta FAIR Chemistry's ML framework for atomistic simulations. Core abstractions: foundation models (UMA) with backbone+heads architecture, ASE calculator integration, Hydra-based config, and multi-task training via TorchTNT.

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

- `torch~=2.13.0`, `e3nn>=0.5` - PyTorch + equivariant neural networks
- `ase>=3.26.0` - Atomic Simulation Environment
- `torchtnt` - PyTorch training framework (TrainUnit/EvalUnit)
- `hydra-core` + `omegaconf` - Configuration management
- `lmdb` - Dataset storage format
- `ray[serve]>=2.53.0` - Distributed computing

## Testing Gotchas

- The canonical Ray Serve batching implementation is
  `src/fairchem/core/components/batch_server.py`. During rebases, do not
  resurrect the legacy `units/mlip_unit/_batch_serve.py`,
  `units/mlip_unit/batch_server.py`, or `calculate/_batch_server.py` paths;
  port their functional changes into the component instead.
- Tests that download registered checkpoints must declare their models with a
  `pretrained` marker. This lets base CI deselect them with `--exclude-models`
  and routes them to the matching model-sweep job.
- Freeze inference parameters after inference-specific module replacement.
  Main's folded-batch linear path removes the former general-backend regression:
  on one H100, freezing improved compiled general inference by 15-17% and cut
  peak allocated memory by 27-29% at 100-2,000 atoms. PyTorch 2.13 CPU checks
  improved by 4% at 32 atoms and were neutral at 1,000 atoms. Custom backward
  paths must preserve input derivatives independently of parameter gradients.
- `umas_fast_gpu` custom backward operators do not implement `vmap` batching.
  Compute Hessians through the per-component loop (`hessian_vmap=False`) when
  exercising that backend, and ensure inference settings forward that option
  into the backbone configuration.
- Set `CI=true` when reproducing CPU CI shards locally. Some multi-GPU graph
  parallel tests rely on that environment variable for skipping instead of the
  `gpu` marker, so the CI marker expression alone will still collect them.
- `graph_parallel_group_size=None` disables graph-parallel setup. A value of
  `1` intentionally initializes singleton graph- and data-parallel groups and
  is used to exercise those paths in tests; do not treat it as disabled.
- Keep the full `AtomicData.clone()` boundary in prediction unless benchmarks
  justify changing it and every model-side mutation has been audited. Graph
  parallelism, MOLE preparation, and conservative gradients can replace or
  mutate input fields, so a selective shallow copy is brittle.

## Numerical Precision

- Model constructors must not mutate process-wide PyTorch precision settings
  such as `torch.set_float32_matmul_precision`. Precision is caller-owned;
  inference applies TF32 temporarily through `InferenceSettings.tf32` and
  restores the prior settings afterward.
- TF32 policy belongs to the training/evaluation unit config or
  `InferenceSettings.tf32`, never to a model config or model attribute.
  Execution callers scope and restore the policy outside compiled `forward`
  methods because precision getters cannot be traced by fullgraph.
- Training and evaluation units default TF32 to disabled. Configs should set
  `tf32` only when overriding that default. Hydra CLI overrides for configs
  that omit the key must use the add syntax, such as
  `+runner.train_eval_unit.tf32=true`.
- Keep one configurable TF32 context manager for scoped matmul precision and
  cuDNN state instead of introducing overlapping context managers.
- Use the `tf32_context_manager` name for that policy; it controls both matmul
  precision and cuDNN TF32, so `matmul_context` is too narrow.
- Training FLOPs profiling invokes the model from `on_train_start`; scoped
  execution settings must cover profiling as well as train/eval step methods.

## Cluster Validation Gotchas

- H100 compute nodes do not have PyPI egress. Provision Python environments on
  the submission host before launching validation jobs.
- Imports from home-backed virtual environments are extremely slow on H100
  nodes. Copy complete environments and large checkpoints to node-local scratch
  before running tests or benchmarks.
- Pretrained checkpoints are cached under `~/.cache/fairchem`. Set
  `HF_HUB_OFFLINE=1` in compute jobs to prevent blocked Hugging Face metadata
  requests when the required files are already cached.
- Separate Hugging Face downloads can populate different snapshots while
  `refs/main` points only to the latest one. Ensure the active snapshot contains
  every checkpoint needed by offline tests.
- Core test collection imports benchmark and calculation modules through the
  shared conftest. Validation environments need the `extras` dependencies,
  including `pandas`, `pyarrow`, and `pymatgen`, even for focused test subsets.
- Importing the benchmark package during core test collection can cause
  `matbench_discovery` to download its WBM summary data. Cache that data before
  running without Figshare access, or collection can fail before tests start.
- Some GPU assertions are stochastic or tolerance-sensitive, and the complete
  GPU matrix is expensive. Reproduce failures with the exact test node (and
  repeat it when appropriate) before rerunning a full GPU shard.

## Hessian Backend Gotchas

- PyTorch's generic `vmap` fallback cannot batch the mutable, output-argument
  Triton operators used by `umas_fast_gpu`. Backend validation rejects requested
  Hessians with `hessian_vmap=True`; set it to `False` until the backward
  operators have explicit batching rules. Automatic backend selection falls
  back to normal mode for this combination. This only changes Hessian
  construction: energy, force, and stress inference are unaffected. The
  fallback computes one vector-Jacobian product per Cartesian force component,
  so it can be slower for large systems while using less memory.
- Explicit `torch.library.register_vmap` rules are possible for mutable custom
  operators. A rule that loops over the mapped dimension would make the
  operator compatible but retain most kernel-launch overhead. Recovering the
  performance value of vectorized Hessians requires rules backed by genuinely
  batched Triton kernels, including every custom backward operator reached by
  the derivative graph.

## Dependency Compatibility

- `pymatgen` and `pymatgen-core` are independently versioned. Slab tests must
  not depend on enumeration order, seeded random coordinates, or atom counts
  unless those values are part of the public contract; prefer composition,
  Miller index, shift, placement, and cell invariants that survive upgrades.

Anytime we learn something that could be beneficial in future coding sessions, automatically add it to CLAUDE.md.

This includes:
- Gotchas that are not obvious
- Subtle bugs that manifest under specific conditions
- Repeat corrections I make to the output of coding agents
