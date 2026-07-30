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

## Running the Test Suite on Windows

- `pip install -e packages/fairchem-core` fails on Windows: `hatch-fancy-pypi-readme`
  resolves `src\fairchem\core\README.md` and `LICENSE.md` relative to the package
  directory and cannot find them. Skip packaging entirely - `fairchem` is a
  namespace package, so `PYTHONPATH=src pytest ... -c packages/fairchem-core/pyproject.toml`
  imports it directly.
- `clusterscope` imports POSIX-only `fcntl`, which blocks the shared
  `tests/core/conftest.py` on Windows. A no-op `fcntl.py` shim in site-packages
  (`flock`/`lockf`/`fcntl`/`ioctl` returning None or 0) unblocks collection.
  Keep it outside the repo.
- `nvalchemiops` is not on PyPI, pypi.nvidia.com or pypi.ngc.nvidia.com, and no
  public Docker image carries it. The `radius_pbc_version=3` tests therefore
  *fail* rather than skip locally - 29 of them in `tests/core/graph`. They fail
  identically on `main`, so diff the failure sets between branches with
  `--junitxml` instead of reading the pass count; the repo's terminal plugin
  suppresses pytest's final tally line, so the tally is not in stdout.
- The project pyproject sets `-x`, so a single pre-existing environment failure
  hides the rest of the suite. Pass `--maxfail=500` when auditing for
  regressions.

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

## Neighbour Truncation Gotchas

- `get_max_neighbors_mask` selects the nearest `max_neighbors` per atom and has
  no global connectivity guarantee. Where two dense regions touch through one
  bridging contact, both endpoints can rank that contact outside their own
  budget and drop it, so a physically connected structure arrives at the model
  as two disconnected components. Nothing downstream detects this. Measured on
  two fcc grains at a 3.2 A gap with cutoff 6 A: the graph splits at every
  budget from 8 to 30, including the shipped `max_neighbors: 30`. Pass
  `preserve_connectivity=True` through `generate_graph` to re-add the shortest
  dropped edges.
- All three radius-graph implementations (v1, v2, nvidia) call
  `get_max_neighbors_mask`, so they share this behaviour identically. Fix it in
  the shared mask, not per implementation.
- Uniformly random dense periodic boxes never fracture (0/400 trials at every
  budget). The defect needs bridge topology - adsorbates, grain boundaries,
  cluster contacts - so random stress tests will not find it.
- ESCAIP's `biknn_radius_graph` builds a *mutual*-kNN mask
  (`env = amax(src_rank/k, dst_rank/k, dist/cutoff)`, keep `env < 1`), which
  requires an edge to be in the top-k of *both* endpoints. On one identical
  case set of 15 structures x 4 budgets it therefore splits 25/60 against the
  shared mask's 15/60. Relaxing it to union-kNN (`minimum` instead of `maximum`
  on the rank pair) only reaches 17/60; it is not a fix. Always quote these
  against the same case set - an earlier probe reported 20 for ESCAIP because
  `knn_pad_size` truncated the candidate list before the count, on a different
  set of structures. `build_radius_graph` is `@torch.jit.script` and discards
  the pre-mask candidate list, so reusing `reconnect_mask` there needs a
  return-signature change, not a one-liner.
- Component labelling and Boruvka both vectorize with `scatter_reduce_` plus
  pointer jumping - no Python loop over edges, so they run on GPU. Skip the
  second labelling pass when the retained graph already has one component per
  system: no edge ever joins two systems, so that is the floor and it proves
  equality with the untruncated graph. Without that fast path the flag costs
  ~2-3.5x `get_max_neighbors_mask`.
- Cost of `preserve_connectivity`, measured on an RTX 4060 with CUDA events
  over 5036 graph builds: 0.99x on a batch with nothing bridged, 1.216x on a
  16-system / 2160-atom batch with 4 bridged. CPU float64 v1 is 0.93-1.00x
  because v1 is dominated by its N^2 build; do not quote the CPU number alone,
  it flatters the change.
- Pointer jumping does not need `log2(num_atoms)` steps per hooking round. The
  round repeats until labels stop changing, so a small constant is correct, and
  each jump is a kernel launch over a batch-sized tensor. At 1372 atoms, 2 syncs
  cost 0.068 ms and 4 scatters over 57624 edges cost 0.132 ms, while 24 jump
  gathers dominated the rest; 4 jumps per round give identical labels and take
  the labeller from 0.45 ms to 0.29 ms.

## Neighbour Truncation: Speedups That Do Work

Four provable work reductions in `reconnect_mask`, all with bitwise-identical
output (130/130 configurations across 13 structures x 5 budgets x CPU and CUDA,
72 of them on the non-trivial repair path). Measure these with **dispatched-op
and sync counts, not wall clock**: on the RTX 4060 laptop the same code A/B'd
between 1.29x and 3.35x depending only on GPU clock state, because the mechanism
is launch and sync count rather than arithmetic. Op counts are exact and
machine-independent.

- **The second component-labelling pass was pure redundancy.** Contracting the
  retained components is a quotient map, so the untruncated graph has fewer
  components than the retained graph *exactly* when some edge crosses two of
  them. `crossing.any()` — already computed inside Boruvka's first round —
  decides it. Deleted a full labelling pass over all edges and all atoms.
- **Boruvka's round count does not need a convergence test.** Every component
  with an outgoing edge merges with at least one other, so each round at least
  halves the count. Rounds past convergence select nothing and are exact no-ops.
  A fixed count removes two device-to-host syncs per round (`crossing.any()` and
  `selected.any()`), which on a 20-node quotient graph were the entire cost:
  1.386 ms for 20 nodes and 256 edges.
  `selected.any()` is also implied by `crossing.any()` — the globally shortest
  crossing edge has both endpoints' minimum equal to its own length, so it always
  clears the threshold.
- **The round bound is the deepest system, not the batch.** Merging happens
  inside a piece of the graph and never across pieces, and no edge joins two
  systems, so a batch's quotient graph is a disjoint union. On 4 bridged + 12
  plain systems: 20 components total but 2 in the deepest system, so 1 round
  instead of 5. Skip the per-system count when `num_systems == 1` — there the two
  bounds are equal and computing it is pure overhead.
- **`unique(return_inverse=True)` does three jobs in one op**: counts the
  components, and its inverse *is* the contraction to `[0, num_kept)`. Because
  `unique` returns sorted values the renumbering is strictly increasing, and
  selection reads labels only through `amin` and `!=`, so no decision changes.
  Counting fixed points of the labelling (`labels == arange`) also gives the
  count, but costs 3 ops against 1 and made the common fast path *slower*
  (25 -> 28 ops) — a regression wall-clock could not resolve and op counts caught
  immediately.
- Per-system component counts cost `num_systems` work, not `num_atoms`: component
  names are the smallest atom index in each component and come back sorted, so
  `searchsorted` over the system boundaries beats `repeat_interleave` +
  `index_add_` over atoms.
- Net: 111 -> 80 ops and 10 -> 5 syncs on a bridged batch, 108 -> 71 ops on a
  single bridged pair, and exactly 25 -> 25 ops / 3 -> 3 syncs on the fast path.
  End-to-end the `preserve_connectivity` tax on `generate_graph` v2 falls from
  1.157-1.182x to 1.117-1.153x; v1 from 1.502-1.701x to 1.390-1.583x. The
  end-to-end effect is small because `reconnect_mask` is a slice of the build.
- The 1.216x tax recorded earlier for this flag did not reproduce on the
  documented harness (1.16-1.18x for v2, 1.50-1.70x for v1 on the same 16-system
  / 2160-atom shape). Treat that number as unscoped until re-measured; state
  which of `generate_graph`, `radius_graph_pbc` or `get_max_neighbors_mask` is
  the denominator, because they differ by more than the effect being measured.

## Neighbour Truncation: Speedups That Do Not Work

Four attempts to make `get_max_neighbors_mask` pay for added work, all measured
on an RTX 4060, all rejected. Do not re-try these without a new argument.

- Skipping the dense `[num_atoms, global_max_degree]` padding: occupancy is
  61-100% even for heterogeneous batches, so the ceiling is ~1.6x on one
  sub-step.
- `torch.topk(k+1)` instead of the full `torch.sort`: 1.33-1.45x on some shapes
  but 0.69-0.81x on others. CUDA's topk is not reliably faster than its sort.
- Replacing `torch.le(distance_sort.T, cutoff)` with a contiguous `[N, D]`
  compare: bitwise identical, 1.00x on most shapes.
- Building and sorting only the rows whose degree exceeds the budget: bitwise
  identical output but **0.69-0.78x**. The sort is only ~5% of the mask's
  runtime, so removing sort work cannot pay for the compaction and index
  lift-back needed to remove it. This is the load-bearing measurement - it
  kills the whole family of "sort less" ideas.
- Under exact geometric degeneracy the minimum spanning forest is not unique
  (two symmetric fcc grains gave 112 bitwise-equal candidate bridges). Any rule
  that picks one is atom-order dependent. Keep the whole degenerate shell, the
  same way the non-strict neighbour budget already does; do not claim a minimal
  forest.
- `enforce_max_strictly=False` (the default) is atom-order invariant (0/54
  permutation trials changed the edge set). `enforce_max_strictly=True` is not
  (38/54 changed). The docstring's warning about unit-cell-dependent formation
  energies applies to the strict path only, so connectivity work must not be
  advertised as fixing it.

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
