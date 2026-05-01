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

### BL torch.compile regression — ROOT CAUSE FOUND AND FIXED
The `@torch.compiler.disable` on `_generate_graph` wrapped the entire function, including the BL (all-gather) code path. This created a larger graph break than main (which only has `@torch.compiler.disable` on `generate_graph` itself). At 64 GPUs the larger non-compiled region caused 12x slower compilation (92 min vs ~8 min on main). Fix: extracted A2A partitioning into `_compute_a2a_partition()` with its own `@torch.compiler.disable`, leaving `_generate_graph` fully compilable for the BL path. Verified at 1-GPU and 8-GPU: compile time matches main exactly. The "46x compile speedup" claim for A2A in experiment 19 was artificial — both paths should compile similarly fast after the fix.

### Sparse P2P is faster than all_to_all_single for spatial partitioning
At 64 GPUs with spatial partitioning, each rank only communicates with ~10-15 actual neighbors (not all 63). `dist.all_to_all_single` creates P-1=63 send/recv pairs (zero-length pairs are no-ops but still consume NCCL group slots). `dist.batch_isend_irecv` with only non-zero neighbors creates ~25 ops (15 send + 10 recv), reducing NCCL operation count by ~4×. This saves ~1ms/layer × 4 layers = 4ms/step, a 6.4% speedup over standard A2A at 64 GPUs. No effect at 8 GPUs (NVLink has negligible per-op overhead). Enable with `backbone.use_p2p_gp=true`.

### torch.index_select with out= doesn't support autograd
`torch.index_select(x, 0, indices, out=buffer)` raises `RuntimeError: functions with out=... arguments don't support automatic differentiation` when `x.requires_grad=True`. This applies even in eval mode because `pos.requires_grad_(True)` is set for force computation (via autograd). Use regular indexing `x[indices].contiguous()` instead.

### All graph-break-prone code in _generate_graph must be in @torch.compiler.disable methods
The `_generate_graph()` method is in the compiled region. Any code inside it that uses `.item()`, data-dependent conditionals, `SimpleNamespace()`, or `nonzero()` will cause graph breaks, leading to either compilation failures or massive slowdowns. Extract such code into separate static/instance methods decorated with `@torch.compiler.disable`. Examples: `_compute_a2a_partition()` (partitioning logic), `_compute_halo_graph()` (AABB halo filtering + subset graph gen). The caller in `_generate_graph()` should be a simple `result = self._method(...)` call with no graph-break-inducing operations.

### SimpleNamespace doesn't support dict-style assignment
`types.SimpleNamespace` objects support attribute access (`obj.key`) but NOT dict-style assignment (`obj["key"] = val`). The `generate_graph()` function in `compute.py` does `data["node_partition"] = node_partition` for v2. When passing a SimpleNamespace as `data`, this crashes with `TypeError`. Fixed with try/except: try dict assignment, fall back to setattr.

### Speed benchmark YAML uses natoms_list, not num_atoms
The `uma-speed.yaml` config uses `runner.natoms_list: [1000]` (a list), not `runner.num_atoms`. The SLURM config uses `job.run_name` for job naming (not `job.scheduler.slurm.job_name`). Fields not in the YAML but in the SlurmConfig dataclass (like `timeout_hr`) need `+` prefix: `+job.scheduler.slurm.timeout_hr=1`. Fields not in SlurmConfig at all (like `time`, `job_name`) cannot be added — use the correct field names from the dataclass.

### V2 internal edge filtering is faster than post-filtering for send_info
Bypassing `radius_graph_pbc_v2`'s internal `node_partition` filtering to compute `send_info` via `filter_edges_by_node_partition` post-hoc is ~12ms SLOWER at 64 GPUs. The reason: without v2's internal filter, v2 generates edges for ALL 64k atoms instead of 1k local atoms, producing ~64× more edges that are then discarded. The 4.2ms `_fused_index_exchange` NCCL collective is a necessary cost when using v2 — it's cheaper than the alternative. Benchmarked: A2A+P2P dropped from 1.405 to 1.249 ns/day at 64 GPUs when bypassing v2's filter.

### AtomicData.get() requires explicit default argument
`AtomicData.get(key)` (without a `default` kwarg) raises `TypeError: AtomicData.get() missing 1 required positional argument: 'default'`. Unlike a regular Python dict where `.get(key)` defaults to `None`, `AtomicData.get()` mandates the second argument. Always use `data_dict.get("key", default=None)`. This only manifests in distributed runs where `data_dict` is an `AtomicData` object — unit tests using plain dicts pass fine.

### Dict flowing through torch.compile causes compilation hangs
Returning a dict from a `@torch.compiler.disable` function and having it flow through the compiled graph to another disabled function causes the compiler to hang indefinitely. The fix is to store the dict on `self` (instance attribute) inside the disabled function, then retrieve it in the next disabled function. Never return complex non-tensor types (dict, SimpleNamespace, etc.) from `@torch.compiler.disable` functions if they will be consumed later in the compiled region.

### Torchrun entrypoints must not call fairchem CLI main()
When launching via `torchrun` (e.g., profai-cli's SLURM submit), the entrypoint must NOT call `fairchem.core._cli.main()`. The CLI's `local_launch()` calls `distutils.setup_env_local()` which overwrites `RANK=0` and `LOCAL_RANK=0` for all processes, breaking multi-GPU setup. Instead, directly call `distutils.setup(dist_config)` with `submit=False, cpu=False` and `setup_graph_parallel_groups(world_size, "nccl")`, then instantiate the runner directly.

### profai-cli launch-experiment --gpus means per-node GPUs
The `--gpus` flag in `profai-cli launch-experiment` sets `--gpus-per-node` and `--nproc_per_node` in the generated SLURM script. For 64 total GPUs across 8 nodes, use `--gpus 8 --nodes 8`, NOT `--gpus 64 --nodes 8`.

### InferenceBenchRunner.job_config requires OmegaConf DictConfig
`Runner.job_config` uses a property descriptor (`TypedPropertyDescriptor`) that enforces the value must be `omegaconf.DictConfig`. Using `types.SimpleNamespace` or a plain dict raises `ValueError`. When constructing a runner outside the Hydra pipeline, use `OmegaConf.create({"metadata": {"results_dir": path}})` instead of `SimpleNamespace`.

### AABB send_info must compute BOTH send and recv counts symmetrically
When computing send_info from AABB geometry to replace `_fused_index_exchange`, you MUST compute both `send_counts` (local atoms → remote AABBs) AND `recv_counts` (remote atoms → our AABB). The all-to-all requires `rank_A.send_counts[B] == rank_B.recv_counts[A]`. If recv_counts is computed from edge_index (edge-based needed atoms) while send_counts comes from AABB, they won't match — AABB is conservative (includes false positives), so send_counts >= edge-based recv_counts. This mismatch deadlocks `all_to_all_single`. Since all ranks have the full position tensor, each rank can independently compute what every other rank will send to it (checking remote atoms against its own expanded AABB). This is purely local, requires no NCCL, and guarantees consistency.

### global_to_local must map needed_atoms in recv_buf (source-rank) order
The `global_to_local` mapping in `build_gp_context` assigns local indices to received atoms. These local indices index into `[x_local | recv_buf]`. Since `all_to_all` fills `recv_buf` by **source rank** (rank 0's data first, then rank 1's, etc.), `needed_atoms` MUST be sorted by source rank before assigning local indices. `nonzero()` returns indices sorted by global index, which does NOT match source-rank ordering with spatial partitioning. With `index_split`, global-index order == rank order (no bug), so this was latent until spatial partitioning was used. Without the sort, 99.9% of remote atom positions are mismatched, producing wrong embeddings silently (model runs but gives incorrect predictions). Fix: `sort_order = needed_from_ranks.argsort(stable=True); needed_atoms = needed_atoms[sort_order]` before the `global_to_local` assignment.

### Use functional collectives for compile-friendly distributed ops
`torch.distributed._functional_collectives` provides `all_to_all_single`, `all_gather_tensor`, etc. that are registered PyTorch ops — torch.compile can trace through them WITHOUT creating graph breaks (unlike `@torch.compiler.disable` on autograd function forward methods). This is critical for performance: each graph break prevents cross-operation fusion by the compiler. PyTorch 2.8+ has full support. The functional collectives accept `ProcessGroup` as the group argument and support both NCCL and Gloo backends. For eval mode (no autograd needed), use the non-autograd versions. For training, use `all_to_all_single_autograd`. CAVEAT: split sizes are passed as Python `list[int]`, and torch.compile specializes on these concrete values. If split sizes change (e.g., MD atoms crossing partition boundaries), the compiler recompiles. For benchmarks (constant structure), this is fine. For production MD, consider using padded equal-split all-to-all or `torch._dynamo.mark_dynamic` on the output.

### Compute/comm overlap via edge splitting does NOT work with torch.compile
Splitting `forward_chunk` into local edges + boundary edges to overlap the funcoll all_to_all with local edge computation causes -9.4% regression at 8 GPUs and 0% improvement at 64 GPUs. Root cause: torch.compile fuses ALL edges into one efficient kernel (triton/CUDA). Splitting into two `forward_chunk` calls prevents this fusion, creating double kernel launches and double memory allocation. The fused kernel is more valuable than hiding 1-5ms of communication. The inductor's `reorder_for_compute_comm_overlap` pass cannot compensate because the fundamental loss of fusion across all edges. For eSCN/UMA models, keep the sync funcoll path (one `all_to_all_collect_compiled` → one `forward_chunk`).

### PyTorch 2.8+ profiler API changes
`FunctionEventAvg.cuda_time_total` was renamed to `device_time_total` and `sort_by="cuda_time_total"` became `sort_by="self_device_time_total"`. Use `getattr(evt, "device_time_total", 0)` with fallback for compatibility. Also, `torch.compile` drops `record_function` annotations — only code inside `@torch.compiler.disable` regions shows up in the profiler breakdown. Non-compiled profiles show the full `record_function` hierarchy.

### Functional collectives break autograd — use AllToAllCollect for gradient forces
`all_to_all_collect_compiled` (functional collectives) does NOT participate in autograd. When UMA-S computes forces via `compute_forces_and_stress` (autograd gradient of energy w.r.t. positions), the gradient chain through the all-to-all communication is broken — remote atom embeddings are treated as constants, producing wrong forces (0.19 max diff vs 4.9e-7 with the fix). The fix: check `torch.is_grad_enabled() and x.requires_grad` and fall back to the autograd-compatible `all_to_all_collect` (which uses `AllToAllCollect` autograd.Function) when gradients are needed. This applies to both `Edgewise.forward()` and the overlap path. The BL (all-gather) path doesn't have this issue because `GatherFromModelParallelRegionSumGradPadded` IS an autograd.Function with proper backward.

### UMA-S uses MLP_EFS_Head, NOT Linear_Force_Head
UMA-S computes forces via autograd in `MLP_EFS_Head` (gradient of energy w.r.t. positions), NOT via direct prediction in `Linear_Force_Head`. The `DatasetSpecificMoEWrapper` wraps `MLP_EFS_Head`. When `direct_forces=False` (the default for UMA-S), the inference context uses `nullcontext()` (NOT `torch.no_grad()`), and `@conditional_grad(torch.enable_grad())` on the head's forward ensures autograd works. The `Linear_Force_Head` class exists but is used by other model configurations.

### profai-cli launch-experiment needs --partition for correct time limit
The profai-cli SLURM script generator queries `sinfo -h -o "%l"` for partition time limits. Without `--partition`, `sinfo` returns the DEFAULT partition's limit (often `infinite`), which sbatch rejects. Always pass `--partition h200` (or the appropriate partition) alongside `--qos` to get the correct time limit.
