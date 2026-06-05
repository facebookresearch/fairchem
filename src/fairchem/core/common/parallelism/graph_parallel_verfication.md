# A2A Graph Parallel — Test & Benchmark Runbook

All commands are run-ready. Copy-paste from the fairchem repo root:

```bash
cd /home/rgao/fairchem
source .venv/bin/activate
```

---

## 1. Correctness Tests

### 1a. GPContext unit tests (CPU, <1s)

```bash
pytest tests/core/common/parallelism/test_graph_parallel.py::TestBuildGPContext -v
```

5 tests: `test_basic_context_building`, `test_global_to_local_mapping`,
`test_no_cross_partition_edges`, `test_edge_split_indices`, `test_edge_split_no_remote_edges`.

### 1b. Distributed A2A primitives (CPU/Gloo, ~10s)

```bash
pytest tests/core/common/parallelism/test_graph_parallel.py -v -k "not TestBuildGPContext"
```

8 tests: forward equivalence (2 graph topologies), backward gradient, multi-rank (2 and 3),
spatial partition, compiled-vs-autograd (index_split + spatial).

### 1c. A2A embedding correctness (CPU/Gloo, ~30s)

```bash
pytest tests/core/common/parallelism/test_a2a_correctness.py -v
```

8 tests: correctness at 100/500 atoms × 2 strategies, consistency across graph sizes × 2,
1536-dim embeddings × 2.

### 1d. Full-model GPU correctness (8 GPUs, ~5 min)

```bash
torchrun --nproc_per_node=8 tests/core/common/parallelism/test_a2a_correctness.py --natoms 1000
```

Loads UMA-S checkpoint, runs BL vs A2A inference, compares energy/forces/stress (tol=1e-4).

### 1e. Predict pipeline + MD consistency (CPU, PR3 branch, ~2 min)

```bash
pytest tests/core/units/mlip_unit/test_predict.py::test_parallel_predict_unit_cpu -v -k "spatial or index_split"
pytest tests/core/units/mlip_unit/test_predict.py::test_parallel_predict_unit_gpu -v -k "spatial or index_split"
pytest tests/core/units/mlip_unit/test_predict.py::test_parallel_predict_unit_batch -v -k "spatial"
pytest tests/core/units/mlip_unit/test_predict.py::test_merge_mole_md_consistency -v -k "spatial or index_split"
```

### 1f. All correctness at once

```bash
pytest tests/core/common/parallelism/ -v
```

---

## 2. Regression Tests

### 2a. Existing tests must pass unchanged

```bash
pytest tests/core/common/test_gp_utils.py -v
pytest tests/core/models/uma/test_escn_md.py -v
pytest tests/core/models/uma/test_compile.py -v
pytest tests/core/components/test_uma_speed_benchmark.py -v
pytest tests/core/units/mlip_unit/test_predict.py::test_parallel_predict_unit_cpu -v -k "not spatial and not index_split"
```

### 2b. All regression at once

```bash
pytest tests/core/common/test_gp_utils.py tests/core/models/uma/test_escn_md.py tests/core/models/uma/test_compile.py tests/core/components/test_uma_speed_benchmark.py -v
```

---

## 3. Performance Benchmarks

All benchmarks use `InferenceBenchRunner` with FCC crystal systems (deterministic),
`timeiters=100`, `repeats=5`, turbo mode (compile=True, tf32=True).
Results written to `benchmark_results.json` in the run directory under `/checkpoint/ocp/rgao/speed/`.

### 3a. 8-GPU single-node (local, ~10 min each)

```bash
# BL baseline
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=local_8gpu \
  runner.natoms_list=[8000,32000] \
  runner.timeiters=100 \
  runner.repeats=5

# A2A + spatial
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=local_8gpu \
  runner.natoms_list=[8000,32000] \
  runner.timeiters=100 \
  runner.repeats=5 \
  '+runner.overrides={backbone: {use_all_to_all_gp: true, gp_partition_strategy: spatial}}'
```

Expected: A2A within ±5% of BL (NVLink hides comm cost).

### 3b. 16-GPU 2-node (SLURM, ~15 min each)

```bash
# BL baseline
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=2 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=16 \
  job.scheduler.slurm.qos=h200_dev \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[16000,64000] \
  runner.timeiters=100 \
  runner.repeats=5

# A2A + spatial
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=2 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=16 \
  job.scheduler.slurm.qos=h200_dev \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[16000,64000] \
  runner.timeiters=100 \
  runner.repeats=5 \
  '+runner.overrides={backbone: {use_all_to_all_gp: true, gp_partition_strategy: spatial}}'
```

Expected: A2A +2-3%.

### 3c. 32-GPU 4-node (SLURM, ~15 min each)

```bash
# BL baseline
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=4 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=32 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[32000,128000] \
  runner.timeiters=100 \
  runner.repeats=5

# A2A + spatial
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=4 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=32 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[32000,128000] \
  runner.timeiters=100 \
  runner.repeats=5 \
  '+runner.overrides={backbone: {use_all_to_all_gp: true, gp_partition_strategy: spatial}}'
```

Expected: A2A +5-10%.

### 3d. 64-GPU 8-node — CANONICAL BENCHMARK (SLURM, ~30 min each)

```bash
# BL baseline
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=8 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=64 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[64000,256000] \
  runner.timeiters=100 \
  runner.repeats=5

# A2A + spatial
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=8 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=64 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[64000,256000] \
  runner.timeiters=100 \
  runner.repeats=5 \
  '+runner.overrides={backbone: {use_all_to_all_gp: true, gp_partition_strategy: spatial}}'
```

Expected: A2A +10-20%.

### 3e. 128-GPU 16-node (SLURM, ~30 min each)

```bash
# BL baseline
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=16 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=128 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[128000,512000] \
  runner.timeiters=100 \
  runner.repeats=5

# A2A + spatial
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=16 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=128 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[128000,512000] \
  runner.timeiters=100 \
  runner.repeats=5 \
  '+runner.overrides={backbone: {use_all_to_all_gp: true, gp_partition_strategy: spatial}}'
```

Expected: A2A +30-40%.

### 3f. 256-GPU 32-node (SLURM, ~45 min each)

```bash
# BL baseline
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=32 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=256 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[256000,1024000] \
  runner.timeiters=100 \
  runner.repeats=5

# A2A + spatial
fairchem -c configs/uma/speed/uma-speed.yaml \
  job=slurm \
  job.scheduler.num_nodes=32 \
  job.scheduler.ranks_per_node=8 \
  job.graph_parallel_group_size=256 \
  job.scheduler.slurm.qos=h200_alignment_shared \
  job.scheduler.slurm.account=ocp \
  runner.natoms_list=[256000,1024000] \
  runner.timeiters=100 \
  runner.repeats=5 \
  '+runner.overrides={backbone: {use_all_to_all_gp: true, gp_partition_strategy: spatial}}'
```

Expected: A2A +60-80%.

---

## 4. Reading Results

```bash
# Find latest runs
ls -lt /checkpoint/ocp/rgao/speed/ | head -20

# Read a specific result
python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
ws = d['config']['world_size']
for model, data in d['model_to_qps_data'].items():
    print(f'Model: {model}, world_size={ws}')
    for natoms, ns_day in data:
        atoms_per_gpu = natoms // ws
        print(f'  {natoms:>7} atoms ({atoms_per_gpu}/gpu): {ns_day:.3f} ns/day')
" /checkpoint/ocp/rgao/speed/<RUN_DIR>/benchmark_results.json
```

---

## 5. Reference Results (H200 turbo)

### 4k atoms/rank (canonical):

| GPUs | BL ns/day | A2A ns/day | Speedup | BL WS Eff | A2A WS Eff |
|------|-----------|------------|---------|-----------|------------|
| 8 | 0.651 | 0.670 | +2.9% | 100% | 100% |
| 16 | 0.610 | 0.626 | +2.6% | 93.7% | 93.6% |
| 32 | 0.568 | 0.623 | +9.7% | 87.3% | 93.1% |
| 64 | 0.477 | 0.570 | +19.5% | 73.2% | 85.1% |
| 128 | 0.359 | 0.498 | +38.7% | 55.1% | 74.3% |
| 256 | 0.246 | 0.433 | +76.0% | 37.8% | 64.6% |

### 1k atoms/rank:

| GPUs | BL ns/day | A2A ns/day | Speedup |
|------|-----------|------------|---------|
| 8 | 2.088 | 2.010 | -3.7% |
| 16 | 2.043 | 1.990 | -2.6% |
| 32 | 1.769 | 1.848 | +4.5% |
| 64 | 1.488 | 1.626 | +9.3% |
