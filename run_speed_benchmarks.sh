#!/bin/bash
# UMA Speed Benchmark: 6000 atoms, all execution modes
# Usage: bash run_speed_benchmarks.sh
#
# Runs the UMA speed benchmark for each execution mode:
#   1. general (baseline PyTorch)
#   2. umas_fast_pytorch (block-diagonal SO2 GEMM)
#   3. triton_scatter_add (Triton, scatter_add backward)
#   4. triton_atomic (Triton, atomic backward)
#   5. triton_pytorch_bwd (Triton fwd, PyTorch bwd)
#   6. triton_recompute (Triton, memory-optimized recompute)

set -e

cd /home/misko/env/feb17_speed_refactor/fairchem_pr4_triton_kernels_repo

CHECKPOINT="/checkpoint/ocp/shared/uma/release/uma_sm_osc_name_fix.pt"

# Common args shared by all runs
COMMON_ARGS=(
    "-c" "configs/uma/benchmark/uma-speed.yaml"
    "job.scheduler.mode=LOCAL"
    "job.scheduler.num_nodes=1"
    "job.scheduler.ranks_per_node=1"
    "job.graph_parallel_group_size=1"
    "runner.timeiters=100"
    "runner.repeats=5"
    "runner.natoms_list=[6000]"
    "runner.model_checkpoints.uma_sm_cons=${CHECKPOINT}"
    "runner.inference_settings.tf32=True"
    "runner.inference_settings.activation_checkpointing=False"
    "runner.inference_settings.merge_mole=True"
    "runner.inference_settings.compile=True"
    "runner.inference_settings.external_graph_gen=False"
    "runner.inference_settings.internal_graph_gen_version=2"
    "runner.generate_traces=False"
)

MODES=(
    "general"
    "umas_fast_pytorch"
    "triton_scatter_add"
    "triton_atomic"
    "triton_pytorch_bwd"
    "triton_recompute"
)

for MODE in "${MODES[@]}"; do
    echo "============================================"
    echo "Running benchmark: ${MODE}"
    echo "============================================"

    RUN_DIR="/tmp/uma_speed_bench/${MODE}"
    mkdir -p "${RUN_DIR}"

    fairchem \
        "${COMMON_ARGS[@]}" \
        "job.run_dir=${RUN_DIR}" \
        "job.run_name=${MODE}" \
        "runner.inference_settings.execution_mode=${MODE}"

    echo ""
    echo "Results saved to: ${RUN_DIR}"
    echo ""
done

echo "============================================"
echo "All benchmarks complete. Results in /tmp/uma_speed_bench/"
echo "============================================"
