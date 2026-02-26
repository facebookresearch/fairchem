#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Benchmark script for umas_fast_gpu execution backend.
# Runs speed benchmarks + force correctness comparison for all 3 backends.
#
# Usage:
#   bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt
#   bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt --natoms 6000
#   bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt --speed-only
#   bash run_benchmarks.sh --checkpoint /path/to/checkpoint.pt --forces-only
# CHECKPOINT : /checkpoint/ocp/shared/uma/release/uma_sm_osc_name_fix.pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/benchmark_logs"

# Defaults
CHECKPOINT="/checkpoint/ocp/shared/uma/release/uma_sm_osc_name_fix.pt"
NATOMS=2000
TIMEITERS=40
REPEATS=5
SPEED_ONLY=false
FORCES_ONLY=false
COMPILE=false

# All execution backends to benchmark
MODES=(
    "umas_fast_gpu"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --natoms)
            NATOMS="$2"
            shift 2
            ;;
        --timeiters)
            TIMEITERS="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --speed-only)
            SPEED_ONLY=true
            shift
            ;;
        --forces-only)
            FORCES_ONLY=true
            shift
            ;;
        --compile)
            COMPILE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --checkpoint PATH [--natoms N] [--timeiters N] [--repeats N] [--log-dir DIR] [--speed-only] [--forces-only]"
            exit 1
            ;;
    esac
done

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Error: --checkpoint is required"
    echo "Usage: $0 --checkpoint PATH [--natoms N] [--timeiters N] [--repeats N] [--log-dir DIR] [--speed-only] [--forces-only]"
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "UMA Execution Backend Benchmarks"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Atoms:      ${NATOMS}"
echo "Modes:      ${MODES[*]}"
echo "Log dir:    ${LOG_DIR}"
echo "=========================================="

# --- Speed benchmarks ---
if [[ "${FORCES_ONLY}" == "false" ]]; then
    echo ""
    echo "=========================================="
    echo "Speed Benchmarks (timeiters=${TIMEITERS}, repeats=${REPEATS})"
    echo "=========================================="

    CONFIG="${SCRIPT_DIR}/configs/uma/benchmark/uma-speed.yaml"

    for mode in "${MODES[@]}"; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Speed benchmark: ${mode}"
        echo "------------------------------------------"

        stdout_log="${LOG_DIR}/speed_${mode}.stdout.log"
        stderr_log="${LOG_DIR}/speed_${mode}.stderr.log"
        run_dir="/tmp/uma_speed_bench/${mode}"
        mkdir -p "${run_dir}"

        if fairchem -c "${CONFIG}" \
            "job.scheduler.mode=LOCAL" \
            "job.scheduler.num_nodes=1" \
            "job.scheduler.ranks_per_node=1" \
            "job.graph_parallel_group_size=null" \
            "job.run_dir=${run_dir}" \
            "job.run_name=${mode}" \
            "runner.timeiters=${TIMEITERS}" \
            "runner.repeats=${REPEATS}" \
            "runner.natoms_list=[${NATOMS}]" \
            "runner.model_checkpoints.uma_sm_cons=${CHECKPOINT}" \
            "runner.inference_settings.tf32=True" \
            "runner.inference_settings.activation_checkpointing=False" \
            "runner.inference_settings.merge_mole=True" \
            "runner.inference_settings.compile=${COMPILE}" \
            "runner.inference_settings.external_graph_gen=True" \
            "runner.inference_settings.internal_graph_gen_version=2" \
            "+runner.inference_settings.execution_mode=${mode}" \
            "runner.generate_traces=False" \
            > "${stdout_log}" 2> "${stderr_log}"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${mode}: COMPLETED"
            # Extract key metrics from logs
            if grep -q "qps:" "${stderr_log}"; then
                grep "qps:" "${stderr_log}" | tail -1
            fi
            if grep -q "Peak CUDA memory" "${stderr_log}"; then
                grep "Peak CUDA memory" "${stderr_log}" | tail -1
            fi
        else
            exit_code=$?
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${mode}: FAILED (exit code ${exit_code})"
            echo "  Check ${stderr_log} for details"
        fi

        echo "  stdout -> ${stdout_log}"
        echo "  stderr -> ${stderr_log}"
    done

    echo ""
    echo "=========================================="
    echo "Speed benchmarks finished."
    echo "=========================================="
fi

# --- Force comparison ---
if [[ "${SPEED_ONLY}" == "false" ]]; then
    echo ""
    echo "=========================================="
    echo "Force Correctness Comparison"
    echo "=========================================="

    force_log="${LOG_DIR}/force_comparison.log"

    COMPILE_FLAG=""
    if [[ "${COMPILE}" == "true" ]]; then
        COMPILE_FLAG="--compile"
    fi

    if python "${SCRIPT_DIR}/compare_forces.py" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${LOG_DIR}" \
        --natoms "${NATOMS}" \
        ${COMPILE_FLAG} \
        2>&1 | tee "${force_log}"; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Force comparison: COMPLETED"
    else
        exit_code=$?
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Force comparison: FAILED (exit code ${exit_code})"
    fi

    echo "  log -> ${force_log}"
fi

echo ""
echo "=========================================="
echo "All done. Logs in: ${LOG_DIR}"
echo "=========================================="
