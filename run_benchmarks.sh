#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/uma/benchmark/uma-speed.yaml"
LOG_DIR="${SCRIPT_DIR}/benchmark_logs"
CHECKPOINT="/checkpoint/ocp/shared/uma/release/uma_sm_osc_name_fix.pt"

MODES=(
    "general"
    "umas_fast_pytorch"
    #"triton_so2_and_rotate_in"
    "triton_so2_and_rotate_in_emit"
    #"triton_so2_and_rotate_in_all_fused"
    "triton_so2_and_rotate_in_outfused"
    #"triton_rotate_in"
    #"triton_gating"
    #"triton_scatter_add"
    #"triton_atomic"
    #"triton_pytorch_bwd"
    #"triton_recompute"
)

mkdir -p "${LOG_DIR}"

echo "Benchmark logs will be saved to: ${LOG_DIR}"
echo "Config: ${CONFIG}"
echo "Modes: ${MODES[*]}"
echo "=========================================="

# --- Speed benchmarks ---
for mode in "${MODES[@]}"; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running speed benchmark: ${mode}"
    echo "------------------------------------------"

    stdout_log="${LOG_DIR}/${mode}.stdout.log"
    stderr_log="${LOG_DIR}/${mode}.stderr.log"

    if fairchem -c "${CONFIG}" \
        "+runner.inference_settings.execution_mode=${mode}" \
        > "${stdout_log}" 2> "${stderr_log}"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${mode}: COMPLETED"
    else
        exit_code=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${mode}: FAILED (exit code ${exit_code})"
    fi

    echo "  stdout -> ${stdout_log}"
    echo "  stderr -> ${stderr_log}"
done

echo ""
echo "=========================================="
echo "Speed benchmarks finished."
echo "=========================================="

# --- Force comparison (no PBC) ---
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running force comparison across modes..."
echo "------------------------------------------"

force_log="${LOG_DIR}/force_comparison.log"

if python "${SCRIPT_DIR}/compare_forces.py" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${LOG_DIR}" \
    2>&1 | tee "${force_log}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Force comparison: COMPLETED"
else
    exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Force comparison: FAILED (exit code ${exit_code})"
fi

echo "  log -> ${force_log}"
echo ""
echo "=========================================="
echo "All done. Logs in: ${LOG_DIR}"
