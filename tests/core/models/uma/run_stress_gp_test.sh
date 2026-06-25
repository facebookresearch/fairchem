#!/bin/bash
#
# Runner script for stress GP correctness test.
# Tests 1, 2, 4, 8 GPUs on a single node.
# For 16 GPUs (2 nodes), use SLURM with the appropriate multi-node launcher.
#
# Usage:
#   bash tests/core/models/uma/run_stress_gp_test.sh [num_molecules]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
PYTHON="${PYTHON:-python}"
TEST_SCRIPT="tests/core/models/uma/test_stress_gp.py"
NUM_MOLECULES="${1:-100}"
REF_FILE=$(mktemp /tmp/stress_ref_XXXXXX.pt)

cd "$REPO_DIR"

echo "============================================================"
echo "Stress GP Correctness Test"
echo "Num molecules: ${NUM_MOLECULES} ($(( NUM_MOLECULES * 3 )) atoms)"
echo "============================================================"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Available GPUs: $NUM_GPUS"

FAIL=0

# Generate reference (1 GPU, no GP)
echo ""
echo "--- Generate reference (1 GPU, no GP) ---"
CUDA_VISIBLE_DEVICES=0 $PYTHON -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29500 \
    "$TEST_SCRIPT" \
    --save-ref "$REF_FILE" --num-molecules "$NUM_MOLECULES" --gp-size 1

if [ ! -f "$REF_FILE" ]; then
    echo "FATAL: Reference file not created!"
    exit 1
fi

# Test each GPU count
for GP_SIZE in 2 4 8; do
    if [ "$GP_SIZE" -gt "$NUM_GPUS" ]; then
        echo ""
        echo "--- Skipping ${GP_SIZE} GPUs (only ${NUM_GPUS} available) ---"
        continue
    fi

    DEVICES=$(seq -s, 0 $((GP_SIZE - 1)))

    echo ""
    echo "--- Test with ${GP_SIZE} GPUs ---"
    CUDA_VISIBLE_DEVICES=$DEVICES $PYTHON -m torch.distributed.run \
        --nproc_per_node=$GP_SIZE --master_port=29500 \
        "$TEST_SCRIPT" \
        --ref-file "$REF_FILE" --num-molecules "$NUM_MOLECULES" --gp-size "$GP_SIZE" || FAIL=1
done

rm -f "$REF_FILE"

echo ""
echo "============================================================"
if [ $FAIL -eq 0 ]; then
    echo "ALL STRESS GP TESTS PASSED!"
else
    echo "SOME STRESS GP TESTS FAILED!"
fi
echo "============================================================"

exit $FAIL
