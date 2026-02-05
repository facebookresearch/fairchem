#!/bin/bash
# =============================================================================
# run-docker-tests.sh
# Runs fairchem tests in a Docker container with bind mounts
# All dependencies are installed at runtime and cached for subsequent runs
# =============================================================================

set -e  # Exit on error

# Get the directory where this script is located (fairchem repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
IMAGE_NAME="fairchem-dev"
VENV_DIR="${SCRIPT_DIR}/.docker-venv"
PIP_CACHE_DIR="${SCRIPT_DIR}/.docker-pip-cache"

# Create cache directories if they don't exist
mkdir -p "${VENV_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

# Default: run all core tests
# You can override by passing arguments: ./run-docker-tests.sh tests/core/test_specific.py
#TEST_PATH="${1:-tests/core}"

echo "=============================================="
echo "Running fairchem in Docker container"
echo "=============================================="
echo "Image:      ${IMAGE_NAME}"
#echo "Test path:  ${TEST_PATH}"
echo "Venv cache: ${VENV_DIR}"
echo "Pip cache:  ${PIP_CACHE_DIR}"
echo "=============================================="

docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e PIP_CACHE_DIR=/tmp/pip-cache \
  --mount type=bind,src="${SCRIPT_DIR}",dst=/app/fairchem \
  --mount type=bind,src="${PIP_CACHE_DIR}",dst=/tmp/pip-cache \
  --mount type=bind,src="${VENV_DIR}",dst=/opt/venv \
  "${IMAGE_NAME}" \
  bash -c "
    # Create venv if it doesn't exist
    if [ ! -f /opt/venv/bin/python ]; then
      echo '>>> Creating virtual environment...'
      python -m venv /opt/venv
    fi

    # Install/update fairchem-core with dev and extras dependencies
    echo '>>> Installing fairchem-core[dev,extras]...'
    /opt/venv/bin/pip install -e '/app/fairchem/packages/fairchem-core[dev,extras]'
  "
