FROM python:3.12-slim

# Install git (required for setuptools-scm version detection)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy fairchem repo into /app/fairchem (including .git for version detection)
COPY . /app/fairchem

# Create virtual environment
RUN uv venv --python 3.12 fc_env

# Add Python to PATH
ENV PATH="/app/fc_env/bin:$PATH"

# Install fairchem-core with dev dependencies
RUN uv pip install --python fc_env/bin/python -e fairchem/packages/fairchem-core[dev]

# Verify installation
RUN fc_env/bin/python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Default command
CMD ["bash"]
