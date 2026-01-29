FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy fairchem repo into /app/fairchem
COPY . /app/fairchem

# Add Python to PATH
ENV PATH="/app/fc_env/bin:$PATH"

# Create venv and install fairchem at runtime (RUN commands don't work on this cluster)
CMD ["sh", "-c", "\
    echo 'Creating virtual environment...' && \
    uv venv --python 3.12 fc_env && \
    echo 'Installing fairchem...' && \
    uv pip install --python fc_env/bin/python -e fairchem/packages/fairchem-core[dev] && \
    echo '\nInstallation complete! Testing PyTorch...' && \
    fc_env/bin/python -c \"import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')\" \
"]
