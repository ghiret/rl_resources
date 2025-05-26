# Base image
ARG CUDA_VERSION="12.6.2"
ARG UBUNTU_VERSION="24.04"
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV VENV_PATH=/home/ubuntu/venv

# Install base dependencies and Python
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git gnupg ca-certificates sudo python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev ffmpeg imagemagick && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && mv /root/.local/bin/uv /usr/local/bin/

# Switch to ubuntu user
USER ubuntu
WORKDIR /tmp/project_deps

# Create venv in user's home directory
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy project files
COPY --chown=ubuntu:ubuntu pyproject.toml ./
COPY --chown=ubuntu:ubuntu src ./src

# Install Python dependencies using uv into venv
RUN UV_HTTP_TIMEOUT=600 uv pip install --python $VENV_PATH/bin/python --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio && \
    uv pip install --python $VENV_PATH/bin/python --no-cache-dir ".[dev]" && \
    rm -rf /tmp/project_deps/*

# Set workspace directory
WORKDIR /workspaces/rl_resources

# Default command
CMD [ "sleep", "infinity" ]
