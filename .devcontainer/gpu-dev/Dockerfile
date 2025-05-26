# Dockerfile for Python 3.11, CUDA, uv, and pre-installed project dependencies

ARG CUDA_VERSION="12.4.1"
ARG UBUNTU_VERSION="22.04"
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Arguments for non-root user creation
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION="3.11"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"
# Path to the shared virtual environment
ENV VENV_PATH=/opt/venv

# Install prerequisites, Python 3.11, global uv, create user, and venv structure as root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    gnupg \
    ca-certificates \
    sudo && \
    # Add deadsnakes PPA for newer Python versions
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev && \
    # Set python3 to point to python3.11
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    # Install pip for Python 3.11
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && \
    # Install/Upgrade pip and install uv system-wide
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir uv && \
    # Create the virtual environment using the installed Python and global uv
    uv venv $VENV_PATH --python /usr/bin/python${PYTHON_VERSION} && \
    # Create non-root user and grant sudo privileges
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME --shell /bin/bash && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Add user to video group (optional)
    usermod -aG video $USERNAME && \
    # Clean up apt caches now
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick

# Create a temporary directory for installing project dependencies
WORKDIR /tmp/project_deps

COPY /pyproject.toml ./
COPY /uv.lock ./
#

# Install project dependencies (including 'dev' extras) into the shared venv using global uv
# This uses the python from the venv to ensure dependencies are installed there.
# The ".[dev]" syntax assumes your pyproject.toml is set up for this.
RUN uv pip install --python $VENV_PATH/bin/python --no-cache-dir ".[dev]" && \
    # Clean up the temporary directory after installation
    rm -rf /tmp/project_deps/*

# Verify installations (as root, uv should be global, nvcc from base)
RUN echo "Python version (system): $(python3 --version)" && \
    echo "Pip version (system): $(pip --version)" && \
    echo "uv version (global): $(uv --version)" && \
    echo "NVCC version: $(nvcc --version)" && \
    echo "Venv Python version: $($VENV_PATH/bin/python --version)" && \
    echo "Venv Pip version: $($VENV_PATH/bin/pip --version)"

# Switch to the non-root user
USER $USERNAME

# Add the venv's bin directory to the PATH for the non-root user
# This makes `python`, `pip`, and any installed CLIs from dependencies available directly.
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Set a default working directory for the user
WORKDIR /home/$USERNAME

# Default command to keep the container running.
CMD ["sleep", "infinity"]
