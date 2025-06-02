#!/bin/bash

set -euo pipefail

ENV_NAME="cs224r-project"
PYTHON_VERSION="3.12"

PLATFORM=$(uname -s)
ARCH=$(uname -m)

if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    CONDA_SUBDIR="osx-arm64"
    PYTORCH_INSTALL_COMMAND="conda install -y -c pytorch pytorch torchvision torchaudio"
    MPS_SUPPORT=true
else
    CONDA_SUBDIR=""
    if command -v nvidia-smi &> /dev/null; then
        PYTORCH_INSTALL_COMMAND="conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1"
    else
        PYTORCH_INSTALL_COMMAND="conda install -y -c pytorch pytorch torchvision torchaudio cpuonly"
    fi
    MPS_SUPPORT=false
fi

if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Installing Miniconda..."
    
    if [[ "$PLATFORM" == "Linux" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$PLATFORM" == "Darwin" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        echo "Unsupported platform: $PLATFORM"
        exit 1
    fi
    
    INSTALLER_NAME="miniconda_installer.sh"
    echo "Downloading Miniconda installer..."
    curl -L "$MINICONDA_URL" -o "$INSTALLER_NAME"
    
    echo "Installing Miniconda..."
    bash "$INSTALLER_NAME" -b -p "$HOME/miniconda3"
    rm "$INSTALLER_NAME"
    
    echo "Initializing conda..."
    "$HOME/miniconda3/bin/conda" init bash
    
    export PATH="$HOME/miniconda3/bin:$PATH"
    source "$HOME/.bashrc" 2>/dev/null || true
    
    echo "Conda installation completed!"
else
    echo "Conda is already installed."
fi

if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    eval "${CONDA_SUBDIR:+CONDA_SUBDIR=$CONDA_SUBDIR }conda create -n $ENV_NAME python=$PYTHON_VERSION -y"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if [[ "$MPS_SUPPORT" == true ]]; then
    echo "Configuring conda environment for $CONDA_SUBDIR..."
    conda config --env --set subdir "$CONDA_SUBDIR"
fi

eval "$PYTORCH_INSTALL_COMMAND"
"$CONDA_PREFIX/bin/python" -m pip install -r requirements.txt --no-cache-dir
"$CONDA_PREFIX/bin/python" -m pip install -e . --no-cache-dir