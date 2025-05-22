#!/bin/bash

set -euo pipefail

ENV_NAME="cs224r-project"
PYTHON_VERSION="3.10"

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
    echo "Conda is not installed. Please install Miniconda or Anaconda first from https://docs.conda.io/en/latest/miniconda.html"
    exit 1
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