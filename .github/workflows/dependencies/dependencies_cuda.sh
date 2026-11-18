#!/usr/bin/env bash

set -eu -o pipefail

CUDA_VERSION=12-6

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y \
    cuda-command-line-tools-${CUDA_VERSION} \
    cuda-compiler-${CUDA_VERSION}           \
    cuda-cupti-dev-${CUDA_VERSION}          \
    cuda-minimal-build-${CUDA_VERSION}      \
    cuda-nvml-dev-${CUDA_VERSION}           \
    cuda-nvtx-${CUDA_VERSION}               \
    libcurand-dev-${CUDA_VERSION}           \
    libcusolver-dev-${CUDA_VERSION}         \
    libcusparse-dev-${CUDA_VERSION}         \
    libcublas-dev-${CUDA_VERSION}           \
    libnvjitlink-${CUDA_VERSION}

CUDA_HOME=/usr/local/cuda-${CUDA_VERSION/-/.}
export PATH=${CUDA_HOME}/bin:${PATH}
which nvcc
nvcc --version

# Make the toolkit available to the CI job's later steps
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "CUDA_HOME=${CUDA_HOME}" >> "${GITHUB_ENV}"
  echo "LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> "${GITHUB_ENV}"
  echo "${CUDA_HOME}/bin" >> "${GITHUB_PATH}"
fi
