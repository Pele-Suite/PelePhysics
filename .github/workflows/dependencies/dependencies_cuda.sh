#!/usr/bin/env bash
#
# Copyright 2020 Axel Huebl
#
# License: BSD-3-Clause-LBNL

# search recursive inside a folder if a file contains tabs
#
# @result 0 if no files are found, else 1
#

set -eu -o pipefail

CUDA_VERSION=11-2

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" \
    | sudo tee /etc/apt/sources.list.d/cuda.list
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
    libcurand-dev-${CUDA_VERSION}

export PATH=/usr/local/cuda-11.2/bin:${PATH}
which nvcc
nvcc --version
