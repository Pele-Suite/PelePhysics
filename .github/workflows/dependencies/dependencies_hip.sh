#!/usr/bin/env bash
#
# Copyright 2020 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

# search recursive inside a folder if a file contains tabs
#
# @result 0 if no files are found, else 1
#

set -eu -o pipefail

sudo wget http://repo.radeon.com/rocm/rocm.gpg.key
sudo apt-key add rocm.gpg.key
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ ubuntu main' \
  | sudo tee /etc/apt/sources.list.d/rocm.list
echo 'export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin:$PATH' \
  | sudo tee -a /etc/profile.d/rocm.sh

sudo apt-get update
sudo apt-get install -y \
    rocm-dev        \
    roctracer-dev   \
    rocprofiler-dev \
    rocrand-dev     \
    rocprim-dev     \
    rocm-libs

source /etc/profile.d/rocm.sh
which hipcc
which clang
which clang++
which flang
hipcc --version
