#!/usr/bin/env bash

# For the HIP jobs, which run in AMD's rocm/dev-ubuntu image: it already has
# the ROCm repository and rocm-dev, so only the libraries are installed here.
# The ROCm version is the image tag in ci.yml.

set -eu -o pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  rocrand-dev rocprim-dev hiprand-dev rocsparse-dev
apt-get clean

# The image does not put ROCm on PATH: add it for the CI job's later steps
# (each GITHUB_PATH entry is prepended, so llvm/bin goes last to come first)
export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:${PATH}
if [ -n "${GITHUB_PATH:-}" ]; then
  printf '%s\n' /opt/rocm/bin /opt/rocm/llvm/bin >> "${GITHUB_PATH}"
fi

which hipcc clang clang++
hipcc --version
hipconfig --full
