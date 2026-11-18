#!/usr/bin/env bash

set -eu -o pipefail

sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.1 noble main" | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600
echo 'export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin:$PATH' | sudo tee -a /etc/profile.d/rocm.sh
sudo apt-get update
sudo apt-get install -y rocm-dev rocrand-dev rocprim-dev hiprand-dev rocsparse-dev

source /etc/profile.d/rocm.sh
which hipcc
which clang
which clang++
which flang
hipcc --version
hipconfig --full

# Make the toolchain available to the CI job's later steps (each GITHUB_PATH
# entry is prepended, so add them in reverse to keep rocm.sh's order)
if [ -n "${GITHUB_PATH:-}" ]; then
  printf '%s\n' /opt/rocm/opencl/bin /opt/rocm/profiler/bin /opt/rocm/bin /opt/rocm/llvm/bin >> "${GITHUB_PATH}"
fi
