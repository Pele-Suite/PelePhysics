#!/usr/bin/env bash

set -eu -o pipefail

sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel

# Make the oneAPI environment available to the CI job's later steps: bash
# sources BASH_ENV at the start of every non-interactive shell. setvars.sh
# exits non-zero on harmless conditions, hence the || true.
if [ -n "${GITHUB_ENV:-}" ]; then
  echo 'source /opt/intel/oneapi/setvars.sh > /dev/null || true' | sudo tee /opt/intel/oneapi/ci-env.sh > /dev/null
  echo "BASH_ENV=/opt/intel/oneapi/ci-env.sh" >> "${GITHUB_ENV}"
fi
