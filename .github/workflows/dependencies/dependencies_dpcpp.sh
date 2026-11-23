#!/usr/bin/env bash

# The oneAPI compiler and MKL from Intel's apt repository, as AMReX's
# dependencies_dpcpp.sh installs them (Intel's basekit and hpckit images are
# larger than this download).

set -eu -o pipefail

# Retry failed downloads (man apt.conf)
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo mkdir --parents --mode=0755 /etc/apt/keyrings
curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /etc/apt/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update

# Retry the install too, against connection drops
status=1
for itry in {1..5}; do
  sudo apt-get install -y --no-install-recommends \
    intel-oneapi-compiler-dpcpp-cpp \
    intel-oneapi-mkl-devel \
    && { sudo apt-get clean; status=0; break; } \
    || sleep 10
done
if [ ${status} -ne 0 ]; then exit 1; fi

# Make the oneAPI environment available to the CI job's later steps: bash
# sources BASH_ENV at the start of every non-interactive shell. setvars.sh
# exits non-zero on harmless conditions, hence the || true.
if [ -n "${GITHUB_ENV:-}" ]; then
  echo 'source /opt/intel/oneapi/setvars.sh > /dev/null || true' | sudo tee /opt/intel/oneapi/ci-env.sh > /dev/null
  echo "BASH_ENV=/opt/intel/oneapi/ci-env.sh" >> "${GITHUB_ENV}"
fi
