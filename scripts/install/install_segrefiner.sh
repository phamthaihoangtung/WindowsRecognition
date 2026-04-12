#!/bin/bash
set -e

git submodule update --init segrefiner

# mmcv 2.x requires CUDA dev headers to compile ops:
#   sudo apt-get install -y libcusparse-dev-12-6 libcublas-dev-12-6 libcurand-dev-12-6 libcufft-dev-12-6 libcusolver-dev-12-6
uv pip install --no-build-isolation mmcv mmengine
uv pip install --no-build-isolation -e segrefiner/
