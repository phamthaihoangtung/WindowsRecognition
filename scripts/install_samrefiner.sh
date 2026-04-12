#!/bin/bash
set -e

git submodule update --init samrefiner
uv sync
