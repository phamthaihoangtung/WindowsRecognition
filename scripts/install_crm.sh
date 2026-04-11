#!/bin/bash
set -e

git submodule update --init crm
uv sync
