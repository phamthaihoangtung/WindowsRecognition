#!/usr/bin/env bash
set -euo pipefail
uv run python utils/crawl_interiors.py "$@"
