#!/usr/bin/env bash
set -euo pipefail
uv run python src/crawl_interiors.py "$@"
