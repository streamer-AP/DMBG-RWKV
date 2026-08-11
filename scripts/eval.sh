#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/acdc.json}"
shift || true

python scripts/run_config.py eval "${CONFIG}" "$@"
