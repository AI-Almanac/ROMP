#!/bin/bash
set -euo pipefail

CONFIG_PATH="${HCWF_CONFIG_PATH:-/tmp/hcwf_job.yaml}"

echo "==> Generating config from environment..."
python /app/scripts/generate_config.py

echo "==> Starting HCWF..."
hcwf-run -p "$CONFIG_PATH"

echo "==> HCWF completed successfully."
