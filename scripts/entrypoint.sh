#!/bin/bash
set -euo pipefail

CONFIG_PATH="${ROMP_CONFIG_PATH:-/tmp/romp_job.yaml}"

echo "==> Generating config from environment..."
python /app/scripts/generate_config.py

echo "==> Starting ROMP..."
momp-run -p "$CONFIG_PATH"

echo "==> ROMP completed successfully."
