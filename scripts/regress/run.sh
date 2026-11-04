#!/bin/bash
# usage: run.sh <out_root> [repo_dir]
# Runs the four regression slices (configs/slice_*.yaml) with outputs under <out_root>/<slice>/ and logs in
# <out_root>/<slice>.log. repo_dir is the checkout whose hcwf package is run (default: the repo of this script).
# Environment:
#   PY                   Python interpreter (default: <repo>/.venv/bin/python)
#   HCWF_REGRESS_DATA    root of the zarr stores, holding india/ and ethiopia/ (default: <repo>/data)
#   ALMANAC_DATASETS     root of the NetCDF datasets (default: /home/hcwf/code/ai-almanac/datasets)
# Last line: "RUN OK" (exit 0) or "RUN FAILED: <slices>" (exit 1).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: run.sh <out_root> [repo_dir]}
REPO=$(cd "${2:-$HERE/../..}" && pwd)
MAIN=$(cd "$HERE/../.." && pwd)
PY=${PY:-$MAIN/.venv/bin/python}
ZARR_ROOT=${HCWF_REGRESS_DATA:-$MAIN/data}
ALMANAC=${ALMANAC_DATASETS:-/home/hcwf/code/ai-almanac/datasets}
SLICES="slice_c_eth_aifsens slice_d_eth_aifs slice_b_india_graphcast slice_a_india_aifsens"

mkdir -p "$OUT/cfg"
OUT=$(cd "$OUT" && pwd)
failed=""
for c in $SLICES; do
  sed -e "s#@ZARR_ROOT@#$ZARR_ROOT#g" -e "s#@ALMANAC_DATASETS@#$ALMANAC#g" -e "s#@OUT@#$OUT#g" \
    "$HERE/configs/$c.yaml" > "$OUT/cfg/$c.yaml"
  ( cd "$REPO" && PYTHONPATH="$REPO" /usr/bin/time -f "$c %e s %M KB" "$PY" -c \
      "from hcwf.cli import main; main()" -p "$OUT/cfg/$c.yaml" --quiet ) > "$OUT/$c.log" 2>&1 \
    || { echo "$c FAILED (see $OUT/$c.log)"; failed="$failed $c"; }
  tail -1 "$OUT/$c.log"
done
if [ -n "$failed" ]; then echo "RUN FAILED:$failed"; exit 1; fi
echo "RUN OK"
