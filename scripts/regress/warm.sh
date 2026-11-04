#!/bin/bash
# usage: warm.sh <out_root> [repo_dir]
# After run.sh: re-runs each slice on its warm cache and compares the result with the cold outputs.
# Environment: PY (default: <repo>/.venv/bin/python). Needs the configs run.sh wrote under <out_root>/cfg.
# Last line: "WARM OK" (exit 0) or "WARM FAILED: <slices>" (exit 1: a run failed or its output changed).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(cd "${1:?usage: warm.sh <out_root> [repo_dir]}" && pwd)
REPO=$(cd "${2:-$HERE/../..}" && pwd)
PY=${PY:-$HERE/../../.venv/bin/python}
failed=""
for c in slice_c_eth_aifsens slice_d_eth_aifs slice_b_india_graphcast slice_a_india_aifsens; do
  rm -rf "$OUT/$c.cold" && mkdir "$OUT/$c.cold" && cp "$OUT/$c"/*.nc "$OUT/$c"/*.csv "$OUT/$c.cold/" 2>/dev/null
  ( cd "$REPO" && PYTHONPATH="$REPO" /usr/bin/time -f "$c warm %e s %M KB" "$PY" -c \
      "from hcwf.cli import main; main()" -p "$OUT/cfg/$c.yaml" --overwrite ) > "$OUT/$c.warm.log" 2>&1 \
    || { echo "$c FAILED (see $OUT/$c.warm.log)"; failed="$failed $c"; continue; }
  tail -1 "$OUT/$c.warm.log"
  echo "  cache: $(grep -ci ' hit' "$OUT/$c.warm.log") hit lines, $(grep -ci ' miss' "$OUT/$c.warm.log") miss lines"
  "$PY" "$HERE/compare_tol.py" "$OUT/$c.cold" "$OUT/$c" | grep -v ' SAME$'
  [ "${PIPESTATUS[0]}" = 0 ] || failed="$failed $c"
done
if [ -n "$failed" ]; then echo "WARM FAILED:$failed"; exit 1; fi
echo "WARM OK"
