#!/bin/bash
# usage: cmp.sh <base_out> <new_out>
# Compares the slice outputs of two run.sh out roots with compare_tol.py (floats rtol 1e-9, discrete values exact),
# printing anything not SAME per slice and the number of files compared.
# Last line: "ALL SAME" (exit 0) or "DIFFERENCES FOUND" (exit 1; also when a slice compared no files).
# Environment: PY (default: <repo>/.venv/bin/python), RTOL (default 1e-9).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HERE/../../.venv/bin/python}
[ $# -eq 2 ] || { echo "usage: cmp.sh <base_out> <new_out>"; exit 2; }
bad=0
for c in slice_c_eth_aifsens slice_d_eth_aifs slice_b_india_graphcast slice_a_india_aifsens; do
  echo "== $c"
  out=$("$PY" "$HERE/compare_tol.py" "$1/$c" "$2/$c" 2>&1); status=$?
  echo "$out" | grep -v ' SAME$'
  [ $status = 0 ] || bad=1
done
if [ $bad = 0 ]; then echo "ALL SAME"; else echo "DIFFERENCES FOUND"; exit 1; fi
