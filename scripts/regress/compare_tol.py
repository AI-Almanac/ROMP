"""Compare two output folders: discrete values exact, floats within rtol (default 1e-9); NaN equals NaN.

usage: compare_tol.py <base_dir> <new_dir>

Compares every .nc and .csv file of <base_dir> with the file of the same name in <new_dir>. Prints one line per file
(``  name: SAME`` or the differences), then ``compared N files`` and a verdict: ``ALL SAME`` (exit 0) or
``DIFFERENCES FOUND`` (exit 1; also when no file was compared, or a file is missing from <new_dir>). Environment:
RTOL, the relative tolerance for floats.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

RTOL = float(os.environ.get("RTOL", "1e-9"))


def same(x, y):
    """Return None if the arrays agree (floats within RTOL, rest exactly, NaN equal NaN), else a description."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return f"shape {x.shape} vs {y.shape}"
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        x, y = x.astype(float), y.astype(float)
        bad = ~(np.isclose(x, y, rtol=RTOL, atol=0.0) | (np.isnan(x) & np.isnan(y)))
        if bad.any():
            rel = np.nanmax(np.abs(x - y)[bad] / np.maximum(np.abs(y[bad]), 1e-300))
            return f"{int(bad.sum())}/{bad.size} floats beyond rtol {RTOL:g}, max rel {rel:.3g}"
        return None
    eq = (x == y) | (pd.isna(x) & pd.isna(y)) if x.dtype.kind in "mM" else (x == y)
    return None if np.all(eq) else f"{int((~eq).sum())}/{eq.size} discrete values differ"


def compare_netcdf(path_a, path_b):
    """Return the list of differences between two NetCDF files (variables and coordinates)."""
    diffs = []
    with xr.open_dataset(path_a) as da, xr.open_dataset(path_b) as db:
        for v in list(da.data_vars) + list(da.coords):
            if v not in db:
                diffs.append(f"{v} missing")
                continue
            d = same(da[v].values, db[v].values)
            if d:
                diffs.append(f"{v}: {d}")
    return diffs


def compare_csv(path_a, path_b):
    """Return the list of differences between two CSV files, column by column."""
    ta, tb = pd.read_csv(path_a), pd.read_csv(path_b)
    if list(ta.columns) != list(tb.columns) or len(ta) != len(tb):
        return ["columns or rows differ"]
    return [f"{c}: {d}" for c in ta.columns if (d := same(ta[c].values, tb[c].values))]


def main(a_dir, b_dir):
    """Compare the folders, print the report and return the exit status."""
    compared, bad = 0, False
    for fa in sorted(glob.glob(f"{a_dir}/*.nc") + glob.glob(f"{a_dir}/*.csv")):
        name = os.path.basename(fa)
        fb = f"{b_dir}/{name}"
        if not os.path.exists(fb):
            print(f"  MISSING in new: {name}")
            bad = True
            continue
        diffs = compare_netcdf(fa, fb) if name.endswith(".nc") else compare_csv(fa, fb)
        compared += 1
        bad = bad or bool(diffs)
        print(f"  {name}: " + ("SAME" if not diffs else "; ".join(diffs)))
    print(f"  compared {compared} files")
    if compared == 0:
        print("  no files compared")
        bad = True
    print("DIFFERENCES FOUND" if bad else "ALL SAME")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
