"""A small deterministic dataset and flat-key run configs for the end-to-end check of `hcwf-run convert`.

The configs use the flat onset keys (wet_init, wet_spell, ..., earliest_onset_date, thresh_file) that the old
"legacy mode" read. The outputs of that mode for these configs were produced once, from the code of commit 1ffc0cb^
(the last one that still had the mode), and are committed under tests/data/legacy_reference/. `test_legacy_reference.py`
converts each config with today's converter, runs it and compares every output with them.

To make the references again: check out 1ffc0cb^ in a worktree and run
`PYTHONPATH=<worktree> python tests/data/legacy_reference/make_reference.py <worktree>/..` against that code (the script
says what it needs); the data below must not change meanwhile.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAT = [4.0, 7.0, 10.0, 13.0]
LON = [35.0, 39.0, 43.0]
YEARS = (2019, 2020, 2021)
DATA_YEARS = (2019, 2020, 2021, 2022)
"""Observations and forecasts exist one year beyond the evaluated years, for seasons that run into the next year."""
STEPS, MEMBERS = 24, 4
BOUNDARY = "00 UTC, labelled by start"
SEASON_PEAKS = ((200, 45, 0.6), (320, 20, 0.5))
"""(day of year, width, extra chance of rain) of the two rainy seasons the rain follows: June to September, November."""


def _rain(year: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Daily rain in mm of 1 April to 31 March of the next year, (time, lat, lon): rainy spells in two seasons."""
    time = pd.date_range(f"{year}-04-01", f"{year + 1}-03-31")
    rng = np.random.default_rng(year)
    doy = np.arange(len(time)) + 91.0
    chance = 0.1 + sum(extra * np.exp(-(((doy - centre) / width) ** 2)) for centre, width, extra in SEASON_PEAKS)
    shape = (len(time), len(LAT), len(LON))
    spells = np.cumsum(rng.random(shape) < 0.04, axis=0) % 2   # per-cell dry and wet stretches
    wet = (rng.random(shape) < (chance[:, None, None] + 0.25 * spells)).astype(float)
    amount = np.round(-np.log(1 - rng.random(shape)) * 14.0, 1)
    return time, wet * amount


def _write_obs(folder: Path) -> None:
    """One observation file per data year."""
    folder.mkdir(parents=True)
    for year in DATA_YEARS:
        time, rain = _rain(year)
        xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm"})},
                   coords={"time": time, "lat": LAT, "lon": LON}).to_netcdf(folder / f"{year}.nc")


def _write_forecast(folder: Path, ensemble: bool) -> None:
    """Daily-init forecasts (1 April to 15 December) per year: lead k holds the observed rain of day init + k plus noise
    (lead k is labelled by its end), one array per member for the ensemble."""
    folder.mkdir(parents=True)
    steps = np.arange(1, STEPS + 1)
    for year in DATA_YEARS:
        time, rain = _rain(year)
        inits = pd.date_range(f"{year}-04-01", f"{year}-12-15")
        rng = np.random.default_rng(1000 + year)
        truth = np.stack([[rain[i + k] for k in steps] for i in range(len(inits))])   # (init, step, lat, lon)
        members = MEMBERS if ensemble else 1
        noise = rng.random((members, *truth.shape)) < 0.12
        values = np.round(np.where(noise, 0.0, truth * (0.7 + 0.6 * rng.random((members, *truth.shape)))), 1)
        dims, coords = ("time", "step", "lat", "lon"), {"time": inits, "step": steps, "lat": LAT, "lon": LON}
        if ensemble:
            dims, coords["number"] = ("time", "number", "step", "lat", "lon"), np.arange(members)
            data = np.moveaxis(values, 0, 1)
        else:
            data = values[0]
        xr.Dataset({"tp": (dims, data.astype("float32"), {"units": "mm"})}, coords=coords).to_netcdf(
            folder / f"{year}.nc")


def _write_threshold(path: Path) -> None:
    """A per-cell threshold (mm) of the wet spell: a grid with variable `mwset`."""
    rng = np.random.default_rng(7)
    amount = np.round(12 + 18 * rng.random((len(LAT), len(LON))), 1)
    xr.Dataset({"mwset": (("lat", "lon"), amount)}, coords={"lat": LAT, "lon": LON}).to_netcdf(path)


def write_dataset(root: Path) -> dict[str, str]:
    """Write the observations, a deterministic and an ensemble forecast and a threshold grid under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    _write_obs(root / "obs")
    _write_forecast(root / "det", ensemble=False)
    _write_forecast(root / "ens", ensemble=True)
    _write_threshold(root / "thresholds.nc")
    return {name: str(root / name) for name in ("obs", "det", "ens", "thresholds.nc")}


def _config(paths: dict[str, str], out: Path, *, ensemble: bool, onset: dict, evaluation: dict) -> dict:
    """An old-style run config: flat onset keys at the top level, the rest in sections."""
    return {
        "project_name": "legacy reference", "region": "ethiopia", "parallel": False, **onset,
        "observations": {"name": "OBS", "path": paths["obs"], "file_pattern": ["{}.nc"], "variable": "precip",
                         "units": "mm", "day_boundary": BOUNDARY},
        "models": [{"name": "FC", "path": paths["ens" if ensemble else "det"], "file_pattern": "{}.nc",
                    "variable": "tp", "units": "mm"}],
        "evaluation": {"start_year_clim": 2019, "end_year_clim": 2021, "init_days": [0, 2, 4],
                       "verification_windows": [[1, 10]], "tolerances": [3], "max_forecast_day": 10,
                       "day_bins": [[1, 5], [6, 10]], "probabilistic": ensemble, "members": "All", **evaluation},
        "output": {"data_dir": str(out), "figure_dir": str(out / "fig"), "save_fig": False,
                   "save_nc_spatial_far_mr_mae": True, "save_csv_score": True, "save_nc_climatology": True,
                   "plot_spatial_far_mr_mae": False, "plot_heatmap_bss_auc": False, "plot_reliability": False,
                   "plot_climatology_onset": False, "plot_panel_heatmap_error": False,
                   "plot_panel_heatmap_skill": False, "plot_bar_bss_rpss_auc": False, "show_plot": False,
                   "show_panel": False},
    }


_PLAIN = dict(wet_init=1, wet_threshold=20, wet_spell=3, dry_threshold=1, dry_spell=7, dry_extent=0)
_SEASON = {"start_date": [2019, 5, 1], "end_date": [2021, 7, 31]}
CASES = {
    "det_plain": dict(ensemble=False, onset=_PLAIN, evaluation=_SEASON),
    "det_veto_text_dates": dict(ensemble=False, onset={**_PLAIN, "dry_extent": 10, "earliest_onset_date": "05-15"},
                                evaluation={"start_date": "2019-05-01", "end_date": "2021-07-31"}),
    "det_threshold_file": dict(ensemble=False, onset={**_PLAIN, "wet_threshold": None, "thresh_file": "{thresholds}",
                                                      "thresh_var": "mwset", "wet_init": 0.5, "wet_spell": 4,
                                                      "earliest_onset_date": [6, 1]}, evaluation=_SEASON),
    "ens_veto": dict(ensemble=True, onset={**_PLAIN, "dry_extent": 12, "dry_spell": 5, "earliest_onset_date": [5, 15]},
                     evaluation=_SEASON),
    "det_years_subset": dict(ensemble=False, onset=_PLAIN,
                             evaluation={**_SEASON, "years": [2019, 2021], "years_clim": [2019, 2021]}),
    "det_season_into_next_year": dict(ensemble=False, onset={**_PLAIN, "earliest_onset_date": [10, 1]},
                                      evaluation={"start_date": [2019, 10, 1], "end_date": [2021, 11, 20]}),
}
"""Name -> arguments of `config`: the flat-key configs whose old outputs are the references."""


def flat_config(name: str, paths: dict[str, str], out: Path) -> dict:
    """The old-style config `name` over the dataset at `paths`, writing its outputs to `out`."""
    case = CASES[name]
    onset = {k: (paths["thresholds.nc"] if v == "{thresholds}" else v) for k, v in case["onset"].items()
             if v is not None}
    evaluation = dict(case["evaluation"])
    top = {key: evaluation.pop(key) for key in ("years", "years_clim") if key in evaluation}   # top-level keys of the old schema
    # the old outputs keep every climatology year as a member (the default now leaves the scored year out)
    return {**_config(paths, out, ensemble=case["ensemble"], onset=onset, evaluation=evaluation), **top,
            "reference": {"climatology_cv": "none"}}


def condense(out: Path) -> dict:
    """Every .nc data variable and .csv table of a run folder as plain lists, keyed by file name."""
    files = {}
    for path in sorted(out.glob("*.nc")):
        with xr.open_dataset(path) as ds:
            files[path.name] = {name: {"dims": list(da.dims), "dtype": str(da.dtype),
                                       "values": np.where(pd.isna(da.values), None, da.values.astype(object)).tolist()
                                       if da.dtype.kind != "M" else [str(v) for v in da.values.ravel()]}
                                for name, da in ds.data_vars.items()}
    for path in sorted(out.glob("*.csv")):
        table = pd.read_csv(path)
        files[path.name] = {"columns": list(table.columns),
                            "rows": table.astype(object).where(table.notna(), None).values.tolist()}
    return files


def same(actual, expected, where: str) -> list[str]:
    """What differs between two condensed runs: floats within rtol 1e-9 (missing equals missing), everything else exact."""
    problems = []
    if set(actual) != set(expected):
        problems.append(f"{where}: files {sorted(set(actual) ^ set(expected))} are in only one run")
    for name in sorted(set(actual) & set(expected)):
        a, e = actual[name], expected[name]
        if "columns" in e:   # a CSV table
            if a["columns"] != e["columns"]:
                problems.append(f"{name}: columns {a['columns']} != {e['columns']}")
            elif not _equal_rows(a["rows"], e["rows"]):
                problems.append(f"{name}: rows differ")
            continue
        if set(a) != set(e):
            problems.append(f"{name}: variables {sorted(set(a) ^ set(e))} are in only one run")
        for var in sorted(set(a) & set(e)):
            if a[var]["dims"] != e[var]["dims"] or a[var]["dtype"] != e[var]["dtype"]:
                problems.append(f"{name}:{var}: dims or dtype differ")
            elif not _equal_rows(a[var]["values"], e[var]["values"]):
                problems.append(f"{name}:{var}: values differ")
    return problems


def _equal_rows(a, e) -> bool:
    """Nested lists equal: floats to rtol 1e-9 (None equals None), other values exactly."""
    if isinstance(a, list) or isinstance(e, list):
        return isinstance(a, list) and isinstance(e, list) and len(a) == len(e) and all(
            _equal_rows(x, y) for x, y in zip(a, e))
    if isinstance(a, float) and isinstance(e, float):
        return bool(np.isclose(a, e, rtol=1e-9, atol=0.0))
    return a == e and type(a) is type(e)
