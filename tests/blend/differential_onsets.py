"""Differential check: blending's onsets and weekly features against hcwf's, on the same synthetic data and configs.

    python tests/blend/differential_onsets.py dump hcwf OUT.pkl                # in HCWF's env
    <onset_blending>/.venv/bin/python tests/blend/differential_onsets.py dump onset_blending OUT.pkl
    python tests/blend/differential_onsets.py compare A.pkl B.pkl

`dump` writes yearly observations and forecast files, runs several blending configs on them (plain grid cells with a
reference onset and rain predictors; a cell transform with the "drop" policy and a regrid; thresholds computed from the
observations with the "zero" policy and climatology windows), and pickles every array of the observed onsets, each
model's forecast onsets, the climatology and the weekly features (with the legacy tables `frame()` rebuilds), as
onset_blending's own code makes them (`onset_blending.core.onsets`, `features`) or as hcwf's (`hcwf.onset.detect`,
`hcwf.blend.features`); `compare` demands them bit-identical. tests/blend/test_blend_onsets_differential.py runs all
three steps when the onset_blending environment exists.
"""
import datetime as dt
import pickle
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

LATS, LONS = [10.0, 10.25, 10.5], [70.0, 70.25]
OBS_YEARS, FORECAST_YEARS, LEAD_DAYS = range(2000, 2014), range(2010, 2014), 40


def onset_date(year, cell, shift=0):
    return dt.date(year, 5, 25) + dt.timedelta(days=(year * 7 + cell * 5) % 40 + shift)


def rain(dates, onset, rng=None):
    days = np.array([(d - onset).days for d in dates])
    out = np.where(days < 0, 0.0, np.where(days < 3, 10.0, 6.0))
    if rng is not None:
        out = out * rng.uniform(0.8, 1.4, len(out))
    return out.astype("float32")


def write_data(root: Path):
    rng = np.random.default_rng(5)
    (root / "obs").mkdir()
    for year in OBS_YEARS:
        dates = [dt.date(year, 4, 1) + dt.timedelta(days=i) for i in range(200)]
        data = np.stack([np.stack([rain(dates, onset_date(year, i * 2 + j), rng) for j in range(len(LONS))], axis=-1)
                         for i in range(len(LATS))], axis=1)
        data[rng.random(data.shape) < 0.002] = np.nan
        xr.Dataset({"precip": (("time", "lat", "lon"), data)},
                   coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS}).to_netcdf(root / "obs" / f"{year}.nc")
    (root / "ens").mkdir()
    for year in FORECAST_YEARS:
        issues = [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
        data = np.zeros((len(issues), 3, LEAD_DAYS, len(LATS), len(LONS)), dtype="float32")
        for t, issue in enumerate(issues):
            leads = [issue + dt.timedelta(days=d) for d in range(1, LEAD_DAYS + 1)]
            for m in range(3):
                for i in range(len(LATS)):
                    for j in range(len(LONS)):
                        data[t, m, :, i, j] = rain(leads, onset_date(year, i * 2 + j, shift=3 * (m - 1)), rng)
        xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": pd.to_datetime(issues), "number": np.arange(3),
                           "day": np.arange(1, LEAD_DAYS + 1), "lat": LATS, "lon": LONS}).to_netcdf(root / "ens" / f"{year}.nc")
    rows = [("10.00_70.00", "north", 0.5), ("10.00_70.25", "north", 0.25), ("10.25_70.00", "south", 1.0),
            ("10.25_70.25", "south", 0.5), ("10.25_70.25", "north", 0.25), ("10.50_70.00", "south", 0.5)]
    pd.DataFrame(rows, columns=["source_id", "target_id", "weight"]).to_csv(root / "weights.csv", index=False)
    grid = [(a, b) for a in LATS for b in LONS]
    pd.DataFrame([(a, b, "north" if a < 10.5 else "south", 1.0 + 0.5 * (b > 70)) for a, b in grid],
                 columns=["lat", "lon", "target_id", "weight"]).to_csv(root / "regrid.csv", index=False)
    pd.DataFrame({"lat": [a for a, _ in grid], "lon": [b for _, b in grid], "onset_thresh": [15 + i for i in range(6)]}
                 ).to_csv(root / "thresholds.csv", index=False)


def configs(root: Path) -> dict:
    base = {
        "name": "t", "data_root": str(root), "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}},
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": "ens", "files": {"folder": "ens"}, "lead_days": {"max": LEAD_DAYS},
                    "rain_predictors": [{"agg": "diff"}, {"agg": "max", "window": 5}]}],
        "onset": {"threshold": {"mm": 15}, "reference_onset": {"month_day": "06-15"}, "fixed_cutoff": "06-10",
                  "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}},
        "climatology": {"forecast_window": LEAD_DAYS, "min_onset_years": 5, "cv_by_year": False},
        "evaluation": {"years": {"training": "2010:2012", "cv_holdout": "2010:2012", "true_holdout": 2013},
                       "formulas": {"b": "outcome ~ prob_clim_qx + diff_ens_qx"}},
    }

    def edit(**sections):
        out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
        for key, value in sections.items():
            out[key] = {**out[key], **value} if isinstance(out[key], dict) else value
        return out

    transform = edit(
        observations={"files": {"folder": "obs"}, "variable": "precip", "missing_rain": "drop",
                      "cell_transform": "weights.csv", "date_marks": "start"},
        models=[{**base["models"][0], "cell_transform": "weights.csv"}],
        onset={"threshold": {"file": "thresholds.csv", "scale": {"factor": 0.0, "offset": 18}},
               "reference_onset": {"month_day": "06-01"}, "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}})
    transform["onset"]["threshold"] = {"mm": 14}
    computed = edit(
        observations={"files": {"folder": "obs"}, "variable": "precip", "missing_rain": "zero"},
        onset={"threshold": {"climatology": {"years": "2000:2013", "start": "05-01", "end": "08-31"}},
               "season_start": "05-01", "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}},
        climatology={"forecast_window": LEAD_DAYS, "min_onset_years": 5, "cv_by_year": True, "windows": [[2000, 2005]]})
    table = edit(onset={"threshold": {"file": "thresholds.csv"}, "reference_onset": {"month_day": "06-01"},
                        "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}})
    regrid = edit(domain={"grid_ids": {"decimal_digits": 2, "format": "fixed"}, "regrid": {"weights": "regrid.csv"}},
                  observations={"files": {"folder": "obs"}, "variable": "precip", "missing_rain": "drop"})
    minlead = edit(models=[{**base["models"][0], "lead_days": {"min": 4, "max": LEAD_DAYS}, "rain_predictors": []}],
                  evaluation={"years": base["evaluation"]["years"], "formulas": {"b": "outcome ~ prob_clim_qx"}})
    return {"regrid": regrid, "plain": base, "transform": transform, "computed": computed, "table": table,
            "minlead": minlead}


def pack(value):
    """A result as plain arrays."""
    if isinstance(value, xr.Dataset):
        return {name: (np.asarray(value[name].values), tuple(value[name].dims)) for name in value.variables}
    return value


def run_onset_blending(config_dict) -> dict:
    sys.path.insert(0, "/home/hcwf/code/onset_blending")
    from onset_blending import BlendConfig
    from onset_blending.core import climatology, features, onsets
    from onset_blending.core.reference import Reference

    config = BlendConfig.from_dict(config_dict)
    reference = Reference.resolve(config)
    observed = onsets.observed(config, reference=reference)
    forecasts = {m.name: onsets.forecast(config, m, reference=reference) for m in config.models}
    years = sorted({y for f in forecasts.values() for y in f.years})
    clim = climatology.build(config, observed, years)
    weekly = features.weekly(config, clim, observed, forecasts)
    daily = features.daily_inputs(config, clim, observed, forecasts, max_day=28)
    return finish(observed, forecasts, clim, weekly, daily)


def run_hcwf(config_dict) -> dict:
    from _recipes import converted

    from hcwf.blend import features
    from hcwf.io.thresholds import thresholds_of

    t = converted(config_dict)
    rc, recipe, setup = t.rc(), t.recipe, t.setup
    thresholds = thresholds_of(rc, rc.models[0])
    reference = features.reference_onset(recipe)
    observed = features.observed(rc, recipe, thresholds, reference)
    forecasts = {m.name: features.forecast(rc, recipe, m.name, thresholds, reference) for m in recipe.models}
    years = sorted({y for f in forecasts.values() for y in f.years})
    clim = features.climatology(recipe, setup, observed, years)
    weekly = features.weekly(recipe, setup.definition, clim, observed, forecasts)
    daily = features.daily_inputs(recipe, clim, observed, forecasts, max_day=28)
    return finish(observed, forecasts, clim, weekly, daily)


def finish(observed, forecasts, clim, weekly, daily) -> dict:
    return {
        "observed": pack(observed.data), "observed_frame": observed.frame(),
        "forecasts": {name: pack(f.data) for name, f in forecasts.items()},
        "climatology": pack(clim.data), "windows": {str(k): pack(v.to_dataset(name="v")) for k, v in clim.windows.items()},
        "weekly": pack(weekly.data), "columns": weekly.columns, "attrs": weekly.attrs, "frame": weekly.frame(),
        "daily": pack(daily.data)}


def dump(package: str, target: str) -> None:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_data(root)
        started = time.time()
        for name, config in configs(root).items():
            out[name] = run_hcwf(config) if package == "hcwf" else run_onset_blending(config)
        out["_seconds"] = time.time() - started
        out["_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with open(target, "wb") as file:
        pickle.dump(out, file)


def differences(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(map(str, a))} vs {sorted(map(str, b))}"]
        return [d for k in a if not str(k).startswith("_") for d in differences(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
        if a[1] != b[1]:
            return [f"{path}: dims {a[1]} vs {b[1]}"]
        return differences(a[0], b[0], path)
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            return []
        except AssertionError as error:
            return [f"{path}: {str(error)[:200]}"]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.dtype.kind == y.dtype.kind == "M":  # pandas 3 (the oracle's environment) keeps microseconds
            x, y = x.astype("datetime64[ns]"), y.astype("datetime64[ns]")
        same = x.shape == y.shape and x.dtype == y.dtype and (
            np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
        return [] if same else [f"{path}: arrays differ ({x.shape} {x.dtype} vs {y.shape} {y.dtype})"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3])
    else:
        with open(sys.argv[2], "rb") as first, open(sys.argv[3], "rb") as second:
            x, y = pickle.load(first), pickle.load(second)
        found = differences(x, y)
        print("\n".join(found) or "identical")
        print(f"seconds {x['_seconds']:.1f} vs {y['_seconds']:.1f}; peak RSS MB {x['_rss_mb']:.0f} vs {y['_rss_mb']:.0f}",
              file=sys.stderr)
