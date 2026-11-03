"""Differential check: the same synthetic NetCDF inputs through onset_blending's readers and hcwf's single input path.

    python tests/blend/differential_input.py dump hcwf OUT.pkl                 # in HCWF's env
    <onset_blending>/.venv/bin/python tests/blend/differential_input.py dump onset_blending OUT.pkl
    python tests/blend/differential_input.py compare A.pkl B.pkl

`dump` writes synthetic observations (with gaps of NaN and negative values), an admin-unit file, forecast files with
members and lead days, and weight tables, reads each as the package does (observations plain, regridded and under a
cell transform with every missing-rain policy; forecasts plain, regridded with the scale applied after the averaging,
and cell-transformed with it applied first, reading only the tiles that feed the units) and pickles every array;
`compare` demands them bit-identical. tests/blend/test_input_differential.py runs all three steps when the
onset_blending environment exists.
"""
import datetime as dt
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/: typed_io
from _inputs import DAYS, LATS, LONS, grid_weights, id_weights, write_adm3, write_forecast, write_observations  # noqa: E402

INITS = [dt.date(2014, 6, 2), dt.date(2014, 6, 9), dt.date(2014, 6, 16)]
SCALE = 0.001
TARGETS = ["north", "south"]


def make_inputs(root: Path) -> dict:
    """The files and tables both packages read."""
    write_observations(root / "obs", years=(2003, 2004))
    with xr.open_dataset(root / "obs" / "2004.nc") as opened:
        data = opened.load()
    data["precip"].values[3:5, 0, 1] = np.nan  # an incomplete source in 2004
    data.to_netcdf(root / "obs" / "2004.nc")
    write_adm3(root / "adm3_2003.nc")
    root.joinpath("fc").mkdir(exist_ok=True)
    write_forecast(root / "fc" / "undeclared.nc", INITS, members=3, units=None)
    big_lats, big_lons = np.arange(10.0, 12.0, 0.25), np.arange(70.0, 72.0, 0.25)
    write_forecast(root / "fc" / "chunked.nc", INITS, members=2, lats=big_lats, lons=big_lons, units=None,
                   chunks=(1, 2, DAYS, 4, 4))
    id_weights().to_csv(root / "weights_id.csv", index=False)
    grid_weights().to_csv(root / "weights_grid.csv", index=False)
    return {"root": root}


def dump_onset_blending(root: Path) -> dict:
    """Read with onset_blending's own readers; the lines of onsets._forecast_onto_units that transform are its own."""
    from onset_blending.core import domain, ids, io, onsets

    out = {}
    convention = ids.GridIdConvention(2, "fixed", "explicit")

    def context(policy="keep", regrid=None, transform=None):
        return {"missing_rain": policy, "variable": "precip", "rename": {}, "shift_days": 0, "regridder": regrid,
                "cell_transform": transform, "convention": convention, "area": lambda df: df}

    plain = io.read_observations(root / "obs" / "2003.nc", "precip", {}, shift_days=0)
    out["obs_plain"] = plain.rain.reshape(-1, len(LATS), len(LONS))
    adm3 = io.read_observations(root / "adm3_2003.nc", "precip", {}, shift_days=0)
    out["adm3"] = adm3.rain
    out["adm3_ids"] = adm3.points["id"].tolist()
    regridder = domain.Regridder.from_csv(root / "weights_grid.csv")
    series = onsets._observed_series(root / "obs" / "2003.nc", context(regrid=regridder))
    out["obs_regrid"], out["obs_regrid_ids"] = series.rain, series.ids.tolist()
    weights = pd.read_csv(root / "weights_id.csv", dtype={"source_id": str, "target_id": str})
    for year in (2003, 2004):
        for policy in ("keep", "zero", "drop"):
            series = onsets._observed_series(root / "obs" / f"{year}.nc", context(policy, transform=weights))
            out[f"obs_cell_{year}_{policy}"] = series.rain
            out[f"obs_cell_{year}_{policy}_supported"] = series.supported
    leads = io.LeadSelection(min_day=1, max_day=DAYS)
    for name, keep_fn in (("undeclared", None),):
        path = root / "fc" / f"{name}.nc"
        points = io.forecast_points(path, "tp", {})
        with io.ForecastFile(path, "tp", {}, leads, np.ones(len(points), dtype=bool)) as forecast:
            out["fc_plain"] = np.concatenate([rain for _, _, rain in forecast.issue_runs()])
            out["fc_plain_members"] = forecast.members
    with io.ForecastFile(root / "fc" / "undeclared.nc", "tp", {}, leads, np.ones(6, dtype=bool), scale=1000.0) as forecast:
        out["fc_scaled"] = np.concatenate([rain for _, _, rain in forecast.issue_runs()])
    # cell transform: scale first, one run of issues at a time (onsets._forecast_onto_units)
    path = root / "fc" / "undeclared.nc"
    points = io.forecast_points(path, "tp", {})
    transform, source = onsets._cell_transform(points, weights, convention, path.name)
    kept_units = np.arange(len(transform.target_ids))
    feeding = np.isin(source, domain.Regridder.sources_of(transform, kept_units))
    with io.ForecastFile(path, "tp", {}, leads, feeding, scale=SCALE) as forecast:
        out["fc_cell"] = _onto_units(forecast, transform, source, feeding, kept_units, domain, post_scale=None)
    # regrid: scale after the averaging
    transform = regridder.transform(*domain.grid_axes(points))
    source = domain.point_sources(points, transform)
    kept_units = np.arange(len(regridder.targets))
    feeding = np.isin(source, domain.Regridder.sources_of(transform, kept_units))
    with io.ForecastFile(path, "tp", {}, leads, feeding, scale=1.0) as forecast:
        out["fc_regrid"] = _onto_units(forecast, transform, source, feeding, kept_units, domain, post_scale=SCALE)
    return out


def _onto_units(forecast, transform, source, feeding, kept_units, domain, post_scale):
    """The body of onsets._forecast_onto_units: each run of issues read at the feeding points and transformed."""
    runs = []
    rows = source[feeding]
    for _, _, rain in forecast.issue_runs():
        n_run, n_member, _, n_day = rain.shape
        values = np.full((len(transform.source_ids), n_run * n_member * n_day), np.nan)
        values[rows] = rain.transpose(2, 0, 1, 3).reshape(len(rows), -1)
        regridded = domain.Regridder.apply(transform, values)[kept_units]
        regridded = regridded.reshape(len(kept_units), n_run, n_member, n_day).transpose(1, 2, 0, 3)
        if post_scale is not None and post_scale != 1.0:
            regridded = regridded * post_scale
        runs.append(np.ascontiguousarray(regridded))
    return np.concatenate(runs)


def dump_hcwf(root: Path) -> dict:
    import typed_io

    from hcwf.io import input as io_input
    from hcwf.spatial import EvaluationUnits
    from hcwf.spatial.ids import GridIdConvention
    from hcwf.io.dims import dim_fmt, dim_fmt_model_ensemble

    out = {}
    convention = GridIdConvention(2, "fixed", "explicit")

    def observations(year, name=None):
        return typed_io.observations(year, {"obs_var": "precip", "obs_unit_cvt": None, "region": None,
                                            "obs_paths": [str(root / "obs" / (name or f"{year}.nc"))]})

    out["obs_plain"] = observations(2003).values
    adm3 = typed_io.observations(2003, {"obs_var": "precip", "obs_unit_cvt": None, "region": None,
                                        "obs_paths": [str(root / "adm3_2003.nc")]})
    out["adm3"], out["adm3_ids"] = adm3.values, adm3["unit"].values.tolist()
    grid = EvaluationUnits.from_weights(root / "weights_grid.csv")
    cells = EvaluationUnits.from_weights(root / "weights_id.csv")

    def lazy_obs(year):
        return dim_fmt(xr.open_dataset(root / "obs" / f"{year}.nc"))["precip"]

    got = io_input.read_on_units(lazy_obs(2003), grid, block_dim="time")
    out["obs_regrid"], out["obs_regrid_ids"] = got.values, got["unit"].values.tolist()
    for year in (2003, 2004):
        for policy in ("keep", "zero", "drop"):
            got, supported, _ = io_input.read_on_units(lazy_obs(year), cells, block_dim="time", missing=policy,
                                                       convention=convention, support=True)
            out[f"obs_cell_{year}_{policy}"] = got.values
            out[f"obs_cell_{year}_{policy}_supported"] = supported
    path = root / "fc" / "undeclared.nc"

    def forecast(file=path, **kwargs):
        da = dim_fmt_model_ensemble(xr.open_dataset(file))["tp"]
        return da.transpose("init_time", "member", "step", "lat", "lon")

    def blend_order(da):  # (issue, member, unit, day)
        return da.transpose("init_time", "member", "unit", "step").values

    plain = forecast().load()
    out["fc_plain"] = plain.transpose("init_time", "member", "lat", "lon", "step").values.reshape(
        len(INITS), -1, len(LATS) * len(LONS), DAYS)
    out["fc_plain_members"] = plain["member"].values
    out["fc_scaled"] = (plain.transpose("init_time", "member", "lat", "lon", "step").values * np.float32(1000.0)).reshape(
        len(INITS), -1, len(LATS) * len(LONS), DAYS)
    out["fc_cell"] = blend_order(io_input.read_on_units(forecast(), cells, block_dim="init_time", scale=SCALE,
                                                        convention=convention))
    out["fc_regrid"] = blend_order(io_input.read_on_units(forecast(), grid, block_dim="init_time", post_scale=SCALE))
    return out


def dump(package: str, target: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = make_inputs(Path(tmp))["root"]
        out = dump_hcwf(root) if package == "hcwf" else dump_onset_blending(root)
    with open(target, "wb") as file:
        pickle.dump(out, file)


def compare(first: str, second: str) -> list[str]:
    """The names whose arrays differ between two dumps (empty when every array is bit-identical)."""
    a, b = (pickle.load(open(p, "rb")) for p in (first, second))
    bad = [f"keys differ: {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        if isinstance(x, list) or isinstance(y, list):
            same = x == y
        else:
            x, y = np.asarray(x), np.asarray(y)
            same = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
        if not same:
            bad.append(f"{name}: {getattr(x, 'shape', None)} {getattr(x, 'dtype', None)} vs "
                       f"{getattr(y, 'shape', None)} {getattr(y, 'dtype', None)}")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3])
    else:
        problems = compare(sys.argv[2], sys.argv[3])
        print("\n".join(problems) or f"identical ({len(pickle.load(open(sys.argv[2], 'rb')))} arrays)")
        sys.exit(1 if problems else 0)
