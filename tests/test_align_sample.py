"""Tests of the sampled alignment check (momp.lib.align_sample) against the whole-year check."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.lib import align_sample
from momp.onset.alignment import check_forecast_alignment, forecast_alignment_from_samples
from momp.onset.ingest import ingest, ingest_forecast

START = "00 UTC, labelled by start"


def _synthetic(shift_days=0, n_init=30, n_cell=60, seed=1):
    """Observations (time, lat, 1) and forecasts whose lead k is the observed day init + k - 1 + shift, plus noise."""
    rng = np.random.default_rng(seed)
    n_t = n_init * 3 + 40
    z = np.zeros((n_t, n_cell))
    for t in range(1, n_t):
        z[t] = 0.6 * z[t - 1] + 0.8 * rng.normal(size=n_cell)
    rain = np.maximum(z - 0.3, 0.0) ** 2 * 5
    times = pd.date_range("2001-05-01", periods=n_t)
    lat = np.arange(n_cell) * 0.25
    obs_da = xr.DataArray(rain[:, :, None], dims=("time", "lat", "lon"), coords={"time": times, "lat": lat, "lon": [1.0]},
                          attrs={"units": "mm"})
    inits = np.arange(5, 5 + 3 * n_init, 3)
    steps = np.arange(1, 5)
    fc = np.stack([[rain[i + k - 1 + shift_days] for k in steps] for i in inits])
    fc = fc + rng.normal(scale=0.3, size=fc.shape) ** 2
    fc_da = xr.DataArray(fc[..., None], dims=("init_time", "step", "lat", "lon"),
                         coords={"init_time": times[inits], "step": steps, "lat": lat, "lon": [1.0]},
                         attrs={"units": "mm"})
    return obs_da, fc_da


@pytest.mark.parametrize("shift", [0, 1, -1])
@pytest.mark.parametrize("label", ["start", "end"])
def test_vectorised_matches_pairwise(shift, label):
    """lagged_correlations through forecast_alignment_from_samples equals check_forecast_alignment lag by lag."""
    obs_da, fc_da = _synthetic(shift)
    obs = ingest(obs_da, day_boundary=START, source="obs")
    forecast = ingest_forecast(fc_da, day_boundary=START, lead_label=label, source="m")
    full = check_forecast_alignment(obs, forecast)
    pad = 3
    first = np.searchsorted(obs.coords["time"], np.datetime64(fc_da.init_time.values[0], "ns"))
    sqrt_obs = np.sqrt(obs.tenths[:, 0, :].T / 10.0)
    days = np.stack([sqrt_obs[first + 3 * i - pad: first + 3 * i - pad + 10] for i in range(fc_da.sizes["init_time"])])
    fcs = np.sqrt(forecast.tenths[:, :, 0, :3] / 10.0).transpose(0, 2, 1)
    report = forecast_alignment_from_samples(days, fcs, leads=(1, 2, 3), pad=pad, lead_label=label, obs="o",
                                             source="m", n_inits=len(fcs))
    assert report.verdict == full.verdict
    for lead in (1, 2, 3):
        for lag, r in full.per_lead[lead].correlations.items():
            assert report.per_lead[lead].correlations[lag] == pytest.approx(r, abs=1e-4)


def test_fingerprint_changes_with_label_and_data(tmp_path):
    """The cache fingerprint depends on the lead label and on the dataset files."""
    folder = tmp_path / "obs"
    folder.mkdir()
    (folder / "2001.nc").write_bytes(b"a")
    kw = {"obs_dir": str(folder), "model_dir": str(folder), "obs_var": "x", "model_var": "y"}
    base = align_sample.fingerprint(kw, "start", "b", "f")
    assert base == align_sample.fingerprint(kw, "start", "b", "f")
    assert base != align_sample.fingerprint(kw, "end", "b", "f")
    (folder / "2001.nc").write_bytes(b"abc")
    assert base != align_sample.fingerprint(kw, "start", "b", "f")


DATA = Path(os.environ.get("ROMP_TEST_DATA", "/nonexistent"))
ZARR = Path(__file__).resolve().parents[1] / "data"

CASES = {
    "ethiopia": dict(obs="ethiopia/obs", pattern="{}.nc", var="total_precipitation_24hr", model="ethiopia/models/AIFS_v2",
                     region="Ethiopia", boundary=START, good="start", zarr_obs="ethiopia/obs.zarr",
                     zarr_model="ethiopia/models/AIFS_v2.zarr", dims={"init_time": "time", "step": "prediction_timedelta_daily"}),
    "india": dict(obs="india/obs", pattern="data_{}.nc", var="RAINFALL", model="india/models/AIFS_v2",
                  region="India", boundary="03 UTC, labelled by end", good="end", zarr_obs="india/obs.zarr",
                  zarr_model="india/models/AIFS_v2.zarr", dims={"init_time": "time", "step": "prediction_timedelta_daily"}),
}


def _sample_kw(case, root, zarr):
    """Keyword arguments for the sampled check of a case from NetCDF folders or zarr stores."""
    return dict(obs_dir=str(root / (case["zarr_obs"] if zarr else case["obs"])), obs_file_pattern=[case["pattern"]],
                obs_var=case["var"], obs_unit_cvt=None,
                model_dir=str(root / (case["zarr_model"] if zarr else case["model"])), model_var="tp",
                unit_cvt=1000.0, file_pattern="{}.nc", model_dims=case["dims"], probabilistic=False,
                region=case["region"], start_date=(2000, 5, 1), end_date=(2024, 8, 31), init_days=(0, 3),
                years=tuple(range(2000, 2025)))


def _full_year(case, label, year=2012):
    """The whole-year check on the NetCDF files: (verdict, best lags per lead)."""
    obs = ingest(xr.open_dataset(DATA / case["obs"] / case["pattern"].format(year))[case["var"]],
                 day_boundary=case["boundary"], source="obs")
    tp = xr.open_dataset(DATA / case["model"] / f"{year}.nc")["tp"]
    report = check_forecast_alignment(obs, ingest_forecast(tp, day_boundary=START, lead_label=label, source="m"))
    return report.verdict, [r.best_lag for r in report.per_lead.values()]


@pytest.mark.skipif(not DATA.exists(), reason="set ROMP_TEST_DATA to run real-data checks")
@pytest.mark.parametrize("name", ["ethiopia", "india"])
@pytest.mark.parametrize("label", ["start", "end"])
@pytest.mark.parametrize("source", ["netcdf", "zarr"])
def test_sample_agrees_with_full_year(name, label, source):
    """The sampled check gives the whole-year check's verdict and best lags, from NetCDF files and zarr copies."""
    case = CASES[name]
    zarr = source == "zarr"
    if zarr and not (ZARR / case["zarr_obs"]).exists():
        pytest.skip("no zarr copy of the data in the repository")
    root = ZARR if zarr else DATA
    reports = align_sample.check_labels(_sample_kw(case, root, zarr), (label,), obs_boundary=case["boundary"],
                                        use_cache=False)
    report, cached = reports[label]
    assert not cached
    verdict, lags = _full_year(case, label)
    assert (report.verdict, [r.best_lag for r in report.per_lead.values()]) == (verdict, lags)
    assert (verdict == "aligned") == (label == case["good"])
