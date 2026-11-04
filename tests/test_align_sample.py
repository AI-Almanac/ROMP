"""Tests of the sampled alignment check (hcwf.onset.sample_alignment) against the whole-year check."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.onset import sample_alignment as align_sample
from hcwf.onset.alignment import check_forecast_alignment, forecast_alignment_from_samples
from hcwf.onset.ingest import ingest, ingest_forecast
from real_data import DATA, needs_data

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


def test_cache_parts_change_with_label_and_data(tmp_path):
    """The cache fingerprint depends on the lead label and on the dataset files."""
    folder = tmp_path / "obs"
    folder.mkdir()
    (folder / "2001.nc").write_bytes(b"a")
    kw = {"obs_dir": str(folder), "model_dir": str(folder), "obs_var": "x", "model_var": "y"}
    store = align_sample._store()

    def fingerprint(label):
        return store.fingerprint(align_sample.cache_parts(kw, label, "b", "f"))

    base = fingerprint("start")
    assert base == fingerprint("start")
    assert base != fingerprint("end")
    (folder / "2001.nc").write_bytes(b"abc")
    assert base != fingerprint("start")


def test_cache_parts_follow_listed_files(tmp_path):
    """A dataset given by paths is fingerprinted by those files, not by a folder."""
    one, two = tmp_path / "a_2001.nc", tmp_path / "b_2001.nc"
    one.write_bytes(b"a")
    two.write_bytes(b"b")
    kw = {"obs_paths": [str(one)], "model_paths": [str(two)], "obs_var": "x", "model_var": "y"}
    store = align_sample._store()
    base = store.fingerprint(align_sample.cache_parts(kw, "start", "b", "f"))
    two.write_bytes(b"bcd")
    assert base != store.fingerprint(align_sample.cache_parts(kw, "start", "b", "f"))


def _write_listed(tmp_path):
    """Synthetic observations and forecasts as yearly files in two layouts: a folder, and files named in paths."""
    obs_da, fc_da = _synthetic(0)
    (tmp_path / "obs").mkdir()
    (tmp_path / "model").mkdir()
    obs_da.to_dataset(name="rain").to_netcdf(tmp_path / "obs" / "2001.nc")
    fc_da.to_dataset(name="tp").to_netcdf(tmp_path / "model" / "2001.nc")
    return tmp_path / "obs" / "2001.nc", tmp_path / "model" / "2001.nc"


def test_datasets_given_by_paths_are_read(tmp_path, monkeypatch):
    """The sampled check reads datasets listed in obs_paths and model_paths, and gives the folder read's report."""
    monkeypatch.setenv("HCWF_CACHE_DIR", str(tmp_path / "cache"))
    obs_file, model_file = _write_listed(tmp_path)
    common = dict(obs_var="rain", model_var="tp", probabilistic=False, region=None)
    by_folder = align_sample.check_labels(
        dict(obs_dir=str(obs_file.parent), obs_file_pattern=["{}.nc"], model_dir=str(model_file.parent),
             file_pattern="{}.nc", **common), ("start",), obs_boundary=START, use_cache=False)["start"][0]
    listed = dict(obs_paths=[str(obs_file)], model_paths=[str(model_file)], **common)
    by_paths, cached = align_sample.check_labels(listed, ("start",), obs_boundary=START)["start"]
    assert not cached
    assert by_paths.verdict == by_folder.verdict
    for lead, report in by_folder.per_lead.items():
        for lag, r in report.correlations.items():
            assert by_paths.per_lead[lead].correlations[lag] == pytest.approx(r, rel=1e-9)
    again, cached = align_sample.check_labels(listed, ("start",), obs_boundary=START)["start"]
    assert cached and again.verdict == by_paths.verdict


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


@needs_data
@pytest.mark.parametrize("name", ["ethiopia", pytest.param("india", marks=pytest.mark.slow)])  # india: 8-12 s
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
