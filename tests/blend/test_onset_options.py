"""The parity options of the onset engine: dry_day, beginning within, horizon, missing policies, exact rain,
search_start variants, onset_count and Definition.windows().

Backends must agree on every option (the stitched search against the reference on the explicit concatenation);
the engine-level options are checked against the blending oracle (tests/blend/_oracle).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from _oracle.onset_rule import find_onsets
from test_onset_parity import BACKENDS, HEAD, assert_same, random_rain, rule
from hcwf.onset import definition as D
from hcwf.onset import engine
from hcwf.onset import plan as P
from hcwf.onset.backends import reference
from hcwf.onset.ingest import ingest, ingest_forecast
from hcwf.onset.model import Jobs, Status, StitchedJobs

BOUNDARY = "00 UTC, labelled by start"
DEFINITIONS = {
    "dry_day_runs": "wet_day: at least 1 mm\ndry_day: less than 0.6 mm\nthresholds: {thr: {mean total over: 3 days}}\n"
                    "onset:\n  window: 3 days\n  require: [{all days: wet}, {total: more than thr}]\n"
                    "  reject if: [{dry run: at least 3 days, within: 9 days after window}]\n",
    "begins": "wet_day: more than 1 mm\ndry_day: at most 1.4 mm\nthresholds: {thr: {mean total over: 3 days}}\n"
              "onset:\n  window: 3 days\n  require: [{first day: wet}, {total: at least thr}]\n"
              "  reject if: [{dry run: at least 3 days, beginning within: 8 days after window}]\n",
    "begins_no_dry_day": "wet_day: at least 1 mm\nthresholds: {thr: {mean total over: 3 days}}\n"
                         "onset:\n  window: 3 days\n  require: [{first day: wet}, {total: at least thr}]\n"
                         "  reject if: [{dry run: at least 2 days, beginning within: days 1 to 6}]\n",
    "dry_window_and_run": "wet_day: at least 1 mm\ndry_day: less than 0.5 mm\n"
                          "thresholds: {thr: {mean total over: 4 days}}\n"
                          "onset:\n  window: 4 days\n  require: [{all days: wet}, {total: more than thr}]\n"
                          "  reject if: [{dry window: 4 days with less than 3 mm, starting within: 7 days after window},"
                          " {dry run: at least 2 days, beginning within: days 5 to 9}]\n",
    "starting_within": "wet_day: at least 1 mm\nthresholds: {thr: {mean total over: 3 days}}\n"
                       "onset:\n  window: 3 days\n  require: [{first day: wet}, {total: at least thr}]\n"
                       "  reject if: [{dry run: at least 3 days, starting within: 8 days after window}]\n",
}


def load(name: str, **kwargs):
    return D.loads(HEAD + DEFINITIONS[name], **kwargs)


def tenths_case(rng, n, n_days, nan=0.04):
    """int32 tenths with wet and dry days, ties on the limits, and missing days."""
    series = random_rain(rng, n, n_days, nan=nan)
    missing = np.isnan(series)
    return np.rint(np.where(missing, 0, series) * 10).astype(np.int32), missing


def detect_all(plan, rain, missing, thr, jobs, backend):
    amounts, invalid = P.resolve_amounts(plan, {"thr": thr}, n_series=len(rain), exact=rain.dtype.kind == "f")
    out = backend.detect(plan, rain, missing, amounts, jobs)
    return out.day, out.status


OPTION_PLANS = [(name, horizon, missing) for name in DEFINITIONS for horizon in ("censor", "truncate", "reject")
                for missing in ("flag", "keep")]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,horizon,missing", OPTION_PLANS)
@pytest.mark.parametrize("exact", [False, True])
def test_options_agree_across_backends(backend, name, horizon, missing, exact):
    rng = np.random.default_rng(len(name) + 7 * (horizon == "reject") + exact)
    n, n_days = 300, 40
    rain, miss = tenths_case(rng, n, n_days)
    thr = np.round(rng.uniform(1, 25, n), 1)
    if exact:
        rain = rain / 10.0
    plan = P.compile(load(name), horizon=horizon, missing=missing)
    start = rng.integers(0, 12, n)
    jobs = Jobs(np.arange(n), start, np.full(n, n_days - 1), rng.integers(n_days - 12, n_days + 1, n))
    want = detect_all(plan, rain, miss, thr, jobs, reference)
    got = detect_all(plan, rain, miss, thr, jobs, backend)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def stitched_case(rng, exact_cells=4, n_hist=60, n_members=3, n_leads=25, n_jobs=120):
    hist, hist_missing = tenths_case(rng, exact_cells, n_hist)
    fc, fc_missing = tenths_case(rng, exact_cells * n_members, n_leads)
    series = rng.integers(0, exact_cells * n_members, n_jobs)
    split = rng.integers(8, n_hist - 5, n_jobs)
    start = np.maximum(split - rng.integers(0, 20, n_jobs), 0)
    end = split + rng.integers(0, n_leads, n_jobs)
    data_end = split + rng.integers(n_leads // 2, n_leads + 1, n_jobs)
    jobs = StitchedJobs(cell=(series // n_members).astype(np.int64), series=series.astype(np.int64),
                        split=split.astype(np.int64), start=start.astype(np.int64), end=end.astype(np.int64),
                        data_end=data_end.astype(np.int64))
    thr = np.round(rng.uniform(1, 25, exact_cells * n_members), 1)
    return hist, hist_missing, fc, fc_missing, jobs, thr


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.id != "reference"])
@pytest.mark.parametrize("name,horizon,missing", OPTION_PLANS)
def test_stitched_options_match_the_explicit_concatenation(backend, name, horizon, missing):
    rng = np.random.default_rng(3 + len(name))
    hist, hist_missing, fc, fc_missing, jobs, thr = stitched_case(rng)
    plan = P.compile(load(name), horizon=horizon, missing=missing)
    amounts, _ = P.resolve_amounts(plan, {"thr": thr}, n_series=len(fc))
    got = backend.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    want = reference.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    np.testing.assert_array_equal(got.day, want.day)
    np.testing.assert_array_equal(got.status, want.status)
    # and the reference's own concatenation shortcut agrees with detect() on the explicit series
    for j in range(0, len(jobs.series), 9):
        c, s, k = int(jobs.cell[j]), int(jobs.series[j]), int(jobs.split[j])
        x = np.concatenate([hist[c, :k], fc[s]])
        m = np.concatenate([hist_missing[c, :k], fc_missing[s]])
        one = reference.detect(plan, x[None], m[None], {key: v[s:s + 1] for key, v in amounts.items()},
                               Jobs(np.zeros(1, dtype=np.int64), jobs.start[j:j + 1], jobs.end[j:j + 1],
                                    jobs.data_end[j:j + 1]))
        assert (int(one.day[0]), int(one.status[0])) == (int(want.day[j]), int(want.status[j]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_rain_cannot_be_stitched(backend):
    plan = P.compile(load("starting_within"))
    hist, hist_missing = np.zeros((1, 10)), np.zeros((1, 10), dtype=bool)
    fc, fc_missing = np.zeros((1, 10)), np.zeros((1, 10), dtype=bool)
    jobs = StitchedJobs(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), np.full(1, 5), np.zeros(1, dtype=np.int64),
                        np.full(1, 8), np.full(1, 15))
    amounts, _ = P.resolve_amounts(plan, {"thr": np.array([5.0])}, n_series=1, exact=True)
    with pytest.raises(NotImplementedError, match="exact"):
        backend.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)


# ---- definition language ----------------------------------------------------------------------------------------

def test_dry_day_and_begins_are_part_of_the_definition_hash():
    base = D.definition_hash(load("starting_within"))
    with_dry = D.loads(HEAD + DEFINITIONS["starting_within"].replace("wet_day: at least 1 mm\n",
                                                                     "wet_day: at least 1 mm\ndry_day: less than 0.5 mm\n"))
    begins = D.loads(HEAD + DEFINITIONS["starting_within"].replace("starting within", "beginning within"))
    assert len({base, D.definition_hash(with_dry), D.definition_hash(begins)}) == 3


def test_definitions_without_the_new_fields_keep_their_hash():
    """Fields added for the parity options are left out of the canonical form when unset."""
    defn = D.load("icpac")
    assert "mm" not in D._canonical(defn.wet_day.amount) and defn.dry_day is None
    assert set(D._canonical(defn.wet_day.amount)) == {"_type", "tenths", "threshold"}


def test_dry_day_must_be_a_fixed_less_than():
    for bad in ("at least 1 mm", "less than thr"):
        with pytest.raises(D.DefinitionError, match="dry_day"):
            D.loads(HEAD + DEFINITIONS["dry_day_runs"].replace("less than 0.6 mm", bad))


def test_beginning_within_is_for_dry_runs_only():
    text = DEFINITIONS["dry_window_and_run"].replace("starting within: 7 days after window", "beginning within: 7 days after window")
    with pytest.raises(D.DefinitionError, match="beginning within.*dry run"):
        D.loads(HEAD + text)


def test_fine_amounts_need_the_flag_and_exact_rain():
    text = "wet_day: at least 1 mm\nonset:\n  window: 3 days\n  require: [{total: at least 20.05 mm}]\n  reject if: []\n"
    with pytest.raises(D.DefinitionError, match="0.1 mm"):
        D.loads(HEAD + text)
    defn = D.loads(HEAD + text, fine_amounts=True)
    plan = P.compile(defn)
    amounts, _ = P.resolve_amounts(plan, {}, n_series=2, exact=True)
    assert amounts["fixed:20.05mm"].tolist() == [20.05, 20.05]
    with pytest.raises(ValueError, match="exact"):
        P.resolve_amounts(plan, {}, n_series=2)


def test_windows_come_from_the_rule_spans():
    assert D.load("icpac").windows() == {"trigger": 3, "follow": 21, "dry_run": 7, "dry_window": None}
    w = D.load("moron_robertson").windows()
    assert w["trigger"] == 5 and w["dry_window"] is not None and w["dry_run"] is None
    assert load("dry_window_and_run").windows() == {"trigger": 4, "follow": 7, "dry_run": 2, "dry_window": 4}
    assert all(v is None for v in D.load("sivakumar").windows().values()) is False


# ---- engine-level options against the blending oracle -------------------------------------------------------------

ENGINE_RULE = ("wet_day: at least 1 mm\ndry_day: less than 1 mm\nthresholds: {thr: {mean total over: 3 days}}\n"
               "onset:\n  window: 3 days\n  require: [{all days: wet}, {total: more than thr}]\n"
               "  reject if: [{dry run: at least 4 days, beginning within: 10 days after window}]\n")


def observed_case(rng, n_years=2, n_cell=6, nan=0.01):
    """Two years of daily rain on a line of cells (float mm with 0.1 resolution and NaN gaps)."""
    time = pd.date_range("2001-01-01", "2002-12-31")
    values = random_rain(rng, n_cell * 1, len(time), nan=nan)
    da = xr.DataArray(values.reshape(n_cell, len(time)), dims=("lat", "time"),
                      coords={"lat": np.arange(n_cell, dtype=float), "time": time}, attrs={"units": "mm"})
    return da, np.round(rng.uniform(5, 30, n_cell), 1)


def oracle_year(da, thr, year, starts, short=True, window=3):
    sel = da.sel(time=str(year)).values
    return find_onsets(sel, thr, starts, rule(window=window, follow_days=10, min_dry_days=4, dry_day_min_mm=1.0),
                       reject_if_short_followup=short)


def onset_days(ds, year0=2001):
    """Onset day of year (NaN for none) per (year, cell) of a detect_observed Dataset with season 'main'."""
    idx = ds["onset_index"].isel(season=0).values.astype(float)
    ok = ds["status"].isel(season=0).values == Status.ONSET
    out = np.full(idx.shape, np.nan)
    for i, year in enumerate(ds["year"].values):
        offset = (pd.Timestamp(f"{year}-01-01") - pd.Timestamp(f"{year0}-01-01")).days
        out[i] = np.where(ok[i], idx[i] - offset + 1, np.nan)
    return out


@pytest.mark.parametrize("backend", ["reference", "numba", "gpu"])
@pytest.mark.parametrize("missing", ["keep", "zero", "drop"])
def test_missing_policies_match_blending(backend, missing):
    if backend == "gpu":
        pytest.importorskip("cupy")
        from hcwf.onset.backends import gpu
        if not gpu.available():
            pytest.skip("needs a GPU")
    rng = np.random.default_rng(31)
    da, thr = observed_case(rng, nan=0.004)
    rain = ingest(da, day_boundary=BOUNDARY, rain_resolution="exact", missing=missing)
    ds = engine.detect_observed(D.loads(HEAD + ENGINE_RULE), rain, thresholds={"thr": thr}, backend=backend,
                                horizon="reject")
    assert ds.attrs["rain_resolution"] == "exact" and ds.attrs["missing"] == missing and ds.attrs["horizon"] == "reject"
    got = onset_days(ds)
    for i, year in enumerate((2001, 2002)):
        series = da.sel(time=str(year)).values.copy()
        if missing == "zero":
            series = np.where(np.isnan(series), 0.0, series)
        want = oracle_year(da.copy(data=np.where(np.isnan(da.values), 0.0, da.values)) if missing == "zero" else da,
                           thr, year, np.ones(len(thr)))[0]
        dropped = np.isnan(series).any(axis=1)
        if missing == "drop":
            want = np.where(dropped, np.nan, want)
            assert (ds["status"].isel(season=0).values[i][dropped] == Status.INSUFFICIENT_DATA).all()
        assert_same(got[i], want)
    assert np.isfinite(got).any()


def test_default_attrs_record_the_default_options():
    rng = np.random.default_rng(1)
    da, thr = observed_case(rng)
    ds = engine.detect_observed(D.loads(HEAD + ENGINE_RULE.replace("dry_day: less than 1 mm\n", "")),
                                ingest(da, day_boundary=BOUNDARY), thresholds={"thr": thr}, backend="reference")
    assert (ds.attrs["rain_resolution"], ds.attrs["missing"], ds.attrs["horizon"]) == ("tenths", "flag", "censor")
    assert "start" not in ds.dims and "search_start" not in ds.attrs


@pytest.mark.parametrize("backend", ["reference", "numba"])
def test_observed_search_start_variants(backend, monkeypatch):
    rng = np.random.default_rng(32)
    da, thr = observed_case(rng, nan=0.0)
    rain = ingest(da, day_boundary=BOUNDARY, rain_resolution="exact", missing="keep")
    dates = np.array([[f"{y}-0{m}-15" for y in (2001, 2002)] for m in (3, 4, 5, 6, 3, 4)], dtype="datetime64[ns]")
    dates[0, 1] = np.datetime64("NaT", "ns")
    per_cell = xr.DataArray(dates, dims=("lat", "year"), coords={"lat": np.arange(6.0), "year": [2001, 2002]})
    calls = []
    real = engine._load_backend(backend).detect
    module = engine._load_backend(backend)
    monkeypatch.setattr(module, "detect", lambda *a, **k: calls.append(1) or real(*a, **k))
    ds = engine.detect_observed(D.loads(HEAD + ENGINE_RULE), rain, thresholds={"thr": thr}, backend=backend,
                                horizon="reject", search_start={"cutoff": "05-01", "ref": per_cell})
    assert len(calls) == 1, "all start variants share one backend call"
    assert list(ds["start"].values) == ["base", "cutoff", "ref"] and ds["status"].dims == ("year", "season", "start", "lat")
    assert ds.attrs["search_start"] == "cutoff=05-01; ref=per-cell dates"
    for i, year in enumerate((2001, 2002)):
        doy_cutoff = pd.Timestamp(f"{year}-05-01").dayofyear
        ref = per_cell.sel(year=year).values
        doy_ref = np.where(np.isnat(ref), 1, pd.DatetimeIndex(ref).dayofyear)
        starts = np.vstack([np.ones(6), np.full(6, doy_cutoff), doy_ref])
        want = oracle_year(da, thr, year, starts)
        for v in range(3):
            idx = ds["onset_index"].isel(season=0, start=v).values[i].astype(float)
            ok = ds["status"].isel(season=0, start=v).values[i] == Status.ONSET
            offset = (pd.Timestamp(f"{year}-01-01") - pd.Timestamp("2001-01-01")).days
            assert_same(np.where(ok, idx - offset + 1, np.nan), want[v])


def forecast_case(rng, n_init=6, n_member=4, n_cell=5, n_leads=45):
    inits = pd.date_range("2020-05-20", periods=n_init, freq="3D")
    values = np.stack([random_rain(rng, n_member * n_cell, n_leads, nan=0.0).reshape(n_member, n_cell, n_leads)
                       for _ in range(n_init)])
    da = xr.DataArray(values, dims=("init_time", "member", "lat", "lead"),
                      coords={"init_time": inits, "member": np.arange(n_member), "lat": np.arange(n_cell, dtype=float),
                              "lead": np.arange(1, n_leads + 1)}, attrs={"units": "mm"})
    return da, np.round(rng.uniform(3, 20, n_cell), 1), inits


@pytest.mark.parametrize("backend", ["reference", "numba"])
def test_forecast_truncate_variants_and_counts_match_blending(backend):
    rng = np.random.default_rng(33)
    da, thr, inits = forecast_case(rng)
    fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation="daily",
                         rain_resolution="exact", missing="keep")
    cutoff = {"fixed_cutoff": "06-05"}
    ds = engine.detect_forecast(D.loads(HEAD + ENGINE_RULE), fc, thresholds={"thr": thr}, lead_window=(1, 45),
                                backend=backend, horizon="truncate", search_start=cutoff, onset_count=True)
    n_init, n_member, n_cell, n_leads = da.shape[0], da.shape[1], da.shape[2], da.shape[3]
    assert ds["status"].dims == ("init_time", "member", "season", "start", "lat")
    assert ds["onset_count"].dims == ("init_time", "season", "start", "lead", "lat")
    assert list(ds["onset_count"]["lead"].values) == list(range(0, 46)) and (ds["onset_count"].sel(lead=0) == 0).all()
    assert (ds["members"] == n_member).all() and ds.attrs["horizon"] == "truncate"
    cutoff_lead = np.array([max(1, int((pd.Timestamp(f"{i.year}-06-05") - i).days) + 1) for i in inits])
    for t in range(n_init):
        series = da.values[t].reshape(n_member * n_cell, n_leads)
        starts = np.vstack([np.ones(n_member * n_cell), np.full(n_member * n_cell, cutoff_lead[t])])
        want = find_onsets(series, np.tile(thr, n_member), starts,
                           rule(window=3, follow_days=10, min_dry_days=4, dry_day_min_mm=1.0))
        for v in range(2):
            lead = ds["onset_lead"].isel(init_time=t, season=0, start=v).values.reshape(n_member * n_cell)
            assert_same(lead, want[v])
            hist = np.zeros((n_leads, n_cell), dtype=int)
            for row, onset in enumerate(want[v]):
                if np.isfinite(onset):
                    hist[int(onset) - 1, row % n_cell] += 1
            np.testing.assert_array_equal(
                ds["onset_count"].isel(init_time=t, season=0, start=v).sel(lead=slice(1, None)).values, hist)
    assert ds["onset_count"].sum() > 0


def test_forecast_search_start_rejects_history_and_drop():
    rng = np.random.default_rng(34)
    da, thr, _ = forecast_case(rng, n_init=2)
    fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation="daily")
    defn = D.loads(HEAD + ENGINE_RULE)
    with pytest.raises(ValueError, match="drop"):
        engine.detect_forecast(defn, fc, thresholds={"thr": thr}, missing="drop", backend="reference")
    with pytest.raises(ValueError, match="base"):
        engine.detect_forecast(defn, fc, thresholds={"thr": thr}, search_start={"base": "06-01"}, backend="reference")
    deterministic = dataclasses.replace(fc, tenths=fc.tenths[:, 0], missing=fc.missing[:, 0],
                                        dims=("init_time", "lat", "lead"),
                                        coords={k: v for k, v in fc.coords.items() if k != "member"})
    with pytest.raises(ValueError, match="member"):
        engine.detect_forecast(defn, deterministic, thresholds={"thr": thr}, onset_count=True, backend="reference")


# ---- ingest -----------------------------------------------------------------------------------------------------

def test_exact_ingest_keeps_float_millimetres_and_the_default_is_unchanged():
    da = xr.DataArray(np.array([[0.04, 0.06, np.nan, -0.04, 1.2345]]), dims=("lat", "time"),
                      coords={"lat": [0.0], "time": pd.date_range("2001-01-01", periods=5)}, attrs={"units": "mm"})
    default = ingest(da, day_boundary=BOUNDARY)
    exact = ingest(da, day_boundary=BOUNDARY, rain_resolution="exact")
    assert default.tenths.dtype == np.int32 and default.tenths.tolist() == [[0, 1, 0, 0, 12]]
    assert exact.tenths.dtype == np.float64 and exact.tenths.tolist() == [[0.04, 0.06, 0.0, -0.04, 1.2345]]
    np.testing.assert_array_equal(exact.missing, default.missing)
    zero = ingest(da, day_boundary=BOUNDARY, missing="zero")
    assert not zero.missing.any() and zero.missing_policy == "zero"
    with pytest.raises(ValueError, match="rain_resolution"):
        ingest(da, day_boundary=BOUNDARY, rain_resolution="coarse")
    with pytest.raises(ValueError, match="missing"):
        ingest(da, day_boundary=BOUNDARY, missing="fill")


def test_exact_curve_rules_are_refused():
    defn = D.load("sivakumar") if False else None
    plan = P.compile(D.loads(HEAD + "wet_day: at least 1 mm\nthresholds: {base: {mean daily rain: true}}\n"
                                    "curve: {accumulate: rain minus base, onset: day after minimum}\n"))
    with pytest.raises(NotImplementedError, match="curve"):
        P.resolve_amounts(plan, {"base": np.array([3.0])}, n_series=1, exact=True)



def test_forecast_reach_counts_a_dry_run_beginning_inside_like_one_starting_inside():
    from types import SimpleNamespace
    from hcwf.lib.onset_config import forecast_reach

    text = DEFINITIONS["begins"]
    def reach(t):
        return forecast_reach(SimpleNamespace(definition=D.loads(HEAD + t, fine_amounts=True)))
    assert reach(text) == reach(text.replace("beginning within", "starting within")) > 11
