"""The refactored momp.stats.detect (rule-based) reproduces the original behaviour.

References are frozen copies of the pre-refactor scalar functions
(tests/_reference_detect.py) plus small reference loops written here that apply
them the way the original day-loop implementations did.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import xarray as xr

import momp.stats.detect as detect
import tests._reference_detect as ref
from momp.stats.onset_rule import LegacyRule, TwoStageAccumulation
from tests.test_onset_rule_legacy_equivalence import LEGACY_CONFIGS, _dry_active, _kw, compare_observed


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _tiny_onset_inputs():
    init_time = pd.to_datetime(["2001-05-01", "2001-05-20"])
    members = [1, 2, 3]
    steps = np.arange(1, 8)
    lats = [10.0, 11.0]
    lons = [20.0, 21.0]

    rain = np.zeros((len(init_time), len(members), len(steps), len(lats), len(lons)))
    rain[0, 0, 1:4, 0, 0] = 3.0
    rain[0, 1, 2:5, 0, 0] = 3.0
    rain[0, 2, 4:7, 0, 0] = 3.0
    rain[0, 0, 0:3, 1, 0] = 3.0
    rain[0, 1, 1:4, 1, 0] = 3.0
    rain[1, 0, 0:3, 0, 0] = 3.0
    rain[1, 1, 1:4, 0, 0] = 3.0
    rain[1, 2, 2:5, 0, 0] = 3.0
    rain[0, :, 0:3, 1, 1] = 3.0
    rain[1, :, 0:3, 1, 1] = 3.0

    p_model = xr.DataArray(
        rain,
        dims=("init_time", "member", "step", "lat", "lon"),
        coords={"init_time": init_time, "member": members, "step": steps, "lat": lats, "lon": lons},
        name="rain",
    )
    thresh_slice = xr.DataArray(np.full((2, 2), 5.0), dims=("lat", "lon"), coords={"lat": lats, "lon": lons})
    onset_da = xr.DataArray(
        np.array([["2001-05-15", "NaT"], ["2001-05-19", "2001-06-10"]], dtype="datetime64[ns]"),
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
        name="onset_date",
    )
    return p_model, thresh_slice, onset_da


def _random_inputs(seed, n_steps=60):
    rng = np.random.default_rng(seed)
    init_time = pd.to_datetime(["2001-05-01", "2001-05-08", "2001-05-20"])
    members = [1, 2, 3, 4]
    steps = np.arange(1, n_steps + 1)
    lats, lons = [10.0, 11.0, 12.0], [20.0, 21.0]
    shape = (len(init_time), len(members), len(steps), len(lats), len(lons))
    rain = rng.exponential(scale=6.0, size=shape)
    rain[rng.random(shape) < 0.45] = 0.0
    p_model = xr.DataArray(
        rain,
        dims=("init_time", "member", "step", "lat", "lon"),
        coords={"init_time": init_time, "member": members, "step": steps, "lat": lats, "lon": lons},
    )
    thresh = xr.DataArray(np.full((3, 2), 15.0), dims=("lat", "lon"), coords={"lat": lats, "lon": lons})
    onset_da = xr.DataArray(
        np.array([["2001-06-15", "NaT"], ["2001-06-19", "2001-06-10"], ["2001-05-15", "2001-07-01"]], dtype="datetime64[ns]"),
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )
    return p_model, thresh, onset_da


def _ens_kwargs(cfg=(1.0, 3, 0, 0), **overrides):
    kwargs = {
        **_kw(*cfg),
        "members": None,
        "onset_percentage_threshold": 0.5,
        "max_forecast_day": 5,
        "mok": None,
        "end_date": (2001, 7, 31),
    }
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------- #
# reference loops (how the original implementations applied detect_onset)
# --------------------------------------------------------------------------- #
def _ref_onset_day(series, thresh, init_date, *, max_forecast_day, mok, **rule_kw):
    for day in range(1, max_forecast_day + 1):
        if ref.detect_onset(day, series, thresh, **rule_kw):
            if mok and (init_date + pd.Timedelta(days=day)).date() < datetime(init_date.year, *mok).date():
                continue
            return day
    return None


def _ref_members(p_model, thresh, onset_da, *, members, onset_percentage_threshold, max_forecast_day, mok, **rule_kw):
    members = list(members) if members else p_model.member.values.tolist()
    rows, mean_rows = [], []
    for init_time in p_model.init_time.values:
        init_date = pd.to_datetime(init_time)
        for lat in p_model.lat.values:
            for lon in p_model.lon.values:
                obs = onset_da.sel(lat=lat, lon=lon).values
                if pd.isna(obs) or init_date >= pd.to_datetime(obs):
                    continue
                th = float(thresh.sel(lat=lat, lon=lon)) if not np.isscalar(thresh) else thresh
                days = []
                for m in members:
                    series = p_model.sel(init_time=init_time, member=m, lat=lat, lon=lon).values
                    d = _ref_onset_day(series, th, init_date, max_forecast_day=max_forecast_day, mok=mok, **rule_kw)
                    days.append(d)
                    rows.append(dict(init_time=init_time, lat=lat, lon=lon, member=m, onset_day=d,
                                     obs_onset_date=pd.to_datetime(obs).strftime("%Y-%m-%d")))
                valid = [d for d in days if d is not None]
                pct = len(valid) / len(members)
                ens = int(round(float(np.mean(valid)))) if pct >= onset_percentage_threshold else None
                mean_rows.append(dict(
                    init_time=init_time, lat=lat, lon=lon, onset_day=ens,
                    onset_date=(init_date + pd.Timedelta(days=ens)).strftime("%Y-%m-%d") if ens is not None else None,
                    member_onset_count=len(valid), total_members=len(members), onset_percentage=pct,
                    obs_onset_date=pd.to_datetime(obs).strftime("%Y-%m-%d"),
                ))
    return pd.DataFrame(rows), pd.DataFrame(mean_rows)


def _ref_deterministic(p_model, thresh, onset_da, *, max_forecast_day, mok, **rule_kw):
    rows = []
    for init_time in p_model.init_time.values:
        init_date = pd.to_datetime(init_time)
        for lat in p_model.lat.values:
            for lon in p_model.lon.values:
                obs = onset_da.sel(lat=lat, lon=lon).values
                if pd.isna(obs) or init_date >= pd.to_datetime(obs):
                    continue
                th = float(thresh.sel(lat=lat, lon=lon)) if not np.isscalar(thresh) else thresh
                series = p_model.sel(init_time=init_time, lat=lat, lon=lon).values
                d = _ref_onset_day(series, th, init_date, max_forecast_day=max_forecast_day, mok=mok, **rule_kw)
                rows.append(dict(
                    init_time=init_time, lat=lat, lon=lon, onset_day=d,
                    onset_date=(init_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d") if d is not None else None,
                    obs_onset_date=pd.to_datetime(obs).strftime("%Y-%m-%d"),
                ))
    return pd.DataFrame(rows)


def _norm(df, keys):
    df = df.copy()
    df["init_time"] = pd.to_datetime(df["init_time"])
    df["onset_day"] = pd.to_numeric(df["onset_day"], errors="coerce")
    if "onset_date" in df:
        df["onset_date"] = df["onset_date"].astype(object).where(df["onset_date"].notna(), None)
    for c in ("member_onset_count", "total_members"):
        if c in df:
            df[c] = df[c].astype(int)
    return df.sort_values(keys).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mok", [None, (5, 4)])
def test_ensemble_matches_reference_on_tiny_inputs(mok):
    p_model, thresh, onset_da = _tiny_onset_inputs()
    kw = _ens_kwargs(mok=mok, members=(1, 2, 3))
    got_m, got_mean = detect.compute_onset_for_all_members(p_model, thresh, onset_da, **kw)
    exp_m, exp_mean = _ref_members(p_model, thresh, onset_da, **kw)
    pdt.assert_frame_equal(_norm(got_m, ["init_time", "lat", "lon", "member"]), _norm(exp_m, ["init_time", "lat", "lon", "member"]), check_dtype=False)
    pdt.assert_frame_equal(_norm(got_mean, ["init_time", "lat", "lon"]), _norm(exp_mean, ["init_time", "lat", "lon"]), check_dtype=False)


@pytest.mark.parametrize("cfg", [(1.0, 3, 0, 0), (1.0, 3, 7, 21), (1.0, 3, 5, 10), (2.0, 4, 3, 8)], ids=str)
@pytest.mark.parametrize("mok", [None, (5, 10)])
def test_ensemble_matches_reference_incl_dry_veto(cfg, mok):
    """The original vectorized path refused dry_extent > wet_spell and fell back to a loop;
    the rule-based path handles both identically to the reference scalar logic."""
    p_model, thresh, onset_da = _random_inputs(2)
    kw = _ens_kwargs(cfg, mok=mok, max_forecast_day=30)
    got_m, got_mean = detect.compute_onset_for_all_members(p_model, thresh, onset_da, **kw)
    exp_m, exp_mean = _ref_members(p_model, thresh, onset_da, **kw)
    assert exp_m["onset_day"].notna().any(), "fixture never triggers onset; test is vacuous"
    pdt.assert_frame_equal(_norm(got_m, ["init_time", "lat", "lon", "member"]), _norm(exp_m, ["init_time", "lat", "lon", "member"]), check_dtype=False)
    pdt.assert_frame_equal(_norm(got_mean, ["init_time", "lat", "lon"]), _norm(exp_mean, ["init_time", "lat", "lon"]), check_dtype=False)


def test_ensemble_scalar_threshold_and_member_subset():
    p_model, _, onset_da = _tiny_onset_inputs()
    kw = _ens_kwargs(members=(1, 3))
    got_m, got_mean = detect.compute_onset_for_all_members(p_model, 5.0, onset_da, **kw)
    exp_m, exp_mean = _ref_members(p_model, 5.0, onset_da, **kw)
    assert set(got_m["member"]) == {1, 3}
    pdt.assert_frame_equal(_norm(got_m, ["init_time", "lat", "lon", "member"]), _norm(exp_m, ["init_time", "lat", "lon", "member"]), check_dtype=False)
    with pytest.raises(ValueError, match="members"):
        detect.compute_onset_for_all_members(p_model, 5.0, onset_da, **_ens_kwargs(members=(1, 9)))


# --------------------------------------------------------------------------- #
# deterministic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", [(1.0, 3, 0, 0), (1.0, 3, 7, 21)], ids=str)
@pytest.mark.parametrize("mok", [None, (5, 10)])
def test_deterministic_matches_reference(cfg, mok):
    p_model, thresh, onset_da = _random_inputs(3)
    det = p_model.isel(member=0, drop=True)
    kw = dict(_kw(*cfg), max_forecast_day=30, mok=mok, end_date=(2001, 7, 31))
    got = detect.compute_onset_for_deterministic_model(det, thresh, onset_da, **kw)
    exp = _ref_deterministic(det, thresh, onset_da, **kw)
    assert exp["onset_day"].notna().any()
    assert list(got.columns) == ["init_time", "lat", "lon", "onset_day", "onset_date", "obs_onset_date"]
    pdt.assert_frame_equal(_norm(got, ["init_time", "lat", "lon"]), _norm(exp, ["init_time", "lat", "lon"]), check_dtype=False)


# --------------------------------------------------------------------------- #
# observed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", LEGACY_CONFIGS, ids=str)
@pytest.mark.parametrize("mok", [None, (5, 15)])
def test_observed_equals_legacy_rule_for_all_configs(cfg, mok):
    """New detect_observed_onset is exactly LegacyRule on ROMP's search window."""
    n_bad, detail, n_onsets = compare_observed(LegacyRule(*cfg), cfg, mok, observed_fn=detect.detect_observed_onset)
    assert n_bad == 0, f"cfg={cfg} mok={mok}: {n_bad} cells differ\n{detail}"
    assert n_onsets > 0


@pytest.mark.parametrize("cfg", [c for c in LEGACY_CONFIGS if not _dry_active(c)], ids=str)
@pytest.mark.parametrize("mok", [None, (5, 15)])
def test_observed_matches_reference_when_veto_inactive(cfg, mok):
    """Bit-for-bit with the original observed path wherever the two original paths agreed."""
    rng = np.random.default_rng(7)
    year = 2001
    time = pd.date_range("2001-04-15", "2001-09-30", freq="D")
    lats, lons = np.arange(5.0, 10.0), np.arange(30.0, 36.0)
    vals = rng.exponential(6.0, size=(len(lats), len(lons), len(time)))
    vals[rng.random(vals.shape) < 0.45] = 0
    vals[rng.random(vals.shape) < 0.02] = np.nan
    rain = xr.DataArray(vals, dims=("lat", "lon", "time"), coords={"lat": lats, "lon": lons, "time": time})
    thresh = xr.DataArray(rng.uniform(10, 30, size=(len(lats), len(lons))), dims=("lat", "lon"), coords={"lat": lats, "lon": lons})
    common = dict(**_kw(*cfg), start_date=(year, 5, 1), end_date=(year, 7, 31), fallback_date=None, mok=mok)
    exp = ref.detect_observed_onset(rain, thresh, year, **common)
    got = detect.detect_observed_onset(rain, thresh, year, **common)
    assert got.dims == exp.dims and got.dtype.kind == "M"
    np.testing.assert_array_equal(got.values.astype("datetime64[ns]"), exp.values.astype("datetime64[ns]"))


# --------------------------------------------------------------------------- #
# plumbing: rule selection flows through kwargs
# --------------------------------------------------------------------------- #
def test_detect_onset_shim_matches_reference():
    rng = np.random.default_rng(9)
    series = rng.exponential(6.0, size=40)
    series[rng.random(40) < 0.5] = 0
    for cfg in LEGACY_CONFIGS:
        for day in range(1, 41):
            assert detect.detect_onset(day, series, 20.0, **_kw(*cfg)) == bool(ref.detect_onset(day, series, 20.0, **_kw(*cfg)))


def test_named_rule_and_params_flow_through_kwargs():
    p_model, thresh, onset_da = _random_inputs(4, n_steps=60)
    rule = TwoStageAccumulation(stage1_days=3, stage1_mm=15, stage2_days=7, stage2_mm=10)
    kw = _ens_kwargs(max_forecast_day=20, onset_rule="two_stage", onset_rule_params=rule.params())
    got_m, _ = detect.compute_onset_for_all_members(p_model, thresh, onset_da, **kw)

    # independent expectation straight from the rule object
    rain = p_model.transpose("init_time", "member", "lat", "lon", "step")
    mask = rule(rain, None, dim="step").where(rain.step <= 20, False)
    first = mask.argmax("step").where(mask.any("step"), -1)
    for row in got_m.itertuples(index=False):
        exp_idx = int(first.sel(init_time=pd.Timestamp(row.init_time), member=row.member, lat=row.lat, lon=row.lon))
        exp = None if exp_idx < 0 else exp_idx + 1
        got = None if row.onset_day is None or pd.isna(row.onset_day) else int(row.onset_day)
        assert got == exp

    # rule instance passthrough gives the same result
    got_inst, _ = detect.compute_onset_for_all_members(p_model, thresh, onset_da, **_ens_kwargs(max_forecast_day=20, onset_rule=rule))
    pdt.assert_frame_equal(_norm(got_m, ["init_time", "lat", "lon", "member"]), _norm(got_inst, ["init_time", "lat", "lon", "member"]), check_dtype=False)


def test_insufficient_steps_for_lookahead_raises():
    p_model, thresh, onset_da = _random_inputs(5, n_steps=30)
    kw = _ens_kwargs(max_forecast_day=10, onset_rule="two_stage")  # lookahead 29 => needs 39 steps
    with pytest.raises(ValueError, match="lookahead"):
        detect.compute_onset_for_all_members(p_model, thresh, onset_da, **kw)
    with pytest.raises(ValueError, match="lookahead"):
        detect.compute_onset_for_deterministic_model(p_model.isel(member=0, drop=True), thresh, onset_da, **kw)


def test_observed_extends_search_end_for_long_lookahead():
    """A rule whose lookahead exceeds extend_end_day can still evaluate end_date itself.

    With the default extend_end_day=47 and lookahead 89 the data window is
    extended to end_date + 90, so an onset *on* end_date+1 is detectable; the
    reference code would have cut the data at end_date + 47 and found nothing.
    """
    year = 2001
    time = pd.date_range("2001-05-01", "2001-12-31", freq="D")
    rain = xr.DataArray(np.full((1, 1, len(time)), 10.0), dims=("lat", "lon", "time"), coords={"lat": [1.0], "lon": [1.0], "time": time})
    rain.loc[:, :, : "2001-07-31"] = 0.0  # first possible onset 2001-08-01
    # lookahead 89 > 47; thresholds require every day of both windows at 10 mm
    rule = TwoStageAccumulation(stage1_days=30, stage1_mm=299, stage2_days=60, stage2_mm=599)
    common = dict(start_date=(year, 5, 1), end_date=(year, 7, 31), fallback_date=None, mok=None, onset_rule=rule)
    got = detect.detect_observed_onset(rain, None, year, **common)
    assert pd.Timestamp(got.values[0, 0]) == pd.Timestamp("2001-08-01")
    # and the extension is only as large as needed: a candidate one day later is not evaluable
    rain.loc[:, :, : "2001-08-01"] = 0.0
    got = detect.detect_observed_onset(rain, None, year, **common)
    assert pd.isna(got.values[0, 0])
