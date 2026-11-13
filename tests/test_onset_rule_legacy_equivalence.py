"""Does LegacyRule reproduce ROMP's original onset criterion?

Reference implementations are frozen in tests/_reference_detect.py (verbatim
copies from before onset detection was routed through OnsetRule). Three
comparisons:

1. LegacyRule vs reference ``detect_onset``            — per day, per series. Exact.
2. LegacyRule vs reference ``detect_observed_onset``   — exact when the dry veto is off.
   When it is on, the *original* observed path differed from the original
   forecast path (dry window one day longer; veto skipped when the window
   overran the series). LegacyRule follows the forecast semantics, so those
   configs are a strict xfail here, and an ObservedPathLegacyRule variant
   proves the contract could reproduce the old observed behaviour too.
3. The refactored ``momp.stats.detect`` public functions vs the references
   (see test_detect_uses_onset_rule.py).

Known, intentional deviation: with the dry veto active, a NaN inside the
dry-extent window used to be treated as "not dry" and let onset through.
LegacyRule fails closed instead. Fixtures with the veto on therefore use
NaN-free rain, and ``test_nan_in_dry_window_is_the_only_deviation`` pins the
difference explicitly.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tests._reference_detect as ref
from momp.stats.onset_rule import LegacyRule, first_onset_index, lead_sum

LEGACY_CONFIGS = [
    # (wet_init, wet_spell, dry_spell, dry_extent)
    (1.0, 3, 0, 0),  # ROMP / ai-almanac default, dry check off
    (1.0, 3, 7, 0),  # dry_spell set but dry_extent=0 => still off
    (1.0, 3, 7, 21),  # ICPAC-style false-start veto
    (1.0, 5, 10, 30),  # docs/package_configuration example
    (1.0, 3, 5, 10),
    (2.0, 4, 3, 8),
]


def _dry_active(cfg):
    return cfg[3] > cfg[1]


def _kw(wet_init, wet_spell, dry_spell, dry_extent):
    return dict(wet_init=wet_init, wet_spell=wet_spell, dry_spell=dry_spell, dry_threshold=1.0, dry_extent=dry_extent)


def _random_rain(rng, shape, p_dry=0.45, nan_frac=0.02):
    rain = rng.exponential(scale=6.0, size=shape)
    rain[rng.random(shape) < p_dry] = 0.0
    drizzle = rng.random(shape) < 0.1
    rain[drizzle] = rng.uniform(0.0, 0.99, size=drizzle.sum())
    if nan_frac:
        rain[rng.random(shape) < nan_frac] = np.nan
    return rain


# --------------------------------------------------------------------------- #
# 1. Forecast scalar path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", LEGACY_CONFIGS, ids=str)
def test_legacy_rule_matches_reference_detect_onset_per_day(cfg):
    rng = np.random.default_rng(0)
    rule = LegacyRule(*cfg)
    n_series, n_steps = 300, 60
    rain = _random_rain(rng, (n_series, n_steps), nan_frac=0.0 if _dry_active(cfg) else 0.02)
    thresh = 20.0
    da = xr.DataArray(rain, dims=("series", "step"), coords={"step": np.arange(1, n_steps + 1)})
    mask = rule(da, thresh, dim="step").values

    for s in range(n_series):
        for day in range(1, n_steps + 1):
            expected = bool(ref.detect_onset(day, rain[s], thresh, **_kw(*cfg)))
            assert mask[s, day - 1] == expected, f"cfg={cfg} series={s} day={day}: rule={mask[s, day - 1]} ref={expected}"
    assert mask.any(), f"cfg={cfg}: fixture never triggers onset; test is vacuous"


def test_nan_in_dry_window_is_the_only_deviation():
    """With the veto on and NaN present, every disagreement is a NaN inside the dry window
    where the reference said onset and the rule (failing closed) says no."""
    cfg = (1.0, 3, 7, 21)
    rng = np.random.default_rng(5)
    rule = LegacyRule(*cfg)
    n_series, n_steps = 400, 60
    rain = _random_rain(rng, (n_series, n_steps), nan_frac=0.03)
    da = xr.DataArray(rain, dims=("series", "step"))
    mask = rule(da, 20.0, dim="step").values
    window_has_nan = lead_sum(da, "step", rule.dry_extent).isnull().values

    n_diff = 0
    for s in range(n_series):
        for day in range(1, n_steps + 1):
            expected = bool(ref.detect_onset(day, rain[s], 20.0, **_kw(*cfg)))
            got = mask[s, day - 1]
            if got != expected:
                n_diff += 1
                assert expected and not got, "rule must only ever be stricter than the reference"
                assert window_has_nan[s, day - 1], "deviation without NaN in the dry window"
    assert n_diff > 0, "fixture produced no NaN-in-dry-window cases; test is vacuous"


# --------------------------------------------------------------------------- #
# 2. Observed xarray path
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservedPathLegacyRule(LegacyRule):
    """LegacyRule with the *original observed path's* dry-veto quirks:
    window [d, d+dry_extent] (one day longer) and veto skipped when it overruns."""

    name = "legacy_observed_path_reference"

    def __call__(self, rain, thresh, *, dim):
        from momp.stats.onset_rule import finalize, lead_all

        w = self.wet_spell
        ok = lead_all(rain >= self.wet_init, dim, w) & (lead_sum(rain, dim, w) > thresh)
        if self.dry_active:
            ds, de = self.dry_spell, self.dry_extent
            dry_end = (rain < self.wet_init).astype(float).rolling({dim: ds}, min_periods=ds).sum() == ds
            n_runs = dry_end.astype(float).rolling({dim: de + 1}, min_periods=1).sum().shift({dim: -de})
            ok = ok & ~(n_runs > 0).fillna(False)
        return finalize(ok)


def _observed_fixture(seed=1, year=2001, nan_frac=0.02):
    rng = np.random.default_rng(seed)
    time = pd.date_range(f"{year}-04-15", f"{year}-09-30", freq="D")
    lats, lons = np.arange(5.0, 13.0), np.arange(30.0, 42.0)
    rain = xr.DataArray(
        _random_rain(rng, (len(lats), len(lons), len(time)), nan_frac=nan_frac),
        dims=("lat", "lon", "time"),
        coords={"lat": lats, "lon": lons, "time": time},
    )
    thresh = xr.DataArray(rng.uniform(10, 30, size=(len(lats), len(lons))), dims=("lat", "lon"), coords={"lat": lats, "lon": lons})
    return rain, thresh


def compare_observed(rule, cfg, mok, *, observed_fn=ref.detect_observed_onset, year=2001):
    """Run ROMP-style observed detection and the rule on the same search window.

    Returns (n_cells_differing, detail, n_expected_onsets).
    """
    rain, thresh = _observed_fixture(year=year, nan_frac=0.0 if _dry_active(cfg) else 0.02)
    start_date, end_date = (year, 5, 1), (year, 7, 31)
    expected = observed_fn(rain, thresh, year, **_kw(*cfg), start_date=start_date, end_date=end_date, fallback_date=None, mok=mok)

    search_start = datetime(year, *(mok or start_date[1:]))
    search_end = datetime(year, *end_date[1:]) + pd.Timedelta(days=47)  # extend_end_day default
    subset = rain.sel(time=slice(search_start, search_end))
    idx = first_onset_index(rule(subset, thresh, dim="time"), "time").values
    times = subset.time.values
    got = np.where(idx >= 0, times[np.clip(idx, 0, None)], np.datetime64("NaT")).astype("datetime64[ns]")
    exp = expected.values.astype("datetime64[ns]")

    equal = (np.isnat(exp) & np.isnat(got)) | (exp == got)
    bad = np.argwhere(~equal)
    detail = "\n".join(
        f"  lat={rain.lat.values[i]} lon={rain.lon.values[j]} observed={exp[i, j].astype('datetime64[D]')} rule={got[i, j].astype('datetime64[D]')}"
        for i, j in bad[:10]
    )
    return len(bad), detail, int((~np.isnat(exp)).sum())


@pytest.mark.parametrize("cfg", LEGACY_CONFIGS, ids=str)
@pytest.mark.parametrize("mok", [None, (5, 15)])
def test_observed_path_variant_matches_reference_exactly(cfg, mok):
    n_bad, detail, n_onsets = compare_observed(ObservedPathLegacyRule(*cfg), cfg, mok)
    assert n_bad == 0, f"cfg={cfg} mok={mok}: {n_bad} cells differ\n{detail}"
    assert n_onsets > 0, "fixture never triggers observed onset; test is vacuous"


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(
            cfg,
            marks=pytest.mark.xfail(
                strict=True,
                reason="original observed path used a dry window one day longer than the forecast "
                "path and skipped the veto at the series tail; LegacyRule follows the forecast path",
            ),
        )
        if _dry_active(cfg)
        else cfg
        for cfg in LEGACY_CONFIGS
    ],
    ids=str,
)
@pytest.mark.parametrize("mok", [None, (5, 15)])
def test_legacy_rule_vs_reference_observed(cfg, mok):
    n_bad, detail, _ = compare_observed(LegacyRule(*cfg), cfg, mok)
    assert n_bad == 0, f"cfg={cfg} mok={mok}: {n_bad} cells differ\n{detail}"
