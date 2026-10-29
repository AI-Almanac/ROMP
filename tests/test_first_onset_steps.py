"""The block-wise first-onset search gives exactly what the rolling-window form gave."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.io.input import onset_steps_needed
from momp.stats import detect
from momp.stats.detect import _first_onset_steps


def _rain(seed=0, init=5, member=4, lat=3, lon=4, step=12, nan_fraction=0.05):
    """Build a random gamma-distributed rainfall array with some NaNs, dims (init_time, member, lat, lon, step)."""
    rng = np.random.default_rng(seed)
    values = rng.gamma(0.8, 4.0, size=(init, member, lat, lon, step)).astype(np.float32)
    values[rng.random(values.shape) < nan_fraction] = np.nan
    return xr.DataArray(
        values,
        dims=("init_time", "member", "lat", "lon", "step"),
        coords={
            "init_time": pd.date_range("2010-05-01", periods=init, freq="3D"),
            "member": np.arange(member),
            "lat": np.linspace(5.0, 7.0, lat),
            "lon": np.linspace(35.0, 38.0, lon),
            "step": np.arange(1, step + 1),
        },
    )


def _rolling_reference(rain, thresh, valid_step, *, wet_init, wet_spell, max_forecast_day):
    """The rolling-window formulation the block-wise search replaced, verbatim in effect."""
    wet_spell_ok = (
        (rain >= wet_init)
        .rolling(step=wet_spell, min_periods=wet_spell, center=False)
        .reduce(np.all)
        .shift(step=-(wet_spell - 1))
        .fillna(False)
        .astype(bool)
    )
    rolling_sum = (
        rain.rolling(step=wet_spell, min_periods=wet_spell, center=False)
        .sum()
        .shift(step=-(wet_spell - 1))
    )
    condition = wet_spell_ok & (rolling_sum > thresh)
    condition = condition.where(condition.step <= max_forecast_day, False)
    if valid_step is not None:
        condition = condition & xr.DataArray(
            valid_step,
            dims=("init_time", "step"),
            coords={"init_time": condition.init_time, "step": condition.step},
        )
    cond = condition.fillna(False).values.astype(bool)
    steps = condition.step.values.astype(float)
    return np.where(cond.any(axis=-1), steps[cond.argmax(axis=-1)], np.nan)


@pytest.mark.parametrize("wet_spell", [1, 3, 5])
@pytest.mark.parametrize("block", [1, 2, 1_000_000])
def test_matches_the_rolling_window_form(monkeypatch, wet_spell, block):
    """The block-wise search matches the rolling-window reference for several wet-spell lengths and block sizes."""
    monkeypatch.setattr(detect, "_ONSET_BLOCK_ELEMENTS", block)
    rain = _rain()
    kwargs = dict(wet_init=1, wet_spell=wet_spell, max_forecast_day=8)

    got = _first_onset_steps(rain, 6.0, None, **kwargs)

    np.testing.assert_array_equal(got, _rolling_reference(rain, 6.0, None, **kwargs))


def test_matches_with_a_threshold_field_and_a_lead_day_mask(monkeypatch):
    """The block-wise search matches the reference with a spatial threshold field and a per-lead-day validity mask."""
    monkeypatch.setattr(detect, "_ONSET_BLOCK_ELEMENTS", 50)
    rain = _rain(seed=1)
    rng = np.random.default_rng(2)
    thresh = xr.DataArray(
        rng.uniform(2.0, 12.0, size=(rain.sizes["lat"], rain.sizes["lon"])),
        dims=("lat", "lon"),
        coords={"lat": rain.lat, "lon": rain.lon},
    )
    valid_step = rng.random((rain.sizes["init_time"], rain.sizes["step"])) > 0.3
    kwargs = dict(wet_init=1, wet_spell=3, max_forecast_day=10)

    got = _first_onset_steps(rain, thresh, valid_step, **kwargs)

    np.testing.assert_array_equal(got, _rolling_reference(rain, thresh, valid_step, **kwargs))


def test_no_onset_anywhere_is_all_nan():
    """Dry data yields NaN for every first-onset step."""
    rain = _rain() * 0

    got = _first_onset_steps(rain, 1.0, None, wet_init=1, wet_spell=3, max_forecast_day=8)

    assert np.isnan(got).all()


@pytest.mark.parametrize(
    ("max_forecast_day", "wet_spell", "dry_extent", "needed"),
    [(30, 3, 0, 32), (30, 3, 7, 37), (30, 1, 0, 30), (30, 3, None, None)],
)
def test_the_loader_keeps_every_lead_day_either_detection_path_reads(
    max_forecast_day, wet_spell, dry_extent, needed
):
    """onset_steps_needed returns the number of lead days the onset detection needs, or None when dry_extent is None."""
    assert (
        onset_steps_needed(max_forecast_day=max_forecast_day, wet_spell=wet_spell, dry_extent=dry_extent)
        == needed
    )
