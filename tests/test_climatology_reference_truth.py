"""The climatology references are scored against the run's own observed onsets and leave the scored year out by default."""

import numpy as np
import pytest

import hcwf.references.conditional as conditional
from pair_frames import pairs_equal
from hcwf.config.mapping import build_config
from hcwf.io.forecasts import initialization_dates
from hcwf.metrics.bins import create_climatological_forecast_obs_pairs
from hcwf.references.climatology import climatological_onset_dataset
from hcwf.runs import check as check_mod
from hcwf.runs.resolved import resolved_config_text
from hcwf.scoring.pairs import prepare_pairs
from test_hcwf_conditional_climatology import CLIM_YEARS, DAY_BINS, TARGET, _observed, typed
import test_hcwf_conditional_climatology as tiny_world

config = tiny_world.config   # the fixture of those tests: the flat config of model A against a climatology

OUTSIDE = tuple(range(2000, 2012))
"""Climatology years that leave the target year 2013 outside the climatology period."""


def _prepared(config, **changes):
    """(rc, case, select, config, prepared) of the config with the changes."""
    config = dict(config, **changes)
    rc, case, select = typed(config)
    prepared = prepare_pairs(rc, case.model, select)
    return rc, case, select, config, prepared


@pytest.fixture
def no_second_detection(monkeypatch):
    """Fail when a reference detects a year again instead of using the onsets the model's pass made."""
    def boom(*args, **kwargs):
        raise AssertionError("the reference detected the observed onsets again")
    monkeypatch.setattr(conditional, "_detect_observed", boom)


@pytest.mark.parametrize("reference", ["climatology", "conditional_climatology"])
def test_a_year_outside_the_climatology_period_is_scored_against_the_runs_observed_onsets(config, reference,
                                                                                         no_second_detection):
    """Target 2013 is not a climatology year: it is scored, truth = the observed onsets the models are scored against."""
    rc, case, select, config, prepared = _prepared(config, ref_model=reference, years_clim=OUTSIDE)
    clim_onset = climatological_onset_dataset(rc, case, select)
    assert TARGET not in clim_onset.year.values
    pairs = prepared.climatology.select(DAY_BINS)
    assert pairs.n_cases
    observed = _observed(config)
    inits = initialization_dates(TARGET, rc.evaluation)
    if reference == "climatology":
        direct = create_climatological_forecast_obs_pairs(clim_onset, TARGET, inits, observed, day_bins=DAY_BINS,
                                                          leave_one_out=True)
        assert pairs_equal(pairs, direct.select(DAY_BINS))
        assert set(pairs.members) == {len(OUTSIDE)}   # nothing to leave out
        with pytest.raises(ValueError, match="not found"):   # the climatology years alone cannot give the truth
            create_climatological_forecast_obs_pairs(clim_onset, TARGET, inits, day_bins=DAY_BINS)
    else:
        direct = conditional.create_conditional_climatological_forecast_obs_pairs(
            clim_onset, TARGET, inits, observed, day_bins=DAY_BINS)
        assert pairs_equal(pairs, direct.select(DAY_BINS))


def test_inside_the_period_the_truth_is_the_climatology_datasets_slice(config):
    """For a climatology year the run's observed onsets equal clim_onset.sel(year) exactly."""
    rc, case, select, config, prepared = _prepared(config, ref_model="climatology")
    clim_onset = climatological_onset_dataset(rc, case, select)
    observed = _observed(config)
    inside = clim_onset.sel(year=TARGET)
    assert observed.dims == inside.dims and observed.dtype == inside.dtype
    assert np.array_equal(observed.values, inside.values, equal_nan=True)
    assert all(np.array_equal(observed[d].values, inside[d].values) for d in ("lat", "lon"))
    inits = initialization_dates(TARGET, rc.evaluation)
    for leave_one_out in (False, True):
        default = create_climatological_forecast_obs_pairs(clim_onset, TARGET, inits, day_bins=DAY_BINS,
                                                           leave_one_out=leave_one_out)
        given = create_climatological_forecast_obs_pairs(clim_onset, TARGET, inits, observed, day_bins=DAY_BINS,
                                                         leave_one_out=leave_one_out)
        assert pairs_equal(default, given)


def test_leave_one_out_drops_exactly_the_target_years_member(config, no_second_detection):
    """Inside the period 13 of 14 years are members (14 with none); outside all 12 are, with either setting."""
    members = {}
    for label, years_clim in (("inside", CLIM_YEARS), ("outside", OUTSIDE)):
        for mode in ("leave_one_year_out", "none"):
            *_, prepared = _prepared(config, ref_model="climatology", years_clim=years_clim, climatology_cv=mode)
            members[label, mode] = set(prepared.climatology.members)
    assert members == {("inside", "leave_one_year_out"): {13}, ("inside", "none"): {14},
                       ("outside", "leave_one_year_out"): {12}, ("outside", "none"): {12}}
    # the default is leave_one_year_out
    assert set(_prepared(config, ref_model="climatology")[-1].climatology.members) == {13}


def test_the_target_year_is_not_a_member_of_its_own_climatology(config):
    """Leaving the year out removes its own onset from the members: the pairs equal those of a climatology without it."""
    rc, case, select, config, prepared = _prepared(config, ref_model="climatology")
    clim_onset = climatological_onset_dataset(rc, case, select)
    inits = initialization_dates(TARGET, rc.evaluation)
    others = clim_onset.sel(year=[y for y in clim_onset.year.values if y != TARGET])
    expected = create_climatological_forecast_obs_pairs(others, TARGET, inits, clim_onset.sel(year=TARGET),
                                                        day_bins=DAY_BINS)
    assert pairs_equal(prepared.climatology.select(DAY_BINS), expected.select(DAY_BINS))


def test_with_the_option_off_the_pairs_are_the_old_ones(config):
    """climatology_cv none: every climatology year is a member and the truth is the target year's slice, as before."""
    rc, case, select, config, prepared = _prepared(config, ref_model="climatology", climatology_cv="none")
    clim_onset = climatological_onset_dataset(rc, case, select)
    old = create_climatological_forecast_obs_pairs(clim_onset, TARGET, initialization_dates(TARGET, rc.evaluation),
                                                   day_bins=DAY_BINS)   # the call as it was: no truth, no leave-out
    assert pairs_equal(prepared.climatology.select(DAY_BINS), old.select(DAY_BINS))


def test_the_option_is_in_the_resolved_config_and_the_check_with_its_source():
    """The effective climatology_cv is written to resolved_config.yaml and listed by --check, with where it came from."""
    base = {"region": "ethiopia",
            "observations": {"path": "/nowhere", "variable": "p", "units": "mm",
                             "day_boundary": "00 UTC, labelled by start", "file_pattern": "{}.nc"},
            "models": [{"name": "A", "path": "/nowhere", "variable": "tp", "units": "mm", "lead_label": "start",
                        "file_pattern": "{}.nc"}],
            "evaluation": {"start_date": [2019, 5, 1], "end_date": [2019, 6, 1]}}
    for reference, value, source in (({}, "leave_one_year_out", "default"),
                                     ({"climatology_cv": "none"}, "none", "run config")):
        flat = build_config({**base, "reference": reference})
        assert f"  climatology_cv: {value}  # {source}" in resolved_config_text(flat).splitlines()
        report = check_mod.CheckReport()
        check_mod._settings(flat, report)
        assert any(line.startswith("climatology_cv") and f": {value}  [{source}" in line
                   for line in report.text().splitlines()), report.text()
