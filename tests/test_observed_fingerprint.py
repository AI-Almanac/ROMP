"""The observed-onset cache key from typed inputs equals the kwargs version it replaces (step S7 of
docs/review/runconfig_design.md), so the on-disk caches written by earlier runs are still hits.

`_oracle_config.fingerprint_parts(rain, thresh, kwargs)` is the frozen kwargs version. The typed
`observed_cache.fingerprint_parts(rain, thresh, obs, area, onset, options, select, years)` (and `parts_for`, which takes
them from a RunConfig) must give the same parts on grid cells, units, masks and a reference onset.
"""
import typed_io
import flat_values
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import _oracle_config as old
from typed_io import flat_config
from hcwf.config.model import RunConfig, Selection
from hcwf.onset.climatology import ReferenceOnset
from hcwf.onset import observed_cache

sys.path.insert(0, str(Path(__file__).parent))
from test_oracle_config import build_world, fixture_configs  # noqa: E402
from test_run_experience import LAT, LON  # noqa: E402

THRESHOLDS = {"onset_thresh": 20.0}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{name: RunConfig} of the fixture configs (grid cells, two models, a model reference, an ensemble)."""
    world = build_world(tmp_path_factory.mktemp("fp_world"))
    paths = fixture_configs(world, tmp_path_factory.mktemp("fp_cfgs"))
    return {name: RunConfig.from_flat(flat_config(path)) for name, path in paths.items()}


@pytest.fixture(scope="module")
def rain():
    return xr.DataArray(np.zeros((2, len(LAT), len(LON))), dims=("time", "lat", "lon"),
                        coords={"lat": LAT, "lon": LON})


def typed(rain, thresh, kwargs):
    """The typed fingerprint parts of the inputs behind `kwargs`, (`typed_io.from_case_kwargs` builds them)."""
    rc, _, select = typed_io.from_case_kwargs(kwargs)
    return observed_cache.parts_for(rain, thresh, rc, select)


def assert_same(rain, kwargs, thresh=THRESHOLDS):
    """The kwargs version, the typed one built from the flat inputs and `parts_for` agree; returns the parts."""
    expected = old.fingerprint_parts(rain, thresh, kwargs)
    assert typed(rain, thresh, kwargs) == expected
    return expected


def test_cells_parts_equal_the_kwargs_version_for_every_fixture_run_and_case(runs, rain):
    for name, rc in runs.items():
        for case in rc.cases():
            kwargs = flat_values.case_kwargs(rc, case, "deterministic")
            parts = assert_same(rain, kwargs)
            assert set(parts) == {"obs_files", "obs_ingest", "cells", "definition", "thresholds", "years"}, name
            # directly, from the run's own specs
            assert observed_cache.fingerprint_parts(
                rain, THRESHOLDS, rc.observations, rc.area, rc.onset, rc.onset_options, Selection(),
                observed_cache.observed_years(rc.evaluation)) == parts


def test_the_parts_do_not_depend_on_the_model_or_window(runs, rain):
    rc = runs["two_models_windows"]
    parts = {observed_cache.fingerprint(typed(rain, THRESHOLDS, flat_values.case_kwargs(rc, case, "deterministic")))
             for case in rc.cases()}
    assert len(parts) == 1


MASKS = [{"land": True}, {"country": ["Ethiopia", "Kenya"], "resolution": "110m"},
         {"file": "/nowhere/mask.nc", "variable": "m", "keep_value": 2},
         {"country": ["Kenya"], "condition": {"file": "/nowhere/f.nc", "variable": "v", "above": 10},
          "cell_rule": "overlap_fraction", "overlap_fraction": 0.3}]


@pytest.mark.parametrize("mask", MASKS, ids=["land", "country", "file", "condition"])
def test_mask_parts_equal_the_kwargs_version(runs, rain, mask):
    rc = runs["det_climatology"]
    kwargs = {**flat_values.case_kwargs(rc, rc.cases()[0], "deterministic"), "mask": mask,
              "nc_mask": "/nowhere/nc.nc", "polygon": True}
    parts = assert_same(rain, kwargs)
    bare = assert_same(rain, {**kwargs, "mask": None, "nc_mask": None, "polygon": False})
    assert parts["cells"] != bare["cells"]


EXPLICIT_DEFAULTS = [{"land": True, "cell_rule": "centre"}, {"land": True, "keep_value": 1, "resolution": "50m"},
                     {"country": ["Kenya"], "land": False, "overlap_fraction": 0.5}]


@pytest.mark.parametrize("mask", EXPLICIT_DEFAULTS, ids=["cell_rule", "keep_value", "land_false"])
def test_a_default_written_explicitly_in_the_mask_keeps_the_kwargs_key(runs, rain, mask):
    """The key hashes the `mask:` mapping as given, so a key written with its default value is a different key from the
    mapping without it, exactly as before the typed AreaSpec (the AreaSpec keeps the mapping as given)."""
    rc = runs["det_climatology"]
    kwargs = {**flat_values.case_kwargs(rc, rc.cases()[0], "deterministic"), "mask": mask}
    parts = assert_same(rain, kwargs)
    without = {key: value for key, value in mask.items() if key in ("land", "country") and value}
    assert without != mask and parts["cells"] != assert_same(rain, {**kwargs, "mask": without})["cells"]
    run = typed_io.from_case_kwargs(kwargs)[0]
    assert run.area.mask_given == mask
    assert observed_cache.parts_for(rain, THRESHOLDS, run)["cells"] == parts["cells"]


def test_option_parts_equal_the_kwargs_version(runs, rain):
    rc = runs["det_climatology"]
    kwargs = flat_values.case_kwargs(rc, rc.cases()[0], "deterministic")
    base = assert_same(rain, kwargs)
    for change in ({"missing": "zero"}, {"onset_horizon": "truncate"}, {"rain_resolution": "exact", "gaps": "refuse"},
                   {"negatives": "keep", "missing": "keep", "onset_horizon": "censor"}):
        parts = assert_same(rain, {**kwargs, **change})
        assert "options" in parts and parts["options"] != base.get("options")


def test_grid_point_and_reference_onset_parts_equal_the_kwargs_version(runs, rain):
    rc = runs["det_climatology"]
    kwargs = flat_values.case_kwargs(rc, rc.cases()[0], "deterministic")
    point = assert_same(rain, {**kwargs, "grid_point": True, "lat_select": 8.0, "lon_select": 40.0})
    other = assert_same(rain, {**kwargs, "grid_point": True, "lat_select": 6.0, "lon_select": 40.0})
    assert point["cells"] != other["cells"] != assert_same(rain, kwargs)["cells"]
    table = pd.DataFrame({"id": ["a"], "onset_date": [pd.Timestamp("2020-06-01")]})
    for reference in (ReferenceOnset(month_day="06-02"), ReferenceOnset(table=table)):
        parts = assert_same(rain, {**kwargs, "reference_onset": reference})
        assert "options" in parts
        assert typed(rain, THRESHOLDS, {**kwargs, "reference_onset": reference}) == parts


def test_units_parts_equal_the_kwargs_version(runs, rain):
    rc = runs["det_climatology"]
    kwargs = flat_values.case_kwargs(rc, rc.cases()[0], "deterministic")
    for units in ({"units_mode": "cells", "region": "kenya"},
                  {"units_mode": "cells", "region": "kenya", "missing": "zero", "keep_unit_ids": ["b", "a"]}):
        parts = assert_same(None, {**kwargs, **units})
        assert "units" in parts and "cells" not in parts


def test_the_parts_follow_the_years_the_thresholds_and_the_files(runs, rain):
    rc = runs["det_climatology"]
    kwargs = flat_values.case_kwargs(rc, rc.cases()[0], "deterministic")
    base = assert_same(rain, kwargs)
    changed = assert_same(rain, {**kwargs, "years": (2019,), "years_clim": (2006, 2007)})
    assert changed["years"] != base["years"] and changed["obs_files"] != base["obs_files"]
    assert assert_same(rain, kwargs, {"onset_thresh": 25.0})["thresholds"] != base["thresholds"]


def test_obs_files_equal_the_oracles(runs):
    rc = runs["det_climatology"]
    kwargs = flat_values.case_kwargs(rc, rc.cases()[0], "deterministic")
    years = observed_cache.observed_years(rc.evaluation)
    assert old._fp_obs_files(kwargs, years) == observed_cache.observation_files(rc.observations, None, years)
