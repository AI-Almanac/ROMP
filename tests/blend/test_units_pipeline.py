"""One units specification, one reader, one threshold path, one detection adapter: the `units:` section of a run config,
UnitsSpec, reading observations and forecasts onto units, lead selection, and the observed-onset fingerprint."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.config import flat_bridge
from hcwf.io import input as io_input
from hcwf.lib import config_schema as schema
from hcwf.onset import regions as rg
from hcwf.spatial import EvaluationUnits, UnitsSpec, UnitsSpecError
from hcwf.stats import observed_cache

LAT, LON = np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0])


def test_a_spec_needs_a_source_for_its_mode_and_names_one_source():
    assert UnitsSpec().how == "cells" and UnitsSpec(weights="w.csv").how == "regrid"
    assert UnitsSpec(weights="w.csv", mode="cell_transform").how == "cell_transform"
    for bad in (dict(mode="regrid"), dict(mode="cells", weights="w.csv"), dict(shapefile="a", weights="b"),
                dict(shapefile="a.shp"), dict(mode="nearest"), dict(boundary="edge")):
        with pytest.raises(UnitsSpecError):
            UnitsSpec(**bad)


def test_cells_units_are_cut_by_box_and_ids_and_ordered_by_id():
    spec = UnitsSpec(box={"lat_min": 1.0}, id_digits=1, id_format="fixed")
    units, positions = spec.select(lat=LAT, lon=LON)
    assert units.ids[positions].tolist() == ["1.0_10.0", "1.0_11.0", "2.0_10.0", "2.0_11.0"]
    _, only = spec.select(lat=LAT, lon=LON, keep_ids=["2.0_11.0", "1.0_10.0"])
    assert units.ids[only].tolist() == ["1.0_10.0", "2.0_11.0"]
    with pytest.raises(ValueError, match="No cells fall"):
        UnitsSpec(box={"lat_min": 50.0}).select(lat=LAT, lon=LON)


def test_dissemination_restricts_the_inputs_only_in_scope_inputs(tmp_path):
    listed = tmp_path / "d.csv"
    pd.DataFrame({"id": ["b", "c"]}).to_csv(listed, index=False)
    ids = ["a", "b", "c"]
    for scope, kept in (("inputs", ["b", "c"]), ("training", ["a", "b", "c"])):
        units, positions = UnitsSpec(dissemination=str(listed), dissemination_scope=scope).select(ids=ids)
        assert units.ids[positions].tolist() == kept
        assert units.subsets["dissemination"].tolist() == [False, True, True]


def test_the_region_units_block_is_the_same_spec():
    assert rg.UnitsSpec is UnitsSpec


def test_the_units_section_flattens_to_flat_keys_and_extends_the_regions():
    errors = []
    flat, _, _ = schema.flatten_sections({"region": "kenya", "units": {"weights": "w.csv", "mode": "cell_transform",
                                                                       "box": {"lat_min": 1}}}, errors)
    assert not errors and flat["units_weights"] == "w.csv" and flat["units_mode"] == "cell_transform"
    assert UnitsSpec.from_flat({"region": "kenya"}) is None   # no units keys: grid cells
    spec = UnitsSpec.from_flat({**flat, "region": "kenya"})
    assert spec.how == "cell_transform" and spec.box == (("lat_min", 1.0),)


def test_units_mode_and_weights_may_differ_by_source():
    kwargs = {"units_mode": {"observations": "cell_transform", "ens": "regrid"},
              "units_weights": {"observations": "o.csv", "ens": "m.csv"}, "units_shapefile": "s.shp",
              "units_unit_id_col": "id"}
    obs, ens = UnitsSpec.from_flat(kwargs, "observations"), UnitsSpec.from_flat(kwargs, "ens")
    assert (obs.how, obs.weights, obs.shapefile) == ("cell_transform", "o.csv", None)
    assert (ens.how, ens.weights) == ("regrid", "m.csv")


def test_strict_lead_selection_orders_renumbers_and_checks_the_horizon():
    da = xr.DataArray(np.arange(6.0), dims="step", coords={"step": [3, 1, 2, 5, 4, 6]})
    out = io_input.select_leads(da, 2, 5, strict=True)
    assert out["lead_day"].values.tolist() == [2, 3, 4, 5] and out["step"].values.tolist() == [1, 2, 3, 4]
    assert out.values.tolist() == [2.0, 0.0, 4.0, 3.0]
    with pytest.raises(ValueError, match="exactly days 1..4"):
        io_input.select_leads(da, 1, 6, strict=True, rain_horizon=(4, "exact"))
    io_input.select_leads(da, 1, 6, strict=True, rain_horizon=(4, "truncate"))
    gap = xr.DataArray(np.arange(3.0), dims="step", coords={"step": [1, 2, 4]})
    with pytest.raises(ValueError, match="consecutive"):
        io_input.select_leads(gap, 1, 4, strict=True)


def test_hcwfs_lead_selection_drops_step_zero_and_cuts_at_the_last_step():
    da = xr.DataArray(np.arange(6.0), dims="step", coords={"step": [0, 1, 2, 3, 4, 5]})
    assert io_input.select_leads(da, None, 3)["step"].values.tolist() == [1, 2, 3]
    assert io_input.select_leads(da)["step"].values.tolist() == [1, 2, 3, 4, 5]


def test_members_are_named_or_cut_by_number():
    da = xr.DataArray(np.zeros(4), dims="member", coords={"member": [0, 1, 2, 3]})
    assert io_input.select_members(da, [1, 3])["member"].values.tolist() == [1, 3]
    assert io_input.select_members(da, at_most=2, among=[1, 2, 3])["member"].values.tolist() == [1, 2]
    with pytest.raises(ValueError, match="none of its members"):
        io_input.select_members(da, at_most=-1)


def test_declared_units_win_over_the_configured_factor():
    declared = xr.DataArray(np.zeros(2), attrs={"units": "m"})
    assert io_input.declared_scale(declared, 7.0) == 1000.0
    assert io_input.declared_scale(xr.DataArray(np.zeros(2), attrs={"units": "kg m**-2"}), 7.0) is None
    assert io_input.declared_scale(xr.DataArray(np.zeros(2)), 25.4) == 25.4


def test_the_fingerprint_of_a_grid_run_has_its_six_parts_and_units_and_options_add_their_own(monkeypatch):
    from hcwf.lib.onset_config import onset_setup

    base = {"obs_dir": "/nowhere", "obs_file_pattern": ("{}.nc",), "obs_var": "p", "obs_unit_cvt": None,
            "region": "kenya", "years": [2001], "onset_rule": {
                "id": "x", "version": 1, "title": "t", "citation": "t", "wording": "t", "status": "draft",
                "notes": "t", "season": {"from": "01-01", "to": "12-31"}, "wet_day": "at least 1 mm",
                "dry_day": "less than 1 mm", "onset": {"window": "3 days", "require": [{"first day": "wet"}],
                                                      "reject if": []}},
            "obs_day_boundary": "00 UTC, labelled by start", "model_lead_label": "start", "threshold_years": [2000, 2001]}
    rain = xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "lat", "lon"), coords={"lat": [0.0, 1.0], "lon": [2.0, 3.0]})
    def parts(rain, kwargs):
        rc, _, select = flat_bridge.from_case_kwargs(kwargs)
        return observed_cache.fingerprint_parts(rain, {}, rc.observations, rc.area, rc.onset, rc.onset_options, select,
                                                observed_cache.observed_years(rc.evaluation))

    plain = parts(rain, base)
    assert set(plain) == {"obs_files", "obs_ingest", "cells", "definition", "thresholds", "years"}
    keyed = parts(rain, {**base, "missing": "zero"})
    assert set(keyed) == set(plain) | {"options"} and keyed["options"] != plain.get("options")
    unit = parts(None, {**base, "units_mode": "cells"})
    assert "units" in unit and "cells" not in unit
    assert onset_setup(base) is not None
