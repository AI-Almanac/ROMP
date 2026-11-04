"""Blending against every onset definition in the registry: supported, or refused early with a clear message."""
import copy

import pytest

from hcwf.config.convert_specs import convert_blend
from hcwf.blend import recipe as R
from hcwf.blend.training import train
from hcwf.errors import HCWFConfigError, OnsetConfigError, RecipeError
from hcwf.onset import definition as definition_mod
from hcwf.onset.model import DerivedSeasons

from test_blend_forecast import blending_config, write_forecasts, write_observations

BASE = {"model_list": ("m",), "blend_years": {"training": [2000, 2001], "cv_holdout": [2001]}}


def _recipe(predictor: dict, formula: str) -> R.Recipe:
    """Return a one-model recipe with one rain predictor and one formula."""
    cfg = {**BASE, "blend_models": {"m": {"rain_predictors": [predictor]}}, "blend_formulas": {"f": formula}}
    return R.recipe_from_config(cfg)


CASES = {
    "numeric": ({"agg": "max", "window": 5}, "outcome ~ prob_clim_qx + max_m_5day_qx"),
    "token": ({"agg": "max", "window": "trigger"}, "outcome ~ prob_clim_qx"),
    "diff": ({"agg": "diff"}, "outcome ~ prob_clim_qx + diff_m_qx"),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("name", definition_mod.registry_ids())
def test_every_definition_passes_or_is_refused_clearly(name, case):
    definition = definition_mod.load(name)
    recipe = _recipe(*CASES[case])
    try:
        recipe.check(definition)
    except RecipeError as error:
        assert error.problems and all(isinstance(p, str) and len(p) > 20 for p in error.problems)
        return
    assert not isinstance(definition.seasons, DerivedSeasons)
    if case == "token":
        assert definition.windows()["trigger"] is not None
    if case == "diff":
        assert definition.thresholds and definition.thresholds[0][1].kind in ("mean_total", "quantile_total")


def _loaded(root, tmp_path, definition, override=None, predictor=None, **onset):
    """Return the synthetic region's run on a registry definition (a RunConfig and its OnsetSetup)."""
    from hcwf.config.build import resolve
    from hcwf.config.onset import onset_setup
    from hcwf.config.model import RunConfig

    mapping = copy.deepcopy(convert_blend(blending_config(root)).mapping)
    mapping["onset"] = {"definition": definition, **onset, **({"override": override} if override else {})}
    if predictor:
        mapping["blend"]["models"]["ens"]["rain_predictors"] = [predictor[0]]
        mapping["blend"]["formulas"] = {"blended_model": f"outcome ~ prob_clim_qx + {predictor[1]}"}
    cfg = dict(resolve(mapping, write=False, tmp=tmp_path).values)
    cfg["cache_dir"] = str(tmp_path / "cache")
    setup = onset_setup(cfg)
    R.recipe_from_config(cfg, setup)
    return RunConfig.from_flat({**cfg, "parallel_workers": 1}), setup


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    root = tmp_path_factory.mktemp("region")
    write_observations(root / "obs")
    write_forecasts(root / "ens")
    return root


def _write_full_years(folder):
    """Write whole-year observation files (the curve's baseline is a whole-year mean)."""
    import datetime as dt

    import numpy as np
    import pandas as pd
    import xarray as xr

    from test_blend_forecast import LATS, LONS, OBS_YEARS, onset_date, rain

    folder.mkdir()
    for year in OBS_YEARS:
        dates = [dt.date(year, 1, 1) + dt.timedelta(days=i) for i in range(365 + (year % 4 == 0))]
        data = np.stack([np.stack([rain(dates, onset_date(year, i * 2 + j)) for j in range(len(LONS))], axis=-1)
                         for i in range(len(LATS))], axis=1)
        xr.Dataset({"precip": (("time", "lat", "lon"), data)},
                   coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def _expected(first_day: tuple[int, int], shift: int):
    """Return the observed onset days over (cell, year) the synthetic rain gives: the onset date plus `shift` days,
    counted from `first_day` of the year."""
    import datetime as dt

    import numpy as np

    from test_blend_forecast import OBS_YEARS, onset_date

    return np.array([[(onset_date(year, cell) + dt.timedelta(days=shift) - dt.date(year, *first_day)).days
                      for year in OBS_YEARS] for cell in range(4)], dtype=float)


def test_a_first_passage_definition_not_otherwise_tested_blends(region, tmp_path):
    # sivakumar: 3 days with at least 20 mm, so the candidate is the day before the 10-10-10 mm days begin
    rc, setup = _loaded(region, tmp_path, "sivakumar",
                        predictor=({"agg": "max", "window": "trigger"}, "max_ens_3day_qx"))
    result = train(rc)
    assert result.inputs.observed.season_start == "05-01"
    assert (result.inputs.observed.data["onset_day"].values == _expected((5, 1), -1)).all()
    assert "max_ens_3day_week1" in result.inputs.weekly.columns


def test_a_curve_definition_blends_with_numeric_windows(tmp_path_factory, tmp_path):
    # liebmann_marengo: the onset is the day after the minimum of the cumulative anomaly, here the first rainy day
    root = tmp_path_factory.mktemp("full")
    _write_full_years(root / "obs")
    write_forecasts(root / "ens")
    rc, setup = _loaded(root, tmp_path, "liebmann_marengo", threshold_years=[2000, 2013],
                        predictor=({"agg": "max", "window": 5}, "max_ens_5day_qx"))
    result = train(rc)
    assert result.inputs.observed.season_start == "01-01"
    assert (result.inputs.observed.data["onset_day"].values == _expected((1, 1), 0)).all()
    assert "max_ens_5day_week1" in result.inputs.weekly.columns


def test_blending_refuses_a_curve_diff_and_derived_seasons_before_reading_data(region, tmp_path):
    with pytest.raises(HCWFConfigError, match=r"'diff' predictor .*cumulative-anomaly curve"):
        _loaded(region, tmp_path, "liebmann_marengo", threshold_years=[2000, 2013],
                predictor=({"agg": "diff"}, "diff_ens_qx"))
    with pytest.raises(HCWFConfigError, match="harmonic season analysis"):
        _loaded(region, tmp_path, "dunning", threshold_years=[2000, 2013], season="1",
                predictor=({"agg": "max", "window": 5}, "max_ens_5day_qx"))


def _season_setup(season: str):
    """Return a stand-in onset setup whose definition has a May and a November season."""
    import dataclasses
    from types import SimpleNamespace

    from hcwf.onset.model import DayRef, Season

    seasons = (Season("first", DayRef(5, 1), DayRef(7, 31)), Season("second", DayRef(11, 15), DayRef(2, 28)))
    return SimpleNamespace(definition=dataclasses.replace(definition_mod.load("icpac"), seasons=seasons), season=season)


@pytest.mark.parametrize("season, start", [("first", "05-01"), ("second", "11-15")])
def test_blending_works_in_the_season_the_onset_setup_selects(season, start):
    assert R.blend_season_start(_season_setup(season)) == start


def test_a_season_that_starts_at_a_climatological_date_is_refused():
    import dataclasses

    from hcwf.onset.model import DayRef, Season

    setup = _season_setup("first")
    seasons = (Season("first", DayRef(climatological="onset", offset_days=-50), DayRef(7, 31)),)
    setup.definition = dataclasses.replace(setup.definition, seasons=seasons)
    with pytest.raises(RecipeError, match="climatological date"):
        R.blend_season_start(setup)


def test_the_onset_config_already_refuses_several_seasons_without_a_choice(tmp_path):
    from hcwf.config.onset import onset_setup

    text = (definition_mod.REGISTRY_DIR / "icpac.yaml").read_text()
    path = tmp_path / "two.yaml"
    path.write_text(text.replace("season:\n  from: 05-01\n  to: 09-30",
                                 "seasons:\n  first: {from: 05-01, to: 07-31}\n  second: {from: 11-15, to: 02-28}"))
    cfg = {"onset_definition": str(path), "obs_day_boundary": "00 UTC, labelled by start",
           "model_lead_label": "start"}
    with pytest.raises(OnsetConfigError, match="several seasons"):
        onset_setup(cfg)
    assert onset_setup({**cfg, "onset_season": "second"}).season == "second"
