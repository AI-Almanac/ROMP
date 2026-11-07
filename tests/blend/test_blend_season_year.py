"""Blend year labels are the season's year (the year the season starts): a season crossing New Year (here 1 October to
28 February) keeps a January issue with the October ones of its season in the weekly features, the cross-validation folds
and a live forecast; a season inside one calendar year is labelled as before."""
import datetime as dt
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.blend import TrainedBlend
from hcwf.blend import features
from hcwf.blend import forecast as forecast_module
from hcwf.config.build import resolve
from hcwf.config.convert_specs import convert_blend
from hcwf.onset.climatology import ReferenceOnset
from hcwf.references import kde

LATS, LONS = [10.0, 10.25], [70.0, 70.25]
OBS_YEARS = range(2000, 2014)
FORECAST_YEARS = range(2010, 2014)
LEAD_DAYS = 40


# -- the one helper --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("start, end", [("05-01", "07-31"), ("03-01", "12-31"), ("01-01", "12-31"), ("06-15", "06-15")])
def test_a_season_inside_one_calendar_year_is_labelled_by_the_year_of_its_dates(start, end):
    rng = np.random.default_rng(5)
    dates = (np.datetime64("1996-01-01") + rng.integers(0, 365 * 30, 400).astype("timedelta64[D]")).astype("datetime64[ns]")
    np.testing.assert_array_equal(kde.season_years(dates, start, end), pd.DatetimeIndex(dates).year.to_numpy())


def test_a_january_date_of_a_season_crossing_new_year_is_in_the_year_before():
    dates = np.array(["2020-10-01", "2020-12-31", "2021-01-05", "2021-02-28", "2021-03-01"], dtype="datetime64[ns]")
    assert kde.season_years(dates, "10-01", "02-28").tolist() == [2020, 2020, 2020, 2020, 2021]
    assert kde.season_years(dates.reshape(1, -1), "10-01", "02-28").shape == (1, 5)
    assert kde.season_years([dt.date(2021, 1, 5)], "10-01", "01-31").tolist() == [2020]


def test_the_reference_onset_of_a_table_is_looked_up_by_the_season_year_of_each_issue():
    table = pd.DataFrame({"id": ["a", "a"], "year": [2020, 2021],
                          "ref_onset_date": [dt.date(2020, 11, 20), dt.date(2021, 11, 25)]})
    reference = ReferenceOnset(table=table)
    issues = np.array(["2020-12-01", "2021-01-05"], dtype="datetime64[ns]")
    got = features.forecast_starts("10-01", reference, issues, ["a"], ("ref",), season=("10-01", "02-28"))["ref"]
    assert got.values[:, 0].astype("datetime64[D]").tolist() == [dt.date(2020, 11, 20)] * 2   # both are in season 2020
    plain = features.forecast_starts("10-01", reference, issues, ["a"], ("ref",))["ref"]       # the calendar year, as before
    assert plain.values[:, 0].astype("datetime64[D]").tolist() == [dt.date(2020, 11, 20), dt.date(2021, 11, 25)]


# -- the issue date of a forecast ------------------------------------------------------------------------------------

def _blend(monkeypatch, start, end):
    monkeypatch.setattr(forecast_module, "blend_season_start", lambda setup: start)
    return SimpleNamespace(setup=None, recipe=SimpleNamespace(climatology=SimpleNamespace(issue_end=end)))


def test_a_january_issue_is_in_the_season_of_the_year_before(monkeypatch):
    blend = _blend(monkeypatch, "10-01", "02-28")
    held = {"m": [dt.date(2021, 1, 5)]}
    assert forecast_module._issue_problems(blend, None, held) == (dt.date(2021, 1, 5), 2020, [])
    assert forecast_module._issue_problems(blend, dt.date(2021, 1, 5), {"m": [dt.date(2021, 1, 5), dt.date(2021, 1, 12)]}
                                           )[1:] == (2020, [])


def test_files_holding_a_whole_season_crossing_new_year_are_one_season(monkeypatch):
    blend = _blend(monkeypatch, "10-01", "02-28")
    issues = [dt.date(2020, 10, 1) + dt.timedelta(days=7 * k) for k in range(22)]   # to 25 February 2021
    assert issues[-1].year == 2021
    assert forecast_module._issue_problems(blend, None, {"a": issues, "b": issues}) == (None, 2020, [])
    other = [dt.date(2021, 10, 1), dt.date(2021, 10, 8)]
    problems = forecast_module._issue_problems(blend, None, {"a": issues, "b": other})[2]
    assert "one initialisation per model, or one season" in problems[0]


def test_files_holding_a_season_inside_one_year_are_labelled_as_before(monkeypatch):
    blend = _blend(monkeypatch, "05-01", "07-31")
    issues = [dt.date(2020, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
    assert forecast_module._issue_problems(blend, None, {"a": issues}) == (None, 2020, [])
    assert forecast_module._issue_problems(blend, dt.date(2020, 6, 5), {"a": issues})[1] == 2020


# -- a blend on a season crossing New Year ---------------------------------------------------------------------------

def onset_date(year: int, cell: int, shift: int = 0) -> dt.date:
    """Onsets between 20 December and the end of January, varying by year and cell."""
    return dt.date(year, 12, 20) + dt.timedelta(days=(year * 7 + cell * 5) % 40 + shift)


def rain(dates, onset: dt.date) -> np.ndarray:
    days = np.array([(d - onset).days for d in dates])
    return np.where(days < 0, 0.0, np.where(days < 3, 10.0, 6.0)).astype("float32")


def write_observations(folder: Path):
    folder.mkdir()
    for year in OBS_YEARS:
        dates = [dt.date(year, 10, 1) + dt.timedelta(days=i) for i in range(180)]   # a season's file runs to March
        data = np.stack([np.stack([rain(dates, onset_date(year, i * 2 + j)) for j in range(len(LONS))], axis=-1)
                         for i in range(len(LATS))], axis=1)
        xr.Dataset({"precip": (("time", "lat", "lon"), data)},
                   coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def write_forecasts(folder: Path, years=FORECAST_YEARS, members=3):
    folder.mkdir(exist_ok=True)
    for year in years:
        issues = [dt.date(year, 10, 1) + dt.timedelta(days=7 * k) for k in range(18)]   # to 28 January of the next year
        data = np.zeros((len(issues), members, LEAD_DAYS, len(LATS), len(LONS)), dtype="float32")
        for t, issue in enumerate(issues):
            leads = [issue + dt.timedelta(days=d) for d in range(1, LEAD_DAYS + 1)]
            for m in range(members):
                for i in range(len(LATS)):
                    for j in range(len(LONS)):
                        data[t, m, :, i, j] = rain(leads, onset_date(year, i * 2 + j, shift=3 * (m - 1)))
        xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": pd.to_datetime(issues), "number": np.arange(members),
                           "day": np.arange(1, LEAD_DAYS + 1), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def crossing_config(root) -> dict:
    return {
        "name": "crossing", "data_root": str(root), "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}},
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": "ens", "files": {"folder": "ens"}, "lead_days": {"max": LEAD_DAYS},
                    "rain_predictors": [{"agg": "diff"}]}],
        "onset": {"threshold": {"mm": 20}, "season_start": "10-01",
                  "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}},
        "climatology": {"forecast_window": LEAD_DAYS, "min_onset_years": 5, "cv_by_year": False, "issue_end": "01-31"},
        "evaluation": {"years": {"training": "2010:2012", "cv_holdout": "2010:2012", "true_holdout": 2013},
                       "formulas": {"blended_model": "outcome ~ prob_clim_qx + diff_ens_qx"}},
    }


def _trained(rc):
    """What `hcwf.blend.training.train` does, with the observed onsets given (a season's observation file cannot cross
    New Year yet, so they are the known onsets of the synthetic rain)."""
    from dataclasses import replace

    from hcwf.blend import evaluation as evaluation_stage
    from hcwf.blend.recipe import blend_season_start
    from hcwf.blend.samples import ObservedOnsets
    from hcwf.blend.training import Inputs, TrainingResult
    from hcwf.io.thresholds import thresholds_of

    recipe, setup = rc.blend, rc.onset
    ids = np.array([f"{lat:.2f}_{lon:.2f}" for lat in LATS for lon in LONS])
    parts = []
    for year in OBS_YEARS:
        dates = np.array([np.datetime64(onset_date(year, k)) for k in range(len(ids))])
        days = kde.season_days(dates, (10, 1), label_years=year)[:, None]
        parts.append({"ids": ids, "years": np.array([year]), "onset_day": days, "onset_idx": days.copy(),
                      "present": np.ones((len(ids), 1), dtype=bool)})
    observed = ObservedOnsets.from_parts(parts, blend_season_start(setup))
    thresholds = thresholds_of(replace(rc, evaluation=replace(rc.evaluation, years=())), rc.models[0])
    blendable = observed.blendable(recipe.climatology.min_onset_years)
    forecasts = {"ens": features.forecast(rc, recipe, "ens", thresholds, None, keep_ids=blendable,
                                          years=rc.evaluation.years, workers=1, needed_only=True)}
    climatology = features.climatology(recipe, setup, observed, sorted(forecasts["ens"].years), workers=1)
    weekly = features.weekly(recipe, setup.definition, climatology, observed, forecasts)
    evaluation = evaluation_stage.evaluate(weekly, recipe, evaluation_stage.dissemination_ids(rc.observations.units, weekly),
                                           workers=1, budget_bytes=None, final=True)
    inputs = Inputs(thresholds=thresholds, reference=None, observed=observed, forecasts=forecasts,
                    climatology=climatology, weekly=weekly)
    return TrainingResult(rc=rc, inputs=inputs, evaluation=evaluation)


@pytest.fixture(scope="module")
def crossing(tmp_path_factory):
    from hcwf.config.model import RunConfig

    root = tmp_path_factory.mktemp("crossing")
    write_observations(root / "obs")
    write_forecasts(root / "ens")
    conversion = convert_blend(crossing_config(root))
    conversion.mapping["onset"]["rule"]["season"] = {"from": "10-01", "to": "02-28"}
    cfg = dict(resolve(conversion.mapping, write=False, tmp=Path(tempfile.mkdtemp(prefix="blend-run-"))).values)
    cfg["cache_dir"] = str(root / "cache")
    result = _trained(RunConfig.from_flat({**cfg, "parallel_workers": 1}))
    path = result.save(root / "trained") / "trained.blend"
    return root, result, TrainedBlend.load(path)


def test_the_weekly_year_is_the_season_year_so_a_cv_fold_keeps_a_whole_season(crossing):
    _, result, _ = crossing
    frame = result.inputs.weekly.frame()
    time = pd.to_datetime(frame["time"])
    assert (time.dt.month.isin([1, 2])).any() and (time.dt.month >= 10).any()   # issues on both sides of New Year
    assert (frame["year"] == kde.season_years(time.to_numpy(), "10-01", "01-31")).all()
    for year, issues in time.groupby(frame["year"]):
        assert issues.min() >= pd.Timestamp(f"{year}-10-01") and issues.max() <= pd.Timestamp(f"{year + 1}-01-31")
    january = time.dt.month == 1
    assert january.any() and (frame["year"][january] == time[january].dt.year - 1).all()   # the January issues are in the year before
    assert all(set(time[frame["year"] == y].dt.month) & {1} and set(time[frame["year"] == y].dt.month) & {10}
               for y in set(frame["year"][january]))                                  # a season in one fold: October and January
    folds = set(result.evaluation.yearly["year"])
    assert folds == {2010, 2011, 2012, 2013}   # no fold for 2014: January 2014 is in the season of 2013
    assert set(frame["year"]) == folds


def test_a_forecast_for_a_january_issue_date_is_made_in_its_season(crossing):
    root, _, blend = crossing
    live = root / "ens" / "2013.nc"
    january = blend.forecast({"ens": live}, issue_date="2014-01-07", workers=1)
    assert set(pd.to_datetime(january.weekly_table["time"]).dt.date) == {dt.date(2014, 1, 7)}
    season = blend.forecast({"ens": live}, workers=1)          # the whole season: October 2013 to January 2014
    times = pd.to_datetime(season.weekly_table["time"])
    assert times.min() < pd.Timestamp("2013-12-01") and times.max() >= pd.Timestamp("2014-01-01")
    row = season.weekly_table[times == pd.Timestamp("2014-01-07")].reset_index(drop=True)
    pd.testing.assert_frame_equal(row, january.weekly_table.reset_index(drop=True))
