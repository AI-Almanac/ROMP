"""Operational forecasts: a trained blend applied to live forecast files.

Only the live files (and, optionally, the issue date) come from the caller. The run config, onset definition, recipe,
reference data and observed onsets all come from the `TrainedBlend`, and the live files must have the structure the
blend was trained on (`momp.lib.dataset_detect.signature_differences`); every difference is reported before any work
starts. The live files are read through `momp.io.input` (only the tiles of the trained units), onsets are detected by
`momp.stats.detect` with the frozen definition and options, and the features are `momp.blend.features`'.

Operationally each raw model gives one file holding one initialisation, and that is the issue date forecast. Files
holding a season of issues (a replay) forecast every issue in it, or the one `issue_date` names.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from momp.blend import features, training
from momp.blend.recipe import CLIMATOLOGY_FEATURE
from momp.lib import dataset_detect

WEEKLY_FILENAME = "blended_forecast_probabilities.csv"
DAILY_FILENAME = "daily_onset_probabilities.csv"
WEEKLY_NETCDF = "blended_forecast_probabilities.nc"
DAILY_NETCDF = "daily_onset_probabilities.nc"

__all__ = ["DAILY_FILENAME", "Forecast", "IncompatibleForecast", "WEEKLY_FILENAME", "check_files", "run"]


class IncompatibleForecast(ValueError):
    """Live forecasts the trained blend cannot be applied to; lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Cannot forecast with this blend:\n" + "\n".join(f"  - {p}" for p in self.problems))


def _dataset(table: pd.DataFrame, columns: list[str]) -> xr.Dataset:
    """A forecast table (rows of id and issue date) as arrays over (time, id), NaN where a unit has no row."""
    time = pd.to_datetime(table["time"])
    times = np.sort(time.unique())
    ids = np.sort(table["id"].unique().astype(str))
    row, col = np.searchsorted(times, time.to_numpy()), np.searchsorted(ids, table["id"].astype(str).to_numpy())
    variables = {}
    for name in columns:
        grid = np.full((len(times), len(ids)), np.nan)
        grid[row, col] = table[name].to_numpy(dtype=float)
        variables[name] = (("time", "id"), grid)
    first = table.drop_duplicates("id").set_index(table.drop_duplicates("id")["id"].astype(str)).reindex(ids)
    for name in ("lat", "lon", "onset_threshold"):
        if name in table.columns:
            variables[name] = (("id",), first[name].to_numpy(dtype=float))
    return xr.Dataset(variables, coords={"time": times, "id": ids})


@dataclass
class Forecast:
    """Blended onset probabilities for the live issue dates.

    `weekly_table` holds one row per unit and issue date as blending's contract has it (id, time, onset_threshold, lat,
    lon, cv_week1..N, cv_later, then the climatology's clim_*); `daily_table` (the random forest blend's) id, time, lat,
    lon, p_day_1..D, p_later and the week bins cv_*. `weekly` and `daily` are the same probabilities as xarray
    Datasets over (time, id); `features` holds the weekly feature rows the probabilities come from."""

    weekly_table: pd.DataFrame
    features: pd.DataFrame
    daily_table: pd.DataFrame | None = None
    folder: Path | None = None          # where `save` wrote it

    @property
    def issue_dates(self) -> list[dt.date]:
        """The issue dates forecast, in order."""
        return sorted({pd.Timestamp(t).date() for t in self.weekly_table["time"]})

    @property
    def label(self) -> str:
        """The issue date, or first_last for a season."""
        dates = self.issue_dates
        return dates[0].isoformat() if len(dates) == 1 else f"{dates[0]}_{dates[-1]}"

    @property
    def weekly(self) -> xr.Dataset:
        columns = [c for c in self.weekly_table.columns if c.startswith(("cv_", "clim_"))]
        return _dataset(self.weekly_table, columns)

    @property
    def daily(self) -> xr.Dataset | None:
        if self.daily_table is None:
            return None
        table = self.daily_table
        days = [c for c in table.columns if c.startswith("p_day_")]
        data = _dataset(table, [c for c in table.columns if c.startswith(("cv_", "p_later"))])
        stack = _dataset(table, days)
        data["p_day"] = xr.concat([stack[c] for c in days], dim=pd.Index(np.arange(1, len(days) + 1), name="day"),
                                  ).transpose("time", "id", "day")
        return data

    def save(self, directory, *, netcdf: bool = False) -> Path:
        """Write the forecast's files into a new directory, all or nothing: blending's contract CSVs (weekly, and daily
        for a random forest blend). `netcdf` also writes the probabilities as NetCDF (off by default: nothing reads it)."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            self.weekly_table.to_csv(staging / WEEKLY_FILENAME, index=False)
            if netcdf:
                self.weekly.to_netcdf(staging / WEEKLY_NETCDF)
            if self.daily_table is not None:
                self.daily_table.to_csv(staging / DAILY_FILENAME, index=False)
                if netcdf:
                    self.daily.to_netcdf(staging / DAILY_NETCDF)
            if directory.exists():
                directory.rmdir()
            os.replace(staging, directory)
            self.folder = directory
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return directory


def _files(forecasts: dict) -> dict[str, list[Path]]:
    from momp.blend.trained import expand_files

    return {name: expand_files(value) for name, value in forecasts.items()}


def _model_spec(blend, name: str):
    return blend.recipe.model(name)


def _read(blend, name: str, path, *, selected: bool) -> tuple[dict, list]:
    model = _model_spec(blend, name)
    return dataset_detect.read_forecast_file(
        path, blend.entries[name], lead_days=model.lead_days, max_member=model.max_member, selected=selected)


def check_files(blend, forecasts: dict[str, list[Path]], *, fewer_members: bool = False,
                held: dict | None = None) -> list[str]:
    """Every way the live files differ from what the blend was trained on. With a `held` dict, the issue dates each
    model's files hold are collected into it."""
    problems = []
    trained, given = set(blend.recipe.model_names), set(forecasts)
    if given != trained:
        missing, extra = sorted(trained - given), sorted(given - trained)
        problems.append("forecasts must be given for exactly the trained models (" + ", ".join(sorted(trained)) + ")"
                        + (f"; missing {', '.join(missing)}" if missing else "")
                        + (f"; not trained on {', '.join(extra)}" if extra else ""))
    for name in sorted(trained & given):
        dates = []
        for path in forecasts[name]:
            try:
                live, issued = _read(blend, name, path, selected=False)
            except (OSError, ValueError, KeyError) as exc:
                problems.append(f"{name}: {Path(path).name}: {exc}")
                continue
            dates += issued
            differences = dataset_detect.signature_differences(blend.signatures[name], live,
                                                               fewer_members=fewer_members)
            if differences:
                problems.append(f"{name}: {Path(path).name}: " + "; ".join(differences))
        if held is not None:
            held[name] = sorted(set(dates))
            if not held[name]:
                problems.append(f"{name}: the live files hold no issue dates")
    return problems


def _issue_problems(blend, issue_date: dt.date | None, held: dict[str, list[dt.date]]):
    """(issue date, season year, problems). With no issue date given, files holding one initialisation each forecast
    it; files holding more forecast every issue of their one season."""
    problems = []
    if issue_date is None and held and all(len(dates) == 1 for dates in held.values()):
        inits = {name: dates[0] for name, dates in held.items()}
        if len(set(inits.values())) == 1:
            issue_date = next(iter(inits.values()))
        else:
            listed = ", ".join(f"{name} {date}" for name, date in inits.items())
            return None, None, [f"the live files are initialised on different dates ({listed}); every model needs "
                                "the same one"]
    if issue_date is not None:
        year = issue_date.year
        season = blend.setup.definition.seasons[0].start
        start = dt.date.fromisoformat(f"{year}-{season.month:02d}-{season.day:02d}")
        end = dt.date.fromisoformat(f"{year}-{blend.recipe.climatology.issue_end}")
        if not start <= issue_date <= end:
            problems.append(f"issue date {issue_date} is outside the season ({start} to {end})")
        for name, dates in held.items():
            if dates and issue_date not in dates:
                nearby = [d.isoformat() for d in dates if abs((d - issue_date).days) <= 7]
                problems.append(f"{name}: the live files have no {issue_date} issue"
                                + (f" (nearby: {', '.join(nearby)})" if nearby else ""))
        return issue_date, year, problems
    years = {d.year for dates in held.values() for d in dates}
    if len(years) != 1:
        return None, None, ["the live files must hold one initialisation per model, or one season (or give "
                            "issue_date)"]
    return None, years.pop(), problems


def run(blend, forecasts: dict, *, issue_date=None, fewer_members: bool = False, workers: int | None = None):
    """Blended probabilities for the live forecasts (model name -> file, glob or list).

    fewer_members lets an ensemble's live files hold only some of the members the blend was trained on: its onset
    probabilities are shares of fewer members, so coarser and noisier than the blend's training saw."""
    from momp.blend.random_forest import feature_names, feature_matrix, predict, weekly_from_daily

    if isinstance(issue_date, str):
        issue_date = dt.date.fromisoformat(issue_date)
    live = _files(forecasts)
    held: dict[str, list] = {}
    problems = check_files(blend, live, fewer_members=fewer_members, held=held)
    issue_date, year, issue_problems = _issue_problems(blend, issue_date, held)
    if problems or issue_problems:
        raise IncompatibleForecast(problems + issue_problems)

    cfg = {**blend.cfg, "model_paths_list": tuple(tuple(str(p) for p in live[m.name]) for m in blend.recipe.models)}
    count, budget = training.resources(cfg)
    workers = workers or count
    recipe, setup = blend.recipe, blend.setup
    thresholds = _thresholds(blend)
    keep = frozenset(blend.units)
    forecast_onsets = {}
    for model in recipe.models:
        forecast_onsets[model.name] = features.forecast(
            cfg, recipe, setup, model.name, thresholds, blend.reference, keep_ids=keep,
            member_ids=blend.signatures[model.name].get("members"), units=blend.frozen_units[model.name],
            issue_dates=None if issue_date is None else (issue_date,), workers=workers, budget_bytes=budget)
    # One season: the climatology has nothing to spread over a worker pool (starting it costs more than the fit).
    climatology = features.climatology(recipe, setup, blend.observed, [year], test_years=(year, year), workers=1)
    # No trimming: a forecast must not know when the onset came.
    weekly = features.weekly(recipe, setup.definition, climatology, blend.observed, forecast_onsets, trim=False)
    rows = np.arange(len(weekly))
    if issue_date is not None:
        rows = rows[np.asarray(weekly.take("time")).astype("datetime64[D]") == np.datetime64(issue_date, "D")]
    issue_rows = rows
    for column in _input_columns(blend.model, weekly):
        rows = rows[pd.notna(weekly.take(column, rows))]
    if not len(rows):
        raise IncompatibleForecast(["no unit and issue date has every feature the blend needs"])
    table = weekly.frame(rows)
    daily = None
    if blend.forest is not None:
        # Every sample of the issue dates: the forest imputes missing inputs.
        fitted = blend.forest
        inputs = features.daily_inputs(recipe, climatology, blend.observed, forecast_onsets,
                                       max_day=fitted.params.max_day, trim=False)
        models = list(recipe.model_names)
        names = feature_names(inputs, models)
        if [names[i] if i < len(names) else None for i in fitted.columns] != list(fitted.features):
            raise IncompatibleForecast(["the live forecasts do not give the random forest's trained inputs"])
        probabilities = predict(fitted, feature_matrix(weekly, inputs, issue_rows, models))
        daily = weekly.frame(issue_rows, ["id", "time", "lat", "lon"])
        for day in range(probabilities.shape[1] - 1):
            daily[f"p_day_{day + 1}"] = probabilities[:, day]
        daily["p_later"] = probabilities[:, -1]
        weeks = weekly_from_daily(probabilities, fitted.params.days_per_week)
        for week, label in enumerate(recipe.outcome_labels()):
            daily[f"cv_{label}"] = weeks[:, week]
    return Forecast(weekly_table=_probabilities(blend, table), features=table, daily_table=daily)


def _thresholds(blend):
    """The frozen thresholds as the detection reads them."""
    from momp.lib.onset_config import Thresholds

    return Thresholds(blend.thresholds)


def _input_columns(model, weekly) -> list[str]:
    """The weekly columns the fitted model's terms are made of."""
    names = {part for feature in model.features for part in str(feature).split(":")}
    return [c for c in weekly.columns if c in names]


def _probabilities(blend, rows: pd.DataFrame) -> pd.DataFrame:
    labels = blend.recipe.outcome_labels()
    predicted = blend.model.predict(rows)
    out = rows[["id", "time", "onset_threshold", "lat", "lon"]].copy()
    for label in labels:
        out[f"cv_{label}"] = predicted[f"cv_{label}"].to_numpy()
    # The climatology the blend starts from, as probabilities.
    weeks = [label for label in labels if label != "later"]
    clim = 1.0 / (1.0 + np.exp(-rows[[f"{CLIMATOLOGY_FEATURE}_{w}" for w in weeks]].to_numpy()))
    for index, week in enumerate(weeks):
        out[f"clim_{week}"] = clim[:, index]
    out["clim_later"] = np.maximum(0.0, 1.0 - clim.sum(axis=1))
    return out


def parse_forecast_arguments(items) -> dict[str, list[str]]:
    """`MODEL=PATH` items (as the command line gives them) as a mapping of model to its paths."""
    out: dict[str, list[str]] = {}
    for item in items:
        model, sep, path = str(item).partition("=")
        if not sep or not model or not path:
            raise ValueError(f"--forecast expects MODEL=PATH, got {item!r}")
        out.setdefault(model, []).append(path)
    return out
