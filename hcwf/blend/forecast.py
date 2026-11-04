"""Operational forecasts: a trained blend applied to live forecast files.

Only the live files (and, optionally, the issue date) come from the caller. The run config, onset definition, recipe,
reference data and observed onsets all come from the `~hcwf.blend.trained.TrainedBlend`, and the live files must have the structure the
blend was trained on (`hcwf.config.detect.signature_differences`); every difference is reported before any work
starts. The live files are read through ``hcwf.io.input`` (only the tiles of the trained units), onsets are detected by
`hcwf.onset.detect` with the frozen definition and options, and the features are `hcwf.blend.features`'.

Operationally each raw model gives one file holding one initialisation, and that is the issue date forecast. Files
holding a season of issues (a replay) forecast every issue in it, or the one ``issue_date`` names.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.blend import features, training
from hcwf.blend.recipe import CLIMATOLOGY_FEATURE, blend_season_start
from hcwf.config import detect as dataset_detect
from hcwf.errors import IncompatibleForecast

WEEKLY_FILENAME = "blended_forecast_probabilities.csv"
"""The name of the weekly probabilities CSV that `Forecast.save` writes."""
DAILY_FILENAME = "daily_onset_probabilities.csv"
"""The name of the daily probabilities CSV (random forest blends only)."""
WEEKLY_NETCDF = "blended_forecast_probabilities.nc"
"""The name of the weekly probabilities NetCDF file (written only on request)."""
DAILY_NETCDF = "daily_onset_probabilities.nc"
"""The name of the daily probabilities NetCDF file (written only on request)."""

__all__ = ["DAILY_FILENAME", "Forecast", "IncompatibleForecast", "WEEKLY_FILENAME", "check_files", "run"]


def _dataset(table: pd.DataFrame, columns: list[str]) -> xr.Dataset:
    """Return a forecast table (rows of id and issue date) as arrays over (time, id).

    Parameters
    ----------
    table : pandas.DataFrame
        One row per unit and issue date, with ``id``, ``time``, the `columns`, and optionally ``lat``, ``lon`` and
        ``onset_threshold`` (taken from the first row of each unit).
    columns : list of str
        The columns to turn into ``float64`` variables over ``(time, id)``.

    Returns
    -------
    xarray.Dataset
        The `columns` over ``(time, id)``, ``NaN`` where a unit has no row for a time, and the per-unit
        ``lat``, ``lon`` and ``onset_threshold`` over ``(id,)`` where the table has them. Times and ids are sorted.
    """
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

    `weekly_table` holds one row per unit and issue date as blending's contract has it (``id``, ``time``,
    ``onset_threshold``, ``lat``, ``lon``, ``cv_week1`` .. ``cv_weekN``, ``cv_later``, then the climatology's
    ``clim_*``); `daily_table` (the random forest blend's) ``id``, ``time``, ``lat``, ``lon``, ``p_day_1`` ..
    ``p_day_D``, ``p_later`` and the week bins ``cv_*``. `Forecast.weekly` and `Forecast.daily` are the same
    probabilities as `xarray.Dataset` objects over ``(time, id)``; `features` holds the weekly feature rows the
    probabilities come from.
    """

    weekly_table: pd.DataFrame
    """The weekly probabilities, one row per unit and issue date."""
    features: pd.DataFrame
    """The weekly feature rows the probabilities were computed from."""
    daily_table: pd.DataFrame | None = None
    """The daily onset probabilities; ``None`` unless the blend is a random forest."""
    folder: Path | None = None
    """Where `Forecast.save` wrote the forecast; ``None`` before."""

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
        """The weekly probabilities (the ``cv_*`` and ``clim_*`` columns) as arrays over ``(time, id)``."""
        columns = [c for c in self.weekly_table.columns if c.startswith(("cv_", "clim_"))]
        return _dataset(self.weekly_table, columns)

    @property
    def daily(self) -> xr.Dataset | None:
        """The daily probabilities as arrays over ``(time, id)`` (``p_day`` over ``(time, id, day)``), ``None`` if none."""
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
        """Write the forecast's files into a new directory, all or nothing.

        The files are blending's contract CSVs (weekly, and daily for a random forest blend). They are written to a
        staging directory next to ``directory`` and moved into place at the end, so a failure leaves nothing behind.
        `Forecast.folder` is set to ``directory``.

        Parameters
        ----------
        directory : str or pathlib.Path
            The directory to create; it may exist if it is empty.
        netcdf : bool, default False
            Also write the probabilities as NetCDF (off by default: nothing reads it).

        Returns
        -------
        pathlib.Path
            The directory written.

        Raises
        ------
        FileExistsError
            ``directory`` exists and is not empty.
        BaseException
            Whatever the writing raised; the staging directory is removed first.
        """
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
    """Expand each model's live files (a file, a glob or a list) to paths.

    Parameters
    ----------
    forecasts : dict
        The model name and its file, glob or list of them.

    Returns
    -------
    dict of str to list of pathlib.Path
        The paths of each model.
    """
    from hcwf.blend.trained import expand_files

    return {name: expand_files(value) for name, value in forecasts.items()}


def _model_spec(blend, name: str):
    """Return a model's recipe entry.

    Parameters
    ----------
    blend : hcwf.blend.trained.TrainedBlend
        The trained blend.
    name : str
        The model name.

    Returns
    -------
    object
        ``blend.recipe.model(name)``.
    """
    return blend.recipe.model(name)


def _read(blend, name: str, path, *, selected: bool) -> tuple[dict, list]:
    """Read one live forecast file as the blend's detection does.

    Parameters
    ----------
    blend : hcwf.blend.trained.TrainedBlend
        The trained blend (its entries and recipe give how to read the file).
    name : str
        The model the file belongs to.
    path : str or pathlib.Path
        The forecast file.
    selected : bool
        Whether to keep only the selected members and units, see `hcwf.config.detect.read_forecast_file`.

    Returns
    -------
    tuple of (dict, list)
        What `hcwf.config.detect.read_forecast_file` returns: the file's signature and its issue dates.
    """
    model = _model_spec(blend, name)
    return dataset_detect.read_forecast_file(
        path, blend.entries[name], lead_days=model.lead_days, max_member=model.max_member, selected=selected)


def check_files(blend, forecasts: dict[str, list[Path]], *, fewer_members: bool = False,
                held: dict | None = None) -> list[str]:
    """Find every way the live files differ from what the blend was trained on.

    Parameters
    ----------
    blend : hcwf.blend.trained.TrainedBlend
        The trained blend.
    forecasts : dict of str to list of pathlib.Path
        The live files of each model; they must be given for exactly the trained models.
    fewer_members : bool, default False
        Accept live files that hold only some of the members the blend was trained on.
    held : dict, optional
        If given, filled with the sorted issue dates that each model's files hold.

    Returns
    -------
    list of str
        The problems found; empty when the files are compatible.
    """
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
    """Settle the issue date and the season year of a forecast and check them against the live files.

    With no issue date given, files holding one initialisation each forecast it (the models must agree); files
    holding more forecast every issue of their one season.

    Parameters
    ----------
    blend : hcwf.blend.trained.TrainedBlend
        The trained blend (its season and the end of the climatology's issue window).
    issue_date : datetime.date or None
        The issue date asked for, or ``None``.
    held : dict of str to list of datetime.date
        The issue dates each model's files hold.

    Returns
    -------
    tuple of (datetime.date or None, int or None, list of str)
        The issue date (``None`` for a whole season), the season year (``None`` if it is ambiguous) and the
        problems found.
    """
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
        start = dt.date.fromisoformat(f"{year}-{blend_season_start(blend.setup)}")
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
    """Compute blended probabilities for the live forecasts.

    The files are checked first (`check_files`); then onsets are detected, the features built without trimming (a
    forecast must not know when the onset came), and the fitted blend (and the random forest, if the blend has one)
    applied.

    Parameters
    ----------
    blend : hcwf.blend.trained.TrainedBlend
        The trained blend.
    forecasts : dict
        The live files: model name to a file, a glob or a list of them.
    issue_date : datetime.date or str, optional
        The issue to forecast (an ISO date string is parsed); ``None`` forecasts the one initialisation of the files,
        or every issue of their season.
    fewer_members : bool, default False
        Let an ensemble's live files hold only some of the members the blend was trained on: its onset
        probabilities are shares of fewer members, so coarser and noisier than the blend's training saw.
    workers : int, optional
        The worker processes; ``None`` takes the number the run's resources allow.

    Returns
    -------
    Forecast
        The probabilities for the units and issue dates that have every feature the blend needs.

    Raises
    ------
    IncompatibleForecast
        The live files differ from the training structure, the issue date does not fit them or the season, the
        random forest's inputs do not match, or no unit and issue date has every feature.
    """
    from hcwf.blend.random_forest import feature_names, feature_matrix, predict, weekly_from_daily

    if isinstance(issue_date, str):
        issue_date = dt.date.fromisoformat(issue_date)
    live = _files(forecasts)
    held: dict[str, list] = {}
    problems = check_files(blend, live, fewer_members=fewer_members, held=held)
    issue_date, year, issue_problems = _issue_problems(blend, issue_date, held)
    if problems or issue_problems:
        raise IncompatibleForecast(problems + issue_problems)

    rc = replace(blend.rc, models=tuple(replace(m, paths=tuple(str(p) for p in live[m.name])) if m.name in live else m
                                        for m in blend.rc.models))   # the live files stand in for the training files
    count, budget = training.resources(rc)
    workers = workers or count
    recipe, setup = blend.recipe, blend.setup
    thresholds = _thresholds(blend)
    keep = frozenset(blend.units)
    forecast_onsets = {}
    for model in recipe.models:
        forecast_onsets[model.name] = features.forecast(
            rc, recipe, model.name, thresholds, blend.reference, keep_ids=keep,
            member_ids=blend.signatures[model.name].get("members"), units=blend.frozen_units[model.name],
            issue_dates=None if issue_date is None else (issue_date,), workers=workers, budget_bytes=budget,
            needed_only=True)
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
    """Return the frozen thresholds as the detection reads them.

    Parameters
    ----------
    blend : hcwf.blend.trained.TrainedBlend
        The trained blend, whose ``thresholds`` are frozen.

    Returns
    -------
    hcwf.config.onset.Thresholds
        The thresholds wrapped for detection.
    """
    from hcwf.config.onset import Thresholds

    return Thresholds(blend.thresholds)


def _input_columns(model, weekly) -> list[str]:
    """Return the weekly columns the fitted model's terms are made of.

    Parameters
    ----------
    model : object
        The fitted logistic blend; its ``features`` are the terms (``a:b`` is made of ``a`` and ``b``).
    weekly : hcwf.blend.samples.Weekly
        The weekly samples, whose ``columns`` are searched.

    Returns
    -------
    list of str
        The columns of `weekly` that appear in a term, in the order of its columns.
    """
    names = {part for feature in model.features for part in str(feature).split(":")}
    return [c for c in weekly.columns if c in names]


def _probabilities(blend, rows: pd.DataFrame) -> pd.DataFrame:
    """Predict the week-bin probabilities of the weekly feature rows, with the climatology they start from.

    Parameters
    ----------
    blend : hcwf.blend.trained.TrainedBlend
        The trained blend.
    rows : pandas.DataFrame
        The weekly feature rows, with ``id``, ``time``, ``onset_threshold``, ``lat``, ``lon`` and the climatology
        feature of each week.

    Returns
    -------
    pandas.DataFrame
        ``id``, ``time``, ``onset_threshold``, ``lat``, ``lon``, ``cv_<label>`` for every outcome label and
        ``clim_<week>`` (the inverse logit of the climatology feature) with ``clim_later``, the rest to 1 (at least 0).
    """
    labels = blend.recipe.outcome_labels()
    predicted = dict(zip(blend.model.classes, blend.model.predict(rows).T))
    out = rows[["id", "time", "onset_threshold", "lat", "lon"]].copy()
    for label in labels:
        out[f"cv_{label}"] = predicted[label]
    # The climatology the blend starts from, as probabilities.
    weeks = [label for label in labels if label != "later"]
    clim = 1.0 / (1.0 + np.exp(-rows[[f"{CLIMATOLOGY_FEATURE}_{w}" for w in weeks]].to_numpy()))
    for index, week in enumerate(weeks):
        out[f"clim_{week}"] = clim[:, index]
    out["clim_later"] = np.maximum(0.0, 1.0 - clim.sum(axis=1))
    return out


def parse_forecast_arguments(items) -> dict[str, list[str]]:
    """Turn ``MODEL=PATH`` items (as the command line gives them) into a mapping of model to its paths.

    Parameters
    ----------
    items : iterable of str
        The ``MODEL=PATH`` items; a model may repeat.

    Returns
    -------
    dict of str to list of str
        The paths given for each model, in order.

    Raises
    ------
    ValueError
        An item has no ``=``, or an empty model or path.
    """
    out: dict[str, list[str]] = {}
    for item in items:
        model, sep, path = str(item).partition("=")
        if not sep or not model or not path:
            raise ValueError(f"--forecast expects MODEL=PATH, got {item!r}")
        out.setdefault(model, []).append(path)
    return out
