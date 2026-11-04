"""Features: onsets collected over files, and the weekly features the blend trains on, built from them.

The onsets come from `hcwf.onset.detect` (the one adapter between HCWF and the onset engine: evaluation units,
search-start variants, member onset counts); this module only collects its per-file arrays, in parallel on the run's
pool, into `~hcwf.blend.samples.ObservedOnsets` and `~hcwf.blend.samples.ForecastOnsets`, and takes the members' mean
rainfall for the rain predictors (`member_mean`, the one blend-specific piece of that read).

A sample is a unit and issue date that the climatology and every model cover (the inner join of the legacy combine),
in the climatology's order (units, then issue dates), dropping issue dates after the observed onset. Its weekly
features (climatology logits, binned forecast probabilities, rain predictors, the outcome) are computed from the
arrays directly and held as arrays (`~hcwf.blend.samples.Weekly`); the legacy table is ``Weekly.frame()``. The random
forest reads its day-level inputs as arrays too (`daily_inputs`).

Weeks are `hcwf.metrics.bins.LeadBins` (column slices for the weekly sums, ``codes`` for the outcome); rain-predictor
windows are read from the onset definition (``Definition.windows()``, via the recipe). The numerics are the legacy
connector's (week sums, winsorised logits, rolling rain sums over the week's start days, rain transforms), operation
for operation, without its day-level table.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logit

from hcwf.blend.recipe import CLIMATOLOGY_FEATURE, Recipe, probability_feature_base, window_tag
from hcwf.blend.samples import DailyInputs, ForecastOnsets, ObservedOnsets, Weekly
from hcwf.config.model import Case, Selection, Window
from hcwf.io import forecasts as forecasts_io
from hcwf.io import observations as observations_io
from hcwf.config.datasets import file_year
from hcwf.onset.climatology import ReferenceOnset
from hcwf.references import kde
from hcwf.references.kde import Climatology
from hcwf.onset.thresholds import unit_thresholds
from hcwf.onset import detect
from hcwf.parallel import map_items
from hcwf.parallel import workers_within

log = logging.getLogger(__name__)

_CHUNK_ROWS = 262_144
"""The rows of the weekly features computed at a time, which bounds the size of the intermediate arrays."""


# ---------------------------------------------------------------------------------------------------------------
# Onsets collected over files
# ---------------------------------------------------------------------------------------------------------------
#
# The run (`rc`, a RunConfig) says where the files are, how they become units (`units:`), the onset rule, its thresholds
# and the reading options; the thresholds come from `hcwf.io.input.thresholds_of`. What follows only adds what blending
# needs of that: the reference onset, the fixed cutoff, a year or a file at a time (a per-call `Selection`, and the
# run's specs replaced where blending reads differently: the horizon, the missing-rain policy, the lead window).


def reference_onset(recipe: Recipe) -> ReferenceOnset | None:
    """Return the reference onset (``onset.reference_onset``): a month-day, or a table read from its file.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.

    Returns
    -------
    hcwf.onset.climatology.ReferenceOnset or None
        The reference onset; None when the recipe has none.

    Raises
    ------
    FileNotFoundError
        The reference table's file does not exist.
    """
    spec = recipe.onset.reference
    if spec is None:
        return None
    if spec.month_day is not None:
        return ReferenceOnset(month_day=spec.month_day)
    if not os.path.exists(spec.file):
        raise FileNotFoundError(f"ref_onset file not found: {spec.file}")
    return ReferenceOnset.read(spec.file, date_col=spec.date_col, year_col=spec.year_col, day_col=spec.day_col,
                               base_date=spec.base_date, unit_col=spec.unit_col)


def _season_start(setup) -> str:
    """Return the start of the first season of an onset setup.

    Parameters
    ----------
    setup : hcwf.config.model.OnsetSetup
        The run's onset setup (``rc.onset``); its definition's first season is read.

    Returns
    -------
    str
        The start day as ``"MM-DD"``.
    """
    start = setup.definition.seasons[0].start
    return f"{start.month:02d}-{start.day:02d}"


def observation_years(rc) -> list[int]:
    """Return the years of the observation files (the season a file holds is the year in its name).

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration; ``rc.observations.paths`` are read.

    Returns
    -------
    list of int
        The distinct years, ascending.

    Raises
    ------
    ValueError
        A file name has no year.
    """
    paths = rc.observations.paths
    years = [file_year(p) for p in paths]
    if None in years:
        raise ValueError("an observation file name has no year (19xx or 20xx); observations are read a year at a "
                         "time: " + ", ".join(p for p, y in zip(paths, years) if y is None))
    return sorted(set(years))


def _observed_year(year, rc, select, thresholds, season_start):
    """Detect one year's observed onsets and summarise them as a file.

    Parameters
    ----------
    year : int
        The year (season) to read.
    rc : hcwf.config.model.RunConfig
        The run configuration, with the years and horizon set for blending.
    select : hcwf.config.model.Selection
        The per-call selection (reference onset, units to keep).
    thresholds : dict
        The onset thresholds.
    season_start : str
        The season start, ``"MM-DD"``; ``onset_day`` counts days from it.

    Returns
    -------
    dict
        ``ids`` (``str``, shape (unit,)), ``years`` (shape (1,)), and ``onset_day`` (``float64``, days after the season
        start, ``NaN`` for none), ``onset_idx`` and ``present``, each of shape (unit, 1).
    """
    onset = detect.detect_observed(lambda: observations_io.observations_of(year, rc, select), thresholds, year, rc, select)
    ids = np.asarray(onset["unit"].values).astype(str)
    dates = onset.values.astype("datetime64[D]")
    cutoff = np.datetime64(f"{year}-{season_start}", "D")
    onset_day = np.where(np.isnat(dates), np.nan, (dates - cutoff).astype("timedelta64[D]").astype(float))
    return {"ids": ids, "years": np.array([year]), "onset_day": onset_day[:, None],
            "onset_idx": onset["onset_index"].values[:, None], "present": onset["present"].values[:, None]}


def observed(rc, recipe: Recipe, thresholds, reference, *, workers: int = 1, keep_ids=None) -> ObservedOnsets:
    """Collect the observed onsets of every observation year, the years in parallel.

    The search starts at the later of the season start and the reference onset; each year's onsets are kept in the
    observed-onset cache while it is active.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration.
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe (its observed horizon is used).
    thresholds : dict
        The onset thresholds.
    reference : hcwf.onset.climatology.ReferenceOnset or None
        The reference onset.
    workers : int, default 1
        The most processes, one year each.
    keep_ids : iterable of str, optional
        Keep only these units; None keeps all.

    Returns
    -------
    hcwf.blend.samples.ObservedOnsets
        The onsets over (unit, year).
    """
    years = observation_years(rc)
    run = replace(rc, evaluation=replace(rc.evaluation, years=(), climatology_years=tuple(years)),
                  onset_options=replace(rc.onset_options, horizon=recipe.onset.observed_horizon))
    select = Selection(reference_onset=reference, keep_unit_ids=None if keep_ids is None else sorted(keep_ids))
    parts = map_items(_observed_year, years, (run, select, thresholds, _season_start(rc.onset)),
                      workers=max(1, min(workers, len(years))))
    return ObservedOnsets.from_parts(parts, _season_start(rc.onset))


def member_mean(rain: np.ndarray) -> np.ndarray:
    """Return the mean over members of rainfall, as the legacy pipeline computed it.

    That is pandas' compiled grouped mean: a compensated (Kahan) sum over the members in order, in the rainfall's own
    dtype, skipping NaN.

    Parameters
    ----------
    rain : numpy.ndarray of float, shape (issue, member, unit, day)
        Rainfall in mm.

    Returns
    -------
    numpy.ndarray of float, shape (issue, unit, day)
        The mean in the dtype of `rain`; ``NaN`` where no member has a value.
    """
    shape = (rain.shape[0],) + rain.shape[2:]
    total = np.zeros(shape, rain.dtype)
    compensation = np.zeros(shape, rain.dtype)
    count = np.zeros(shape, np.int64)
    for member in range(rain.shape[1]):
        value = rain[:, member]
        present = ~np.isnan(value)
        y = value - compensation
        t = total + y
        c = (t - total) - y
        c[np.isnan(c)] = 0  # an infinite value
        np.copyto(compensation, c, where=present)
        np.copyto(total, t, where=present)
        count += present
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count.astype(rain.dtype)
    out[count == 0] = np.nan
    return out


def _rain_mean(values: np.ndarray) -> np.ndarray:
    """Return the members' mean rainfall; one member is its own mean, with no spread.

    Parameters
    ----------
    values : numpy.ndarray of float, shape (issue, member, unit, day)
        Rainfall in mm.

    Returns
    -------
    numpy.ndarray of float, shape (issue, unit, day)
        The mean over members.
    """
    return values[:, 0] if values.shape[1] == 1 else member_mean(values)


_DETECT_VALUES = 1 << 24
"""The most forecast values (members x units x lead days x issues) the onset search of one worker holds at a time.

A run is read whole, its issues are searched a few at a time (the detection holds several copies of what it searches).
"""

_PARALLEL_VALUES = 1 << 25
"""Forecast values (members x units x lead days x issues) below which a model's files are done in this process.

That is about 2 s of work, less than starting the workers costs.
"""

SERIES = {"base": "base", "fixed_cutoff": "fixed_cutoff", "ref_onset": "ref"}
"""The count series a probability variant is binned from (`count_<series>` in a forecast's data)."""


def forecast_starts(fixed_cutoff, reference: ReferenceOnset | None, issues, ids, series=("fixed_cutoff", "ref")) -> dict:
    """Return the search starts after the base series, for the `series` asked for.

    The starts are the fixed cutoff date, and the reference onset (``"MM-DD"``, or dates over ``init_time`` and
    ``unit``).

    Parameters
    ----------
    fixed_cutoff : str
        The fixed cutoff date, as the onset engine takes it.
    reference : hcwf.onset.climatology.ReferenceOnset or None
        The reference onset; None for none.
    issues : sequence of numpy.datetime64
        The issue dates.
    ids : sequence of str
        The unit ids.
    series : tuple of str, default ("fixed_cutoff", "ref")
        The series to give starts for.

    Returns
    -------
    dict
        The starts by series name: ``"fixed_cutoff"`` and ``"ref"`` (a month-day string or an ``xarray.DataArray``
        of dates with dims ``(init_time, unit)``).
    """
    starts = {"fixed_cutoff": fixed_cutoff} if "fixed_cutoff" in series else {}
    if reference is not None and "ref" in series:
        if reference.month_day is not None:
            starts["ref"] = reference.month_day
        else:
            years = pd.DatetimeIndex(issues).year.to_numpy(dtype=int)
            dates = reference.dates(np.tile(np.asarray(ids).astype(str), len(issues)), np.repeat(years, len(ids)))
            starts["ref"] = xr.DataArray(dates.reshape(len(issues), len(ids)), dims=("init_time", "unit"))
    return starts


def _forecast_needs(recipe: Recipe, model, needed_only: bool) -> tuple[tuple[str, ...], int | None]:
    """Return the count series and lead days of a model's forecast onsets to keep.

    Every series and all days unless `needed_only`; else the base series and those the recipe bins, and the days its
    features read: the probability days and the rain predictors' horizon (every day for the random forest, which reads
    them all; none for a model without rain predictors).

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    model : hcwf.blend.recipe.ModelSpec
        The model's entry in the recipe.
    needed_only : bool
        Keep only what the features read.

    Returns
    -------
    series : tuple of str
        The count series.
    days : int or None
        The number of lead days to keep; None for all.
    """
    if not needed_only:
        return tuple(SERIES.values()), None
    series = ("base", *(SERIES[v] for v in recipe.probability_variants()))
    if recipe.evaluation.random_forest is not None:
        return series, None
    rain = recipe.rain_horizon_days(model) if model.rain_predictors else 0
    return series, max(recipe.features.probability_days, rain)


def _forecast_file_info(path, case, select, spec):
    """Read what the runs of one forecast file need to be placed.

    Parameters
    ----------
    path : str or os.PathLike
        The forecast file.
    case : hcwf.config.model.Case
        The model and window.
    select : hcwf.config.model.Selection
        The per-call selection (units, issue dates, leads).
    spec : hcwf.config.UnitsSpec
        The units specification of the model.

    Returns
    -------
    dict or None
        ``ids``, ``days``, ``issues``, ``bounds`` (the runs), ``bytes`` (per run) and ``values`` (members x units x lead
        days x issues); None when none of the issues asked for is in the file.
    """
    with forecasts_io.UnitForecast(path, case.model, spec, keep_ids=select.keep_unit_ids,
                                   issue_dates=select.issue_dates, select=select) as file:
        if not len(file.issues):
            return None
        return {"ids": file.ids, "days": np.asarray(file.days), "issues": file.issues, "bounds": file.run_bounds(),
                "bytes": file.run_bytes(),
                "values": len(file.issues) * file.n_member * len(file.ids) * len(file.days)}


def _forecast_runs(item, rc, case, select, spec, thresholds, fixed_cutoff, reference, series, keep_days):
    """Count the members with their onset on each lead day, for some runs of issues of one forecast file.

    The counts come from the search starting at lead 1, at the fixed cutoff date and at the reference onset.

    Parameters
    ----------
    item : tuple
        ``(path, bounds)``: the forecast file and the ``(start, stop)`` issue ranges of its runs to do.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    case : hcwf.config.model.Case
        The model and window.
    select : hcwf.config.model.Selection
        The per-call selection.
    spec : hcwf.config.UnitsSpec
        The units specification of the model.
    thresholds : dict
        The onset thresholds.
    fixed_cutoff : str
        The fixed cutoff date.
    reference : hcwf.onset.climatology.ReferenceOnset or None
        The reference onset.
    series : tuple of str
        The count series to keep.
    keep_days : int or None
        The lead days to keep; None for all.

    Returns
    -------
    list of tuple
        One ``(start, stop, counts, n_member, rain)`` per run: the issue range, the counts by series (``uint8``, shape
        (issue, unit, day)), the number of members and the members' mean rainfall (in the file's dtype, shape
        (issue, unit, day)).
    """
    path, bounds = item
    with forecasts_io.UnitForecast(path, case.model, spec, keep_ids=select.keep_unit_ids, issue_dates=select.issue_dates,
                                   select=select) as file:
        ids, issues, n_day = file.ids, file.issues, len(file.days)
        days = n_day if keep_days is None else min(n_day, keep_days)
        run = replace(rc, evaluation=replace(rc.evaluation, max_forecast_day=n_day))   # each file has its own leads
        out = []
        for start, stop, rain in file.runs(bounds):
            values = np.asarray(rain.values)
            n_member = values.shape[1]
            step = max(1, _DETECT_VALUES // max(1, n_member * len(ids) * n_day))   # issues detected at a time
            counts = {v: np.zeros((stop - start, len(ids), days), dtype=np.uint8) for v in series}
            for first in range(0, stop - start, step):
                last = min(stop - start, first + step)
                found, order, _ = detect.count_members(
                    rain.isel(init_time=slice(first, last)), thresholds, run, case, select,
                    search_start=forecast_starts(fixed_cutoff, reference, issues[start + first:start + last], ids, series))
                found = found.sel(lead=slice(1, None))
                for v in series:
                    counts[v][first:last] = found.isel(start=order.index(v) if v in order else 0).values[..., :days]
                del found
            out.append((start, stop, counts, n_member, _rain_mean(values[..., :days])))
            del values, rain
    return out


def forecast(rc, recipe: Recipe, model: str, thresholds, reference, *, keep_ids=None, member_ids=None,
             units=None, issue_dates=None, workers: int = 1, budget_bytes: int | None = None,
             needed_only: bool = False) -> ForecastOnsets:
    """Collect a model's forecast onsets over every one of its files, runs of issues in parallel as far as memory allows.

    The files are read one run of issues at a time (`hcwf.io.forecasts.UnitForecast`), a few runs to a worker task, so
    that a file is shared by several workers and all of them stay busy to the end; each run's arrays are placed into the
    result as it arrives.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration.
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    model : str
        The model name.
    thresholds : dict
        The onset thresholds.
    reference : hcwf.onset.climatology.ReferenceOnset or None
        The reference onset.
    keep_ids : iterable of str, optional
        Limit the units to these (the blendable units under the inner join; their results are unchanged).
    member_ids : iterable of int, optional
        Limit the members read (a forecast reads the trained ones).
    units : hcwf.config.UnitsSpec, optional
        Override the model's units specification (a forecast passes the trained blend's).
    issue_dates : iterable of datetime-like, optional
        Read only the issues on these days (files without any are left out).
    workers : int, default 1
        The most worker processes.
    budget_bytes : int, optional
        The memory the workers may hold; None for no bound.
    needed_only : bool, default False
        Keep only the count series and lead days the recipe's features read (see `_forecast_needs`; the results of those
        are unchanged).

    Returns
    -------
    hcwf.blend.samples.ForecastOnsets
        The onsets; its ``data`` is None when no file has an issue asked for.

    Raises
    ------
    ValueError
        The files cover different units or lead days, or several issues fall on one date.
    """
    spec, setup = recipe.model(model), rc.onset
    source = rc.model(model)
    case = Case(source, rc.evaluation.windows[0] if rc.evaluation.windows else Window(1, rc.evaluation.max_forecast_day, 0))
    paths = source.paths
    units = units if units is not None else source.units
    run = replace(rc, onset_options=replace(rc.onset_options, missing=recipe.onset.forecast_missing,
                                            horizon=rc.onset_options.horizon or "truncate"))
    select = Selection(
        first_lead=spec.lead_days[0], last_lead=spec.lead_days[1], max_member=spec.max_member,
        member_ids=None if member_ids is None else [int(m) for m in member_ids],
        rain_horizon=(spec.rain_horizon.max_day, spec.rain_horizon.policy), check_units=False,
        keep_unit_ids=None if keep_ids is None else sorted(keep_ids),
        issue_dates=None if issue_dates is None else tuple(issue_dates))
    series, keep_days = _forecast_needs(recipe, spec, needed_only)
    infos = [_forecast_file_info(path, case, select, units) for path in paths]
    files = [(path, info) for path, info in zip(paths, infos) if info is not None]
    if not files:
        return ForecastOnsets(model=model, data=None)
    ids, days = files[0][1]["ids"], files[0][1]["days"]
    if any(not np.array_equal(info["ids"], ids) for _, info in files[1:]):
        raise ValueError(f"model {model}: its files cover different units")
    if any(not np.array_equal(info["days"], days) for _, info in files[1:]):
        raise ValueError(f"model {model}: its files cover different lead days")
    issues = np.concatenate([info["issues"] for _, info in files])
    stamps = pd.DatetimeIndex(issues)
    repeated = sorted({d.date().isoformat() for d in stamps[stamps.duplicated()]})
    if repeated:
        raise ValueError(f"model {model}: several issues on the same date ({', '.join(repeated[:5])}"
                         f"{'…' if len(repeated) > 5 else ''}): one initialisation per date")
    offsets = np.cumsum([0] + [len(info["issues"]) for _, info in files])

    # starting a pool of workers costs seconds: a small forecast is done here; a task is a few consecutive runs of one
    # file, about 6 tasks to a worker
    if sum(info["values"] for _, info in files) < _PARALLEL_VALUES:
        workers = 1
    per_task = max(1, -(-len(issues) // (6 * workers)))
    tasks = []
    for number, (path, info) in enumerate(files):
        group = []
        for bounds in info["bounds"]:
            group.append(bounds)
            if sum(stop - start for start, stop in group) >= per_task:
                tasks.append((number, (path, group)))
                group = []
        if group:
            tasks.append((number, (path, group)))
    count = max(1, min(workers, len(tasks)))
    count = workers_within([info["bytes"] for _, info in files], count, budget_bytes)
    name = setup.definition.thresholds[0][0] if setup.definition.thresholds else None
    n_day = len(days) if keep_days is None else min(len(days), keep_days)
    held = {"counts": {}, "members": np.zeros(len(issues), dtype=np.uint8), "rain": None}

    def place(position, result):
        """Place one task's runs into the arrays held for the result."""
        number = tasks[position][0]
        for start, stop, counts, n_member, rain in result:
            rows = slice(offsets[number] + start, offsets[number] + stop)
            for v, values in counts.items():
                if v not in held["counts"]:
                    held["counts"][v] = np.zeros((len(issues), len(ids), n_day), dtype=np.uint8)
                held["counts"][v][rows] = values
            held["members"][rows] = n_member
            if held["rain"] is None:
                held["rain"] = np.empty((len(issues), len(ids), n_day), dtype=rain.dtype)
            held["rain"][rows] = rain

    map_items(_forecast_runs, [item for _, item in tasks],
              (run, case, select, units, thresholds, recipe.onset.fixed_cutoff, reference, series, keep_days),
              workers=count, sink=place)
    order = None if stamps.is_monotonic_increasing else np.argsort(issues, kind="stable")
    take = (lambda a: a) if order is None else (lambda a: a[order])
    issues = take(issues)
    threshold = unit_thresholds(thresholds.get(name), ids, units.convention)
    variables = {"members": (("issue",), take(held["members"])), "onset_threshold": (("unit",), threshold),
                 "year": (("issue",), pd.DatetimeIndex(issues).year.to_numpy(dtype=np.int64)),
                 **{f"count_{v}": (("issue", "unit", "day"), take(held["counts"][v])) for v in series},
                 "rain_mean": (("issue", "unit", "day"), take(held["rain"]))}
    data = xr.Dataset(variables, coords={"issue": issues, "unit": ids.astype(object), "day": days[:n_day]})
    return ForecastOnsets(model=model, data=data)


# ---------------------------------------------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------------------------------------------


def climatology(recipe: Recipe, setup, observed_onsets: ObservedOnsets, forecast_years, *,
                test_years: tuple[int, int] | None = None, workers: int = 1) -> Climatology:
    """Build the KDE climatologies for the issue dates of the test years (`hcwf.references.kde`).

    These are the training window's and one per extra year window of the recipe.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    setup : hcwf.config.model.OnsetSetup
        The run's onset setup (for the season start).
    observed_onsets : hcwf.blend.samples.ObservedOnsets
        The observed onsets.
    forecast_years : iterable of int
        The years the forecasts cover.
    test_years : tuple of (int, int), optional
        Override the recipe's test years (a live season scores one year).
    workers : int, default 1
        The most processes; the recipe's own worker count, if any, wins.

    Returns
    -------
    hcwf.references.kde.Climatology
        The climatologies.
    """
    spec = recipe.climatology
    train, test = kde.climatology_years(observed_onsets.years, forecast_years, train_years=spec.train_years,
                                        test_years=test_years or spec.test_years)
    grid, by_year, max_h = kde.plan_issues(test, _season_start(setup), spec.issue_end,
                                           recipe.climatology_forecast_window(), spec.horizon_dicts(), spec.cv_by_year)
    return kde.build_climatology(observed_onsets.onsets(), grid, by_year, max_h, train=train, test=test,
                                 min_onset_years=spec.min_onset_years, cv_by_year=spec.cv_by_year,
                                 windows=spec.windows, workers=spec.workers or workers)


@dataclass
class Rows:
    """The feature rows: unit and issue positions into each source."""

    units: np.ndarray
    """The unit id of each row."""
    times: np.ndarray
    """The issue date of each row, ``datetime64[ns]``."""
    years: np.ndarray
    """The year of each row."""
    season_day: np.ndarray
    """The season day of each row's issue."""
    clim_unit: np.ndarray
    """The position of each row's unit in the climatology's units."""
    model_unit: dict[str, np.ndarray]
    """The position of each row's unit in each model's units, by model name."""
    model_issue: dict[str, np.ndarray]
    """The position of each row's issue in each model's issues, by model name."""
    true_onset: np.ndarray
    """The observed onset date of each row, ``datetime64[D]``, ``NaT`` where there is none."""
    true_onset_day: np.ndarray
    """The observed onset day of each row (days after the season start), ``NaN`` where there is none."""

    def __len__(self) -> int:
        """Return the number of rows.

        Returns
        -------
        int
            The number of rows.
        """
        return len(self.units)


def rows(recipe: Recipe, climatology: Climatology, observed_onsets: ObservedOnsets,
         forecasts: dict[str, ForecastOnsets], *, trim: bool | None = None) -> Rows:
    """Find the units and issue dates every source covers, in the climatology's order.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    climatology : hcwf.references.kde.Climatology
        The climatologies; their units and issues set the order.
    observed_onsets : hcwf.blend.samples.ObservedOnsets
        The observed onsets, for the true onset of each row.
    forecasts : dict of str to hcwf.blend.samples.ForecastOnsets
        Each model's forecast onsets.
    trim : bool, optional
        Drop the issue dates after the observed onset; None reads ``features.trim_after_true_onset``.

    Returns
    -------
    Rows
        The rows.
    """
    clim = climatology.data
    clim_units = clim["unit"].values.astype(str)
    unit_ok = np.ones(len(clim_units), dtype=bool)
    issue_ok = np.ones(clim.sizes["issue"], dtype=bool)
    issues = clim["issue"].values
    for onsets in forecasts.values():
        unit_ok &= np.isin(clim_units, onsets.data["unit"].values.astype(str))
        issue_ok &= np.isin(issues, onsets.data["issue"].values)
    unit_pos, issue_pos = np.flatnonzero(unit_ok), np.flatnonzero(issue_ok)
    clim_unit = np.repeat(unit_pos, len(issue_pos))
    grid = np.tile(issue_pos, len(unit_pos))
    units, times, years = clim_units[clim_unit], issues[grid], clim["issue_year"].values[grid]

    observed_units = observed_onsets.data["unit"].values.astype(str)
    order = np.argsort(observed_units)
    found = order[np.searchsorted(observed_units, units, sorter=order).clip(0, len(order) - 1)]
    has_unit = observed_units[found] == units
    observed_years = observed_onsets.data["year"].values
    year_pos = np.searchsorted(observed_years, years).clip(0, observed_onsets.data.sizes["year"] - 1)
    known = has_unit & (observed_years[year_pos] == years)
    true_onset = np.where(known, observed_onsets.onset_dates()[found, year_pos], np.datetime64("NaT"))
    true_onset_day = np.where(known, observed_onsets.data["onset_day"].values[found, year_pos], np.nan)

    if trim if trim is not None else recipe.features.trim_after_true_onset:
        day = times.astype("datetime64[D]")
        keep = np.isnat(true_onset) | (day <= true_onset)
    else:
        keep = np.ones(len(units), dtype=bool)

    model_unit, model_issue = {}, {}
    for name, onsets in forecasts.items():
        m_units = onsets.data["unit"].values.astype(str)
        m_order = np.argsort(m_units)
        model_unit[name] = m_order[np.searchsorted(m_units, units[keep], sorter=m_order)]
        model_issue[name] = np.searchsorted(onsets.data["issue"].values, times[keep])
    return Rows(units=units[keep], times=times[keep], years=years[keep],
                season_day=clim["issue_season_day"].values[grid][keep], clim_unit=clim_unit[keep],
                model_unit=model_unit, model_issue=model_issue, true_onset=true_onset[keep],
                true_onset_day=true_onset_day[keep])


def _week_sums(daily: np.ndarray, bins) -> list[np.ndarray]:
    """Sum daily values over the weeks, as the legacy connector sums them.

    Parameters
    ----------
    daily : numpy.ndarray of float, shape (row, day)
        Daily values for days 1..D.
    bins : hcwf.metrics.bins.LeadBins
        The week bins (``column_slices()`` gives the day columns of each).

    Returns
    -------
    list of numpy.ndarray of float, shape (row,)
        The sum of each week.
    """
    daily = np.asfortranarray(daily)
    return [daily[:, window].sum(axis=1) for window in bins.column_slices()]


def _lead_days(r: Rows) -> np.ndarray:
    """Return the days from each row's issue date to the observed onset.

    Parameters
    ----------
    r : Rows
        The feature rows.

    Returns
    -------
    numpy.ndarray of float64, shape (row,)
        The lead in days; ``NaN`` where there is no onset.
    """
    lead = (r.true_onset - r.times.astype("datetime64[D]")).astype("timedelta64[D]").astype(float)
    lead[np.isnat(r.true_onset)] = np.nan
    return lead


def _grid_lat_lon(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parse latitude and longitude from ``"<lat>_<lon>"`` ids, once per distinct id.

    Parameters
    ----------
    units : numpy.ndarray of object, shape (n,)
        Unit ids.

    Returns
    -------
    lat : numpy.ndarray of float64, shape (n,)
        Latitude in degrees; ``NaN`` for an id that is not a grid id (lat within -90..90, lon within -180..360).
    lon : numpy.ndarray of float64, shape (n,)
        Longitude in degrees, likewise.
    """
    codes, distinct = pd.factorize(units)
    text = pd.Series(distinct, dtype=str)
    parts = text.str.split("_", n=1, expand=True)
    lat = pd.to_numeric(parts[0], errors="coerce")
    lon = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else pd.Series(np.nan, index=text.index)
    grid = (text.str.count("_") == 1) & lat.between(-90, 90) & lon.between(-180, 360)
    return lat.where(grid).to_numpy()[codes], lon.where(grid).to_numpy()[codes]


def weekly(recipe: Recipe, definition, climatology: Climatology, observed_onsets: ObservedOnsets,
           forecasts: dict[str, ForecastOnsets], *, trim: bool | None = None) -> Weekly:
    """Build the weekly features the legacy connector built from the day-level table.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    definition : hcwf.config.model.Definition
        The onset definition the rain-predictor windows are read from.
    climatology : hcwf.references.kde.Climatology
        The climatologies.
    observed_onsets : hcwf.blend.samples.ObservedOnsets
        The observed onsets.
    forecasts : dict of str to hcwf.blend.samples.ForecastOnsets
        Each model's forecast onsets.
    trim : bool, optional
        Override ``features.trim_after_true_onset`` (a forecast never trims).

    Returns
    -------
    hcwf.blend.samples.Weekly
        The features, one sample per unit and issue date.
    """
    bins = recipe.features.bins
    r = rows(recipe, climatology, observed_onsets, forecasts, trim=trim)
    first = recipe.models[0].name
    threshold = forecasts[first].data["onset_threshold"].values[r.model_unit[first]]
    units, sample_unit = np.unique(r.units, return_inverse=True)
    lat, lon = _grid_lat_lon(units)
    unit_threshold = np.full(len(units), np.nan)
    unit_threshold[sample_unit] = threshold  # one model unit per id
    labels = recipe.outcome_labels()
    codes = bins.codes(_lead_days(r), earlier=False)
    variables: dict[str, tuple] = {
        "sample_unit": ("sample", sample_unit.astype(np.int32)), "time": ("sample", r.times),
        "year": ("sample", pd.DatetimeIndex(r.times).year.to_numpy()), "lat": ("unit", lat), "lon": ("unit", lon),
        "onset_threshold": ("unit", unit_threshold), "outcome": ("sample", codes, {"labels": labels})}
    attrs = {"rain_horizons": {}, "unavailable_rain_predictors": {}}
    features: dict[str, np.ndarray] = {}
    for start in range(0, max(len(r), 1), _CHUNK_ROWS):
        here = slice(start, min(len(r), start + _CHUNK_ROWS))
        for name, values in _row_features(recipe, definition, climatology, forecasts, r, here, threshold,
                                          attrs).items():
            if name not in features:
                features[name] = np.empty(len(r), dtype=values.dtype)
            features[name][here] = values
    for name, values in features.items():
        variables[name] = ("sample", values)
    data = xr.Dataset(variables, coords={"unit": units.astype(object)})
    return Weekly(data=data, columns=["id", "time", "year", "lat", "lon", "onset_threshold", "outcome", *features],
                  attrs=attrs)


def _row_features(recipe, definition, climatology, forecasts, r: Rows, here: slice, threshold, attrs) -> dict:
    """Compute the weekly features of a slice of rows, in the legacy column order.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    definition : hcwf.config.model.Definition
        The onset definition the rain-predictor windows are read from.
    climatology : hcwf.references.kde.Climatology
        The climatologies.
    forecasts : dict of str to hcwf.blend.samples.ForecastOnsets
        Each model's forecast onsets.
    r : Rows
        The feature rows.
    here : slice
        The rows to compute.
    threshold : numpy.ndarray of float64, shape (row,)
        The onset threshold of each row (mm), indexed with `here`.
    attrs : dict
        Receives the rain horizons and the unavailable rain predictors (see `_rain_predictors`).

    Returns
    -------
    dict of str to numpy.ndarray
        The feature columns, each of shape ``(here,)``.
    """
    features, bins = recipe.features, recipe.features.bins
    day_max = features.probability_days
    unit, years, season_day = r.clim_unit[here], r.years[here], r.season_day[here]
    out = {}
    feature = CLIMATOLOGY_FEATURE
    cond = climatology.at("conditional", unit, years, season_day)
    unc = climatology.at("unconditional", unit, years, season_day)
    cond_weeks = _week_sums(cond[:, 1: day_max + 1], bins)
    unc_weeks = _week_sums(unc[:, 1: day_max + 1], bins)
    out[f"{feature}_unc_earlier"] = _logit_winsor(unc[:, 0].copy())
    for w in range(bins.n_inner):
        out[f"{feature}_week{w + 1}"] = _logit_winsor(cond_weeks[w])
        out[f"{feature}_unc_week{w + 1}"] = _logit_winsor(unc_weeks[w])
    del cond, unc
    ids = r.units[here]
    for window in recipe.climatology.windows:
        array = climatology.windows[window]
        names = array["unit"].values.astype(str)
        order = np.argsort(names)
        position = order[np.searchsorted(names, ids, sorter=order).clip(0, len(order) - 1)]
        values = _windowed_at(array, position, years, season_day)
        values[names[position] != ids] = np.nan
        for w, week in enumerate(_week_sums(values[:, 1: day_max + 1], bins), start=1):
            out[f"{feature}_{window_tag(window)}_week{w}"] = _logit_winsor(week)

    variants = ["base", *recipe.probability_variants()]
    for model in recipe.models:
        data = forecasts[model.name].data
        issue, unit_m = r.model_issue[model.name][here], r.model_unit[model.name][here]
        members = data["members"].values[issue].astype(float)[:, None]
        for variant in variants:
            daily = data[f"count_{SERIES[variant]}"].values[issue, unit_m, :day_max] / members
            prefix = probability_feature_base(model.name, variant)
            for w, week in enumerate(_week_sums(daily, bins), start=1):
                out[f"{prefix}_week{w}"] = week
    out.update(_rain_predictors(recipe, definition, forecasts, r, here, threshold[here], attrs))
    return out


def _windowed_at(array, unit, year, season_day) -> np.ndarray:
    """Read a windowed climatology at rows.

    Parameters
    ----------
    array : xarray.DataArray
        A year-window climatology with dims ``(unit, [year,] season_day, lead)``.
    unit : numpy.ndarray of int, shape (row,)
        Each row's position along ``unit``.
    year : numpy.ndarray of int, shape (row,)
        Each row's year (used when the array has a ``year`` dim).
    season_day : numpy.ndarray of int, shape (row,)
        Each row's season day.

    Returns
    -------
    numpy.ndarray of float, shape (row, lead)
        The probabilities of each row.
    """
    day = np.searchsorted(array["season_day"].values, season_day)
    if "year" in array.dims:
        return array.values[unit, np.searchsorted(array["year"].values, year), day]
    return array.values[unit, day]


def _rain_predictors(recipe: Recipe, definition, forecasts, r: Rows, here: slice, threshold, attrs) -> dict:
    """Compute the weekly maxima (or minima) of rolling rain sums, as the legacy connector does.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    definition : hcwf.config.model.Definition
        The onset definition the rolling windows are read from.
    forecasts : dict of str to hcwf.blend.samples.ForecastOnsets
        Each model's forecast onsets (their mean rainfall).
    r : Rows
        The feature rows.
    here : slice
        The rows to compute.
    threshold : numpy.ndarray of float64, shape (here,)
        The onset threshold of each row (mm), for the ``diff`` predictors.
    attrs : dict
        Receives ``rain_horizons`` and ``unavailable_rain_predictors`` by model and column.

    Returns
    -------
    dict of str to numpy.ndarray of float64, shape (here,)
        The predictor columns, ``<agg>_<model>_<window>day_week<n>`` or ``diff_<model>_week<n>``.

    Raises
    ------
    ValueError
        A model lacks rainfall lead days the predictors need, or has non-finite rainfall within a strict rain horizon.
    """
    features, bins = recipe.features, recipe.features.bins
    starts_by_week = [list(range(a, b + 1)) for a, b in bins.inner]
    out = {}
    for model in recipe.models:
        name = model.name
        if not model.rain_predictors:
            continue
        transform = rain_transform(model.rain_transform if model.rain_transform is not None
                                   else features.rain_transform)
        policy = model.rain_horizon.resolved_policy
        rain_day_max = recipe.rain_horizon_days(model)
        strict = policy != "legacy"
        rain = forecasts[name].data["rain_mean"]
        days = rain["day"].values
        need = list(range(1, rain_day_max + 1))
        missing = [d for d in need if d not in days]
        if missing:
            raise ValueError(f"{name}: rainfall lead days {missing[:5]} are missing for the rain predictors")
        positions = np.searchsorted(days, need)
        issue, unit = r.model_issue[name][here], r.model_unit[name][here]
        mat = np.asfortranarray(rain.values[issue, unit][:, positions].astype(float))
        if strict and not np.isfinite(mat).all():
            raise ValueError(f"{name}: non-finite rainfall within the strict rain horizon ({rain_day_max} days)")
        attrs["rain_horizons"][name] = {"rain_day_max": rain_day_max, "strict": strict, "policy": policy}
        parsed = [(p.agg, recipe.window_days(p, definition)) for p in model.rain_predictors]
        rolled = {w: _rolling_sums(mat, w) for w in {w for _, w in parsed}}
        cache = {}
        for agg, window in parsed:
            for wi, starts in enumerate(starts_by_week):
                if (agg, window, wi) in cache:
                    continue
                valid = _complete_starts(rain_day_max, window, starts)
                cache[(agg, window, wi)] = _over_starts(rolled[window], valid, np.fmin if agg == "min" else np.fmax)
                if not valid:
                    column = f"diff_{name}_week{wi + 1}" if agg == "diff" else f"{agg}_{name}_{window}day_week{wi + 1}"
                    attrs["unavailable_rain_predictors"][column] = {
                        "model": name, "rain_day_max": rain_day_max, "window": window, "week": wi + 1}
        for agg, window in parsed:
            for w in range(1, bins.n_inner + 1):
                values = cache[(agg, window, w - 1)]
                if agg == "diff":
                    out[f"diff_{name}_week{w}"] = transform(values) - transform(threshold)
                else:
                    out[f"{agg}_{name}_{window}day_week{w}"] = transform(values)
    return out


def _logit_winsor(p, lo: float = 0.0001, hi: float = 0.9999) -> np.ndarray:
    """Return the logit of a probability clipped to ``[lo, hi]``.

    Parameters
    ----------
    p : array-like of float
        The probabilities.
    lo : float, default 0.0001
        The lower clip.
    hi : float, default 0.9999
        The upper clip.

    Returns
    -------
    numpy.ndarray of float64
        The logits, of the shape of `p`.
    """
    return logit(np.clip(np.asarray(p, dtype=float), lo, hi))


def _identity_transform(x) -> np.ndarray:
    """Return the values as a float array, unchanged.

    Parameters
    ----------
    x : array-like of float
        The values.

    Returns
    -------
    numpy.ndarray of float64
        `x`, of its own shape.
    """
    return np.asarray(x, dtype=float)


def _log1p_transform(x) -> np.ndarray:
    """Return ``log1p`` of the values clipped at 0; NaN stays NaN.

    Parameters
    ----------
    x : array-like of float
        The values.

    Returns
    -------
    numpy.ndarray of float64
        The transformed values, of the shape of `x`.
    """
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), x, np.log1p(np.clip(x, 0.0, None)))


def _power_transform(p):
    """Return the transform ``x ** p`` clipped at 0, NaN staying NaN.

    Parameters
    ----------
    p : float or str
        The exponent (a string is converted with ``float``).

    Returns
    -------
    callable
        A function of an array-like of float giving a ``numpy.ndarray`` of float64 of the same shape.
    """
    p = float(p)

    def f(x):
        """Raise the values, clipped at 0, to the power ``p``; NaN stays NaN."""
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x), x, np.clip(x, 0.0, None) ** p)
    return f


def rain_transform(value):
    """Return the rainfall transform that rain predictors apply (not onset detection or the climatology).

    The transforms are ``identity``, ``sqrt``, ``fourth_root``, ``log1p``, ``power:<p>`` or a number ``p``
    (``x ** p``). Powers and ``log1p`` clip at 0; NaN stays NaN.

    Parameters
    ----------
    value : str or int or float or None
        The transform's name or exponent; None, ``"identity"``, ``"none"`` and the empty string are the identity.

    Returns
    -------
    callable
        A function of an array-like of float giving a ``numpy.ndarray`` of float64 of the same shape.

    Raises
    ------
    ValueError
        `value` names no transform.
    """
    if value is None:
        return _identity_transform
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _power_transform(value)
    name = str(value).strip().lower()
    if name in ("identity", "none", ""):
        return _identity_transform
    if name == "sqrt":
        return _power_transform(0.5)
    if name == "fourth_root":
        return _power_transform(0.25)
    if name == "log1p":
        return _log1p_transform
    if name.startswith("power:"):
        return _power_transform(name.split(":", 1)[1])
    raise ValueError(f"Unknown rain_transform {value!r}: identity, sqrt, fourth_root, log1p, 'power:<p>' or a number.")


def _rolling_sums(mat: np.ndarray, k: int) -> np.ndarray:
    """Compute rolling k-day sums along each row.

    Parameters
    ----------
    mat : numpy.ndarray of float64, shape (row, day)
        Daily values.
    k : int
        The window length in days.

    Returns
    -------
    numpy.ndarray of float64, shape (row, day - k + 1)
        The sums; with no columns when there are fewer than `k` days.
    """
    n = mat.shape[1]
    if n < k:
        return np.empty((mat.shape[0], 0))
    return np.stack([mat[:, s:s + k].sum(axis=1) for s in range(n - k + 1)], axis=1)


def _complete_starts(rain_day_max: int, window: int, starts) -> list[int]:
    """Return the week's start days whose whole rolling window lies within ``1..rain_day_max``.

    Parameters
    ----------
    rain_day_max : int
        The last lead day with rainfall.
    window : int
        The rolling window length in days.
    starts : iterable of int
        The week's start days.

    Returns
    -------
    list of int
        The start days whose window fits.
    """
    last_start = int(rain_day_max) - int(window) + 1
    return [int(s) for s in starts if 1 <= int(s) <= last_start] if last_start >= 1 else []


def _over_starts(rolled: np.ndarray, starts, reduce) -> np.ndarray:
    """Reduce the rolling sums at the start days with ``numpy.fmax`` or ``numpy.fmin``, ignoring missing sums.

    Parameters
    ----------
    rolled : numpy.ndarray of float64, shape (row, start)
        Rolling sums, column ``s - 1`` for the window starting at day ``s``.
    starts : iterable of int
        The start days (1-based).
    reduce : numpy.ufunc
        ``numpy.fmax`` or ``numpy.fmin``.

    Returns
    -------
    numpy.ndarray of float64, shape (row,)
        The reduction per row; ``NaN`` without any usable start.
    """
    ok = [s for s in starts if 1 <= s <= rolled.shape[1]]
    if rolled.shape[1] == 0 or not ok:
        return np.full(rolled.shape[0], np.nan)
    return reduce.reduce(rolled[:, [s - 1 for s in ok]], axis=1)


def daily_inputs(recipe: Recipe, climatology: Climatology, observed_onsets: ObservedOnsets,
                 forecasts: dict[str, ForecastOnsets], *, max_day: int, trim: bool | None = None) -> DailyInputs:
    """Build the random forest's inputs for the samples `weekly` builds (with the same trim).

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    climatology : hcwf.references.kde.Climatology
        The climatologies.
    observed_onsets : hcwf.blend.samples.ObservedOnsets
        The observed onsets.
    forecasts : dict of str to hcwf.blend.samples.ForecastOnsets
        Each model's forecast onsets.
    max_day : int
        The last lead day of the climatology and rainfall inputs.
    trim : bool, optional
        Override ``features.trim_after_true_onset``.

    Returns
    -------
    hcwf.blend.samples.DailyInputs
        Per sample the issue time, the onset lead (days), the conditional climatology's daily onset probability and each
        model's mean rainfall by lead day.
    """
    r = rows(recipe, climatology, observed_onsets, forecasts, trim=trim)
    clim = climatology.at("conditional", r.clim_unit, r.years, r.season_day, leads=slice(1, max_day + 1))
    variables: dict[str, tuple] = {"time": ("sample", r.times), "onset_lead": ("sample", _lead_days(r)),
                                   "clim_p_onset": (("sample", "clim_day"), clim)}
    coords = {"clim_day": np.arange(1, clim.shape[1] + 1)}
    for name, onsets in forecasts.items():
        rain = onsets.data["rain_mean"]
        variables[f"rain_{name}"] = (("sample", f"{name}_day"), rain.values[r.model_issue[name], r.model_unit[name]])
        coords[f"{name}_day"] = rain["day"].values
    return DailyInputs(xr.Dataset(variables, coords=coords))
