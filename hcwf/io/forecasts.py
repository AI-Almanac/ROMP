"""Reading a model's forecasts: one year of inits at a time (`load_forecast`), or one file onto evaluation units (`UnitForecast`).

The inits, the leads the onset rule reads, the members asked for and the region's box are selected on the lazy array before any
data is decoded; the cell mask and the unit conversion are then applied in place on the one array that was read. `select_leads`
and `select_members` are the lead-day and member selections of both readers.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.config.model import NO_SELECTION
from hcwf.config.onset import forecast_reach
from hcwf.io.stores import forecast_sources, open_forecast
from hcwf.onset.ingest import IngestInfo, ingest_forecast_info
from hcwf.spatial.onto_units import declared_scale, read_units, select_units
from hcwf.spatial.select import in_memory, region_select, scale_in_place
from hcwf.spatial.units import UnitsSpec

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.config.model import (
        AreaSpec,
        EvaluationSpec,
        ModelSpec,
        OnsetOptions,
        Selection,
    )
    from hcwf.config.onset import OnsetSetup

log = logging.getLogger(__name__)

def steps_needed(max_forecast_day, onset: OnsetSetup | None):
    """The last lead day onset detection reads, or None if it can't be told.

    The largest day any requirement or veto of the compiled plan reads for the last candidate day
    max_forecast_day, so loading never truncates what the rule reads.
    """
    if not isinstance(max_forecast_day, int):
        return None
    return max_forecast_day + forecast_reach(onset) - 1


@dataclass(frozen=True, eq=False)
class ForecastRead:
    """What reading a model's forecasts depends on, besides the year: the typed inputs of one call (see `load_forecast`)."""

    model: ModelSpec
    ev: EvaluationSpec
    area: AreaSpec
    options: OnsetOptions
    select: Selection
    onset: OnsetSetup | None
    dir_fig: object
    ensemble: bool


_reported_ingest = set()


def ingest_forecast(cube, onset, model, options, select=NO_SELECTION, source=None):
    """Normalise a model's forecast cube (already in mm, dims init_time, [member], lat, lon, step) to ForecastRain.

    Uses the model's own `accumulation` ("daily", "since_init" or "auto") and `daily_boundary` (the config's per-model
    values; "daily" when absent), so accumulated fields are differenced and sub-daily steps summed to days
    exactly as hcwf.onset.ingest.ingest_forecast does. source labels the data (default: the model's name); how the
    days were made is logged once per model.
    """
    source = source or str(model.name or "forecast")
    accumulation = model.accumulation or "daily"
    boundary = model.daily_boundary or None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IngestInfo)
        forecast, info = ingest_forecast_info(cube, day_boundary=onset.forecast_boundary, lead_label=onset.lead_label,
                                              units="mm", source=source, accumulation=accumulation,
                                              daily_boundary=boundary,
                                              **options.ingest(True, None, check_units=select.check_units))
    forecast.ingest_info = info
    key = (str(model.name), info.accumulation, info.resolution, info.daily_boundary)
    if key not in _reported_ingest:
        _reported_ingest.add(key)
        text = f"{model.name or source}: accumulation {info.accumulation} [{info.accumulation_evidence}]; steps {info.resolution}"
        if info.daily_boundary:
            text += f"; days cut at {info.daily_boundary}"
        log.info(text + "".join(f"; {note}" for note in info.notes))
    return forecast


def _init_dates(year, date_filter_year, init_days, start_date, end_date):
    """
    Get initialization dates for a given year: the days between start_date and end_date
    (month/day only) whose weekday in date_filter_year is in init_days (0 = Monday).
    """
    start_MMDD = start_date[1:]
    end_MMDD = end_date[1:]

    start_date = datetime(date_filter_year, *start_MMDD)
    end_date = datetime(date_filter_year, *end_MMDD)
    date_range = pd.date_range(start_date, end_date, freq='D')

    filtered_dates = date_range[date_range.weekday.isin(init_days)]

    # Convert to the requested year
    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{year}-%m-%d'))

    return filtered_dates_yr


def initialization_dates(year, ev):
    """The initialization dates of `year`: the days between ev.start_date and ev.end_date (month/day only) whose weekday
    in ev.date_filter_year is in ev.init_days (0 = Monday)."""
    return _init_dates(year, ev.date_filter_year, ev.init_days, ev.start_date, ev.end_date)


def _forecast_on_units(da, spec: UnitsSpec, keep_ids, unit_cvt):
    """The selected issues of a forecast variable on the run's evaluation units, dims (init_time, [member], unit, step).

    The units are those of the observations (their ids, as the same specification builds them for this source), the
    millimetre factor is the file's declared unit or unit_cvt (as `UnitForecast` reads it).
    """
    units, positions = select_units(da, spec, keep_ids)
    out = read_units(da, spec, units, positions, block_dim="init_time", scale=declared_scale(da, unit_cvt))
    return out.transpose("init_time", *[d for d in ("member", "unit", "step") if d in out.dims])


def _load_forecast(year, read: ForecastRead):
    """Read one year of forecasts as a loaded DataArray in mm, reading only what the run needs.

    Inits, leads (up to what the onset rule reads), members and the region's box are selected on the lazy array;
    the selection is read once, the cell mask and the unit conversion (x * unit_cvt, in float32 as before) are
    applied in place on it. The file's step 0, if present, is dropped. A year listed in several files
    (select.model_paths) is read file by file and joined along init_time.
    """
    model, ev, select = read.model, read.ev, read.select
    sources = forecast_sources(year, model.dir, model.file_pattern, select.model_paths, select.model_files_years)
    dates = initialization_dates(year, ev)
    spec = model.units
    parts = []
    for file_path in sources:
        ds, close = open_forecast(year, model.dir, model.file_pattern, read.ensemble, model.dims_map(), dates,
                                   file_path, multi=len(sources) > 1)
        if ds is None:
            continue
        try:
            if 'step' not in ds.dims:
                raise KeyError("'step' dimension not found in dataset")
            # Onset detection reads no further than this lead day, so loading the rest only costs memory.
            ds = select_leads(ds, None, steps_needed(ev.max_forecast_day, read.onset))
            if read.ensemble and ev.members:
                ds = select_members(ds, ev.members)
            if spec is not None:   # evaluation units: the rain is averaged onto them as it is read
                parts.append(_forecast_on_units(ds[model.var], spec, select.keep_unit_ids, model.unit_cvt))
                continue
            da = region_select(ds[[model.var]], area=read.area, select=select, read_by="init_time",
                               dataset_name=str(model.name or "model"), dir_fig=read.dir_fig)[model.var]
            parts.append(in_memory(da))
        finally:
            close()
    if not parts:
        raise ValueError(f"No matching initialization times found for year {year}")
    da = parts[0] if len(parts) == 1 else xr.concat(parts, dim="init_time").sortby("init_time")
    return da if spec is not None else scale_in_place(da, model.unit_cvt)   # units are scaled as they are averaged


def load_forecast(year, model, ev, area, options, onset, *, ensemble, select=NO_SELECTION, dir_fig=None):
    """Load a model's forecasts of one year: the twice-weekly initializations between ev.start_date and ev.end_date.

    Filters for the init days (0 = Monday) of ev.date_filter_year. The forecast file is expected to be named
    model.file_pattern.format(year) in model.dir (or model.dir is a .zarr store of all years, or select.model_paths
    lists the files, any number per year) with the variable model.var, daily accumulated rainfall with dimensions
    (init_time, [member], lat, lon, step). With `ensemble` the members ev.members are selected before the read.
    `onset` says which lead days the onset rule reads (the rest are not loaded); `options` is the run's reading options
    (the ingest takes them: see `ingest_forecast`).

    Returns the DataArray of precipitation in mm, read once with the region, inits and leads selected first.
    """
    return _load_forecast(year, ForecastRead(model, ev, area, options, select, onset, dir_fig, ensemble))


def forecast_of(year, rc, model, *, ensemble, select=NO_SELECTION):
    """`load_forecast` of a run for the source `model`."""
    steps = rc.onset if isinstance(rc.evaluation.max_forecast_day, int) else None
    return load_forecast(year, model, rc.evaluation, rc.area, rc.onset_options, steps, ensemble=ensemble,
                         select=select, dir_fig=rc.output.figure_dir)


def select_leads(ds, first=None, last=None, *, strict=False, rain_horizon=None, where="forecast"):
    """The lead days of a dataset or array with a `step` dimension, `first`..`last` (None: from the first positive
    step, to the last).

    Without `strict` this is HCWF's reading: a step 0 is dropped and the steps up to `last` kept as they come. With
    `strict` (onset_blending's) the days must be whole, distinct and consecutive; they are put in ascending order and
    the step labels become 1..n, the days themselves a `lead_day` coordinate. `rain_horizon` (max_day, policy) checks
    first that the file holds the lead days 1..max_day, exactly ("exact") or with later days permitted ("truncate").
    """
    if not strict and rain_horizon is None:
        if first is None:
            if ds["step"][0].values == 0:
                ds = ds.sel(step=slice(1, None))
        else:
            ds = ds.sel(step=slice(first, None))
        return ds if last is None else ds.sel(step=slice(None, last))
    values = np.asarray(ds["step"].values).reshape(-1)
    numbers = []
    for v in values:
        try:
            numbers.append(int(v))
        except (TypeError, ValueError):
            numbers.append(None)
    if rain_horizon is not None and rain_horizon[0] is not None:
        _check_rain_horizon([v for v, d in zip(values, numbers) if d is not None and (first is None or d >= first)],
                            int(rain_horizon[0]), where, allow_extra=str(rain_horizon[1] or "exact").lower() == "truncate")
    keep = np.array([d is not None and (first is None or d >= first) and (last is None or d <= last) for d in numbers])
    if not keep.any():
        raise ValueError(f"Day filtering removed all columns for: {where}")
    positions = np.flatnonzero(keep)
    days = np.asarray([numbers[p] for p in positions])
    order = np.argsort(days, kind="stable")
    positions, days = positions[order], days[order]
    if len(set(days.tolist())) != len(days) or any(b - a != 1 for a, b in zip(days, days[1:])):
        raise ValueError(f"{where}: lead days must be consecutive and distinct, got {days[:8].tolist()}")
    out = ds.isel(step=positions)
    return out.assign_coords(lead_day=("step", days)).assign_coords(step=np.arange(1, len(days) + 1))


def _whole_day(value) -> int | None:
    """A lead day as a whole number, None for anything else."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if np.isfinite(numeric) and numeric.is_integer() else None


def _check_rain_horizon(day_values, rain_day_max: int, where: str, *, allow_extra: bool) -> None:
    """A strict rain horizon needs the lead days 1..rain_day_max exactly (later days too, with allow_extra)."""
    parsed = [_whole_day(v) for v in day_values]
    invalid = [repr(v) for v, d in zip(day_values, parsed) if d is None]
    valid = [d for d in parsed if d is not None]
    duplicates = sorted({d for d in valid if valid.count(d) > 1})
    actual, required = set(valid), set(range(1, rain_day_max + 1))
    missing, extra = sorted(required - actual), sorted(actual - required)
    problems = []
    if invalid:
        problems.append(f"non-integer day values={invalid[:10]}")
    if duplicates:
        problems.append(f"duplicate days={duplicates[:10]}")
    if missing:
        problems.append(f"missing days={missing[:10]}")
    if [d for d in extra if d < 1]:
        problems.append(f"nonpositive days={[d for d in extra if d < 1][:10]}")
    if [d for d in extra if d > rain_day_max] and not allow_extra:
        problems.append(f"extra days={[d for d in extra if d > rain_day_max][:10]}")
    if problems:
        requirement = (f"days 1..{rain_day_max}, with later days permitted" if allow_extra
                       else f"exactly days 1..{rain_day_max}")
        raise ValueError(f"{where}: strict rainfall horizon requires {requirement}; " + "; ".join(problems))


def select_members(ds, members=None, *, at_most=None, among=None, where="forecast"):
    """The members of a dataset or array with a `member` dimension.

    `members` (HCWF's run setting) names members that must exist. `at_most` and `among` (onset_blending's) keep the
    members present whose number is at most that and among those; none left is an error.
    """
    if members:
        return ds.sel(member=list(members))
    numbers = np.asarray(ds["member"].values).astype(int)
    keep = np.ones(len(numbers), dtype=bool)
    if at_most is not None:
        keep &= numbers <= at_most
    if among is not None:
        keep &= np.isin(numbers, among)
    if not keep.any():
        raise ValueError(f"{where}: none of its members are read")
    return ds if keep.all() else ds.isel(member=np.flatnonzero(keep))


_RUN_VALUES = 1 << 23
"""Values (members x units x lead days) of the issues one run of a forecast file holds at most."""


class UnitForecast:
    """One forecast file read onto evaluation units, a run of issues at a time (memory stays one run).

    Opens the file lazily as `_load_forecast` does (dimension names, lead days, members), keeps every issue, and puts
    the units of `spec` (those among `keep_ids`) in id order. `issues`, `ids`, `days` (the lead days kept, their step
    labels renumbered 1..n) and `members` describe it; `runs()` yields (first, stop, rain) with rain over (init_time,
    member, unit, step) in the file's dtype for cells and float64 for averaged units.
    When the first lead day kept is d (steps are renumbered 1..n) init_time is the issue plus d - 1 days, so every step
    has its true date.
    `issue_dates` (dates or ISO texts) keeps only the issues on those days, and reads nothing of the others: a file
    that has none of them has no issues and `runs()` yields nothing.
    """

    def __init__(self, path, model, spec: UnitsSpec, *, keep_ids=None, issue_dates=None, select=NO_SELECTION):
        """`model` is the forecast's ModelSpec (variable, dimension names, unit factor); `select` picks the lead days
        (first_lead, last_lead, rain_horizon) and members (max_member, member_ids)."""
        self.path, name = path, Path(path).name
        variable = model.var
        ds, self._close = open_forecast(None, None, None, None, model.dims_map(), None, file_path=str(path))
        try:
            if variable not in ds.variables:
                raise ValueError(f"{name} has no {variable!r} variable (it has: {', '.join(map(str, ds.variables))})")
            da = ds[[variable]][variable]
            if "step" not in da.dims or "init_time" not in da.dims:
                raise ValueError(f"{name}: no {variable!r} variable over issue time, lead day, optional member and "
                                 "spatial dimensions only")
            self.scale = declared_scale(da, model.unit_cvt)
            da = select_leads(da, select.first_lead, select.last_lead, strict=True,
                              rain_horizon=select.rain_horizon, where=f"NetCDF file {name}")
            self.days = np.asarray(da["lead_day"].values)
            if "member" not in da.dims:
                da = da.expand_dims(member=[0])
            elif select.max_member is not None or select.member_ids is not None:
                da = select_members(da, at_most=select.max_member, among=select.member_ids, where=name)
            self.issues = pd.DatetimeIndex(da["init_time"].values).normalize().to_numpy()
            if issue_dates is not None:
                wanted = np.array([np.datetime64(d, "D") for d in issue_dates])
                rows = np.flatnonzero(np.isin(self.issues.astype("datetime64[D]"), wanted))
                da, self.issues = da.isel(init_time=rows), self.issues[rows]
            spatial = [d for d in da.dims if d not in ("init_time", "member", "step")]
            self.da = da.transpose("init_time", "member", "step", *spatial)
            self.spec = spec
            self.units, self.positions = select_units(self.da, spec, keep_ids)
            self.ids = self.units.ids[self.positions]
            self.members = np.asarray(self.da["member"].values)
        except BaseException:
            self._close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._close()

    def runs(self):
        """(first, stop, rain) for each run of issues (whole storage chunks along init_time)."""
        da = self.da
        per_issue = max(1, da.sizes["member"] * len(self.ids) * len(self.days))
        rows = max(1, _RUN_VALUES // per_issue)
        chunks = da.encoding.get("chunksizes") or da.encoding.get("chunks")
        if chunks and len(chunks) == da.ndim:
            stored = int(chunks[da.dims.index("init_time")])
            rows = max(1, -(-rows // stored)) * stored
        n_issue = len(self.issues)
        for start in range(0, n_issue, max(1, min(rows, n_issue))):
            stop = min(n_issue, start + rows)
            rain = read_units(da.isel(init_time=slice(start, stop)), self.spec, self.units, self.positions,
                              block_dim="init_time", scale=self.scale)
            labelled = self.issues[start:stop] + np.timedelta64(int(self.days[0]) - 1, "D")
            yield start, stop, rain.transpose("init_time", "member", "unit", "step").assign_coords(init_time=labelled)


def unit_forecast_bytes(path, variable) -> int:
    """Peak memory to process one forecast file on units: one run of issues and its working copies."""
    import math

    import netCDF4

    with netCDF4.Dataset(path) as ds:
        var = ds.variables.get(variable)
        if var is None:
            return 512 * 2**20
        values = math.prod(var.shape)
    return int(min(values, _RUN_VALUES * 4) * 30) + 512 * 2**20
