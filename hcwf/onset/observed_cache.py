"""A cache of observed onset dates, shared by every model, window and reference of a run and kept across runs.

Observed onsets depend only on the observations, how they are read, the evaluated cells, the onset definition, the
thresholds (and season climatology) and the years, so one detection serves all models, verification windows, the
reference and later runs. The entry is keyed by a *fingerprint* of those six parts (see `hcwf.cache.Store`, which keeps
the entries and reports hits and misses); the backend, models, windows and reference are not in it. Evaluation units
replace the cells part by a units part (the units specification, the files it names, the units kept), and run options
that change the numbers (missing-rain policy, resolution, horizon, the reference onset a search starts at) add an
options part when they are set; without them the fingerprint is the one a grid run always had.

While a run is active (`activate`), `hcwf.onset.detect.detect_observed` asks `lookup` first and calls `store` after
detecting, so each year is detected once per fingerprint. The entry is written to the folder once per run, when the run
ends (`flush`, a `hcwf.cache.run_scope` end hook), not once per year.

The fingerprint is computed from typed inputs (``fingerprint_parts(rain, thresh, obs, area, onset, options, select,
years)``; `parts_for` takes them from a RunConfig). Its parts are the same hashes the kwargs form gave, so entries
written by earlier runs are still hits.

On disk (``<cache_dir>/observed_onsets/<fingerprint>/``): ``parts.json`` (the hash of each part) and ``onsets.nc``
(``onset_date`` over year and the spatial dimensions, ``lat`` and ``lon`` or ``unit``, plus any per-unit arrays the
onsets carry as coordinates, such as ``present``). Without a cache folder the cache is in memory for the run only. A
failure of the cache never stops a run: the onsets are then simply detected.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import Codec, Store, file_identity, memo, on_run_end, part_hash
from hcwf.config.model import Selection

__all__ = [
    "SUBFOLDER", "absorb", "activate", "current", "drain_new", "export_state", "fingerprint", "install_state",
    "observation_files", "parts_for", "threshold_hash",
]


log = logging.getLogger(__name__)

SUBFOLDER = "observed_onsets"
"""The folder of the cache folder that holds the observed-onset entries."""


@dataclass
class _Run:
    """The cache of the active run: where it is, its store, the onsets a worker found and the tables not yet written."""

    dir: Path | None = None
    """The cache folder; None keeps the cache in memory."""
    worker: bool = False
    """Whether this is a worker process, which reads the folder but leaves writing to the parent."""
    store: Store | None = None
    """The store of the entries."""
    new: list = field(default_factory=list)
    """The records a worker detected and the parent has not yet taken (see `drain_new`)."""
    dirty: dict = field(default_factory=dict)
    """The ``(parts, table)`` of each fingerprint detected in this run and not yet written (see `flush`)."""


_state = memo("onset.observed_cache.run", maxsize=1)
"""The active run's cache: the one item "run" is a `_Run`; none while no run uses the cache."""
_tables = memo("onset.observed_cache.tables", share=True)
"""The in-memory table of each fingerprint, read from the cache folder or detected in this run (the workers get a copy)."""
_hashes = memo("onset.observed_cache.hashes", maxsize=256)
"""Hashes memoised for the run: the thresholds by object, and the observation files of a dataset."""


def _run() -> _Run | None:
    """Return the active run's cache.

    Returns
    -------
    _Run or None
        The cache of the active run, or None when no run uses the cache.
    """
    return _state.get("run")


def _empty() -> dict:
    """Return an empty table, the in-memory form of an entry (see `_read`).

    Returns
    -------
    dict
        ``years`` and ``extras`` empty dicts, and ``dims``, ``coords``, ``lat`` and ``lon`` None.
    """
    return {"years": {}, "dims": None, "coords": None, "lat": None, "lon": None, "extras": {}}


def _read(path: Path) -> dict:
    """Read a cached entry into its in-memory table.

    Parameters
    ----------
    path : pathlib.Path
        The ``onsets.nc`` file of the entry.

    Returns
    -------
    dict
        The table: ``years`` (``{year: datetime64[ns] values}``), ``dims`` (the spatial dimension names, a tuple),
        ``coords`` (``{dim: values}``), ``lat`` and ``lon`` (the coordinates of a grid, else None) and ``extras``
        (``{year: {name: values}}`` of the per-unit arrays the onsets carry).
    """
    with xr.open_dataset(path) as ds:
        dims = tuple(ds["onset_date"].dims[1:])
        years = {int(y): ds["onset_date"].sel(year=y).values.astype("datetime64[ns]") for y in ds["year"].values}
        extra = [n for n in ds.data_vars if n != "onset_date"]
        extras = {int(y): {n: ds[n].sel(year=y).values for n in extra} for y in ds["year"].values}
        coords = {d: ds[d].values.copy() for d in dims}
        return {"years": years, "dims": dims, "coords": coords, "lat": coords.get("lat"), "lon": coords.get("lon"),
                "extras": extras}


def _write(path: Path, table: dict) -> None:
    """Write an entry: ``onset_date`` over year and the spatial dims, and each extra array over the same.

    Parameters
    ----------
    path : pathlib.Path
        The ``onsets.nc`` file to write.
    table : dict
        The in-memory table (see `_read`).
    """
    years, dims = sorted(table["years"]), table["dims"]
    variables = {"onset_date": (("year", *dims), np.stack([table["years"][y] for y in years]))}
    for name in sorted({n for y in years for n in table["extras"].get(y, {})}):
        variables[name] = (("year", *dims), np.stack([table["extras"][y][name] for y in years]))
    ds = xr.Dataset(variables, coords={"year": years, **{d: table["coords"][d] for d in dims}})
    ds.to_netcdf(path)


def _new_store(root) -> Store:
    """Return the observed-onset store of a cache folder.

    Parameters
    ----------
    root : str or pathlib.Path or None
        The cache folder; None keeps the store in memory.

    Returns
    -------
    hcwf.cache.Store
        The store, with the ``onsets.nc`` codec.
    """
    return Store(SUBFOLDER, root, codec=Codec("onsets.nc", _read, _write), prefix="obs", label="Observed onsets",
                 describe=lambda table: f"{len(table['years'])} years cached",
                 miss_note="; detecting once for all models, windows and references")


def activate(cache_dir=None) -> None:
    """Start using the cache for this run, clearing what an earlier run left.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        The cache folder; None keeps the cache in memory.
    """
    for table in (_state, _tables, _hashes):
        table.clear()
    _state["run"] = _Run(dir=Path(cache_dir) if cache_dir else None, store=_new_store(cache_dir))


@on_run_end
def flush() -> None:
    """Write the tables detected in this run to the cache folder, each once.

    Does nothing without an active run or in a worker process. A failure of the folder is only logged (by the store).
    """
    run = _run()
    if run is None or run.worker:
        return
    dirty, run.dirty = run.dirty, {}
    for parts, table in dirty.values():
        run.store.put(parts, table)


def threshold_hash(thresh) -> str:
    """Hash the threshold values: the named arrays or the season climatology.

    The hash is memoised for the run by object identity.

    Parameters
    ----------
    thresh : dict or xarray.DataArray or object
        The thresholds: a dict of named arrays or tables (a ``season_climatology`` attribute, when set, is hashed too),
        a DataArray (its values and coordinates), or anything `hcwf.cache.part_hash` accepts.

    Returns
    -------
    str
        The hash of the thresholds.
    """
    cached = _hashes.get(("thresh", id(thresh)))
    if cached is not None and cached[0] is thresh:
        return cached[1]
    if isinstance(thresh, dict):
        season = getattr(thresh, "season_climatology", None)
        named = {name: value.to_dict("list") if isinstance(value, pd.DataFrame) else value
                 for name, value in thresh.items()}   # a table is hashed by its contents
        result = part_hash(named, None if season is None else {n: season[n] for n in season.data_vars})
    elif isinstance(thresh, xr.DataArray):   # the values and the grid they sit on
        result = part_hash(thresh, {str(name): thresh.coords[name] for name in thresh.coords})
    else:
        result = part_hash(thresh)
    _hashes[("thresh", id(thresh))] = (thresh, result)
    return result


def observed_years(ev) -> list[int]:
    """Return the years observed onsets are needed for: the evaluated years and the climatology years.

    Parameters
    ----------
    ev : hcwf.config.model.EvaluationSpec
        The evaluation settings, with ``years`` and ``climatology_years``.

    Returns
    -------
    list of int
        The sorted union of both year sets.
    """
    return sorted({int(y) for y in (*ev.years, *ev.climatology_years)})


def observation_files(obs, files_years, years) -> list:
    """Return the identity (name, size, mtime) of the observation file of every year.

    For a zarr store the identity of the store and the years are returned instead. A year without a file gives
    ``"missing <year>"``.

    Parameters
    ----------
    obs : hcwf.config.model.ObsSpec
        The observation settings: listed ``paths``, or a folder ``dir`` with ``file_pattern``, or a zarr store.
    files_years : collection of int or None
        The years of the listed files to keep, as in `hcwf.config.model.Selection.obs_files_years`.
    years : iterable of int
        The years to identify files for.

    Returns
    -------
    list
        One entry per year (or ``[store identity, years]`` for a zarr store); each is a list of file identities or
        the missing-year note.
    """
    from hcwf.config.datasets import is_zarr
    from hcwf.io.stores import year_files

    if obs.paths:   # listed files, any number per year
        found = []
        for year in years:
            files = year_files(obs.paths, obs.dir or None, files_years, year)
            found.append([file_identity(f) for f in files] or [f"missing {year}"])
        return found
    if is_zarr(obs.dir):
        return [file_identity(obs.dir), list(years)]
    found = []
    for year in years:
        for pattern in obs.file_pattern or ():
            path = Path(str(obs.dir)) / pattern.format(year)
            if path.exists():
                found.append(file_identity(path))
                break
        else:
            found.append([f"missing {year}"])
    return found


def _obs_files_hash(obs, files_years, years) -> str:
    """Hash `observation_files`, memoised for the run.

    Parameters
    ----------
    obs : hcwf.config.model.ObsSpec
        The observation settings.
    files_years : collection of int or None
        The years of the listed files to keep.
    years : iterable of int
        The years to identify files for.

    Returns
    -------
    str
        The hash of the file identities.
    """
    key = ("files", str(obs.dir), str(obs.file_pattern), tuple(years), str(obs.paths), str(files_years))
    return _hashes.get_or_make(key, lambda: part_hash(observation_files(obs, files_years, years)))


def _mask_files(area) -> list:
    """Return the identity of the files the cell selection reads.

    Parameters
    ----------
    area : hcwf.config.model.AreaSpec
        The area settings: the mask file and its condition file, ``nc_mask`` and the shapefile folder.

    Returns
    -------
    list
        The `hcwf.cache.file_identity` of each of those that is set.
    """
    paths = []
    if area.mask is not None:
        paths += [area.mask.file, (area.mask.condition or {}).get("file")]
    paths += [area.nc_mask, area.shpfile_dir]
    return [file_identity(p) for p in paths if p]


_MASK_GIVEN = ("land", "country", "file", "variable", "keep_value", "condition", "cell_rule", "overlap_fraction",
               "resolution")
"""The keys of a config's ``mask:`` section (the other ``MaskSpec`` fields are set by the implicit masks only)."""


def _mask_mapping(area):
    """Return the ``mask:`` mapping of the config as it was given.

    The mapping is ``AreaSpec.mask_given``, so a key written with its default value still hashes as written, as before.
    An area built by hand has none: the section's keys that differ from their defaults are used.

    Parameters
    ----------
    area : hcwf.config.model.AreaSpec
        The area settings.

    Returns
    -------
    dict or None
        The mask keys and values, or None when the area has no mask.
    """
    if area.mask_given is not None:
        return area.mask_given
    mask = area.mask
    if mask is None:
        return None
    default = type(mask)()
    return {key: list(value) if key == "country" else value for key in _MASK_GIVEN
            if (value := getattr(mask, key)) != getattr(default, key)}


def _reference_part(reference):
    """Return what identifies a reference onset: its month-day, table or computed field.

    Parameters
    ----------
    reference : object or None
        The reference onset of the selection, with ``month_day``, ``table`` (a DataFrame or None) and ``computed``.

    Returns
    -------
    tuple or None
        ``(month_day, table as dict of lists or None, computed)``, or None without a reference.
    """
    if reference is None:
        return None
    table = None if reference.table is None else reference.table.to_dict("list")
    return (reference.month_day, table, reference.computed)


def _options(options, units, select) -> dict:
    """Return the run options that change observed onsets and are set: reading policies, horizon, reference onset.

    Parameters
    ----------
    options : hcwf.config.model.OnsetOptions
        The onset options of the run.
    units : object or None
        The units specification of the observations (None for a grid run).
    select : hcwf.config.model.Selection
        The call's selection, whose ``reference_onset`` is included when set.

    Returns
    -------
    dict
        The set options by name; empty when none is set.
    """
    out = {**options.ingest(False, units)}
    if options.horizon:
        out["horizon"] = options.horizon
    if select.reference_onset is not None:
        out["reference_onset"] = _reference_part(select.reference_onset)
    return out


def fingerprint_parts(rain, thresh, obs, area, onset, options, select, years) -> dict:
    """Return the hash of each fingerprint part for one observed-onset detection (see the module text).

    Parameters
    ----------
    rain : xarray.DataArray or hcwf.onset.model.DailyRain or None
        Supplies the grid of the cells part. With evaluation units it is not needed (None is fine): the files and the
        units specification say which units there are.
    thresh : dict or xarray.DataArray
        The thresholds (see `threshold_hash`).
    obs : hcwf.config.model.ObsSpec
        The run's observation settings.
    area : hcwf.config.model.AreaSpec
        The run's area settings.
    onset : hcwf.config.onset.OnsetSetup
        The run's onset definition, season and observation day boundary.
    options : hcwf.config.model.OnsetOptions
        The run's onset options.
    select : hcwf.config.model.Selection
        The call's selection.
    years : iterable of int
        The years observed onsets are needed for (see `observed_years`).

    Returns
    -------
    dict of str to str
        The hash of each part: ``obs_files``, ``obs_ingest``, ``definition``, ``thresholds``, ``years``, ``cells`` (or
        ``units`` with evaluation units) and, when options are set, ``options``.
    """
    from hcwf.onset import definition as definition_mod

    spec = obs.units
    parts = {
        "obs_files": _obs_files_hash(obs, select.obs_files_years, years),
        "obs_ingest": part_hash(obs.var, obs.unit_cvt, onset.obs_day_boundary, obs.file_pattern),
        "definition": part_hash(definition_mod.definition_hash(onset.definition), onset.season, onset.dry_check),
        "thresholds": threshold_hash(thresh),
        "years": part_hash(list(years)),
    }
    if spec is None:
        lat, lon = (np.asarray(rain.coords[name] if isinstance(rain.coords, dict) else rain[name].values)
                    for name in ("lat", "lon"))   # rain is a DataArray, or an ingested DailyRain
        parts["cells"] = part_hash(area.region, json.dumps(_mask_mapping(area), sort_keys=True, default=str),
                                   _mask_files(area), bool(area.polygon), bool(select.grid_point),
                                   select.lat if select.grid_point else None,
                                   select.lon if select.grid_point else None, lat, lon)
    else:
        parts["units"] = part_hash(repr(spec), [file_identity(p) for p in spec.files()], obs.dims_map(),
                                   sorted(select.keep_unit_ids or ()), onset.lead_label)
    opts = _options(options, spec, select)
    if opts:
        parts["options"] = part_hash(opts)
    return parts


def parts_for(rain, thresh, rc, select=Selection()) -> dict:
    """Return `fingerprint_parts` of a RunConfig's observations, area, onset and options.

    Parameters
    ----------
    rain : xarray.DataArray or hcwf.onset.model.DailyRain or None
        Supplies the grid of the cells part.
    thresh : dict or xarray.DataArray
        The thresholds.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    select : hcwf.config.model.Selection, default Selection()
        The call's selection.

    Returns
    -------
    dict of str to str
        The hash of each fingerprint part.
    """
    return fingerprint_parts(rain, thresh, rc.observations, rc.area, rc.onset, rc.onset_options, select,
                             observed_years(rc.evaluation))


def fingerprint(parts: dict) -> str:
    """Return the fingerprint (``obs-`` and 12 hex characters) of a dict of part hashes.

    Parameters
    ----------
    parts : dict of str to str
        The part hashes, as from `fingerprint_parts`.

    Returns
    -------
    str
        The fingerprint.
    """
    return _new_store(None).fingerprint(parts)


def _table(parts: dict) -> dict:
    """Return the in-memory table of these parts, read from the cache the first time (which logs a hit or a miss).

    Parameters
    ----------
    parts : dict of str to str
        The part hashes of the fingerprint.

    Returns
    -------
    dict
        The table (see `_read`); empty when the entry is not cached.
    """
    store = _run().store
    fp = store.fingerprint(parts)
    table = _tables.get(fp)
    if table is None:
        table = _tables[fp] = store.get(parts) or _empty()
    return table


def lookup(rain, thresh, year, rc, select=Selection()):
    """Return the cached observed onsets of a year, or None.

    A per-unit array the onsets carried comes back as a coordinate. Any failure of the cache is logged at debug level
    and gives None.

    Parameters
    ----------
    rain : xarray.DataArray or hcwf.onset.model.DailyRain or None
        Supplies the grid of the cells part of the fingerprint.
    thresh : dict or xarray.DataArray
        The thresholds.
    year : int
        The year.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    select : hcwf.config.model.Selection, default Selection()
        The call's selection.

    Returns
    -------
    xarray.DataArray or None
        ``onset_date`` (``datetime64[ns]``) over the spatial dims, or None when the cache is inactive or the year is not
        cached yet.
    """
    if _run() is None:
        return None
    try:
        parts = parts_for(rain, thresh, rc, select)
        table = _table(parts)
        values = table["years"].get(int(year))
        if values is None:
            return None
        dims = table["dims"]
        out = xr.DataArray(values.copy(), coords=[(d, table["coords"][d]) for d in dims], name="onset_date")
        extras = table["extras"].get(int(year), {})
        return out.assign_coords({name: (dims, value.copy()) for name, value in extras.items()}) if extras else out
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache lookup failed (%s)", exc)
        return None


def _entry(onset) -> tuple:
    """Split an onset DataArray into what an entry keeps.

    Parameters
    ----------
    onset : xarray.DataArray
        The observed onsets of a year over the spatial dims.

    Returns
    -------
    tuple of (tuple of str, dict, dict)
        ``(dims, coords, extras)``: the dimension names, the index coordinates by dimension, and the arrays the onsets
        carry as coordinates, by name.
    """
    dims = tuple(onset.dims)
    return (dims, {d: np.asarray(onset[d].values) for d in dims},
            {name: np.asarray(onset.coords[name].values) for name in onset.coords if name not in onset.dims})


def store(rain, thresh, year, rc, select, onset) -> None:
    """Keep the detected onsets of a year (memory, and disk when a cache folder is set).

    The disk write happens once, when the run ends (see `flush`). A worker process keeps the onsets for the parent to
    take (see `drain_new`). Any failure of the cache is logged at debug level.

    Parameters
    ----------
    rain : xarray.DataArray or hcwf.onset.model.DailyRain or None
        Supplies the grid of the cells part of the fingerprint.
    thresh : dict or xarray.DataArray
        The thresholds.
    year : int
        The year.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    select : hcwf.config.model.Selection
        The call's selection.
    onset : xarray.DataArray
        The observed onsets of the year over the spatial dims.
    """
    run = _run()
    if run is None:
        return
    try:
        parts = parts_for(rain, thresh, rc, select)
        fp = run.store.fingerprint(parts)
        table = _table(parts)
        dims, coords, extras = _entry(onset)
        table["years"][int(year)] = np.asarray(onset.values).astype("datetime64[ns]")
        table.update(dims=dims, coords=coords, lat=coords.get("lat"), lon=coords.get("lon"))
        if extras:
            table["extras"][int(year)] = extras
        if run.worker:   # a worker process: the parent writes the entry (see absorb)
            run.new.append((fp, parts, int(year), table["years"][int(year)], dims, coords, extras))
        else:
            run.dirty[fp] = (parts, table)   # written once, when the run ends (see flush)
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache store failed (%s)", exc)


def current() -> dict:
    """Return the in-memory tables of this run's detections, over lat and lon or over the units.

    Returns
    -------
    dict
        ``{fingerprint: table}`` for each table with data (see `_read` for the table).
    """
    return {fp: table for fp, table in _tables.items() if table["dims"] is not None}


def export_state() -> dict:
    """Return the cache as worker processes need it.

    The tables travel as the shared memo `_tables` (see `hcwf.cache.shared_state`); this holds the rest.

    Returns
    -------
    dict
        ``dir`` (the cache folder or None) and ``announced`` (the set of entries already announced in the log).
    """
    run = _run()
    return {"dir": None if run is None else run.dir, "announced": set(run.store.announced) if run else set()}


def install_state(state: dict) -> None:
    """Start a worker process's cache from the parent's state: reads the same folder, never writes to it.

    Parameters
    ----------
    state : dict
        The parent's state, as from `export_state`.
    """
    _state.clear()
    run = _state["run"] = _Run(dir=state["dir"], worker=True, store=_new_store(state["dir"]))
    run.store.announced.update(state["announced"])


def drain_new() -> list:
    """Return the onset tables a worker detected since the last call, and forget them.

    Returns
    -------
    list of tuple
        ``(fingerprint, parts, year, values, dims, coords, extras)`` for each record; empty without an active run.
    """
    run = _run()
    if run is None:
        return []
    out, run.new = run.new, []
    return out


def absorb(records: list) -> None:
    """Take onsets a worker detected into this process's cache (memory, and the folder when one is set), in order.

    Parameters
    ----------
    records : list of tuple
        The records of a worker, as from `drain_new`.
    """
    run = _run()
    if run is None:
        return
    for fp, parts, year, values, dims, coords, extras in records:
        table = _table(parts)
        table["years"][year] = values
        table.update(dims=dims, coords=coords, lat=coords.get("lat"), lon=coords.get("lon"))
        if extras:
            table["extras"][year] = extras
        run.dirty[fp] = (parts, table)   # written once, when the run ends (see flush)
