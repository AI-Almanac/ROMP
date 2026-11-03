"""A cache of observed onset dates, shared by every model, window and reference of a run and kept across runs.

Observed onsets depend only on the observations, how they are read, the evaluated cells, the onset definition,
the thresholds (and season climatology) and the years, so one detection serves all models, verification windows,
the reference and later runs. The entry is keyed by a *fingerprint* of those six parts (see hcwf.cache.Store, which
keeps the entries and reports hits and misses); the backend, models, windows and reference are not in it. Evaluation
units replace the cells part by a units part (the units specification, the files it names, the units kept), and run
options that change the numbers (missing-rain policy, resolution, horizon, the reference onset a search starts at) add
an options part when they are set; without them the fingerprint is the one a grid run always had. While a run is active (``activate``), ``hcwf.stats.detect.detect_observed_onset``
asks ``lookup`` first and calls ``store`` after detecting, so each year is detected once per fingerprint.

On disk (``<cache_dir>/observed_onsets/<fingerprint>/``): ``parts.json`` (the hash of each part) and ``onsets.nc``
(``onset_date`` over year and the spatial dimensions, lat and lon or unit, plus any per-unit arrays the onsets carry as
coordinates, such as ``present``). Without a cache folder the cache is in memory for the run only. A failure of the cache never stops a run: the onsets are then simply detected.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import Codec, Store, file_identity, part_hash

log = logging.getLogger(__name__)

SUBFOLDER = "observed_onsets"
_state = {"on": False, "dir": None, "worker": False, "store": None}
_tables: dict = {}
_new: list = []
_memo: dict = {}


def _empty() -> dict:
    return {"years": {}, "dims": None, "coords": None, "lat": None, "lon": None, "extras": {}}


def _read(path: Path) -> dict | None:
    """Read a cached entry: {"years": {year: values}, "dims", "coords": {dim: values}, "extras": {year: {name: values}}}
    (with "lat" and "lon" too for a grid)."""
    with xr.open_dataset(path) as ds:
        dims = tuple(ds["onset_date"].dims[1:])
        years = {int(y): ds["onset_date"].sel(year=y).values.astype("datetime64[ns]") for y in ds["year"].values}
        extra = [n for n in ds.data_vars if n != "onset_date"]
        extras = {int(y): {n: ds[n].sel(year=y).values for n in extra} for y in ds["year"].values}
        coords = {d: ds[d].values.copy() for d in dims}
        return {"years": years, "dims": dims, "coords": coords, "lat": coords.get("lat"), "lon": coords.get("lon"),
                "extras": extras}


def _write(path: Path, table: dict) -> None:
    """Write an entry: onset_date over year and the spatial dims, and each extra array over the same."""
    years, dims = sorted(table["years"]), table["dims"]
    variables = {"onset_date": (("year", *dims), np.stack([table["years"][y] for y in years]))}
    for name in sorted({n for y in years for n in table["extras"].get(y, {})}):
        variables[name] = (("year", *dims), np.stack([table["extras"][y][name] for y in years]))
    ds = xr.Dataset(variables, coords={"year": years, **{d: table["coords"][d] for d in dims}})
    ds.to_netcdf(path)


def _new_store(root) -> Store:
    """The observed-onset store of a cache folder (None: in memory)."""
    return Store(SUBFOLDER, root, codec=Codec("onsets.nc", _read, _write), prefix="obs", label="Observed onsets",
                 describe=lambda table: f"{len(table['years'])} years cached",
                 miss_note="; detecting once for all models, windows and references")


def activate(cache_dir=None) -> None:
    """Start using the cache for this run; cache_dir None keeps it in memory. Clears what an earlier run left."""
    reset()
    _state.update(on=True, dir=Path(cache_dir) if cache_dir else None, store=_new_store(cache_dir))


def reset() -> None:
    """Stop using the cache and forget the in-memory tables, announcements and memoised parts."""
    _state.update(on=False, dir=None, worker=False, store=None)
    _new.clear()
    _tables.clear()
    _memo.clear()
    from hcwf.io.input import clear_obs_cache

    clear_obs_cache()


def threshold_hash(thresh) -> str:
    """Hash of the threshold values: the named arrays or the season climatology."""
    cached = _memo.get(("thresh", id(thresh)))
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
    _memo[("thresh", id(thresh))] = (thresh, result)
    return result


def _years(kwargs) -> list[int]:
    """The years observed onsets are needed for: the evaluated years and the climatology years."""
    years = set()
    for key in ("years", "years_clim"):
        value = kwargs.get(key)
        if value and not isinstance(value, str):
            years.update(int(y) for y in value)
    return sorted(years)


def obs_files(kwargs, years) -> list:
    """The identity (name, size, mtime) of the observation file of every year (or of the zarr store, with the years)."""
    from hcwf.io.input import is_zarr, year_files

    if kwargs.get("obs_paths"):   # listed files, any number per year
        found = []
        for year in years:
            files = year_files(kwargs["obs_paths"], kwargs.get("obs_dir") or None, kwargs.get("obs_files_years"), year)
            found.append([file_identity(f) for f in files] or [f"missing {year}"])
        return found
    if is_zarr(kwargs["obs_dir"]):
        return [file_identity(kwargs["obs_dir"]), list(years)]
    found = []
    for year in years:
        for pattern in kwargs.get("obs_file_pattern") or ():
            path = Path(str(kwargs["obs_dir"])) / pattern.format(year)
            if path.exists():
                found.append(file_identity(path))
                break
        else:
            found.append([f"missing {year}"])
    return found


def _obs_files_hash(kwargs, years) -> str:
    """Hash of obs_files, memoised for the run."""
    key = ("files", str(kwargs.get("obs_dir")), str(kwargs.get("obs_file_pattern")), tuple(years),
           str(kwargs.get("obs_paths")), str(kwargs.get("obs_files_years")))
    if key not in _memo:
        _memo[key] = part_hash(obs_files(kwargs, years))
    return _memo[key]


def _mask_files(kwargs) -> list:
    """Identity of the files the cell selection reads (mask file, condition file, nc_mask, shapefiles)."""
    paths = []
    mask = kwargs.get("mask")
    if isinstance(mask, dict):
        paths += [mask.get("file"), (mask.get("condition") or {}).get("file")]
    paths += [kwargs.get("nc_mask"), kwargs.get("shpfile_dir")]
    return [file_identity(p) for p in paths if p]


def _reference_part(reference):
    """What identifies a reference onset: its month-day, table or computed field."""
    if reference is None:
        return None
    table = None if reference.table is None else reference.table.to_dict("list")
    return (reference.month_day, table, reference.computed, reference.statistic)


def _options(kwargs) -> dict:
    """The run options that change observed onsets and are set: reading policies, the horizon, the reference onset."""
    from hcwf.io.input import ingest_options

    options = {**ingest_options(kwargs)}
    if kwargs.get("onset_horizon"):
        options["horizon"] = kwargs["onset_horizon"]
    if kwargs.get("reference_onset") is not None:
        options["reference_onset"] = _reference_part(kwargs["reference_onset"])
    return options


def fingerprint_parts(rain, thresh, kwargs) -> dict:
    """The hash of each fingerprint part for one observed-onset detection (see the module text).

    `rain` supplies the grid of the cells part (a DataArray, or an ingested DailyRain); with evaluation units it is not
    needed (None is fine): the files and the units specification say which units there are."""
    from hcwf.io.input import units_files, units_spec
    from hcwf.lib.onset_config import onset_setup
    from hcwf.onset import definition as definition_mod

    setup = onset_setup(kwargs)
    years = _years(kwargs)
    spec = units_spec(kwargs, "observations")
    parts = {
        "obs_files": _obs_files_hash(kwargs, years),
        "obs_ingest": part_hash(kwargs.get("obs_var"), kwargs.get("obs_unit_cvt"), setup.obs_day_boundary,
                                kwargs.get("obs_file_pattern")),
        "definition": part_hash(definition_mod.definition_hash(setup.definition), setup.season, setup.dry_check),
        "thresholds": threshold_hash(thresh),
        "years": part_hash(years),
    }
    if spec is None:
        lat, lon = (np.asarray(rain.coords[name] if isinstance(rain.coords, dict) else rain[name].values)
                    for name in ("lat", "lon"))   # rain is a DataArray, or an ingested DailyRain
        parts["cells"] = part_hash(kwargs.get("region"), json.dumps(kwargs.get("mask"), sort_keys=True, default=str),
                                   _mask_files(kwargs), bool(kwargs.get("polygon")), bool(kwargs.get("grid_point")),
                                   kwargs.get("lat_select") if kwargs.get("grid_point") else None,
                                   kwargs.get("lon_select") if kwargs.get("grid_point") else None, lat, lon)
    else:
        parts["units"] = part_hash(repr(spec), units_files(kwargs), kwargs.get("obs_dims"),
                                   sorted(kwargs.get("keep_unit_ids") or ()), setup.lead_label)
    options = _options(kwargs)
    if options:
        parts["options"] = part_hash(options)
    return parts


def fingerprint(parts: dict) -> str:
    """The fingerprint (``obs-`` and 12 hex characters) of a dict of part hashes."""
    return _new_store(None).fingerprint(parts)


def _table(parts: dict) -> dict:
    """The in-memory table of these parts, read from the cache the first time (which logs a hit or a miss)."""
    fp = _state["store"].fingerprint(parts)
    if fp not in _tables:
        _tables[fp] = _state["store"].get(parts) or _empty()
    return _tables[fp]


def lookup(rain, thresh, year, kwargs):
    """The cached observed onsets of a year as a DataArray over the spatial dims (a per-unit array the onsets carried
    comes back as a coordinate), or None (inactive, or not cached yet)."""
    if not _state["on"]:
        return None
    try:
        parts = fingerprint_parts(rain, thresh, kwargs)
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
    """(dims, coords, extras) of an onset DataArray: its index coordinates and the arrays it carries as coordinates."""
    dims = tuple(onset.dims)
    return (dims, {d: np.asarray(onset[d].values) for d in dims},
            {name: np.asarray(onset.coords[name].values) for name in onset.coords if name not in onset.dims})


def store(rain, thresh, year, kwargs, onset) -> None:
    """Keep the detected onsets of a year (memory, and disk when a cache folder is set)."""
    if not _state["on"]:
        return
    try:
        parts = fingerprint_parts(rain, thresh, kwargs)
        fp = _state["store"].fingerprint(parts)
        table = _table(parts)
        dims, coords, extras = _entry(onset)
        table["years"][int(year)] = np.asarray(onset.values).astype("datetime64[ns]")
        table.update(dims=dims, coords=coords, lat=coords.get("lat"), lon=coords.get("lon"))
        if extras:
            table["extras"][int(year)] = extras
        if _state["worker"]:   # a worker process: the parent writes the entry (see absorb)
            _new.append((fp, parts, int(year), table["years"][int(year)], dims, coords, extras))
        else:
            _state["store"].put(parts, table)
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache store failed (%s)", exc)


def current() -> dict:
    """The in-memory tables of this run's detections, {fingerprint: {"years": {year: values}, "dims", "coords"}}
    (with "lat" and "lon" too for a grid), over lat and lon or over the units."""
    return {fp: table for fp, table in _tables.items() if table["dims"] is not None}


def export_state() -> dict:
    """The cache as worker processes need it: its folder, the tables read or detected so far and which were announced."""
    return {"dir": _state["dir"], "tables": dict(_tables), "announced": set(_state["store"].announced if _state["store"] else ())}


def install_state(state: dict) -> None:
    """Start a worker process's cache from the parent's state: reads the same folder, never writes to it."""
    reset()
    _state.update(on=True, dir=state["dir"], worker=True, store=_new_store(state["dir"]))
    _tables.update(state["tables"])
    _state["store"].announced.update(state["announced"])


def drain_new() -> list:
    """The onset tables a worker detected since the last call, [(fingerprint, parts, year, values, dims, coords, extras)];
    clears them."""
    out = list(_new)
    _new.clear()
    return out


def absorb(records: list) -> None:
    """Take onsets a worker detected into this process's cache (memory, and the folder when one is set), in order."""
    if not _state["on"]:
        return
    for fp, parts, year, values, dims, coords, extras in records:
        table = _table(parts)
        table["years"][year] = values
        table.update(dims=dims, coords=coords, lat=coords.get("lat"), lon=coords.get("lon"))
        if extras:
            table["extras"][year] = extras
        _state["store"].put(parts, table)
