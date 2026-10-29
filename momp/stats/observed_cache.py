"""A cache of observed onset dates, shared by every model, window and reference of a run and kept across runs.

Observed onsets depend only on the observations, how they are read, the evaluated cells, the onset definition,
the thresholds (and season climatology) and the years, so one detection serves all models, verification windows,
the reference and later runs. The entry is keyed by a *fingerprint* of those six parts; the backend, models,
windows and reference are not in it. While a run is active (``activate``), ``momp.stats.detect.detect_observed_onset``
asks ``lookup`` first and calls ``store`` after detecting, so each year is detected once per fingerprint.

On disk (``<cache_dir>/observed_onsets/<fingerprint>/``): ``parts.json`` (the hash of each part) and ``onsets.nc``
(``onset_date`` over year, lat, lon). The first use of a fingerprint in a run logs a hit, a partial hit or a miss;
on a miss it names the parts that differ from the nearest cached entry. Without a cache folder the cache is in
memory for the run only. A failure of the cache never stops a run: the onsets are then simply detected.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

PARTS = ("obs_files", "obs_ingest", "cells", "definition", "thresholds", "years")
"""The parts of a fingerprint, in the order they are reported."""
SUBFOLDER = "observed_onsets"
_state = {"on": False, "dir": None, "worker": False}
_tables: dict = {}
_new: list = []
_announced: set = set()
_memo: dict = {}


def activate(cache_dir=None) -> None:
    """Start using the cache for this run; cache_dir None keeps it in memory. Clears what an earlier run left."""
    reset()
    _state.update(on=True, dir=Path(cache_dir) if cache_dir else None)


def reset() -> None:
    """Stop using the cache and forget the in-memory tables, announcements and memoised parts."""
    _state.update(on=False, dir=None, worker=False)
    _new.clear()
    _tables.clear()
    _announced.clear()
    _memo.clear()
    from momp.io.input import clear_obs_cache

    clear_obs_cache()


def _digest(*chunks) -> str:
    """A 12-hex-character hash of the given text or bytes."""
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk if isinstance(chunk, bytes) else str(chunk).encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


def _file_id(path) -> list:
    """Name, size and modification time of a file, or just the name when it is missing."""
    try:
        stat = os.stat(path)
        return [str(path), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [str(path), None, None]


def _array_bytes(value) -> bytes:
    """The bytes of an array-like (names and dims included), for hashing."""
    if value is None:
        return b"none"
    if isinstance(value, (int, float, np.number)):
        return repr(float(value)).encode()
    da = value if isinstance(value, xr.DataArray) else xr.DataArray(np.asarray(value))
    return (repr(da.dims) + repr(da.shape)).encode() + np.ascontiguousarray(da.values).tobytes()


def _threshold_hash(thresh) -> str:
    """Hash of the threshold values: the named arrays, the season climatology, or the legacy scalar or map."""
    cached = _memo.get(("thresh", id(thresh)))
    if cached is not None and cached[0] is thresh:
        return cached[1]
    chunks = []
    if isinstance(thresh, dict):
        for name in sorted(thresh):
            chunks += [name, _array_bytes(thresh[name])]
        season = getattr(thresh, "season_climatology", None)
        if season is not None:
            for name in sorted(season.data_vars):
                chunks += [name, _array_bytes(season[name])]
    else:
        chunks.append(_array_bytes(thresh))
    result = _digest(*chunks)
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


def _obs_files_hash(kwargs, years) -> str:
    """Hash of the identity (name, size, mtime) of the observation file of every needed year."""
    key = ("files", str(kwargs.get("obs_dir")), str(kwargs.get("obs_file_pattern")), tuple(years))
    if key not in _memo:
        found = []
        from momp.io.input import is_zarr

        if is_zarr(kwargs["obs_dir"]):
            store = Path(str(kwargs["obs_dir"]))
            found = [_file_id(store / name) for name in ("zarr.json", ".zmetadata", ".zgroup")] + [list(years)]
            years = ()
        for year in years:
            for pattern in kwargs.get("obs_file_pattern") or ():
                path = Path(str(kwargs["obs_dir"])) / pattern.format(year)
                if path.exists():
                    found.append(_file_id(path))
                    break
            else:
                found.append([f"missing {year}"])
        _memo[key] = _digest(json.dumps(found))
    return _memo[key]


def _mask_files(kwargs) -> list:
    """Identity of the files the cell selection reads (mask file, condition file, nc_mask, shapefiles)."""
    paths = []
    mask = kwargs.get("mask")
    if isinstance(mask, dict):
        paths += [mask.get("file"), (mask.get("condition") or {}).get("file")]
    paths += [kwargs.get("nc_mask"), kwargs.get("shpfile_dir")]
    return [_file_id(p) for p in paths if p]


def fingerprint_parts(rain, thresh, kwargs) -> dict:
    """The hash of each fingerprint part for one observed-onset detection (see the module text)."""
    from momp.lib.onset_config import onset_setup
    from momp.onset import definition as definition_mod

    setup = onset_setup(kwargs, thresh)
    years = _years(kwargs)
    lat, lon = np.asarray(rain["lat"].values), np.asarray(rain["lon"].values)
    return {
        "obs_files": _obs_files_hash(kwargs, years),
        "obs_ingest": _digest(kwargs.get("obs_var"), kwargs.get("obs_unit_cvt"), setup.ingest_boundary,
                              kwargs.get("obs_file_pattern")),
        "cells": _digest(kwargs.get("region"), json.dumps(kwargs.get("mask"), sort_keys=True, default=str),
                         json.dumps(_mask_files(kwargs)), bool(kwargs.get("polygon")), bool(kwargs.get("grid_point")),
                         kwargs.get("lat_select") if kwargs.get("grid_point") else None,
                         kwargs.get("lon_select") if kwargs.get("grid_point") else None, lat.tobytes(), lon.tobytes()),
        "definition": _digest(definition_mod.definition_hash(setup.definition), setup.season, setup.dry_check,
                              setup.legacy),
        "thresholds": _threshold_hash(thresh),
        "years": _digest(json.dumps(years)),
    }


def fingerprint(parts: dict) -> str:
    """The fingerprint (``obs-`` and 12 hex characters) of a dict of part hashes."""
    return "obs-" + _digest(json.dumps(parts, sort_keys=True))


def _entry_dir(fp: str) -> Path | None:
    """The disk folder of a fingerprint, or None for an in-memory cache."""
    return _state["dir"] / SUBFOLDER / fp if _state["dir"] is not None else None


def _read(folder: Path) -> dict | None:
    """Read a cached entry: {"years": {year: values}, "lat": ..., "lon": ...}, or None if unreadable."""
    try:
        with xr.open_dataset(folder / "onsets.nc") as ds:
            years = {int(y): ds["onset_date"].sel(year=y).values.astype("datetime64[ns]") for y in ds["year"].values}
            return {"years": years, "lat": ds["lat"].values.copy(), "lon": ds["lon"].values.copy()}
    except Exception as exc:  # noqa: BLE001 - a damaged entry is recomputed
        log.debug("observed-onset cache entry %s unreadable (%s)", folder, exc)
        return None


def _write(fp: str, parts: dict, table: dict) -> None:
    """Write an entry (atomically, so parallel workers cannot leave a half file)."""
    folder = _entry_dir(fp)
    if folder is None or not table["years"]:
        return
    folder.mkdir(parents=True, exist_ok=True)
    years = sorted(table["years"])
    ds = xr.Dataset({"onset_date": (("year", "lat", "lon"), np.stack([table["years"][y] for y in years]))},
                    coords={"year": years, "lat": table["lat"], "lon": table["lon"]})
    temp = folder / f"onsets.{os.getpid()}.tmp"
    ds.to_netcdf(temp)
    os.replace(temp, folder / "onsets.nc")
    (folder / "parts.json").write_text(json.dumps({"fingerprint": fp, "parts": parts}, indent=1))


def _nearest(fp: str, parts: dict) -> tuple[str, list[str]] | None:
    """The cached entry sharing most parts with these, and the names of the parts that differ."""
    root = _state["dir"] / SUBFOLDER if _state["dir"] is not None else None
    if root is None or not root.is_dir():
        return None
    best = None
    for folder in root.iterdir():
        try:
            other = json.loads((folder / "parts.json").read_text())["parts"]
        except (OSError, ValueError, KeyError):
            continue
        if folder.name == fp:
            continue
        differ = [name for name in PARTS if other.get(name) != parts.get(name)]
        if best is None or len(differ) < len(best[1]):
            best = (folder.name, differ)
    return best


def _table(fp: str, parts: dict) -> dict:
    """The in-memory table of a fingerprint, read from disk the first time; the first use is announced."""
    if fp in _tables:
        return _tables[fp]
    table = {"years": {}, "lat": None, "lon": None}
    folder = _entry_dir(fp)
    stored = _read(folder) if folder is not None and (folder / "onsets.nc").exists() else None
    if stored:
        table = stored
    _tables[fp] = table
    if fp not in _announced:
        _announced.add(fp)
        where = "in memory" if _state["dir"] is None else str(_state["dir"] / SUBFOLDER)
        if stored:
            log.info("Observed onsets: cache hit %s (%d years cached, %s)", fp, len(stored["years"]), where)
        else:
            near = _nearest(fp, parts)
            if near is None:
                log.info("Observed onsets: cache miss %s (no earlier entry in %s); detecting once for all "
                         "models, windows and references", fp, where)
            else:
                log.info("Observed onsets: cache miss %s; nearest cached entry %s differs in: %s", fp, near[0],
                         ", ".join(near[1]))
    return table


def lookup(rain, thresh, year, kwargs):
    """The cached observed onsets of a year as a (lat, lon) DataArray, or None (inactive, or not cached yet)."""
    if not _state["on"]:
        return None
    try:
        parts = fingerprint_parts(rain, thresh, kwargs)
        table = _table(fingerprint(parts), parts)
        values = table["years"].get(int(year))
        if values is None:
            return None
        return xr.DataArray(values.copy(), coords=[("lat", table["lat"]), ("lon", table["lon"])], name="onset_date")
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache lookup failed (%s)", exc)
        return None


def store(rain, thresh, year, kwargs, onset) -> None:
    """Keep the detected onsets of a year (memory, and disk when a cache folder is set)."""
    if not _state["on"]:
        return
    try:
        parts = fingerprint_parts(rain, thresh, kwargs)
        fp = fingerprint(parts)
        table = _table(fp, parts)
        table["years"][int(year)] = np.asarray(onset.values).astype("datetime64[ns]")
        table["lat"], table["lon"] = np.asarray(onset["lat"].values), np.asarray(onset["lon"].values)
        if _state["worker"]:   # a worker process: the parent writes the entry (see absorb)
            _new.append((fp, parts, int(year), table["years"][int(year)], table["lat"], table["lon"]))
        else:
            _write(fp, parts, table)
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache store failed (%s)", exc)


def current() -> dict:
    """The in-memory tables of this run, {fingerprint: {"years": {year: values}, "lat", "lon"}} (for results)."""
    return _tables


def export_state() -> dict:
    """The cache as worker processes need it: its folder, the tables read or detected so far and which were announced."""
    return {"dir": _state["dir"], "tables": dict(_tables), "announced": set(_announced)}


def install_state(state: dict) -> None:
    """Start a worker process's cache from the parent's state: reads the same folder, never writes to it."""
    reset()
    _state.update(on=True, dir=state["dir"], worker=True)
    _tables.update(state["tables"])
    _announced.update(state["announced"])


def drain_new() -> list:
    """The onset tables a worker detected since the last call, [(fingerprint, parts, year, values, lat, lon)]; clears them."""
    out = list(_new)
    _new.clear()
    return out


def absorb(records: list) -> None:
    """Take onsets a worker detected into this process's cache (memory, and the folder when one is set), in order."""
    if not _state["on"]:
        return
    for fp, parts, year, values, lat, lon in records:
        table = _table(fp, parts)
        table["years"][year] = values
        table["lat"], table["lon"] = lat, lon
        _write(fp, parts, table)
