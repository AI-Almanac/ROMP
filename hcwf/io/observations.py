"""Reading observed daily rainfall a year at a time, from files or a zarr store, on grid cells or on evaluation units.

`load_observations` reads one year. The years a run scores (and the year before each) are read once and held for the
run (`hcwf.cache.reset_run_state` drops them). The region's box is selected on the lazy array before any data is
decoded; the cell mask and the unit conversion are applied in place on the one array that was read. Data on a string
``"unit"`` dimension (admin units) is read without a region box.

The files are located by `hcwf.io.stores`, the dimension names standardised by `hcwf.io.dims`, the region and masks
applied by `hcwf.spatial.select`, and `hcwf.io.reference` and `hcwf.io.thresholds` build on these readers.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import memo
from hcwf.config.model import NO_SELECTION
from hcwf.config.onset import history_needs_previous_year
from hcwf.io.dims import dim_fmt, rename_model_dims
from hcwf.io.stores import observation_source, open_data, open_store
from hcwf.onset.ingest import observation_shift_days
from hcwf.parallel import ordered_prefetch, worker_count
from hcwf.spatial.onto_units import (
    declared_scale,
    read_units,
    select_units,
    units_files,
)
from hcwf.spatial.evalgrid import OBSERVATIONS, region_read
from hcwf.spatial.select import (
    implicit_keep,
    implicit_masks,
    area_box,
    in_memory,
    scale_in_place,
)
from hcwf.spatial.units import UnitsSpec

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.config.model import (
        AreaSpec,
        ObsSpec,
        OnsetOptions,
        ParallelSpec,
        Selection,
    )
    from hcwf.config.onset import OnsetSetup

__all__ = ["history_of", "observation_grid_of", "observation_years_of", "observations_of"]


log = logging.getLogger(__name__)

_obs_years = memo("io.observations.years")
"""The observation years of the run, read once and held (see `years_held`), by file and settings."""
_obs_lock = threading.Lock()
"""Held while a year is looked up and, if absent, read and stored, so that two reader threads do not read it twice."""


@dataclass(frozen=True, eq=False)
class ObsRead:
    """What reading the observations depends on, besides the year: the typed inputs of one call (see `load_observations`)."""

    obs: ObsSpec
    """The observation dataset settings."""
    area: AreaSpec
    """The region and mask settings."""
    options: OnsetOptions
    """The onset options."""
    select: Selection
    """The run's selection settings."""
    onset: OnsetSetup | None
    """The onset setup; needed for observations on units."""
    run_years: tuple
    """The years the run scores (they and the year before each are held)."""
    dir_fig: object
    """The folder for mask figures."""
    parallel: ParallelSpec | None = None
    """The parallel settings, or None for the defaults."""


def mask_text(area) -> str:
    """Return the ``mask`` section as text, for use in a cache key.

    Parameters
    ----------
    area : hcwf.config.model.AreaSpec
        The region and mask settings.

    Returns
    -------
    str
        ``"null"`` without a mask, else the mask's key; followed by the evaluation grid's key when the run has one
        (the cells read are then the evaluation grid's).
    """
    grid = "" if area.evaluation_grid is None else f"|grid:{area.evaluation_grid.key()}"
    return ("null" if not area.mask else area.mask.key()) + grid


def years_held(run_years):
    """Return the years a run scores (and the year before each, for the observed history): read once, held for the run.

    Climatology and threshold years outside these are read, used and dropped (the observed-onset cache already
    shares what they yield), so the held observations stay the size of the evaluation period.

    Parameters
    ----------
    run_years : iterable of int
        The years the run scores.

    Returns
    -------
    set of int
        The years and the year before each.
    """
    years = {int(y) for y in run_years}
    return years | {y - 1 for y in years}


_GROUP_BYTES = 1_500_000_000
"""The most observation data read in one block when years are read together (a block is read, sliced per year, freed)."""


def _locate(year, read: ObsRead):
    """Locate the file or store holding the year's observations.

    Parameters
    ----------
    year : int
        The year to locate.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    tuple of (str or None, str or tuple of str or None)
        The zarr store path or None, and the file path (or paths) or None; see `hcwf.io.stores.observation_source`,
        which raises `FileNotFoundError` when no file holds the year.
    """
    return observation_source(year, read.obs.dir, read.obs.file_pattern, read.obs.paths, read.select.obs_files_years)


def _read_obs_range(first, last, store, read: ObsRead, cells=False):
    """Read the region's box of a zarr store's observations from 1 January of ``first`` to 31 December of ``last``, in one pass.

    Equal, year by year, to `_read_obs_year` (the same cells masked, the same unit factor); an archive chunked over
    the whole time axis is decoded once for all the years instead of once per year.

    Parameters
    ----------
    first : int
        The first year.
    last : int
        The last year, inclusive.
    store : str
        The zarr store path.
    read : ObsRead
        The typed inputs of the observation reading.
    cells : bool, default False
        Read only the cells the region's masks keep (see `_read_obs_year`).

    Returns
    -------
    xarray.DataArray
        The observation variable over the days of the years, in mm, with the unit factor applied.
    """
    obs = read.obs
    ds = dim_fmt(open_store(store))
    ds = ds[[obs.var]].sel(time=slice(f"{first}-01-01", f"{last}-12-31"))
    da = region_read(ds, obs.var, area=read.area, select=read.select, source=OBSERVATIONS, read_by="time",
                     dataset_name=str(obs.name or "obs"), dir_fig=read.dir_fig, skip_masked=cells, compact=cells)
    return scale_in_place(in_memory(da), obs.unit_cvt)


def _long_time_chunks(store, read: ObsRead) -> bool:
    """Tell whether the store's time chunks are longer than a year, so each year read decodes data of other years too.

    Observations on evaluation units (``units:``) are never read in blocks: the block read keeps the grid, and the
    units are built for one year at a time (`_read_obs_units`).

    Parameters
    ----------
    store : str
        The zarr store path.
    read : ObsRead
        The typed inputs of the observation reading (its variable and ``units``).

    Returns
    -------
    bool
        True when the time chunk exceeds 366 days and the observations are on the grid; False also when the layout
        cannot be read.
    """
    if read.obs.units is not None:
        return False
    obs_var = read.obs.var
    try:
        da = dim_fmt(open_store(store))[obs_var]
        chunks = da.encoding.get("chunks")
        return bool(chunks) and chunks[da.dims.index("time")] > 366
    except Exception:  # noqa: BLE001 - an unreadable layout is read year by year, as before
        return False


def _year_slice(block, year, store):
    """Slice one year out of a block read by `_read_obs_range` (a view of it).

    Parameters
    ----------
    block : xarray.DataArray
        The block, with a ``time`` dimension.
    year : int
        The year to slice.
    store : str
        The zarr store path, for the error message.

    Returns
    -------
    xarray.DataArray
        The year's days of the block.

    Raises
    ------
    FileNotFoundError
        The year is not in the block.
    """
    sub = block.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if sub.sizes["time"] == 0:
        raise FileNotFoundError(f"No observations for year {year} in {store}")
    return sub


def _estimated_year_bytes(store, obs_var) -> int:
    """Return an upper bound of one year of the store's variable in memory (its whole grid, 366 days, as stored).

    Parameters
    ----------
    store : str
        The zarr store path.
    obs_var : str
        The observation variable.

    Returns
    -------
    int
        The size in bytes, at least 1.
    """
    da = dim_fmt(open_store(store))[obs_var]
    cells = int(np.prod([n for dim, n in da.sizes.items() if dim != "time"], dtype=np.int64))
    return max(1, cells * da.dtype.itemsize * 366)


def read_obs_years(years, obs, area, options, select=NO_SELECTION, *, onset=None, run_years=(), dir_fig=None,
                   parallel=None, hold=True):
    """Return an iterator over each of the years' observations in order, as `load_observations` gives them.

    The years the run holds (``run_years`` and the year before each) are taken from, and kept in, the run's
    observation cache. A zarr store chunked over many years is read in blocks of about 1.5 GB, each decoded once
    and sliced per year; any other source is read a year at a time, the next few ahead in threads (zarr only;
    NetCDF files are read one at a time, as HDF5 is not thread-safe). Memory stays at one block, or
    a few years, however many years are asked for.

    Parameters
    ----------
    years : iterable of int
        The years to read, in the order of the result.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    options : hcwf.config.model.OnsetOptions
        The onset options.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.
    onset : hcwf.config.onset.OnsetSetup, optional
        The onset setup; needed for observations on units.
    run_years : iterable of int, default ()
        The years the run scores (they and the year before each are held).
    dir_fig : str or pathlib.Path, optional
        The folder for mask figures.
    parallel : hcwf.config.model.ParallelSpec, optional
        The parallel settings; None for the defaults.
    hold : bool, default True
        Keep the held years; False reads without keeping even those.

    Returns
    -------
    collections.abc.Iterator
        The generator of `obs_years_of`: ``(rain, None)``, or ``(None, error)`` for a year that could not be read.
    """
    return obs_years_of(ObsRead(obs, area, options, select, onset, tuple(run_years), dir_fig, parallel), years, hold)


def obs_years_of(read: ObsRead, years, hold=True, cells=False) -> Iterator[tuple]:
    """Yield each of the years' observations in order, or the error that stopped a year.

    The generator behind `read_obs_years`. With ``cells`` a year that is not held and not already in the run's cache is
    read as only the cells the region's masks keep (dims ``time``, ``cell``, see `_read_obs_year`); the held years stay
    whole, as the run's cache has them.

    Parameters
    ----------
    read : ObsRead
        The typed inputs of the observation reading.
    years : iterable of int
        The years to read, in the order of the result.
    hold : bool, default True
        Keep the held years; False reads without keeping even those.
    cells : bool, default False
        Read the not-held years on the kept cells only.

    Yields
    ------
    tuple
        ``(rain, None)`` with the year's `xarray.DataArray`, or ``(None, error)`` where ``error`` is the exception
        that stopped the year (a missing file, for example).
    """
    years = list(years)
    if not years:
        return
    obs, select = read.obs, read.select
    keep = years_held(read.run_years) if hold else set()

    def locate(year):
        """Return the observation-cache key, the zarr store (or None) and the file path (or None) of a year."""
        store, path = _locate(year, read)
        return _obs_key(year, store, path, read), store, path

    def held(year):
        """Return the array the run holds for the year, or None."""
        with _obs_lock:
            return _obs_years.get(locate(year)[0])

    def finish(year, rainfall, view=False):
        """Hold the year if the run does (a copy when it is a view of a block, so it can be freed), then apply the grid point."""
        if year in keep:
            with _obs_lock:
                rainfall = _obs_years.setdefault(locate(year)[0], rainfall.copy(deep=True) if view else rainfall)
        return _at_grid_point(rainfall, select)

    try:
        store = locate(years[0])[1]
    except FileNotFoundError:
        store = None
    if store is not None and len(years) > 1 and _long_time_chunks(store, read):
        per_block = max(1, _GROUP_BYTES // _estimated_year_bytes(store, obs.var))
        for start in range(0, len(years), per_block):
            group = years[start:start + per_block]
            need = [year for year in group if held(year) is None]
            compact = cells and not any(year in keep for year in need)
            block = _read_obs_range(min(need), max(need), store, read, compact) if need else None
            for year in group:
                try:
                    rainfall = held(year)
                    if rainfall is None:
                        yield finish(year, _year_slice(block, year, store), view=True), None
                    else:
                        yield finish(year, rainfall), None
                except FileNotFoundError as exc:
                    yield None, exc
            del block
        return

    # NetCDF/HDF5 readers are not thread-safe: only a zarr store is read by several threads at once
    readers = _workers(read.parallel, len(years)) if store is not None else 1

    def single(year):
        """Read one year on its own (or take the held one), or return the error that stopped it."""
        try:
            rainfall = held(year)
            if rainfall is None:
                _, store, path = locate(year)
                rainfall = _read_obs_year(year, store, path, read, cells and year not in keep)
            return rainfall, None
        except Exception as exc:  # noqa: BLE001 - handed to the caller, which decides what a missing year means
            return None, exc

    for year, (rainfall, error) in zip(years, ordered_prefetch(single, years, readers)):
        yield (finish(year, rainfall), None) if error is None else (None, error)


def _workers(parallel, tasks) -> int:
    """Return the workers for ``tasks`` independent tasks under the run's parallel setting.

    Parameters
    ----------
    parallel : hcwf.config.model.ParallelSpec or None
        The parallel settings; None gives the defaults.
    tasks : int
        The number of independent tasks.

    Returns
    -------
    int
        The worker count (see `hcwf.parallel.worker_count`).
    """
    config = {} if parallel is None else {"parallel": parallel.enabled, "parallel_workers": parallel.workers}
    return worker_count(config, tasks)


def _open_year(year, store, path, read: ObsRead):
    """Open the year's observations lazily, in the package's dimension names.

    Parameters
    ----------
    year : int
        The year to open.
    store : str or None
        The zarr store path (the year's days of it are selected), or None for a file.
    path : str or None
        The NetCDF file path, when ``store`` is None.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    xarray.Dataset
        The lazy dataset with canonical dimension names.

    Raises
    ------
    FileNotFoundError
        The zarr store has no observations of the year.
    """
    if store is not None:
        ds = dim_fmt(open_store(store))
        ds = ds[[read.obs.var]].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if ds.sizes["time"] == 0:
            raise FileNotFoundError(f"No observations for year {year} in {store}")
        return ds
    return dim_fmt(rename_model_dims(open_data(path), read.obs.dims_map()))


def _observation_box(year, read: ObsRead, keep=False):
    """Return the region's box (or the one grid point) of the year's lazy observations.

    Nothing of the rainfall is read.

    Parameters
    ----------
    year : int
        The year to look at.
    read : ObsRead
        The typed inputs of the observation reading.
    keep : bool, default False
        Also find the cells the implicit (land and country) masks keep.

    Returns
    -------
    tuple of (dict, tuple of str, numpy.ndarray of bool or None)
        The ``lat`` and ``lon`` values of the box, the dims of the variable, and the boolean ``(lat, lon)`` array of
        the kept cells (None unless ``keep``).
    """
    store, path = _locate(year, read)
    ds = _open_year(year, store, path[0] if isinstance(path, tuple) else path, read)
    try:
        box = area_box(ds[[read.obs.var]], read.area)
        dims = tuple(box[read.obs.var].dims)
        grid = read.area.evaluation_grid
        if grid is not None and grid.regrids(OBSERVATIONS):   # the cells read are the evaluation grid's
            field = xr.DataArray(np.zeros((len(grid.lat), len(grid.lon)), dtype="float32"), dims=("lat", "lon"),
                                 coords={"lat": grid.lat, "lon": grid.lon})
            box = xr.Dataset({read.obs.var: field})
        box = _at_grid_point(box, read.select)
        kept = implicit_keep(box, read.area, read.select) if keep else None
        return {name: np.asarray(box[name].values) for name in ("lat", "lon")}, dims, kept
    finally:
        if path is not None:
            ds.close()


def observation_grid(year, read: ObsRead):
    """Return the cells `load_observations` gives for the year, from the file's coordinates alone.

    Nothing of the rainfall is read: the region's box (or the one grid point) of the lazy dataset, which is where the
    loaded array's cells come from (the masks set cells to NaN and keep them). Data without a box has no such grid.

    Parameters
    ----------
    year : int
        The year to look at.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    dict of str to numpy.ndarray
        The ``lat`` and ``lon`` coordinate values.
    """
    return _observation_box(year, read)[0]


def reference_cells(year, read: ObsRead):
    """Find the cells to hold when the reference years can be held as only their kept cells.

    That is gridded (not unit) observations of a region's whole box (not one grid point) on the dims ``(time, lat,
    lon)`` in some order, where the land and country masks (not a ``mask`` section or ``nc_mask``) drop some cells and
    keep some.

    Parameters
    ----------
    year : int
        The year whose grid is looked at.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    tuple of (dict, tuple of str, numpy.ndarray of bool) or None
        The ``lat`` and ``lon`` of the box, its spatial dims in the file's order, and the boolean ``(lat, lon)`` array
        of the cells the implicit masks keep; None when the conditions above do not hold.
    """
    if read.obs.units is not None or read.select.grid_point or read.area.region is None:
        return None
    grid, dims, keep = _observation_box(year, read, keep=True)
    if keep is None or not keep.any() or set(dims) != {"time", "lat", "lon"} or len(dims) != 3:
        return None
    return grid, tuple(d for d in dims if d != "time"), keep


def _read_obs_year(year, store, path, read: ObsRead, cells=False):
    """Read one year of one variable: select the year and the region's box on the lazy array, read once, mask in place.

    A tuple of files (a year listed in several files) is read file by file and joined along time. With ``cells`` (gridded
    data whose implicit masks drop cells; see `reference_cells`) only the kept cells are read: the dims are then
    ``(time, cell)`` with the cells' lat and lon as coordinates over ``cell`` (see `hcwf.spatial.select.load_blocks`).

    Parameters
    ----------
    year : int
        The year to read.
    store : str or None
        The zarr store path, or None for a file.
    path : str or tuple of str or None
        The NetCDF file path (or paths), when ``store`` is None.
    read : ObsRead
        The typed inputs of the observation reading.
    cells : bool, default False
        Read only the cells the masks keep.

    Returns
    -------
    xarray.DataArray
        The year in mm (the unit factor applied), dims ``(time, lat, lon)`` in the file's order, or ``(time, unit)`` for
        observations on units (see `_read_obs_units`).
    """
    obs = read.obs
    if isinstance(path, tuple):
        parts = [_read_obs_year(year, store, one, read, cells) for one in path]
        return xr.concat(parts, dim="time").sortby("time")
    ds = _open_year(year, store, path, read)
    if obs.units is not None:
        try:
            return _read_obs_units(ds[obs.var], obs.units, obs.unit_cvt, read)
        finally:
            if path is not None:
                ds.close()
    da = region_read(ds, obs.var, area=read.area, select=read.select, source=OBSERVATIONS, read_by="time",
                     dataset_name=str(obs.name or "obs"), dir_fig=read.dir_fig, skip_masked=cells, compact=cells)
    da = in_memory(da)
    if path is not None:
        ds.close()
    return scale_in_place(da, obs.unit_cvt)


def _units_key(read: ObsRead) -> tuple:
    """Return what reading observations onto units adds to the key of a held year.

    Parameters
    ----------
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    tuple
        Empty for grid cells, else the settings that shape the unit series.
    """
    obs = read.obs
    if obs.units is None:
        return ()
    return (repr(obs.units), repr(units_files(obs.units)), read.options.missing,
            repr(sorted(read.select.keep_unit_ids or ())), repr(obs.dims_map()), str(obs.day_boundary),
            str(None if read.onset is None else read.onset.lead_label))


def _obs_key(year, store, path, read: ObsRead):
    """Return what identifies a loaded observation year: the source (with its size and time), the settings and the cells.

    Parameters
    ----------
    year : int
        The year.
    store : str or None
        The zarr store path, or None.
    path : str or tuple of str or None
        The file path (or paths), when ``store`` is None.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    tuple
        The key of the run's table of held years.
    """
    obs, area = read.obs, read.area
    source = store or path
    try:
        stats = [os.stat(p) for p in (source if isinstance(source, tuple) else (source,))]
        ident = tuple((stat.st_size, stat.st_mtime_ns) for stat in stats)
        ident = ident[0] if len(ident) == 1 else ident
    except OSError:
        ident = None
    land, shp = implicit_masks(read.select)
    return (year, str(source), ident, obs.var, obs.unit_cvt, str(area.region), str(area.box), land, shp, str(area.nc_mask),
            str(obs.name or "obs"), mask_text(area)) + _units_key(read)


def load_observations(year, obs, area, options, select=NO_SELECTION, *, onset=None, run_years=(), dir_fig=None):
    """Load observed daily rainfall for a given year (from the year's NetCDF file, or from a zarr store of all years).

    The region's box and, for a store, the year are selected before any data is read, and the cell mask and the
    unit conversion are applied in place. The years the run scores (``run_years``, and the year before each)
    are read once per run: later calls (other windows, cases, the climatology) get the same array back, so treat
    it as read-only. Other years are read afresh when asked for.

    With ``obs.paths`` (names or globs; ``select.obs_files_years`` keeps the files of those years) the year's files are
    the listed ones with that year in their name, instead of ``obs.file_pattern.format(year)`` in ``obs.dir``. A region
    of None, or data on a ``"unit"`` dimension, is read without a box or cell mask; observations on units need ``onset``
    (the day boundary and lead label their dates are moved by). `select` may pick one grid point and keep some units.

    Parameters
    ----------
    year : int
        The year to load.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    options : hcwf.config.model.OnsetOptions
        The onset options.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.
    onset : hcwf.config.onset.OnsetSetup, optional
        The onset setup; required for observations on units.
    run_years : iterable of int, default ()
        The years the run scores (they and the year before each are held).
    dir_fig : str or pathlib.Path, optional
        The folder for mask figures.

    Returns
    -------
    xarray.DataArray
        The year's daily rain in mm, dims ``(time, lat, lon)`` (``(time, unit)`` for units); read-only by convention.
    """
    return observation_year(year, ObsRead(obs, area, options, select, onset, tuple(run_years), dir_fig))


def observation_year(year, read: ObsRead):
    """Load the year of an `ObsRead`: the body of `load_observations`.

    Parameters
    ----------
    year : int
        The year to load.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    xarray.DataArray
        The year's daily rain (see `load_observations`).
    """
    select = read.select
    store, path = _locate(year, read)
    key = _obs_key(year, store, path, read)
    with _obs_lock:
        rainfall = _obs_years.get(key)
    if rainfall is None and store is not None and year in years_held(read.run_years) \
            and _long_time_chunks(store, read):
        # an archive chunked over many years: read all the years the run holds in one pass, not one decode per year
        for _ in obs_years_of(read, sorted(years_held(read.run_years))):
            pass
        with _obs_lock:
            rainfall = _obs_years.get(key)
    if rainfall is None:
        rainfall = _read_obs_year(year, store, path, read)
        if year in years_held(read.run_years):
            with _obs_lock:
                rainfall = _obs_years.setdefault(key, rainfall)

    return _at_grid_point(rainfall, select)


def _at_grid_point(data, select):
    """Return the data cut to the selected grid point, or as it is when the run has none.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data with ``lat`` and ``lon`` coordinates.
    select : hcwf.config.model.Selection
        The run's selection settings (``grid_point``, ``lat``, ``lon``).

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        ``data`` with the one-cell ``lat`` and ``lon`` of the point kept, or ``data`` itself.
    """
    return data.sel(lat=[select.lat], lon=[select.lon]) if select.grid_point else data


def observation_year_once(year, read: ObsRead):
    """Read one observation year for a single use: the held copy if there is one, else a read that is not kept.

    Parameters
    ----------
    year : int
        The year to read.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    xarray.DataArray
        The year's daily rain (see `load_observations`).
    """
    store, path = _locate(year, read)
    with _obs_lock:
        held = _obs_years.get(_obs_key(year, store, path, read))
    if held is None:
        held = _read_obs_year(year, store, path, read)
    return _at_grid_point(held, read.select)


def _read_obs_units(da, spec: UnitsSpec, unit_cvt, read: ObsRead):
    """Read one year of observations on the units, with dates moved onto the forecasts' labels.

    The coordinate ``present`` says which units have a series this year (a cell transform without a source, or a
    missing day under the ``"drop"`` policy, leave a unit out).

    Parameters
    ----------
    da : xarray.DataArray
        The year's observations as read from the file.
    spec : hcwf.spatial.units.UnitsSpec
        How the files turn into unit series.
    unit_cvt : float or str, optional
        The unit factor of the observations.
    read : ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    xarray.DataArray
        ``float64`` millimetres with dims ``(time, unit)``, a ``unit`` coordinate of string ids and a boolean
        ``present`` coordinate over ``unit``.

    Raises
    ------
    ValueError
        The data holds more than one year, the unit ids repeat, or ``read`` has no onset setup.
    """
    da = da.transpose("time", ...)
    stamps = pd.DatetimeIndex(da["time"].values)
    years = np.unique(stamps.year.to_numpy())
    if len(years) != 1:
        raise ValueError(f"observations on units are read a year per file; this one holds the years {years.tolist()}")
    units, positions = select_units(da, spec, read.select.keep_unit_ids)
    scale, policy = declared_scale(da, unit_cvt), read.options.missing or "flag"
    present = np.ones(len(positions), dtype=bool)
    if spec.how == "cell_transform" and "unit" not in da.dims:
        series, supported, _ = read_units(da, spec, units, positions, block_dim="time", scale=scale, missing=policy,
                                          support=True)
        present, policy = np.asarray(supported)[:, 0].copy(), "keep"  # the policy was applied to the source cells
    else:
        series = read_units(da, spec, units, positions, block_dim="time", scale=scale)
    ids = units.ids[positions]
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("points share the unit ids")
    values = np.asarray(series.transpose("time", "unit").values, dtype=float)
    if policy == "drop":
        present &= ~np.isnan(values).any(axis=0)
    if read.onset is None:
        raise ValueError("observations on units need the onset setup (the day boundary and lead label their dates move by)")
    shift = observation_shift_days(read.onset.obs_day_boundary, read.onset.lead_label)
    dates = (stamps.normalize() + pd.Timedelta(days=shift)).to_numpy()
    return xr.DataArray(values, dims=("time", "unit"), coords={"time": dates, "unit": ids.astype(str),
                                                               "present": ("unit", present)}, attrs={"units": "mm"})


def observed_history(year, current, onset, obs, area, options, select=NO_SELECTION, *, run_years=(), dir_fig=None):
    """Return the observed rain to stitch before each forecast of ``year``, or None when ``forecast_history`` is not observed.

    The previous year's file is added when a season or the rule's look-back reaches before 1 January; a missing
    previous file is skipped, which makes the earliest searches ``INSUFFICIENT_DATA``.

    Parameters
    ----------
    year : int
        The forecast year.
    current : xarray.DataArray
        The year's observations, already loaded.
    onset : hcwf.config.onset.OnsetSetup
        The onset setup.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    options : hcwf.config.model.OnsetOptions
        The onset options.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.
    run_years : iterable of int, default ()
        The years the run scores.
    dir_fig : str or pathlib.Path, optional
        The folder for mask figures.

    Returns
    -------
    xarray.DataArray or None
        None when ``onset.forecast_history`` is not ``"observed"``; ``current`` when no earlier year is needed or
        available; else the previous year and ``current`` joined along ``time``.
    """
    if onset.forecast_history != "observed":
        return None
    if not history_needs_previous_year(onset):
        return current
    try:
        previous = load_observations(year - 1, obs, area, options, select, onset=onset, run_years=run_years,
                                     dir_fig=dir_fig)
    except FileNotFoundError:
        return current
    return xr.concat([previous, current], dim="time")


def observations_of(year, rc, select=NO_SELECTION):
    """Call `load_observations` with the settings of a run.

    Parameters
    ----------
    year : int
        The year to load.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.

    Returns
    -------
    xarray.DataArray
        The year's daily rain (see `load_observations`).
    """
    return load_observations(year, rc.observations, rc.area, rc.onset_options, select,
                             onset=rc.onset if rc.observations.units is not None else None,
                             run_years=rc.evaluation.years, dir_fig=rc.output.figure_dir)


def observation_years_of(years, rc, select=NO_SELECTION, hold=True):
    """Call `read_obs_years` with the settings of a run.

    Parameters
    ----------
    years : iterable of int
        The years to read.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.
    hold : bool, default True
        Keep the held years.

    Returns
    -------
    collections.abc.Iterator
        The iterator of `read_obs_years`.
    """
    return read_obs_years(years, rc.observations, rc.area, rc.onset_options, select,
                          onset=rc.onset if rc.observations.units is not None else None,
                          run_years=rc.evaluation.years, dir_fig=rc.output.figure_dir, parallel=rc.parallel, hold=hold)


def observation_grid_of(year, rc, select=NO_SELECTION):
    """Call `observation_grid` with the settings of a run.

    Parameters
    ----------
    year : int
        The year to look at.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.

    Returns
    -------
    dict of str to numpy.ndarray
        The ``lat`` and ``lon`` coordinate values.
    """
    return observation_grid(year, ObsRead(rc.observations, rc.area, rc.onset_options, select,
                                          rc.onset if rc.observations.units is not None else None,
                                          tuple(rc.evaluation.years), rc.output.figure_dir, rc.parallel))


def history_of(year, current, rc, select=NO_SELECTION):
    """Call `observed_history` with the settings of a run.

    Parameters
    ----------
    year : int
        The forecast year.
    current : xarray.DataArray
        The year's observations, already loaded.
    rc : hcwf.config.model.RunConfig
        The run configuration.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.

    Returns
    -------
    xarray.DataArray or None
        The history (see `observed_history`).
    """
    return observed_history(year, current, rc.onset, rc.observations, rc.area, rc.onset_options, select,
                            run_years=rc.evaluation.years, dir_fig=rc.output.figure_dir)
