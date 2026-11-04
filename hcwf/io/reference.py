"""The reference (threshold and climatology) years of the observations, ingested once as one DailyRain.

The thresholds (`hcwf.io.thresholds`) read and ingest the reference years (`reference_rain`) and leave them held for
the climatological onsets of those years (`reference_rain_year`) until `release_reference` (or the run's state is
reset). The years are read with the observation readers of `hcwf.io.observations`.

Where the region's land and country masks drop cells, the years are read and held as only the cells the masks keep (a
`hcwf.onset.model.DailyRain` with ``cells``, dims ``"cell"`` and ``"time"``), not the whole box: the thresholds and the
harmonic seasons are computed on those cells and come out on the whole grid (`hcwf.onset.climatology`), and a year
handed to the climatological onsets is put on the whole grid for that one year (``DailyRain.on_grid``), with the
dropped cells as ingestion gives them, so that the onsets are those of the whole box.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import xarray as xr

from hcwf.cache import memo
from hcwf.io.observations import (
    mask_text,
    obs_years_of,
    observation_year,
    observation_year_once,
    reference_cells,
    years_held,
)
from hcwf.onset.ingest import ingest
from hcwf.spatial.select import implicit_masks

log = logging.getLogger(__name__)

_reference = memo("io.reference.held", maxsize=1)
"""The reference (threshold and climatology) years as one ingested DailyRain, kept from the thresholds to the climatology:
the one item ``"held"`` is ``(key, rain, first year, last year)``."""


def _reference_key(obs, area, select, boundary):
    """Return what the shared reference rain depends on: the observations as read and cut, and the day boundary.

    Parameters
    ----------
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    select : hcwf.config.model.Selection
        The run's selection settings.
    boundary : str
        The day boundary the observations are ingested with.

    Returns
    -------
    tuple of str
        The settings as strings (and the land and shape-file mask flags), comparable for equality.
    """
    land, shp = implicit_masks(select)
    return (str(obs.dir), str(None if obs.file_pattern is None else list(obs.file_pattern)), str(obs.var),
            str(obs.unit_cvt), str(area.region), land, shp, str(area.nc_mask), str(obs.name or "obs"),
            mask_text(area), bool(select.grid_point), str(select.lat), str(select.lon), str(boundary))


def release_reference():
    """Drop the shared reference rainfall.

    Its memory is the size of the reference years' tenths and flags. Nothing happens when none is held.
    """
    _reference.clear()


def reference_held() -> bool:
    """Tell whether the thresholds left their reference rainfall for the climatology to use.

    Returns
    -------
    bool
        True when a reference rainfall is held (see `reference_rain_year`).
    """
    return "held" in _reference


def reference_rain_year(year, obs, area, select, boundary):
    """Return one year of the shared reference rain as a DailyRain (views of it, nothing copied), or None.

    The thresholds read and ingest the reference years once (see `_stream_reference`); the climatological onsets of
    those years use that same ingested rain instead of reading and converting every year again. The year's rain is
    exactly what ingest gives for the year read on its own, once it is put on the whole grid with ``on_grid`` when the
    reference years are held as only their kept cells (``cells`` is then set on it).

    Parameters
    ----------
    year : int
        The year to slice out.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    select : hcwf.config.model.Selection
        The run's selection settings.
    boundary : str
        The day boundary the observations are ingested with.

    Returns
    -------
    hcwf.onset.model.DailyRain or None
        The year's rain (tenths of mm, ``(..., day)``), or None when nothing is held, the year is outside the held
        years, or the held rain was made for other settings.
    """
    held = _reference.get("held")
    if held is None:
        return None
    key, rain, first, last = held
    if not first <= year <= last or key != _reference_key(obs, area, select, boundary):
        return None
    from hcwf.onset.model import DailyRain

    start = (date(year, 1, 1) - date(first, 1, 1)).days
    days = slice(start, start + (date(year, 12, 31) - date(year, 1, 1)).days + 1)
    return DailyRain(tenths=rain.tenths[..., days], missing=rain.missing[..., days], dims=rain.dims,
                     coords={**rain.coords, "time": rain.coords["time"][days]}, year=rain.year[days],
                     month=rain.month[days], day=rain.day[days], calendar=rain.calendar,
                     day_boundary=rain.day_boundary, source="observations", cells=rain.cells)


class LazyRain:
    """A DailyRain whose data is read only when something other than its grid, source and resolution is asked for.

    The threshold cache keys on those, so a cache hit never loads the reference years.

    Parameters
    ----------
    probe : hcwf.onset.model.DailyRain
        One year of the reference rain, which gives ``dims``, ``coords``, ``source`` and ``resolution``.
    load : callable
        Called with no arguments, returns the full `hcwf.onset.model.DailyRain`; called once, on first use.

    Attributes
    ----------
    dims : tuple of str
        The dimension names of the probe.
    coords : dict
        The coordinates of the probe.
    source : str
        The label of the data source.
    resolution : float
        The grid resolution.
    """

    def __init__(self, probe, load):
        self.dims, self.coords, self.source, self.resolution = probe.dims, probe.coords, probe.source, probe.resolution
        self._load, self._full = load, None

    def __getattr__(self, name):
        """Load the full reference rainfall on first use and read the attribute from it.

        Parameters
        ----------
        name : str
            The attribute asked for.

        Returns
        -------
        object
            The attribute of the full `hcwf.onset.model.DailyRain`.

        Raises
        ------
        AttributeError
            `name` starts with an underscore (private names are never forwarded).
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if self._full is None:
            self._full = self._load()
        return getattr(self._full, name)


def _obs_source_label(read, onset):
    """Return the text that labels the reference observations as a data source.

    The label holds the dataset, folder, pattern, variable, unit factor, and for units what turns the files into series.

    Parameters
    ----------
    read : hcwf.io.observations.ObsRead
        The typed inputs of the observation reading.
    onset : hcwf.config.onset.OnsetSetup
        The onset setup (its day boundary and lead label enter the label for units).

    Returns
    -------
    str
        The label.
    """
    obs = read.obs
    label = (f"{obs.name}:{obs.dir}:{None if obs.file_pattern is None else list(obs.file_pattern)}:"
             f"{obs.var}:cvt={obs.unit_cvt}")
    if obs.units is not None:
        label += (f":units={obs.units!r}:options={sorted(read.options.ingest(False, obs.units).items())}:"
                  f"dims={obs.dims_map()}:labels={onset.obs_day_boundary}/{onset.lead_label}")
    return label


class _NotContiguous(Exception):
    """The reference years do not follow each other day by day, so they cannot be merged year by year."""


def _kept_cells(read, year):
    """Return what `hcwf.io.observations.reference_cells` gives for the year, or None when it cannot be looked at.

    The whole box is then read, and the read reports what is wrong, a missing year or an unknown region, as it always
    did.

    Parameters
    ----------
    read : hcwf.io.observations.ObsRead
        The typed inputs of the observation reading.
    year : int
        The year whose grid is looked at.

    Returns
    -------
    tuple or None
        The result of `hcwf.io.observations.reference_cells`, or None when it raised.
    """
    try:
        return reference_cells(year, read)
    except Exception as exc:  # noqa: BLE001 - only a saving of memory: the ordinary read follows
        log.debug(f"reference cells not looked at: {exc}")
        return None


def _on_kept_cells(rain, rows, cols, flat):
    """Return the year's observations on only the kept cells (dims ``time``, ``cell``).

    Parameters
    ----------
    rain : xarray.DataArray
        The year's observations, either read on the kept cells already (a ``cell`` dimension) or on the whole box.
    rows : numpy.ndarray of int64, shape (cell,)
        The latitude index of each kept cell.
    cols : numpy.ndarray of int64, shape (cell,)
        The longitude index of each kept cell.
    flat : numpy.ndarray of int64, shape (cell,)
        The flat index of each kept cell, the ``cell`` coordinate.

    Returns
    -------
    xarray.DataArray
        `rain` as it was if it has a ``cell`` dimension, else the kept cells picked from the whole box.
    """
    if "cell" in rain.dims:
        return rain
    return rain.isel(lat=xr.DataArray(rows, dims="cell"), lon=xr.DataArray(cols, dims="cell")).assign_coords(cell=flat)


def _cells_of_reference(one, grid, dims, flat, source, options, onset):
    """Return the `hcwf.onset.model.Cells` of reference years ingested as `one` (a year on the kept cells).

    What the masked cells are as ingestion gives them is found by ingesting one all-``NaN`` cell.

    Parameters
    ----------
    one : hcwf.onset.model.DailyRain
        One ingested year on the kept cells.
    grid : dict
        The whole grid's ``lat`` and ``lon`` coordinates.
    dims : tuple of str
        The spatial dimension names of the whole grid.
    flat : numpy.ndarray of int64, shape (cell,)
        The flat index of each kept cell.
    source : str
        The label of the data source.
    options : dict
        The ingestion options (`hcwf.config.model.OnsetOptions.ingest`).
    onset : hcwf.config.onset.OnsetSetup
        The onset setup (the observation day boundary).

    Returns
    -------
    hcwf.onset.model.Cells
        The kept cells with the tenths and missing flag a masked cell gets.
    """
    from hcwf.onset.model import Cells

    time = xr.DataArray(np.full((one.coords["time"].size, 1), np.nan), dims=("time", "cell"),
                        coords={"time": one.coords["time"], "cell": [0]})
    masked = ingest(time, day_boundary=onset.obs_day_boundary, units="mm", source=source, **options)
    return Cells(flat=flat, lat=grid["lat"], lon=grid["lon"], dims=dims, fill_tenths=masked.tenths[0, 0].item(),
                 fill_missing=bool(masked.missing[0, 0]))


def _stream_reference(onset, years, read, source):
    """Ingest the reference years one at a time into one preallocated DailyRain (peak memory: the result plus one year).

    Gives what ingest of the concatenated years gives, for standard-calendar years that follow each other
    day by day on one grid. Years the run also evaluates stay in the observation cache; the others are read,
    ingested and dropped. Where the masks drop cells, only the kept cells are read and held (see the module text): the
    result has ``cells``.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup (the observation day boundary).
    years : tuple of (int, int)
        The first and last reference year.
    read : hcwf.io.observations.ObsRead
        The typed inputs of the observation reading.
    source : str
        The label of the data source.

    Returns
    -------
    hcwf.onset.model.DailyRain
        The years as one ingested rainfall, tenths of mm over all days of `years`.

    Raises
    ------
    _NotContiguous
        The years do not follow each other day by day, use a non-standard calendar or differ in grid.
    error
        An error that reading a year returned (for example a missing file).
    """
    from hcwf.onset.model import DailyRain

    first = date(years[0], 1, 1)
    total = (date(years[1], 12, 31) - first).days + 1
    out, offset = None, 0
    options = read.options.ingest(False, read.obs.units)
    kept = _kept_cells(read, years[0])
    if kept is not None:
        grid, spatial, keep = kept
        flat = np.flatnonzero(keep.reshape(-1))
        rows, cols = np.divmod(flat, keep.shape[1])
    cells = None
    for year, (rain, error) in zip(range(years[0], years[1] + 1),
                                   obs_years_of(read, range(years[0], years[1] + 1), cells=kept is not None)):
        if error is not None:
            raise error
        if kept is not None:
            rain = _on_kept_cells(rain, rows, cols, flat)
        one = ingest(rain, day_boundary=onset.obs_day_boundary, units="mm", source=source, **options)
        del rain
        if kept is not None:
            if "cell" not in one.dims or not np.array_equal(one.coords["cell"], flat):
                raise _NotContiguous
            cells = cells or _cells_of_reference(one, grid, spatial, flat, source, options, onset)
        if one.calendar != "standard" or one.tenths.shape[-1] != (date(year, 12, 31) - date(year, 1, 1)).days + 1 \
                or (date(year, 1, 1) - first).days != offset:
            raise _NotContiguous
        if out is None:
            shape = one.tenths.shape[:-1] + (total,)
            out = DailyRain(tenths=np.empty(shape, dtype=one.tenths.dtype), missing=np.empty(shape, dtype=bool),
                            dims=one.dims, coords={**one.coords, "time": np.empty(total, dtype="datetime64[ns]")},
                            year=np.empty(total, dtype=np.int64), month=np.empty(total, dtype=np.int64),
                            day=np.empty(total, dtype=np.int64), calendar=one.calendar,
                            day_boundary=one.day_boundary, source=source, cells=cells)
        elif one.dims != out.dims or any(not np.array_equal(one.coords[d], out.coords[d]) for d in one.dims[:-1]):
            raise _NotContiguous
        n = one.tenths.shape[-1]
        out.tenths[..., offset:offset + n] = one.tenths
        out.missing[..., offset:offset + n] = one.missing
        for name in ("year", "month", "day"):
            getattr(out, name)[offset:offset + n] = getattr(one, name)
        out.coords["time"][offset:offset + n] = one.coords["time"]
        offset += n
    return out


def _reference_year(year, read, keep=None):
    """Return one reference year's rainfall: held for the run if the run evaluates that year, else read and not kept.

    Parameters
    ----------
    year : int
        The year to read.
    read : hcwf.io.observations.ObsRead
        The typed inputs of the observation reading.
    keep : collection of int, optional
        The years held by the run's observation cache; None takes them from `read`.

    Returns
    -------
    xarray.DataArray
        The year's observations, without the ``present`` coordinate (which units have a series).
    """
    keep = years_held(read.run_years) if keep is None else keep
    rain = observation_year(year, read) if year in keep else observation_year_once(year, read)
    return rain.drop_vars("present") if "present" in rain.coords else rain  # which units have a series, not rain


def reference_rain(onset, years, read):
    """Return the observations of ``years[0]`` to ``years[1]`` as an ingested DailyRain, labelled with their source.

    Years are read and ingested one at a time into one array rather than concatenated as rainfall first, and the
    result is held for the climatological onsets (see `reference_rain_year`).

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup (the observation day boundary).
    years : tuple of (int, int)
        The first and last reference year, inclusive.
    read : hcwf.io.observations.ObsRead
        The typed inputs of the observation reading.

    Returns
    -------
    hcwf.onset.model.DailyRain
        The ingested rainfall in tenths of mm.
    """
    source = _obs_source_label(read, onset)
    if years[0] != years[1]:
        try:
            rain = _stream_reference(onset, years, read, source)
        except _NotContiguous:
            pass
        else:
            _reference["held"] = (_reference_key(read.obs, read.area, read.select, onset.obs_day_boundary), rain,
                                  years[0], years[1])
            return rain
    return ingest(
        xr.concat([_reference_year(year, read) for year in range(years[0], years[1] + 1)], dim="time"),
        day_boundary=onset.obs_day_boundary, units="mm", source=source, **read.options.ingest(False, read.obs.units))
