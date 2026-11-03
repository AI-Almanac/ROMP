"""The reference (threshold and climatology) years of the observations, ingested once as one DailyRain.

The thresholds read and ingest the reference years (`reference_rain`) and leave them held for the climatological onsets of those
years (`reference_rain_year`) until `release_reference` (or the run's state is reset).

Where the region's land and country masks drop cells, the years are read and held as only the cells the masks keep (a
DailyRain with `cells`, dims "cell" and "time"), not the whole box: the thresholds and the harmonic seasons are computed on
those cells and come out on the whole grid (hcwf.onset.climatology), and a year handed to the climatological onsets is put on
the whole grid for that one year (`DailyRain.on_grid`), with the dropped cells as ingestion gives them, so that the onsets
are those of the whole box.
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
the one item "held" is (key, rain, first year, last year)."""


def _reference_key(obs, area, select, boundary):
    """What the shared reference rain depends on: the observations as read and cut, and the day boundary they are ingested with."""
    land, shp = implicit_masks(select)
    return (str(obs.dir), str(None if obs.file_pattern is None else list(obs.file_pattern)), str(obs.var),
            str(obs.unit_cvt), str(area.region), land, shp, str(area.nc_mask), str(obs.name or "obs"),
            mask_text(area), bool(select.grid_point), str(select.lat), str(select.lon), str(boundary))


def release_reference():
    """Drop the shared reference rainfall (its memory is the size of the reference years' tenths and flags)."""
    _reference.clear()


def reference_held() -> bool:
    """Whether the thresholds left their reference rainfall for the climatology to use (see reference_rain_year)."""
    return "held" in _reference


def reference_rain_year(year, obs, area, select, boundary):
    """One year of the shared reference rain as a DailyRain (views of it, nothing copied), or None when it is not held.

    The thresholds read and ingest the reference years once (see _stream_reference); the climatological onsets of
    those years use that same ingested rain instead of reading and converting every year again. The year's rain is
    exactly what ingest gives for the year read on its own, once it is put on the whole grid with `on_grid` when the
    reference years are held as only their kept cells (`cells` is then set on it).
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
    """

    def __init__(self, probe, load):
        """Take dims, coords and source from the probe (one year); `load` returns the full DailyRain."""
        self.dims, self.coords, self.source, self.resolution = probe.dims, probe.coords, probe.source, probe.resolution
        self._load, self._full = load, None

    def __getattr__(self, name):
        """Load the full reference rainfall on first use and read the attribute from it."""
        if name.startswith("_"):
            raise AttributeError(name)
        if self._full is None:
            self._full = self._load()
        return getattr(self._full, name)


def _obs_source_label(read, onset):
    """The text that labels the reference observations as a data source (dataset, folder, pattern, variable, unit
    factor, and for units what turns the files into series)."""
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
    """What reference_cells gives for the year, or None when it cannot be looked at (the whole box is then read, and the read
    reports what is wrong, a missing year or an unknown region, as it always did)."""
    try:
        return reference_cells(year, read)
    except Exception as exc:  # noqa: BLE001 - only a saving of memory: the ordinary read follows
        log.debug(f"reference cells not looked at: {exc}")
        return None


def _on_kept_cells(rain, rows, cols, flat):
    """The year's observations on only the kept cells (dims time, "cell"): as they were read that way, else picked from the
    whole box."""
    if "cell" in rain.dims:
        return rain
    return rain.isel(lat=xr.DataArray(rows, dims="cell"), lon=xr.DataArray(cols, dims="cell")).assign_coords(cell=flat)


def _cells_of_reference(one, grid, dims, flat, source, options, onset):
    """The Cells of reference years ingested as `one` (a year on the kept cells): what the masked cells are as ingestion
    gives them, found by ingesting one of them."""
    from hcwf.onset.model import Cells

    time = xr.DataArray(np.full((one.coords["time"].size, 1), np.nan), dims=("time", "cell"),
                        coords={"time": one.coords["time"], "cell": [0]})
    masked = ingest(time, day_boundary=onset.obs_day_boundary, units="mm", source=source, **options)
    return Cells(flat=flat, lat=grid["lat"], lon=grid["lon"], dims=dims, fill_tenths=masked.tenths[0, 0].item(),
                 fill_missing=bool(masked.missing[0, 0]))


def _stream_reference(onset, years, read, source):
    """Ingest the reference years one at a time into one preallocated DailyRain (peak memory: the result plus one year).

    Gives what ingest of the concatenated years gives, for standard-calendar years that follow each other
    day by day on one grid; raises _NotContiguous otherwise. Years the run also evaluates stay in the
    observation cache; the others are read, ingested and dropped. Where the masks drop cells, only the kept cells are
    read and held (see the module text): the result has `cells`.
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
    """One reference year's rainfall: held for the run if the run evaluates that year, else read and not kept."""
    keep = years_held(read.run_years) if keep is None else keep
    rain = observation_year(year, read) if year in keep else observation_year_once(year, read)
    return rain.drop_vars("present") if "present" in rain.coords else rain  # which units have a series, not rain


def reference_rain(onset, years, read):
    """The observations of years[0]..years[1] as ingested DailyRain, labelled with their source.

    Years are read and ingested one at a time into one array rather than concatenated as rainfall first.
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
