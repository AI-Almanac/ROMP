"""Climatological onset dates from observed rainfall, and climatology used as a baseline forecast.

The functions take the typed inputs of `hcwf.config.model` (the run ``rc``, the ``case`` and the per-call `select`).
`climatological_onset` and `climatological_onset_dataset` compute the climatology from the observed onsets of the
climatology years (`hcwf.onset.detect.detect_observed`), and `compute_climatology_as_forecast` turns it into a baseline
forecast table.
"""

from types import SimpleNamespace

from hcwf.config.model import Selection
from hcwf.io.observations import observation_grid_of, observation_years_of
from hcwf.cache import memo
from hcwf.io.reference import reference_held, reference_rain_year, release_reference
from hcwf.io.thresholds import thresholds_of
from hcwf.metrics.bins import season_relative_days
from hcwf.onset.detect import detect_observed
from hcwf.onset.climatology import ReferenceOnset, _doy_index, reduce_onset_days
from hcwf.spatial import axes

import numpy as np
import xarray as xr
import pandas as pd
import logging

__all__ = [
    "climatological_onset", "climatological_onset_dataset", "compute_climatology_as_forecast", "season_day_dates",
]

log = logging.getLogger(__name__)

_MEMO = memo("references.climatology.onsets", maxsize=32)
"""Climatological onset datasets of this run: ``{key: ((thresholds, onset setup) held so their ids stay unique, dataset)}``."""


class _YearRain:
    """One climatology year's observations, read when the object is called (``detect_observed`` does so on a cache miss).

    `probe` stands in for the rain where only its grid is needed (the observed-onset cache's fingerprint): the cells
    of the year from the file's coordinates, or the held reference rain's own, without reading any rainfall.

    Parameters
    ----------
    year : int
        The climatology year.
    rc : hcwf.config.model.RunConfig
        The run (its observations and area locate the year's files).
    select : hcwf.config.model.Selection
        The call's choices (cells, units).
    read : callable
        ``read(year)`` returns the year's rainfall or raises its read error.
    held : object or None
        The already ingested reference rain of the year (with ``on_grid``, ``coords`` and ``cells``), or None to read it.
    """

    def __init__(self, year, rc, select, read, held):
        self.year, self._rc, self._select, self._read, self._held = year, rc, select, read, held

    def __call__(self):
        """Return the year's rainfall: the held reference rain on the grid, or what `read` returns.

        A read error of the year propagates from `read`.

        Returns
        -------
        object
            The rainfall of the year (what `detect_observed` takes as its rain).

        """
        return self._held.on_grid() if self._held is not None else self._read(self.year)

    @property
    def probe(self):
        """A stand-in for the rain with only ``coords`` (the year's grid; empty when it cannot be found)."""
        try:
            held = self._held
            coords = (held.coords if held.cells is None else held.cells.coords()) if held is not None \
                else observation_grid_of(self.year, self._rc, self._select)
        except Exception:  # noqa: BLE001 - without a grid the cache is skipped and the year is read
            coords = {}
        return SimpleNamespace(coords=coords)


def _year_readers(years, rc, select, shared=True):
    """Make a `_YearRain` for each year: its observations are read only when called, on a cache miss of ``detect_observed``.

    Calling it returns the rain (`hcwf.io.observations.observation_years_of`: read in blocks or ahead) or raises the
    year's read error. A year the thresholds were just computed from comes as the ``DailyRain`` of that already read and
    ingested reference rainfall (`hcwf.io.reference.reference_rain_year`) instead of being read and converted again.
    ``shared=False`` reads every year as rainfall. The years are read through one generator while they are asked for in
    order (so blocks and read-ahead work as before); a year asked for after skipped ones (cached onsets, never read)
    restarts it there.

    Parameters
    ----------
    years : iterable of int
        The climatology years, in the order they will be asked for.
    rc : hcwf.config.model.RunConfig
        The run (observations, area, onset definition).
    select : hcwf.config.model.Selection
        The call's choices.
    shared : bool, default True
        Use the held reference rain for the years the thresholds were computed from, when there is one.

    Returns
    -------
    list of _YearRain
        One reader per year, in the order of ``years``.
    """
    years = list(years)
    held = {year: None for year in years}
    if shared and reference_held():
        held = {year: reference_rain_year(year, rc.observations, rc.area, select, rc.onset.obs_day_boundary)
                for year in years}
    rest = [year for year in years if held[year] is None]
    feed = {"years": None, "next": None}   # the generator over `rest` and the position of the year it yields next

    def read(year):
        """Read one year through the shared generator, restarting it when the year is not the next one."""
        position = rest.index(year)
        if feed["next"] != position:
            if feed["years"] is not None:
                feed["years"].close()
            feed["years"] = observation_years_of(rest[position:], rc, select)
        feed["next"] = position + 1
        rain, error = next(feed["years"])
        if error is not None:
            raise error
        return rain

    return [_YearRain(year, rc, select, read, held[year]) for year in years]


def climatological_onset(rc, case, select=Selection()):
    """Compute climatological onset dates from all available IMD files.

    The onset of each climatology year is detected, converted to a day of the season's 365-day year
    (`hcwf.metrics.bins.season_relative_days`: 29 February shares 1 March, an onset in the year after the season's
    year counts past 365) and the years are averaged (`hcwf.onset.climatology.reduce_onset_days`, rounded to a whole
    day), so a season crossing the year boundary has a mean beyond 365. In grid-point mode the mean daily rainfall by
    day of year (leap day left out) is returned as well.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run (its climatology years, observations, onset definition and thresholds).
    case : hcwf.config.model.Case
        The case whose model the thresholds are loaded for.
    select : hcwf.config.model.Selection, default Selection()
        The call's choices; ``select.grid_point`` (with ``lat`` and ``lon``) selects one grid point and also returns the
        climatological rainfall.

    Returns
    -------
    xarray.DataArray or tuple of (xarray.DataArray, xarray.DataArray)
        The climatological onset day of the 365-day year (``float64``, 366 and over for the next year; over ``lat``, ``lon`` or ``unit``). In grid-point mode a
        tuple of that and the climatological mean rainfall by ``doy`` (1-365).

    Raises
    ------
    ValueError
        The thresholds are not scalar in grid-point mode, or no year gave an onset.
    Exception
        Whatever the detection or reading of a year raised, with a note naming the year.
    """
    years_clim = rc.evaluation.climatology_years
    grid_point = bool(select.grid_point)
    thresh_da = thresholds_of(rc, case.model, select)

    log.debug(f"Computing climatological onset from {len(years_clim)} years_clim: {min(years_clim)}-{max(years_clim)}")

    all_onset_days = []
    rainfall_all_years = []

    for year, reader in zip(years_clim, _year_readers(years_clim, rc, select, shared=not grid_point)):
        try:
            if grid_point:
                if np.ndim(thresh_da) > 0:
                    raise ValueError("thresh_da must be scalar.")

                rainfall_ds = reader()
                rainfall_all_years.append(rainfall_ds)
            else:
                rainfall_ds = reader   # read only when the year's onsets are not cached

            # Detect onset for this year
            onset_da = detect_observed(rainfall_ds, thresh_da, year, rc, select)

            # Convert onset dates to the day of the season's 365-day year (past 365 for an onset in the next year)
            onset_doy = xr.DataArray(season_relative_days(onset_da.values, year) + 1.0, dims=onset_da.dims,
                                     coords={d: onset_da.coords[d] for d in onset_da.dims if d in onset_da.coords},
                                     name="dayofyear")

            all_onset_days.append(onset_doy)

        except Exception as e:   # a mean over fewer years would be a different climatology: stop, naming the year
            e.add_note(f"climatological onset, year {year}")
            raise

    if not all_onset_days:
        raise ValueError("No valid years found for climatology computation")

    release_reference()   # the thresholds' reference rainfall has served; free its memory

    # Stack all years and compute mean day of year
    onset_stack = xr.concat(all_onset_days, dim='year')
    climatological_onset_doy = reduce_onset_days(onset_stack, "mean")   # mean day of year, rounded to a whole day

    log.debug(f"Climatological onset computed from {len(all_onset_days)} valid years")

    if grid_point:
        rainfall_stack = xr.concat(rainfall_all_years, dim='year')

        # 29 February is left out and every other date takes its day of the 365-day year (1 March is day 60 every year)
        doy = _doy_index(rainfall_stack['time'].dt.month.values, rainfall_stack['time'].dt.day.values) + 1
        rainfall_stack = rainfall_stack.isel(time=np.flatnonzero(doy > 0))
        rainfall_stack = rainfall_stack.assign_coords(doy=('time', doy[doy > 0]))

        # Aggregate over years using groupby
        # explicitly reduce the 'time' dimension by taking mean
        rainfall_clim_mean = (
            rainfall_stack.groupby('doy')
            .mean(dim='time')  # <- reduce the concatenated time dimension
            .mean(dim='year')  # <- average across years
        )

        return climatological_onset_doy, rainfall_clim_mean

    return climatological_onset_doy


def season_day_dates(years, days) -> np.ndarray:
    """Place days of the season's 365-day year (as `climatological_onset` gives them) in given years.

    Day 1 is 1 January and 1 March is day 60 every year, so in a leap year the dates from day 60 on are one day later
    than the plain day of year; a day past 365 falls in the next year.

    Parameters
    ----------
    years : int or numpy.ndarray of int
        The season year of each day.
    days : float or numpy.ndarray of float
        The days (1-based, whole numbers), broadcast against ``years``.

    Returns
    -------
    numpy.ndarray of datetime64[D]
        The dates.
    """
    years = np.asarray(years, dtype=np.int64)
    day_index = np.asarray(days).astype(np.int64) - 1
    start = (years - 1970).astype("datetime64[Y]")
    leap = np.asarray(pd.DatetimeIndex(np.atleast_1d(start).ravel()).is_leap_year).reshape(start.shape) & (day_index >= 59)
    return start.astype("datetime64[D]") + (day_index + leap).astype("timedelta64[D]")


def compute_climatology_as_forecast(climatological_onset_doy, init_dates, observed_onset_da,
                                   *, max_forecast_day):
    """Use climatology as a forecast model for the given initialization dates.

    Only processes forecasts initialized before the observed onset date.

    Parameters
    ----------
    climatological_onset_doy : xarray.DataArray or hcwf.onset.climatology.ReferenceOnset
        The climatological onset day of year, or a ``ReferenceOnset`` (its day-of-year field on the grid of
        ``observed_onset_da``).
    init_dates : pandas.DatetimeIndex
        The initialization dates.
    observed_onset_da : xarray.DataArray of datetime64
        The observed onset dates, used for filtering and for the grid.
    max_forecast_day : int
        The maximum forecast day to consider: the lead window in days after the init.

    Returns
    -------
    pandas.DataFrame
        The climatology forecast results, one row per (init, lat, lon) (or (init, unit)) in that order for cells with an
        observed onset and an init before it; the climatological onset falls in the year of its init (the next year past day 365). Empty when no
        cell qualifies.
    """
    if isinstance(climatological_onset_doy, ReferenceOnset):
        climatological_onset_doy = climatological_onset_doy.doy_field(observed_onset_da)
    spatial = axes.spatial_dims(observed_onset_da.dims)
    if "unit" in spatial:   # units are matched by id
        climatological_onset_doy = climatological_onset_doy.reindex(unit=observed_onset_da["unit"].values)
    coords = axes.coords_of(climatological_onset_doy, spatial)
    n_cells = int(np.prod([len(coords[d]) for d in spatial]))

    log.debug(f"Processing climatology as forecast for {len(init_dates)} init times x {n_cells} {'/'.join(spatial)} points...")

    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    init_ns = inits.values.astype("datetime64[ns]")
    obs = observed_onset_da.transpose(*spatial).values.astype("datetime64[ns]")
    no_obs = np.isnat(obs)
    total_potential_inits = len(inits) * obs.size
    skipped_no_obs = len(inits) * int(no_obs.sum())

    late = axes.broadcast_rows(init_ns, obs.ndim + 1) >= obs[None]
    skipped_late_init = int((~no_obs[None] & late).sum())
    valid = ~no_obs[None] & ~late
    i, *where = np.nonzero(valid)
    where = tuple(where)
    valid_inits = len(i)

    doy = climatological_onset_doy.transpose(*spatial).values.astype(float)[where]
    no_clim = np.isnan(doy)
    clim_date = season_day_dates(inits.year.to_numpy()[i], np.where(no_clim, 1, doy))
    init_at = init_ns[i]
    clim_at = clim_date.astype("datetime64[ns]")
    in_window = (~no_clim & (init_at + np.timedelta64(1, "D") <= clim_at)
                 & (clim_at <= init_at + np.timedelta64(int(max_forecast_day), "D")))
    onsets_forecasted = int(in_window.sum())

    if valid_inits:
        onset_day = (clim_at - init_at) // np.timedelta64(1, "D")
        climatology_forecast_df = pd.DataFrame({
            'init_time': [init_dates[n] for n in i],
            **axes.columns_of(spatial, coords, where),
            'onset_day': onset_day if in_window.all() else np.where(in_window, onset_day, np.nan),
            'onset_date': np.where(in_window, axes.date_strings(clim_date), None),
            'climatological_onset_doy': doy,
            'climatological_onset_date': np.where(no_clim, None, axes.date_strings(clim_date)),
            'obs_onset_date': axes.date_strings(obs[where]),
        })
    else:
        climatology_forecast_df = pd.DataFrame([])

    log.debug("\nClimatology Forecast Summary:")
    log.debug(f"Total potential initializations: {total_potential_inits}")
    log.debug(f"Skipped (no observed onset): {skipped_no_obs}")
    log.debug(f"Skipped (initialized after observed onset): {skipped_late_init}")
    log.debug(f"Valid initializations processed: {valid_inits}")
    log.debug(f"Onsets forecasted: {onsets_forecasted}")
    log.debug(f"Forecast rate: {onsets_forecasted/valid_inits:.3f}" if valid_inits > 0 else "Forecast rate: 0.000")

    return climatology_forecast_df

# def compute_climatology_metrics_with_windows same as stats.benchmark.compute_onset_metrics_with_windows
# def compute_climatology_baseline_multiple_years same as stats.benchmark.compute_metrics_multiple_years


###=========  for bin climatology ============

## This function computes onset dates for all available years in IMD folder and creates a climatological onset dataset
def climatological_onset_dataset(rc, case, select=Selection()):
    """Compute onset dates for all climatology years and create a climatological dataset.

    The result is kept for the rest of the run and a copy returned, so a second call with the same inputs reuses it.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run: its observations, area, onset definition and options, and climatology years.
    case : hcwf.config.model.Case
        The case whose model the thresholds are loaded for (the dataset does not depend on it).
    select : hcwf.config.model.Selection, default Selection()
        The call's choices (cells, units, reference onset).

    Returns
    -------
    xarray.DataArray
        3D array with dimensions ``(year, lat, lon)`` (or 2D ``(year, unit)`` on evaluation units) containing onset dates.

    Raises
    ------
    ValueError
        No year was processed.
    Exception
        Whatever the detection or reading of a year raised, with a note naming the year.
    """
    years_clim = rc.evaluation.climatology_years
    thresh_slice = thresholds_of(rc, case.model, select)
    key = repr((id(thresh_slice), id(rc.onset), rc.observations, rc.area, rc.onset_options, years_clim, select))
    if key in _MEMO:
        log.debug("Climatological onset dataset: reusing the one computed earlier in this run")
        return _MEMO[key][1].copy(deep=True)

    log.debug(f"Computing climatological onset from {len(years_clim)} years_clim: {min(years_clim)}-{max(years_clim)}")

    # Initialize lists to store results
    onset_arrays = []
    valid_years = []

    # Process each year
    for year, reader in zip(years_clim, _year_readers(years_clim, rc, select)):
        log.debug(f"\nProcessing year {year}...")

        try:
            # Detect onset for this year (the rainfall is read only when its onsets are not cached)
            onset_da = detect_observed(reader, thresh_slice, year, rc, select)
            spatial = axes.spatial_dims(onset_da.dims)
            values = tuple(np.asarray(onset_da[d].values) for d in spatial)

            # Count valid onsets
            valid_onsets = (~pd.isna(onset_da.values)).sum()
            total_points = onset_da.size

            log.debug(f"Year {year}: Found onset in {valid_onsets}/{total_points} grid points ({valid_onsets/total_points:.1%})")

            # Store the onset array
            onset_arrays.append(onset_da.values)
            valid_years.append(year)

        except Exception as e:   # a dataset over fewer years would be a different one: stop, naming the year
            e.add_note(f"climatological onset dataset, year {year}")
            raise

    if not onset_arrays:
        raise ValueError("No years were successfully processed")

    # Stack all onset arrays into a 3D array
    onset_3d = np.stack(onset_arrays, axis=0)

    # Create the final DataArray
    climatological_onset_da = xr.DataArray(
        onset_3d,
        coords=[('year', valid_years), *zip(spatial, values)],
        name='climatological_onset_dates',
        attrs={
            'description': 'Onset dates for climatological ensemble',
            'method': 'no date filter',
            'years_processed': valid_years,
            'total_years': len(valid_years)
        }
    )

    # Print summary statistics
    total_possible = onset_3d.size
    total_valid = (~pd.isna(climatological_onset_da.values)).sum()

    log.debug(f"\n{'='*60}")
    log.debug("CLIMATOLOGICAL ONSET DATASET SUMMARY")
    log.debug(f"{'='*60}")
    log.debug(f"Years processed: {len(valid_years)} ({min(valid_years)}-{max(valid_years)})")
    log.debug(f"Spatial domain: {' x '.join(f'{len(v)} {d}' for d, v in zip(spatial, values))}")
    log.debug(f"Total valid onsets: {total_valid:,}/{total_possible:,} ({total_valid/total_possible:.1%})")
    log.debug(f"Method: {climatological_onset_da.attrs['method']}")

    release_reference()   # the thresholds' reference rainfall has served; free its memory
    _MEMO[key] = ((thresh_slice, rc.onset), climatological_onset_da)   # held so their ids stay unique
    return climatological_onset_da.copy(deep=True)

