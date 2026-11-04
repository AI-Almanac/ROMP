"""Notebook helpers: the observed-onset map of a year and the rainfall time series at a grid point.

`spatial_onset` maps the observed onset of a year (through `hcwf.figures.func_map.spatial_metrics_map`);
`obs_onset_analysis` plots the rainfall time series with the detected onset at a grid point (through
`hcwf.figures.rainfall_time_series`). Both read the run's observations, thresholds and onset definition.
"""
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.config.model import Selection
from hcwf.io.observations import observations_of
from hcwf.io.thresholds import thresholds_of
from hcwf.onset.detect import detect_observed
from hcwf.figures.func_map import spatial_metrics_map
from hcwf.figures.rainfall_time_series import plot_rainfall_timeseries_with_onset_and_wetspell
from hcwf.onset.climatology import _doy_index
from hcwf.references.climatology import climatological_onset, season_day_dates

log = logging.getLogger(__name__)


def spatial_onset(year, rc):
    """Detect the observed onset of a year, map it as day of year and return it.

    Parameters
    ----------
    year : int
        The observation year.
    rc : hcwf.config.model.RunConfig
        The run: its observations, onset definition and thresholds (those of the first model).

    Returns
    -------
    xarray.DataArray
        The onset day of year, ``float64`` over ``(lat, lon)``, ``NaN`` where there is no onset.
    """
    thresh_slice = thresholds_of(rc, rc.models[0])

    log.debug("Loading observational rainfall data...")
    rainfall_ds = observations_of(year, rc)

    log.debug("Detecting observed onset...")
    onset_da = detect_observed(rainfall_ds, thresh_slice, year, rc)

    onset_doy = onset_da.dt.dayofyear.astype(float)
    onset_doy = onset_doy.where(~onset_da.isnull())


    spatial_metrics_map(onset_doy, "observation year "+str(year), rc.figure_spec(),
                    window=rc.evaluation.windows[0].span, fig=None, ax=None, cmap='YlOrRd', n_colors=10,
                    onset_plot=True, cbar_ssn=True, domain_mask=True, polygon_only=False,
                    show_ylabel=True, title="onset day", rect_box=True)

    return onset_doy


def _season_window(rc):
    """Return the window of the season to plot.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run; its ``onset`` setup names the season.

    Returns
    -------
    tuple of (tuple of (int, int), tuple of (int, int)) or None
        ``((month, day), (month, day))``: the fixed bounds of the chosen season. None (the whole record) for derived
        seasons.
    """
    setup = rc.onset
    if setup.derived_seasons:
        return None
    season = next(s for s in setup.definition.seasons if s.name == setup.season)
    return (season.start.month, season.start.day), (season.end.month, season.end.day)


def obs_onset_analysis(year, rc, lat_select=11, lon_select=39, incl_clim=True):
    """Detect the observed onset of a year and plot the rainfall time series at a grid point.

    The grid point is the one nearest to ``lat_select`` and ``lon_select``. The figure is saved in
    ``rc.output.figure_dir`` as ``onset_time_series_lat<lat>_lon<lon>_<year>.png``.

    Parameters
    ----------
    year : int
        The observation year.
    rc : hcwf.config.model.RunConfig
        The run; its climatology years give the climatological onset.
    lat_select : float, default 11
        Latitude of the grid point, in degrees north.
    lon_select : float, default 39
        Longitude of the grid point, in degrees east.
    incl_clim : bool, default True
        Add the climatological rainfall and onset to the figure.

    Returns
    -------
    da_sub : xarray.DataArray
        The observed rainfall within the season window.
    onset_da : xarray.DataArray
        The detected onset dates.
    """
    # load onset precip threshold, 2-D or scalar
    thresh_da = thresholds_of(rc, rc.models[0])

    log.debug(f"\n{'-'*50}")
    log.debug(f"Processing year {year}")

    # obs onset
    da = observations_of(year, rc)
    onset_da = detect_observed(da, thresh_da, year, rc)

    window = _season_window(rc)
    if window is None:
        da_sub = da
    else:
        start_date, end_date = window
        da_sub = da.where(
            (
                (da.time.dt.month > start_date[0]) |
                ((da.time.dt.month == start_date[0]) & (da.time.dt.day >= start_date[1]))
            ) &
            (
                (da.time.dt.month < end_date[0]) |
                ((da.time.dt.month == end_date[0]) & (da.time.dt.day <= end_date[1]))
            ),
            drop=True
        )

    da_pr_clim_sub = None
    climatological_onset_datetime = None

    if incl_clim:
        climatological_onset_doy, da_pr_clim = climatological_onset(
            rc, rc.first_case(rc.models[0].name), Selection(grid_point=True, lat=lat_select, lon=lon_select))

        climatological_onset_datetime = pd.Timestamp(
            season_day_dates(year, float(climatological_onset_doy.squeeze())))

        # the climatological rain of each day by its day of the 365-day year (29 February takes 28 February's)
        index = _doy_index(da_sub['time'].dt.month.values, da_sub['time'].dt.day.values)
        index = np.where(index < 0, _doy_index(2, 28), index)
        doy = xr.DataArray(index + 1, dims="time")
        da_pr_clim_sub = da_pr_clim.sel(doy=doy).copy()

        # assign the actual time from da_sub
        da_pr_clim_sub = da_pr_clim_sub.assign_coords(time=da_sub['time']).squeeze()

    save_path = os.path.join(rc.output.figure_dir, f"onset_time_series_lat{lat_select}_lon{lon_select}_{year}.png")

    plot_rainfall_timeseries_with_onset_and_wetspell(da_sub, onset_da,
                                                     lat_select=lat_select, lon_select=lon_select,
                                                     year_select=year, save_path=save_path,
                                                     incl_clim=incl_clim, pr_clim=da_pr_clim_sub,
                                                     onset_clim=climatological_onset_datetime)

    return da_sub, onset_da


# ------------------------------------------------------------------------------
