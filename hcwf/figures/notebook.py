"""Notebook helpers: the observed-onset map of a year (`spatial_onset`) and the rainfall time series with the detected
onset at a grid point (`obs_onset_analysis`)."""
import logging
import os

import pandas as pd

from hcwf.config.model import Selection
from hcwf.io.input import observations_of, thresholds_of
from hcwf.onset.detect import detect_observed
from hcwf.figures.func_map import spatial_metrics_map
from hcwf.figures.rainfall_time_series import plot_rainfall_timeseries_with_onset_and_wetspell
from hcwf.references.climatology import climatological_onset

log = logging.getLogger(__name__)


def spatial_onset(year, rc):
    """Detect observed onset for a year, map it as day of year, and return the day-of-year DataArray.

    rc is the run (a RunConfig): its observations, onset definition and thresholds.
    """
    thresh_slice = thresholds_of(rc, rc.models[0])

    log.debug("Loading observational rainfall data...")
    rainfall_ds = observations_of(year, rc)

    log.debug("Detecting observed onset...")
    onset_da = detect_observed(rainfall_ds, thresh_slice, year, rc)

    onset_doy = onset_da.dt.dayofyear.astype(float)
    onset_doy = onset_doy.where(~onset_da.isnull())


    spatial_metrics_map(onset_doy, "observation year "+str(year), rc.figure_spec(),
                    window=rc.evaluation.windows[0].span, fig=None, ax=None, figsize=(8, 6), cmap='YlOrRd', n_colors=10,
                    onset_plot=True, cbar_ssn=True, domain_mask=True, polygon_only=False,
                    show_ylabel=True, title="onset day", rect_box=True)

    return onset_doy


def _season_window(rc):
    """The ((month, day), (month, day)) window to plot: the chosen season's fixed bounds; None (the whole record) for derived seasons."""
    setup = rc.onset
    if setup.derived_seasons:
        return None
    season = next(s for s in setup.definition.seasons if s.name == setup.season)
    return (season.start.month, season.start.day), (season.end.month, season.end.day)


def obs_onset_analysis(year, rc, lat_select=11, lon_select=39, incl_clim=True):
    """Detect observed onset for a year, plot the rainfall time series at the grid point nearest lat_select/lon_select, and return (rainfall within the season window, onset dates).

    rc is the run (a RunConfig); its climatology years give the climatological onset."""
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

        doy = int(climatological_onset_doy.squeeze())

        climatological_onset_datetime = (
            pd.Timestamp(year=year, month=1, day=1)
            + pd.Timedelta(days=doy - 1)
        )

        # get day-of-year for each timestamp
        doy = da_sub['time'].dt.dayofyear

        # select values from rainfall_clim_mean using day-of-year
        da_pr_clim_sub = da_pr_clim.sel(doy=doy).copy()

        # assign the actual time from da_sub
        da_pr_clim_sub = da_pr_clim_sub.assign_coords(time=da_sub['time']).squeeze()

    save_path = os.path.join(rc.output.figure_dir, f"onset_time_series_lat{lat_select}_lon{lon_select}_{year}.png")

    plot_rainfall_timeseries_with_onset_and_wetspell(da_sub, onset_da, None,
                                                     lat_select=lat_select, lon_select=lon_select,
                                                     year_select=year, save_path=save_path,
                                                     incl_clim=incl_clim, pr_clim=da_pr_clim_sub,
                                                     onset_clim=climatological_onset_datetime)

    return da_sub, onset_da


# ------------------------------------------------------------------------------
