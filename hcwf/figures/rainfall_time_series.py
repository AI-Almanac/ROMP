"""Rainfall time series at one grid point with onset date markers.

`plot_rainfall_timeseries_with_onset_and_wetspell` is called by `hcwf.figures.notebook.obs_onset_analysis`.
"""
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import logging

__all__ = []


log = logging.getLogger(__name__)

def plot_rainfall_timeseries_with_onset_and_wetspell(pr, onset_date, lat_select, lon_select, year_select=None, save_path=None, incl_clim=None, pr_clim=None, onset_clim=None):
    """Plot the rainfall time series of a grid point with the onset date marked.

    The grid point is the one nearest to the selected latitude and longitude. The figure is shown, and saved when
    ``save_path`` is given.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation in mm/day, with dims ``(time, lat, lon)``.
    onset_date : xarray.DataArray
        Onset dates over ``(lat, lon)``; ``NaT`` where there is no onset.
    lat_select : float
        Latitude of the selected grid point, in degrees north.
    lon_select : float
        Longitude of the selected grid point, in degrees east.
    year_select : int, optional
        Year shown in the title only; the full series passed in is plotted.
    save_path : str, optional
        Path to save the plot to (300 dpi); None does not save.
    incl_clim : bool, optional
        Add the climatological rainfall ``pr_clim`` and the climatological onset ``onset_clim``.
    pr_clim : xarray.DataArray, optional
        Climatological daily rainfall in mm/day with a ``time`` coordinate; needed with ``incl_clim``.
    onset_clim : datetime.datetime, optional
        The climatological onset date; needed with ``incl_clim``.

    """
    # Select the nearest grid point
    pr_point = pr.sel(lat=lat_select, lon=lon_select, method="nearest")
    onset_point = onset_date.sel(lat=lat_select, lon=lon_select, method="nearest")

    # Get actual coordinates
    actual_lat = float(pr_point.lat.values)
    actual_lon = float(pr_point.lon.values)

    # Plot single year
    pr_year = pr_point
    onset_year = onset_point

    fig, ax = plt.subplots(figsize=(8, 3))

    # Plot rainfall as line with circle markers
    ax.plot(pr_year.time, pr_year.values, marker='o', markersize=4, linewidth=1.5,
            color='blue', markerfacecolor='blue', markeredgecolor='blue',
            markeredgewidth=0.5, alpha=0.8, label='Daily rainfall')

    if incl_clim:
        ax.plot(pr_clim.time, pr_clim.values, marker='s', markersize=2, linewidth=1.2,
            color='dimgray', markerfacecolor='dimgray', markeredgecolor='dimgray',
            markeredgewidth=0.3, alpha=0.8, label='Climatology')

        ax.axvline(x=onset_clim, color='grey', linewidth=1.5, linestyle='-',
                    label=f'Onset clim: {onset_clim.strftime("%b %d")}', alpha=0.8)

    # Mark onset date if it exists
    if not pd.isna(onset_year.values):
        onset_datetime = pd.to_datetime(onset_year.values)
        ax.axvline(x=onset_datetime, color='red', linewidth=1.5, linestyle='-', 
                    label=f'Onset: {onset_datetime.strftime("%b %d")}', alpha=0.8)

    # Formatting
    ax.set_ylabel('Rainfall (mm/day)', fontsize=8, fontweight='normal')
    ax.set_title(f'Daily Rainfall Onset Dates - {year_select}\n'
                f'Location: {actual_lat:.1f}°N, {actual_lon:.1f}°E', 
                fontsize=8, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper right', fontsize=8, frameon=False, fancybox=False, shadow=True)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        log.debug(f"onset time series figure saved to {save_path}")

    plt.show()
