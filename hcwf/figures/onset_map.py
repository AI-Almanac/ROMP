"""Spatial map of climatological onset day of year.

`plot_spatial_climatology_onset` draws the climatological onset with `hcwf.figures.field.draw_field` on the basemap
of `hcwf.figures.visual`, and shares the date formatting of `hcwf.figures.func_map`.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from hcwf.figures.maps import calculate_cmz_averages
from hcwf.naming import tuple_to_str_range
import cartopy.crs as ccrs
from matplotlib import colors as mcolors

from hcwf.figures.visual import cbar_season, set_basemap
from hcwf.spatial.masks import shp_mask
from hcwf.figures.visual import add_polygon
from hcwf.figures.func_map import doy_to_mmm_dd
from hcwf.figures.visual import box_boundary
from hcwf.figures.field import colorbar_extend, draw_field, extent, units_of
from hcwf.runs.folder import file_name
import logging

log = logging.getLogger(__name__)

def plot_spatial_climatology_onset(onset_da_dict, spec, years_clim, *, cbar_ssn=False,
                                   domain_mask=False, rect_box=False):
    """Plot the map of the climatological onset day of year.

    The figure is saved to ``spec.dir_fig`` (300 dpi) and shown if ``spec.show_plot`` is true.

    Parameters
    ----------
    onset_da_dict : dict of str to xarray.DataArray
        The climatological onset day of year, over ``(lat, lon)`` or ``(unit,)``; only the first entry is drawn.
    spec : object
        The run's ``FigureSpec``: ``polygon``, ``dir_fig``, ``region``, ``show_plot`` and what `set_basemap` reads.
    years_clim : tuple of (int, int)
        The climatology years, named in the output file.
    cbar_ssn : bool, default False
        Use the season colormap of `hcwf.figures.visual.cbar_season`.
    domain_mask : bool, default False
        Mask the cells outside the country boundary (grid fields only).
    rect_box : bool, default False
        Draw the rectangular boundary box.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    plot_path : str
        The output file path in ``spec.dir_fig`` (raises ``UnboundLocalError`` here when ``spec.dir_fig`` is empty).
    """
    polygon, dir_fig, region = spec.polygon, spec.dir_fig, spec.region

    # Extract data
    climatological_onset_doy = next(iter(onset_da_dict.values()))

    # Get coordinates (the bounds of the polygons on evaluation units)
    units = units_of(climatological_onset_doy, spec)
    lats, lons = extent(climatological_onset_doy, units)

    # Detect resolution from latitude spacing
    lat_diff = abs(lats[1] - lats[0]) if units is None else np.nan
    log.debug(f"Detected resolution: {lat_diff:.1f} degrees")

    # Plot parameters
    if units is not None or abs(lat_diff) < 0.99:   # no values are written on polygons
        txt_fsize = None
    elif abs(lat_diff - 2.0) < 0.1:
        txt_fsize = 8
    elif abs(lat_diff - 4.0) < 0.1:
        txt_fsize = 10
    elif abs(lat_diff - 1.0) < 0.1:
        txt_fsize = 6
    else:
        txt_fsize = 8

    # Define colormap levels for day of year - more refined levels
    levels = np.arange(135, 245, 3)  # May 15 (135) to late September (270)

    # create figure obj and ax
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # set map extent, country boundary, gridline
    set_basemap(ax, spec)

    # Define Core Monsoon Zone bounding polygon coordinates based on resolution 
    polygon_defined = False

    if polygon and units is None:
        ax, polygon1_lat, polygon1_lon, polygon_defined = \
                                    add_polygon(ax, climatological_onset_doy, polygon, return_polygon=True)

        # Calculate statistics (only calculate CMZ stats if polygon is defined)
        if polygon_defined:
            cmz_onset_mean = calculate_cmz_averages(climatological_onset_doy, polygon1_lon, polygon1_lat)
        else:
            cmz_onset_mean = np.nan

    if cbar_ssn:
        cmap_jjas, norm_jjas, bounds = cbar_season()
    else:
        # Use a colormap (RdYlBu_r or similar) also 'RdYlGn_r', 'Spectral_r', or 'coolwarm'
        cmap_jjas = plt.cm.Spectral
        norm_jjas = mcolors.BoundaryNorm(levels, cmap_jjas.N, extend='max')

    # mask data inside country boundary
    if domain_mask and units is None:
        climatological_onset_doy = shp_mask(climatological_onset_doy, region=region)

    vmin, vmax = climatological_onset_doy.quantile([0.05, 0.95], dim=None, skipna=True).values

    # Plot discrete values at each grid point using pcolormesh with custom colormap
    im = draw_field(ax, climatological_onset_doy, units, cmap=cmap_jjas, norm=norm_jjas, transform=ccrs.PlateCarree())

    # Add colorbar with MMM DD labels for every other tick
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=20,
                        extend=colorbar_extend(im))

    if cbar_ssn:
        # Create tick positions - use every other bound for labeling
        tick_positions = bounds[::2]  # Every other boundary
        tick_labels = [doy_to_mmm_dd(doy) for doy in tick_positions[:-1]]  # Exclude last boundary

        # Set all boundaries as minor ticks (for visual separation)
        cbar.set_ticks(bounds, minor=True)
        # Set every other boundary as major ticks (with labels)
        cbar.set_ticks(tick_positions[:-1])  # Exclude last boundary
    else:
        # Create custom tick labels in MMM DD format
        tick_levels = levels[::4]  # Use every other level to avoid crowding
        tick_labels = [doy_to_mmm_dd(doy) for doy in tick_levels]
        cbar.set_ticks(tick_levels)

    cbar.set_ticklabels(tick_labels)

    cbar.set_label('Mean onset date', fontsize=12, fontweight='normal')
    cbar.ax.tick_params(labelsize=10)    

    # Add text annotations for onset days
    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = climatological_onset_doy.values[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > 200 else 'black'
                    ax.text(lon, lat, f'{value:.0f}', 
                               ha='center', va='center',
                               color=text_color, fontsize=txt_fsize, fontweight='normal')

    # Add CMZ average text (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_onset_mean):
        cmz_text = f'mean onset: {cmz_onset_mean:.0f} days'
        ax.text(0.98, 0.02, cmz_text, transform=ax.transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')

    ax.text(0.98, 0.98, 'onset (day of year)', transform=ax.transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')

    if rect_box:
        box_boundary('rect_boundary', ax, edgecolor='black', linewidth=2,
                    linestyle='-', fill=False, alpha=1.0,
                    zorder=20)

    plt.tight_layout()

    # Save if path provided
    if dir_fig:
        plot_filename = file_name(None, "climatology_map", tuple_to_str_range(years_clim))
        plot_path = os.path.join(dir_fig, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        log.debug(f"Figure saved to: {plot_path}")

    if spec.show_plot:
        plt.show()

    return fig, ax, plot_path
