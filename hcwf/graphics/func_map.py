"""Spatial map plotting of gridded onset or metric fields with CMZ statistics and date helpers."""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from hcwf.utils.mask import polygon_mask
from hcwf.graphics.maps import calculate_cmz_averages
from hcwf.utils.printing import tuple_to_str
import cartopy.crs as ccrs
from matplotlib import colors as mcolors
from matplotlib import colormaps

from hcwf.utils.visual import cbar_season, contrast_text_color, set_basemap
from hcwf.utils.mask import shp_mask
from hcwf.utils.visual import add_polygon
from hcwf.utils.visual import box_boundary
from hcwf.graphics.field import colorbar_extend, draw_field, extent, units_of
from hcwf.lib.run_folder import file_name
import logging

log = logging.getLogger(__name__)


def spatial_metrics_map(da, model_name, spec, *, window=None,
                    fig=None, ax=None, figsize=(8, 6), cmap='YlOrRd', n_colors=10, int_bin=True,
                    onset_plot=False, cbar_ssn=False, domain_mask=False, polygon_only=False,
                    show_ylabel=True, title=None, panel=False, text_scale=1.0, 
                    vmin=None, vmax=None, rect_box=False):
    """
    Plot a spatial map of a gridded 2D field (e.g. onset day of year or a spatial metric).

    Draws the map on the given or a new figure, optionally prints values in cells and the
    CMZ mean onset, and shows it unless ``panel`` is True. The figure is not saved.

    spec is the run's FigureSpec; window the verification window (named in the would-be file).

    Returns
    -------
    fig, ax, im, plot_path
        Figure, axes, the pcolormesh handle and the would-be output file path.
    """

    polygon, region = spec.polygon, spec.region

    # Get coordinates (the bounds of the polygons on evaluation units)
    units = units_of(da, spec)
    lats, lons = extent(da, units)
    var_name = da.name

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

    text_size = 14
    text_size *= text_scale
    if txt_fsize is not None:
        txt_fsize *= text_scale 

    # Define colormap levels 
    if None in (vmin, vmax):
        vmin, vmax = da.quantile([0.1, 0.9], dim=None, skipna=True).values

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    levels = np.linspace(vmin, vmax, n_colors)

    if int_bin and n_colors>0:
        # use integer bins
        vmin = np.floor(vmin)  # floor to nearest integer
        vmax = np.ceil(vmax)   # ceil to nearest integer
        levels = np.arange(vmin, vmax + 1)  # integer boundaries
        n_colors = len(levels)

    cmap_discrete = colormaps.get_cmap(cmap).resampled(n_colors)

    if cbar_ssn:
        cmap_jjas, norm_jjas, bounds = cbar_season()
    elif n_colors > 0:
        # Use a colormap (RdYlBu_r or similar) also 'RdYlGn_r', 'Spectral_r', or 'coolwarm'
        cmap_jjas = cmap_discrete
        norm_jjas = mcolors.BoundaryNorm(levels, cmap_jjas.N, extend='max')
    else:
        cmap_jjas = cmap
        norm_jjas = mcolors.Normalize(vmin=vmin, vmax=vmax)  # set explicit min/max

    # -----------------------------------------------------------------------------
    # create figure obj and ax
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # set map extent, country boundary, gridline
    ax, gl = set_basemap(ax, spec)

    # Define Core Monsoon Zone bounding polygon coordinates based on resolution 

    polygon_defined = False

    if polygon and units is None:
        ax, polygon1_lat, polygon1_lon, polygon_defined = \
                                    add_polygon(ax, da, polygon, return_polygon=True)

    # Calculate statistics (only calculate CMZ stats if polygon is defined)
    if polygon_defined:
        cmz_onset_mean = calculate_cmz_averages(da, polygon1_lon, polygon1_lat)
    else:
        cmz_onset_mean = np.nan

    # mask data inside country boundary
    if domain_mask and units is None:
        da = shp_mask(da, region=region)

    if polygon_only and units is None:
        da = polygon_mask(da)

    # Plot discrete values at each grid point using pcolormesh with custom colormap
    im = draw_field(ax, da, units, cmap=cmap_jjas, norm=norm_jjas, transform=ccrs.PlateCarree())

    # Add colorbar with MMM DD labels for every other tick
    if not panel:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=20,
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
            tick_levels = levels[::2]  # Use every other level to avoid crowding
            tick_labels = [doy_to_mmm_dd(doy) for doy in tick_levels]
            cbar.set_ticks(tick_levels)

        cbar.set_ticklabels(tick_labels)

        cbar.set_label(var_name, fontsize=12, fontweight='normal')
        cbar.ax.tick_params(labelsize=10)    

    # Add model name text in top-right
    ax.text(0.99, 0.97, model_name, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            color='black', fontsize=text_size*1.2, fontweight='bold')

    # Add text annotations for onset days
    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = da.values[i, j]
                if not np.isnan(value):
                    text_color = contrast_text_color(value, cmap_obj, norm)
                    ax.text(lon, lat, f'{value:.0f}' if onset_plot else f'{value:.1f}', 
                               ha='center', va='center',
                               color=text_color, fontsize=txt_fsize, fontweight='normal')

    # Add CMZ average text (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_onset_mean) and onset_plot:
        cmz_text = f'mean onset: {doy_to_mmm_dd(cmz_onset_mean)}'
        ax.text(0.98, 0.02, cmz_text, transform=ax.transAxes,
                    color='black', fontsize=text_size,
                    verticalalignment='bottom', horizontalalignment='right')

        ax.text(0.98, 0.98, 'onset (day of year)', transform=ax.transAxes,
                    color='black', fontsize=text_size, fontweight='normal',
                    verticalalignment='top', horizontalalignment='right')

    if show_ylabel and gl:
        gl.left_labels = True
    elif not show_ylabel and gl:
        gl.left_labels = False
    elif show_ylabel:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
        ax.tick_params(axis='y', labelleft=True)

    if title:
        ax.text(0.02, 1.02, title, transform=ax.transAxes,
                verticalalignment='bottom', fontsize=text_size*1.0, fontweight='normal')

    if rect_box:
        box_boundary('rect_boundary', ax, edgecolor='black', linewidth=2,
                    linestyle='-', fill=False, alpha=1.0,
                    zorder=20)

    plt.tight_layout()

    if not panel:
        plt.tight_layout()
        plt.show()

    window_str = tuple_to_str(window)
    plot_filename = file_name(None, "field_map", model_name, var_name, window_str)
    plot_path = os.path.join(spec.dir_fig, plot_filename)

    return fig, ax, im, plot_path


def doy_to_mmm_dd(doy, date_filter_year=2024):
    """Convert day of year to 'MMM DD' format"""
    # 2024 is a leap year, so every day of year 1-366 exists
    date = pd.to_datetime(f"{date_filter_year}-{int(doy):03d}", format="%Y-%j")
    return date.strftime("%b %d")
