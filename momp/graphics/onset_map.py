"""Spatial map of climatological onset day of year, plus a script entry point that plots it."""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy
from momp.graphics.maps import calculate_cmz_averages
from momp.utils.printing import tuple_to_str_range
import cartopy.crs as ccrs
from matplotlib import colors as mcolors

from momp.utils.visual import cbar_season, set_basemap
from momp.utils.land_mask import shp_mask, add_polygon
from momp.graphics.func_map import spatial_metrics_map
from momp.utils.visual import box_boundary
import logging

log = logging.getLogger(__name__)

def plot_spatial_climatology_onset(onset_da_dict, *, years_clim, shpfile_dir, polygon, dir_fig, 
                                   region, figsize=(18, 6), cbar_ssn=False, domain_mask=False, 
                                   show_plot=True, rect_box=False, **kwargs):
    """
    Plot spatial maps of climatology onset day of year
    """

    # Extract data
    climatological_onset_doy = next(iter(onset_da_dict.values()))

    # Get coordinates
    lats = climatological_onset_doy.lat.values
    lons = climatological_onset_doy.lon.values

    # Detect resolution from latitude spacing
    lat_diff = abs(lats[1] - lats[0])
    log.debug(f"Detected resolution: {lat_diff:.1f} degrees")

    # Plot parameters
    map_lw = 0.75
    polygon_lw = 1.25
    panel_linewidth = 0.5
    tick_length = 3
    tick_width = 0.8
    if abs(lat_diff) < 0.99:
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
    set_basemap(ax, region, shpfile_dir, polygon, **kwargs)

    # Define Core Monsoon Zone bounding polygon coordinates based on resolution 
    polygon_defined = False

    if polygon:
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
    if domain_mask:
        climatological_onset_doy = shp_mask(climatological_onset_doy, region=region)

    vmin, vmax = climatological_onset_doy.quantile([0.05, 0.95], dim=None, skipna=True).values

    # Plot discrete values at each grid point using pcolormesh with custom colormap
    im = ax.pcolormesh(climatological_onset_doy.lon, climatological_onset_doy.lat, climatological_onset_doy.values, 
                     cmap=cmap_jjas, norm=norm_jjas, transform=ccrs.PlateCarree(), shading='auto')

    # Add colorbar with MMM DD labels for every other tick
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=20)

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
        plot_filename = f"climatology_onset_{tuple_to_str_range(years_clim)}.png"
        plot_path = os.path.join(dir_fig, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        log.debug(f"Figure saved to: {plot_path}")

    if show_plot:
        plt.show()

    return fig, ax, plot_path


def doy_to_date_string(doy, date_filter_year=2024):
    """Convert day of year to dd/mm format"""
    # Assuming non-leap year for consistency
    date = datetime(date_filter_year, 1, 1) + timedelta(days=int(doy) - 1)
    return date.strftime('%d/%m')

def doy_to_mmm_dd(doy, date_filter_year=2024):
    """Convert day of year to 'MMM DD' format"""
    # Use 2018 (not a leap year) as reference to handle all possible DOYs
    date = pd.to_datetime(f"{date_filter_year}-{int(doy):03d}", format="%Y-%j")
    return date.strftime("%b %d")


if __name__ == "__main__":
    from itertools import product
    from momp.stats.benchmark import compute_metrics_multiple_years
    from momp.lib.control import iter_list, make_case
    from momp.lib.convention import Case
    from momp.lib.loader import get_cfg, get_setting
    from dataclasses import asdict

    cfg, setting = get_cfg(), get_setting()

    cfg.ref_model = 'climatology'
    cfg.probabilistic = False

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    layout_pool = iter_list(vars(cfg_ref))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))
        log.debug(f"processing model onset evaluation for {case.case_name}")

        case_ref = {'model_dir': setting.ref_model_dir,
                    'model_var': case.ref_model_var,
                    'file_pattern': setting.ref_model_file_pattern,
                    'unit_cvt': setting.ref_model_unit_cvt
                    }

        case.update(case_ref)

        if case.model == 'climatology':
            case.years = case.years_clim

        log.debug("%s %s", "\n case.file_pattern = ", case.file_pattern)
        log.debug("%s %s", "\n setting.ref_model_file_pattern = ", setting.ref_model_file_pattern)

        case_cfg_ref = {**asdict(case), **asdict(setting)}
        log.debug("%s %s", "\n case_cfg_ref.file_pattern  = ", case_cfg_ref.get('file_pattern'))

        # model-obs onset benchmarking
        log.debug("%s %s", "\n ", case_cfg_ref)
        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg_ref)
        log.debug("%s %s", "\n case_cfg_ref.file_pattern  = ", case_cfg_ref.get('years_clim'))
        break

    da = next(iter(onset_da_dict.values()))

    fig, _, _, _ = spatial_metrics_map(da, case.model, 
                    fig=None, ax=None, figsize=(8, 6), cmap='YlOrRd', n_colors=10,
                    onset_plot=True, cbar_ssn=True, domain_mask=True, polygon_only=False,
                    show_ylabel=True, title="climatology onset", **case_cfg_ref)

    years_clim = case_cfg_ref.get('years_clim')
    dir_fig = case_cfg_ref.get('dir_fig')
    plot_filename = f"climatology_onset_{tuple_to_str_range(years_clim)}.png"
    plot_path = os.path.join(dir_fig, plot_filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
