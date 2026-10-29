"""CMZ averaging helper and a three-panel spatial map of MAE, false alarm rate and miss rate."""
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from momp.params.region_def import polygon_boundary
from momp.utils.land_mask import get_india_outline, shp_outline
from momp.utils.visual import box_boundary


def calculate_cmz_averages(da, polygon_lon, polygon_lat):
    """Calculate spatial average within the CMZ polygon"""
    from matplotlib.path import Path

    # Get coordinates
    lats = da.lat.values
    lons = da.lon.values

    polygon_path = Path(list(zip(polygon_lon, polygon_lat)))

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    inside_polygon = polygon_path.contains_points(points).reshape(lon_grid.shape)

    values_inside = da.values[inside_polygon]

    if len(values_inside) > 0:
        return np.nanmean(values_inside)
    else:
        return np.nan


def plot_spatial_metrics(spatial_metrics, *, case_name, shpfile_dir, polygon, dir_fig, region, 
                         figsize=(18, 6), show_plot=True, rect_box=True, **kwargs):
    """
    Plot spatial maps of Mean MAE, False Alarm Rate, and Miss Rate for one model in a 1x3 subplot
    with region boundary, CMZ polygon, grid values displayed, and CMZ averages.

    Saves the figure to ``dir_fig`` (if given), prints the CMZ averages, and shows the
    figure if ``show_plot`` is True. Returns ``(fig, axes)``.
    """

    # Extract data
    mean_mae = spatial_metrics['mean_mae']
    far = spatial_metrics['false_alarm_rate'] * 100  # Convert to percentage
    miss_rate = spatial_metrics['miss_rate'] * 100   # Convert to percentage

    # Get coordinates
    lats = mean_mae.lat.values
    lons = mean_mae.lon.values

    # Detect resolution from latitude spacing
    lat_diff = abs(lats[1] - lats[0])
    print(f"Detected resolution: {lat_diff:.1f} degrees")

    # Define Core Monsoon Zone bounding polygon coordinates based on resolution 
    polygon_defined = False

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(mean_mae)

        if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
            polygon_defined = True

    def calculate_mae_stats_across_years(spatial_metrics, lons, lats, polygon_lon, polygon_lat):
        """Calculate MAE statistics: spatial average for each year, then mean ± SE across years"""
        yearly_mae_keys = [key for key in spatial_metrics.keys() if key.startswith('mae_') and key != 'mae_combined']

        if not yearly_mae_keys:
            print("Warning: No yearly MAE maps found")
            return np.nan, np.nan, np.nan, np.nan

        cmz_yearly_averages = []
        overall_yearly_averages = []

        for mae_key in yearly_mae_keys:
            year_mae_map = spatial_metrics[mae_key]

            if polygon_defined and polygon_lon is not None:
                cmz_avg = calculate_cmz_averages(year_mae_map, polygon_lon, polygon_lat)
                if not np.isnan(cmz_avg):
                    cmz_yearly_averages.append(cmz_avg)

            overall_avg = np.nanmean(year_mae_map.values)
            if not np.isnan(overall_avg):
                overall_yearly_averages.append(overall_avg)

        if len(cmz_yearly_averages) > 0 and polygon_defined:
            cmz_mean = np.mean(cmz_yearly_averages)
            cmz_se = np.std(cmz_yearly_averages, ddof=1) / np.sqrt(len(cmz_yearly_averages)) if len(cmz_yearly_averages) > 1 else 0
        else:
            cmz_mean, cmz_se = np.nan, np.nan

        if len(overall_yearly_averages) > 0:
            overall_mean = np.mean(overall_yearly_averages)
            overall_se = np.std(overall_yearly_averages, ddof=1) / np.sqrt(len(overall_yearly_averages)) if len(overall_yearly_averages) > 1 else 0
        else:
            overall_mean, overall_se = np.nan, np.nan

        return cmz_mean, cmz_se, overall_mean, overall_se

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Calculate statistics (only calculate CMZ stats if polygon is defined)
    if polygon_defined:
        cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = calculate_mae_stats_across_years(
            spatial_metrics, lons, lats, polygon1_lon, polygon1_lat
        )

        cmz_far = calculate_cmz_averages(spatial_metrics['false_alarm_rate'] * 100, polygon1_lon, polygon1_lat)
        cmz_mr = calculate_cmz_averages(spatial_metrics['miss_rate'] * 100, polygon1_lon, polygon1_lat)
    else:
        cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = calculate_mae_stats_across_years(
            spatial_metrics, lons, lats, None, None
        )
        cmz_far = np.nan
        cmz_mr = np.nan

    # Create edges for pcolormesh (cell boundaries)
    lon_edges = np.concatenate([lons - (lons[1]-lons[0])/2, [lons[-1] + (lons[1]-lons[0])/2]])
    lat_edges = np.concatenate([lats - (lats[1]-lats[0])/2, [lats[-1] + (lats[1]-lats[0])/2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)

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

    # Panel 1: Mean MAE
    masked_mae = np.ma.masked_invalid(mean_mae.values)
    im1 = axes[0].pcolormesh(LON_edges, LAT_edges, masked_mae, 
                             cmap='OrRd', vmin=0, vmax=15, shading='flat')

    # Add India outline
    if shpfile_dir:
        india_boundaries = get_india_outline(shpfile_dir)
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            axes[0].plot(india_lon, india_lat, color='black', linewidth=map_lw)
    else:
        shp_outline(axes[0], region=region)

    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                         fill=False, edgecolor='black', linewidth=polygon_lw)
        axes[0].add_patch(polygon)

    # Add text annotations for MAE values
    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = mean_mae.values[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > 7.5 else 'black'
                    axes[0].text(lon, lat, f'{value:.1f}', 
                               ha='center', va='center',
                           color=text_color, fontsize=txt_fsize, fontweight='normal')

    # Add CMZ average text with mean ± SE across years (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_mae_mean):
        if cmz_mae_se > 0:
            cmz_text = f'MAE: {cmz_mae_mean:.1f}±{cmz_mae_se:.1f} days'
        else:
            cmz_text = f'MAE: {cmz_mae_mean:.1f} days'

        axes[0].text(0.98, 0.02, cmz_text, transform=axes[0].transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')

    axes[0].text(0.98, 0.98, 'MAE (in days)', transform=axes[0].transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')
    axes[0].set_xlabel('Longitude', fontsize=12)
    axes[0].set_ylabel('Latitude', fontsize=12)

    # Panel 2: False Alarm Rate
    masked_far = np.ma.masked_invalid(far.values)
    im2 = axes[1].pcolormesh(LON_edges, LAT_edges, masked_far, 
                             cmap='Reds', vmin=0, vmax=100, shading='flat')

    # Add India outline
    if "india_boundaries" in locals():
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            axes[1].plot(india_lon, india_lat, color='black', linewidth=map_lw)
    else:
        shp_outline(axes[1], region=region)

    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                         fill=False, edgecolor='black', linewidth=polygon_lw)
        axes[1].add_patch(polygon)

    # Add text annotations for FAR values
    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = far.values[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > 50 else 'black'
                    axes[1].text(lon, lat, f'{value:.0f}', 
                               ha='center', va='center',
                               color=text_color, fontsize=txt_fsize, fontweight='normal')

    # Add CMZ average text (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_far):
        cmz_text = f'FAR: {cmz_far:.1f}%'
        axes[1].text(0.98, 0.02, cmz_text, transform=axes[1].transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')

    axes[1].text(0.98, 0.98, 'False Alarm Rate (%)', transform=axes[1].transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')
    axes[1].set_xlabel('Longitude', fontsize=12)

    # Panel 3: Miss Rate
    masked_mr = np.ma.masked_invalid(miss_rate.values)
    im3 = axes[2].pcolormesh(LON_edges, LAT_edges, masked_mr, 
                             cmap='Blues', vmin=0, vmax=100, shading='flat')

    # Add India outline
    if "india_boundaries" in locals():
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            axes[2].plot(india_lon, india_lat, color='black', linewidth=map_lw)
    else:
        shp_outline(axes[2], region=region)

    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                         fill=False, edgecolor='black', linewidth=polygon_lw)
        axes[2].add_patch(polygon)

    # Add text annotations for Miss Rate values
    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = miss_rate.values[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > 50 else 'black'
                    axes[2].text(lon, lat, f'{value:.0f}', 
                               ha='center', va='center',
                               color=text_color, fontsize=txt_fsize, fontweight='normal')

    # Add CMZ average text (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_mr):
        cmz_text = f'MR: {cmz_mr:.1f}%'
        axes[2].text(0.98, 0.02, cmz_text, transform=axes[2].transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')

    axes[2].text(0.98, 0.98, 'Miss Rate (%)', transform=axes[2].transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')

    axes[2].set_xlabel('Longitude', fontsize=12)

    # add colorbar
    ims = [im1, im2, im3]
    for ax, im in zip(axes, ims):
        # Get position of the subplot
        pos = ax.get_position()

        # Shorter than the subplot width
        cbar_width = pos.width * 0.7  # 70% width
        cbar_height = 0.015           # very thin

        # Centered below the subplot
        cbar_x = pos.x0 + (pos.width - cbar_width)/2
        cbar_y = pos.y0 - 0.06        # distance below subplot

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)

        # 2. Append a new axes at the bottom ('bottom')
        # size: height of cbar (5%), pad: distance from plot (0.05)
        cax = divider.append_axes("bottom", size="5%", pad=0.3)

        # 3. Create the colorbar in that specific cax
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        # 4. To make it SHORTER (e.g., 60% of the plot width):
        pos = cax.get_position()
        new_width = pos.width * 0.6
        new_x = pos.x0 + (pos.width - new_width) / 2
        cax.set_position([new_x, pos.y0, new_width, pos.height])
        cax.set_anchor('C') # Center it
        cax.set_box_aspect(0.05) # This forces a thin, short ratio

    # Set consistent axis limits and styling for all panels
    for i, ax in enumerate(axes):
        ax.set_xlim([lons.min()-2, lons.max()+2])
        ax.set_ylim([lats.min()-2, lats.max()+2])

        xticks = np.arange(lons.min(), lons.max()+1, 4)
        xticklabels = [f"{int(x)}°E" for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        if i == 0:
            yticks = np.arange(lats.min(), lats.max()+1, 4)
            yticklabels = [f"{int(y)}°N" for y in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.tick_params(axis='both', which='major', labelsize=10, 
                      length=tick_length, width=tick_width)
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_linewidth(panel_linewidth)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)

    for ax in axes:
        if rect_box:
            box_boundary('rect_boundary', ax, edgecolor='black', linewidth=2,
                    linestyle='-', fill=False, alpha=1.0,
                    zorder=20)

    plt.tight_layout()

    # Save if path provided
    if dir_fig:
        plot_filename = f"spatial_metrics_{case_name}.png"
        plot_path = os.path.join(dir_fig, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {plot_path}")

    # Only print CMZ averages if polygon is defined
    if polygon_defined:
        print(f"\n=== CORE MONSOON ZONE (CMZ) AVERAGES ===")

        if not np.isnan(cmz_mae_mean):
            print(f"CMZ Mean MAE (avg across years): {cmz_mae_mean:.2f} ± {cmz_mae_se:.2f} days")
        else:
            print(f"CMZ Mean MAE: N/A")

        print(f"CMZ False Alarm Rate: {cmz_far:.1f} %")
        print(f"CMZ Miss Rate: {cmz_mr:.1f} %")
    else:
        print(f"\nNote: CMZ averages not calculated (resolution {lat_diff:.1f}° not supported)")

    if show_plot:
        plt.show()

    return fig, axes
