"""Multi-panel maps of a spatial error metric: reference model absolute, other models as differences."""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from dataclasses import asdict
from momp.lib.control import make_case
from momp.lib.control import ref_cfg_layout
from momp.lib.convention import Case
from momp.graphics.func_map import spatial_metrics_map
from momp.utils.printing import tuple_to_str
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors
import logging
from momp.lib.run_folder import file_name

log = logging.getLogger(__name__)


def panel_map_mae_far_mr(model_list, verification_window, var_name, cfg, setting, **kwargs):
    """
    Plot a row of spatial maps of a metric (e.g. mean_mae, false_alarm_rate, miss_rate), one per model.

    Metric fields are read from the ``spatial_metrics_<model>_<window>.nc`` files in ``cfg.dir_out``.
    The first model in ``model_list`` is drawn as absolute values; the others are drawn as
    differences from it. The figure is saved to ``cfg.dir_fig`` and shown.

    Returns
    -------
    fig, axes, ims
        Figure, list of map axes and list of pcolormesh handles.
    """
    window_str = tuple_to_str(verification_window)

    n = len(model_list)
    fig = plt.figure(figsize=(8, 5/4*n))
    axes = []
    ims = []
    fig_width, fig_height = fig.get_size_inches()
    text_scale =  min(fig_width/n, fig_height/1) / 5 * 1.2

    gs = GridSpec(2, n, height_ratios=[1, 0.043], hspace=0.1, wspace=0.01, top=0.85)

    da_all = []

    for i, model in enumerate(model_list):
        fi = os.path.join(cfg.dir_out, file_name(vars(cfg), "spatial", f"{model}_{window_str}"))
        ds = xr.open_dataset(fi)
        da = ds[var_name]
        unit = "days"
        if var_name in ["false_alarm_rate", "miss_rate"]:
            da = da*100
            unit = '$(\%)$'
        ds.close()

        if i == 0:
            da_all.append(da)
        elif i > 0:
            da = da - da_all[0] 
            da_all.append(da)

    da_combined = xr.concat(da_all[1:], dim='model')
    v_low = da_combined.quantile(0.05).item()
    v_high = da_combined.quantile(0.95).item()
    limit = max(abs(v_low), abs(v_high))
    vmin = -limit
    vmax = limit

    for i, model in enumerate(model_list):
        da = da_all[i]

        combi = (model, verification_window)
        if model == cfg.ref_model:
            cfg_ref, _ = ref_cfg_layout(cfg, ref_model=model, verification_window=verification_window)
            case = make_case(Case, combi, vars(cfg_ref))
        else:
            case = make_case(Case, combi, vars(cfg))

        case_cfg = {**asdict(case), **asdict(setting)}

        if i > 0:
            show_ylabel, title, cmap = False, '$\Delta$'+unit, 'RdBu_r'
        else:
            show_ylabel, title, cmap = True, unit, 'YlOrRd'

        log.debug("%s %s %s %s", i, show_ylabel, title, cmap)

        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        ax.set_anchor('N') # Pushes the map to the TOP of the GridSpec cell

        if i > 0:
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = None, None

        fig, ax, im, _ = spatial_metrics_map(da, model, fig=fig, ax=ax, domain_mask=True, n_colors=0,
                                         show_ylabel=show_ylabel, cmap=cmap, title=title, panel=True, 
                                         text_scale=text_scale, vmin=vmin, vmax=vmax, **case_cfg)

        axes.append(ax)
        ims.append(im)
        i += 1

    cax_ref = fig.add_subplot(gs[1, 0])
    pos = cax_ref.get_position()
    cax_ref.set_position([pos.x0 + 0.2 * pos.width, pos.y0-0.03, pos.width * 0.6, pos.height])
    cbar_ref = fig.colorbar(ims[0], cax=cax_ref, orientation='horizontal')

    cax_mod = fig.add_subplot(gs[1, 1:])
    pos = cax_mod.get_position()

    if n > 2:
        cax_mod.set_position([pos.x0+pos.width*0.25, pos.y0-0.03, pos.width*0.5, pos.height]) # shrink width by 20% on each side
    else:
        cax_mod.set_position([pos.x0 + 0.2 * pos.width, pos.y0-0.03, pos.width * 0.6, pos.height])

    cbar_mod = fig.colorbar(ims[1], cax=cax_mod, orientation='horizontal', extend='both')

    for cb in (cbar_ref, cbar_mod):
        cb.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for cb in (cbar_ref, cbar_mod):
        if isinstance(cb.norm, (mcolors.BoundaryNorm,mcolors.Normalize)):
            if cb == cbar_ref:
                vmin = da_all[0].quantile(0.1).item()
                vmax = da_all[0].quantile(0.9).item()
            else:
                vmin = -limit
                vmax = limit

            # Create 7 evenly spaced values between min and max
            # This ignores the "bins" and looks at the actual numbers
            tick_locs_to_show = np.linspace(vmin, vmax, 7).astype(int)

            cb.set_ticks(tick_locs_to_show)
            cb.set_ticklabels([str(v) for v in tick_locs_to_show])

            cb.ax.xaxis.set_minor_locator(plt.NullLocator())

            cb.ax.tick_params(labelsize=7, direction='in', length=1.5)

    fig.suptitle(f"{var_name} {window_str} day forecast", fontsize=12, y=0.98)

    plot_filename = f"map_{var_name}_{window_str}.png"
    plot_path = os.path.join(cfg.dir_fig, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    log.debug(f"Figure saved to: {plot_path}")

    plt.show()

    return fig, axes, ims


if __name__ == "__main__":

    from dataclasses import asdict
    import xarray as xr
    from momp.lib.control import make_case
    from momp.lib.control import ref_cfg_layout
    from momp.lib.convention import Case
    from momp.lib.loader import get_cfg, get_setting
    from momp.graphics.func_map import spatial_metrics_map
    from momp.utils.printing import tuple_to_str
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.ticker import MaxNLocator
    from matplotlib.gridspec import GridSpec
    from matplotlib import colors as mcolors

    cfg, setting = get_cfg(), get_setting()

    model_list = cfg.model_list
    model_list = (cfg.ref_model,) + model_list
    verification_window = cfg.verification_window_list[0]
    log.debug(verification_window)
    window_str = tuple_to_str(verification_window)

    var_name = "mean_mae"

    n = len(model_list)
    fig = plt.figure(figsize=(8, 5/4*n))
    axes = []
    ims = []
    fig_width, fig_height = fig.get_size_inches()
    text_scale =  min(fig_width/n, fig_height/1) / 5 * 1.2

    gs = GridSpec(
        2, n,
        height_ratios=[1, 0.043],   # plots, then colorbars
        hspace=0.02,
        wspace=0.07
    )

    for i, model in enumerate(model_list):
        fi = os.path.join(cfg.dir_out, file_name(vars(cfg), "spatial", f"{model}_{window_str}"))
        ds = xr.open_dataset(fi)
        da = ds[var_name]
        ds.close()

        combi = (model, verification_window)
        if model == cfg.ref_model:
            cfg_ref, _ = ref_cfg_layout(cfg, ref_model=model, verification_window=verification_window)
            case = make_case(Case, combi, vars(cfg_ref))
        else:
            case = make_case(Case, combi, vars(cfg))

        case_cfg = {**asdict(case), **asdict(setting)}

        if i > 0:
            show_ylabel, title, cmap = False, None, 'RdBu_r'
        else:
            show_ylabel, title, cmap = True, f"{var_name} {window_str} day", 'YlOrRd'

        log.debug("%s %s %s %s", i, show_ylabel, title, cmap)

        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())

        fig, ax, im, _ = spatial_metrics_map(da, model, fig=fig, ax=ax, domain_mask=True, 
                                         show_ylabel=show_ylabel, cmap=cmap, title=title, panel=True, 
                                         text_scale=text_scale, **case_cfg)

        axes.append(ax)
        ims.append(im)
        i += 1

    cax_ref = fig.add_subplot(gs[1, 0])
    cbar_ref = fig.colorbar(
        ims[0],
        cax=cax_ref,
        orientation='horizontal'
    )

    gs_cell = gs[1, 1:]  # full width across remaining panels
    pos = gs_cell.get_position(fig)  # bounding box of this cell
    # shrink width by 10% on each side
    cax_mod = fig.add_axes([pos.x0 + 0.1, pos.y0, pos.width * 0.6, pos.height])

    cbar_mod = fig.colorbar(
        ims[1],
        cax=cax_mod,
        orientation='horizontal'
    )

    for cb in (cbar_ref, cbar_mod):
        cb.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for i, label in enumerate(cb.ax.get_xticklabels()):
            if i % 2 == 1:
                label.set_visible(False)

    for cb in (cbar_ref, cbar_mod):
        # Access the bins from BoundaryNorm
        if isinstance(cb.norm, mcolors.BoundaryNorm):
            boundaries = cb.norm.boundaries  # the levels
            # Compute centers of each bin
            tick_locs = 0.5 * (boundaries[:-1] + boundaries[1:])
            tick_locs_to_show = tick_locs[::2]
            cb.set_ticks(tick_locs_to_show)

            cb.ax.xaxis.set_minor_locator(plt.NullLocator())
            tick_labels = [str(int(i)) for i in np.arange(len(tick_locs_to_show))]

            cb.set_ticklabels(tick_labels)
            cb.ax.tick_params(labelsize=7, direction='in', length=1.5)

    fig.suptitle(f"{var_name} (in days) {window_str} day forecast", fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.99, 1])
    plt.show()

    plot_filename = f"map_{var_name}_{model_name}_{window_str}.png"
    plot_path = os.path.join(cfg.dir_fig, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    log.debug(f"Figure saved to: {plot_path}")
