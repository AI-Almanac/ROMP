"""Portrait panels of MAE, FAR and MR differences from climatology, per model and forecast window."""
import os
import numpy as np
import matplotlib.pyplot as plt
from momp.io.output import nested_dict_to_array
from momp.utils.visual import portrait_plot


def panel_portrait_mae_far_mr(results, *, dir_fig, show_panel=True, **kwargs):
    """
    Draw portrait plots of mean MAE, FAR and MR as differences from the last row (climatology).

    Each metric is taken from ``results`` via ``nested_dict_to_array``; the last row is subtracted
    from the others. The figure is shown if ``show_panel`` is True and always saved to ``dir_fig``,
    named from ``kwargs['model_list'][0]`` and ``kwargs['max_forecast_day']``.
    Returns ``(fig, (ax1, ax2, ax3))``.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

    # load mean mae data and calculate ananomly wrt cimatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "mean_mae")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data))
    vrange = (-v, v)

    fig, ax1, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax1, vrange=vrange,
                                 annotate=True, annotate_data=data, title='$\Delta MAE$ (days)', colorbar_off=True)

    # anomaly false alarm rate relative to climatology 
    arr, row_labels, col_labels = nested_dict_to_array(results, "false_alarm_rate")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data))
    vrange = (-v, v)

    fig, ax2, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax2, vrange=vrange,
                                 annotate=True, annotate_data=data, title='$\Delta FAR (\%)$', colorbar_off=True)

    ax2.set_xlabel('Forecast window (days)')

    # anomaly miss rate relative to climatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "miss_rate")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data))
    vrange = (-v, v)

    fig, ax3, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax3, vrange=vrange,
                                 annotate=True, annotate_data=data, title='$\Delta MR (\%)$', colorbar_off=True)

    fig.tight_layout()
    if show_panel:
        plt.show()

    # save figure
    first_model = kwargs.get('model_list')[0]
    figure_filename = f"panel_portrait_mae_far_mr_{first_model}_{kwargs['max_forecast_day']}day.png"
    figure_filename = os.path.join(dir_fig, figure_filename)
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{figure_filename}'")

    return fig, (ax1, ax2, ax3)


if __name__ == "__main__":

    from itertools import product
    from momp.lib.control import iter_list, make_case
    from momp.utils.printing import tuple_to_str
    from momp.lib.convention import Case
    from momp.io.output import set_nested
    from momp.io.output import nested_dict_to_array
    import xarray as xr
    import copy

    cfg, setting = get_cfg(), get_setting()

    result = {}
    var_list = ["mean_mae", "false_alarm_rate", "miss_rate"]

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        window_bin_str = tuple_to_str(case.verification_window) 

        fi = os.path.join(cfg.dir_out,"spatial_metrics_{}_{}.nc")
        fi = fi.format(case.model, window_bin_str)
        ds = xr.open_dataset(fi)

        spatial_metrics_dict = {var: ds[var] for var in var_list}

        result = set_nested(result, combi, spatial_metrics_dict)

        ds.close()

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    layout_pool = iter_list(vars(cfg_ref))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))

        window_bin_str = tuple_to_str(case.verification_window) 

        fi = os.path.join(cfg.dir_out,"spatial_metrics_{}_{}.nc")
        fi = fi.format(case.ref_model, window_bin_str)

        if result.get(case.ref_model, {}).get(window_bin_str) is None:
            ds = xr.open_dataset(fi)
            spatial_metrics_dict = {var: ds[var] for var in var_list}

            result = set_nested(result, combi, spatial_metrics_dict)

            ds.close() 

    panel_portrait_mae_far_mr(result, **vars(cfg))
