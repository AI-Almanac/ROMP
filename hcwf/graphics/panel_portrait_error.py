"""Portrait panels of MAE, FAR and MR differences from climatology, per model and forecast window."""
import os
import numpy as np
import matplotlib.pyplot as plt
from hcwf.io.output import nested_dict_to_array
from hcwf.lib.run_folder import file_name
from hcwf.utils.visual import portrait_plot
import logging

log = logging.getLogger(__name__)


def panel_portrait_mae_far_mr(results, spec):
    """
    Draw portrait plots of mean MAE, FAR and MR as differences from the last row (climatology).

    Each metric is taken from ``results`` via ``nested_dict_to_array``; the last row is subtracted
    from the others. The figure is shown if ``spec.show_panel`` is True and always saved to ``spec.dir_fig``,
    named from the run's first model and max_forecast_day.
    Returns ``(fig, (ax1, ax2, ax3))``.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

    # load mean mae data and calculate ananomly wrt cimatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "mean_mae")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data))
    vrange = (-v, v)

    fig, ax1, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax1, vrange=vrange,
                                 annotate=True, annotate_data=data, title=r'$\Delta MAE$ (days)', colorbar_off=True)

    # anomaly false alarm rate relative to climatology 
    arr, row_labels, col_labels = nested_dict_to_array(results, "false_alarm_rate")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data))
    vrange = (-v, v)

    fig, ax2, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax2, vrange=vrange,
                                 annotate=True, annotate_data=data, title=r'$\Delta FAR (\%)$', colorbar_off=True)

    ax2.set_xlabel('Forecast window (days)')

    # anomaly miss rate relative to climatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "miss_rate")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data))
    vrange = (-v, v)

    fig, ax3, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax3, vrange=vrange,
                                 annotate=True, annotate_data=data, title=r'$\Delta MR (\%)$', colorbar_off=True)

    fig.tight_layout()
    if spec.show_panel:
        plt.show()

    # save figure
    figure_filename = file_name(None, "panel_error", spec.models[0], spec.max_forecast_day)
    figure_filename = os.path.join(spec.dir_fig, figure_filename)
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    log.debug(f"Figure saved as '{figure_filename}'")

    return fig, (ax1, ax2, ax3)
