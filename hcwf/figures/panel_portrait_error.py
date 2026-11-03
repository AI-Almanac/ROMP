"""Portrait panels of MAE, FAR and MR differences from climatology, per model and forecast window."""
import os
import numpy as np
import matplotlib.pyplot as plt
from hcwf.runs.folder import file_name
from hcwf.figures.visual import portrait_plot
import logging

log = logging.getLogger(__name__)


def nested_dict_to_array(nested_dict, metric_key):
    """
    Convert a 2-level nested dict to a 2D NumPy array based on the first two levels.

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary with at least two levels.
        Example: results[model][window]['mae']
    metric_key : str
        The key at the leaf level to extract (e.g., 'mae').

    Returns
    -------
    arr : np.ndarray
        2D array with shape (num_top_keys, num_second_keys)
    row_labels : list
        List of top-level keys (rows)
    col_labels : list
        List of second-level keys (columns)
    """
    # Top-level keys → rows
    row_labels = list(nested_dict.keys())

    # Second-level keys → columns (assume all rows share same keys)
    first_row = nested_dict[row_labels[0]]
    col_labels = list(first_row.keys())

    # Initialize array
    arr = np.full((len(row_labels), len(col_labels)), np.nan)

    # Fill array with metric values
    for i, rkey in enumerate(row_labels):
        for j, ckey in enumerate(col_labels):
            # Safely get the value (default to np.nan if missing)
            da = nested_dict.get(rkey, {}).get(ckey, {}).get(metric_key, np.nan)
            arr[i,j] = da.mean()

    return arr, row_labels, col_labels


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
