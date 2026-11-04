"""Side-by-side portrait (heatmap) panels of binned BSS and AUC for each model.

`panel_portrait_bss_auc` draws two `hcwf.figures.visual.portrait_plot` panels with a climatology row from the binned
scores; `extract_binned_dict` turns those into arrays.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from hcwf.figures.visual import portrait_plot
from hcwf.runs.folder import file_name
from hcwf.naming import tuple_to_str
import logging

log = logging.getLogger(__name__)


def extract_binned_dict(dic, second_level_key, bin_key="clean_bins"):
    """Extract one binned score per model from a 2-level nested dictionary.

    The binned scores are saved as lists without bin keys; the bins are named by the list under ``bin_key``.

    Parameters
    ----------
    dic : dict of str to dict
        The binned scores per model: first-level keys are the models, second-level keys the score names.
    second_level_key : str
        The score to extract, e.g. ``"Fair_Brier_Skill_Score"``; each entry is a sequence over the bins.
    bin_key : str, default "clean_bins"
        The second-level key holding the bin labels, e.g. ``[(1, 5), (6, 10)]``; read from the first model.

    Returns
    -------
    array_2d : numpy.ndarray, shape (model, bin)
        The score of each model in each bin.
    first_level_keys : list of str
        The first-level keys (the models).
    bins : list or None
        The bin labels; None when ``bin_key`` is absent.
    """
    # 1st-level keys
    first_level_keys = list(dic.keys())

    # binned skill scores are saved as a list without bin keys
    # use "clean_bins" key to indicate bins
    second_level_value_as_keys = next(iter(dic.values())).get(bin_key)

    # Build 2-D array (rows = 1st-level, columns = 3rd-level)
    array_2d = np.array([
        dic[first][second_level_key]
        for first in first_level_keys
    ])
    return array_2d, first_level_keys, second_level_value_as_keys


def panel_portrait_bss_auc(result_binned, verification_window, spec):
    """Draw portrait plots of binned BSS (percent) and AUC, with a climatology row, for each model.

    The figure is shown if ``spec.show_panel`` is true and always saved to ``spec.dir_fig`` (300 dpi).

    Parameters
    ----------
    result_binned : dict of str to dict
        The binned scores per model, each with ``Fair_Brier_Skill_Score``, ``AUC``, ``AUC_ref`` and ``clean_bins``.
        The climatology AUC row is read from the first model.
    verification_window : tuple of (int, int)
        The verification window, in days; named in the file.
    spec : object
        The run's ``FigureSpec``: ``show_panel`` and ``dir_fig``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : tuple of (matplotlib.axes.Axes, matplotlib.axes.Axes)
        The BSS and AUC panels.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # load binned BSS, add climatology on top row
    arr, row_labels, col_labels = extract_binned_dict(result_binned, 'Fair_Brier_Skill_Score')
    data = np.vstack([np.zeros(arr.shape[1]), arr])
    data *= 100
    row_labels = ["Climatology"] + row_labels

    fig, ax1, im = portrait_plot(data, col_labels, row_labels, fig=fig, ax=ax1, 
                                 annotate=True, annotate_data=data, title=r'(a) Brier Skill Score ($\%$)', 
                                 colorbar_off=True)
    ax1.set_xlabel('Forecast window (days)')

    # load binned AUC, add climatology on top row
    arr, row_labels, col_labels = extract_binned_dict(result_binned, 'AUC')
    arr_clim, _, _ = extract_binned_dict(result_binned, 'AUC_ref')
    log.debug("%s %s", "arr = ", arr)
    log.debug("%s %s", "arr_clim = ", arr_clim)
    data = np.vstack([arr_clim[0], arr])
    row_labels = ["Climatology"] + row_labels

    fig, ax2, im = portrait_plot(data, col_labels, row_labels, fig=fig, ax=ax2, 
                                 annotate=True, annotate_data=data, title='(b) AUC', 
                                 colorbar_off=True)
    ax2.set_xlabel('Forecast window (days)')

    fig.tight_layout()
    if spec.show_panel:
        plt.show()

    # save figure
    window_str = tuple_to_str(verification_window)
    figure_filename = os.path.join(spec.dir_fig, file_name(None, "panel_skill", window_str))
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    log.debug(f"Figure saved as '{figure_filename}'")

    return fig, (ax1, ax2)
