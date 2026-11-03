"""Side-by-side portrait (heatmap) panels of binned BSS and AUC for each model."""
import os
import numpy as np
import matplotlib.pyplot as plt
from hcwf.utils.visual import portrait_plot
from hcwf.io.dict import extract_binned_dict
from hcwf.lib.run_folder import file_name
from hcwf.utils.printing import tuple_to_str
import logging

log = logging.getLogger(__name__)


def panel_portrait_bss_auc(result_binned, verification_window, spec):
    """
    Draw portrait plots of binned BSS (%) and AUC, with a climatology row, for each model.

    The figure is shown if ``spec.show_panel`` is True and always saved to ``spec.dir_fig``.
    Returns ``(fig, (ax1, ax2))``.
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
