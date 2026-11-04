"""Horizontal bar panel comparing BSS, RPSS and AUC across models for one verification window.

`panel_bar_bss_rpss_auc` draws the panel from the overall scores of the models, which `extract_overall_dict` turns
into arrays; the file is written to the run's figure folder.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from hcwf.runs.folder import file_name
from hcwf.naming import tuple_to_str
import logging

log = logging.getLogger(__name__)


def extract_overall_dict(dic, second_level_key):
    """Extract one value per entry of a 2-level nested dictionary.

    Parameters
    ----------
    dic : dict of str to dict
        The scores per model: first-level keys are the models, second-level keys the score names.
    second_level_key : str
        The score to extract from every first-level entry.

    Returns
    -------
    array_1d : numpy.ndarray, shape (model,)
        The value of ``second_level_key`` of each first-level entry, in the order of the keys.
    first_level_keys : list of str
        The first-level keys (the models).
    """
    # 1st-level keys
    first_level_keys = list(dic.keys())

    # 1-D array of values for the chosen 2nd-level key
    array_1d = np.array([dic[first][second_level_key] for first in first_level_keys])

    return array_1d, first_level_keys


def panel_bar_bss_rpss_auc(result_overall, verification_window, spec, *, legend=True):
    """Draw horizontal bars of BSS, RPSS (bottom axis, percent) and AUC (top axis) for each model.

    The climatology AUC is drawn as a vertical line. The figure is shown if ``spec.show_panel`` is true and always
    saved to ``spec.dir_fig`` (300 dpi).

    Parameters
    ----------
    result_overall : dict of str to dict
        The overall scores per model, each with ``Fair_Brier_Skill_Score``, ``Fair_RPS_Skill_Score``, ``AUC`` and
        ``AUC_ref`` (scalars; the climatology AUC is read from the first model).
    verification_window : tuple of (int, int)
        The verification window, in days; named in the title and the file.
    spec : object
        The run's `~hcwf.config.model.FigureSpec`: ``show_panel`` and ``dir_fig``.
    legend : bool, default True
        Draw the legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : tuple of (matplotlib.axes.Axes, matplotlib.axes.Axes)
        The bottom axis (skill scores) and the top axis (AUC).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    window = verification_window
    title = f"{tuple_to_str(window)} day forecast"

    # load binned BSS, add climatology on top row
    bss, model_list  = extract_overall_dict(result_overall, 'Fair_Brier_Skill_Score')
    rps, _  = extract_overall_dict(result_overall, 'Fair_RPS_Skill_Score')
    auc, _  = extract_overall_dict(result_overall, 'AUC')
    log.debug("result_overall = %s", result_overall)
    auc_ref, _  = extract_overall_dict(result_overall, 'AUC_ref')
    auc_ref = auc_ref[0]

    bss *= 100
    rps *= 100

    # Create second x-axis (top)
    ax2 = ax.twiny()

    # Bar height
    height = 0.2

    y_pos = np.arange(len(model_list))

    # Add colors
    auc_col = np.array([217, 95, 14]) / 256
    rpss_col = np.array([33, 102, 172]) / 256
    bss_col = np.array([146, 197, 222]) / 256

    # Plot AUC bars on top
    ax2.barh(y_pos + height, auc, height,
                    label='AUC', alpha=0.8, color=auc_col)

    # Plot skill scores on bottom axis
    ax.barh(y_pos, bss, height,
                   label='BSS', alpha=0.8, color=bss_col)
    ax.barh(y_pos - height, rps, height,
                   label='RPSS', alpha=0.8, color=rpss_col)

    # Plot Climatology AUC as vertical line
    ax2.axvline(x=auc_ref, color=auc_col, linestyle='-', linewidth=1.25, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.25, alpha=0.8)

    # Customize axes
    ax.set_xlabel(r'BSS/RPSS ($\%$)', fontsize=12)
    ax.set_title(f'{title}', fontsize=15, fontweight='normal', loc='left')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_list, rotation=0, ha='right', fontsize=15)

    min_val = np.min(np.concatenate([rps, bss, auc])) *1.05
    ax.set_xlim(min_val, 25) 
    ax.set_ylim(-0.5, 1.5)

    ax2.set_xlabel('AUC', fontsize=12, color=auc_col)
    ax2.tick_params(axis='x', colors=auc_col)
    ax2.spines['top'].set_color(auc_col)
    ax2.set_xlim(0.6, 1.0)
    ax2.set_xticks(np.arange(0.6, 1.02, 0.05))

    # Control tick label visibility
    if not isinstance(ax, np.ndarray): # single panel plot
        ax.tick_params(labelbottom=True, labeltop=False)
        ax2.tick_params(labelbottom=False, labeltop=True)
        ax.tick_params(labelleft=True)
    else:
        ax.tick_params(labelbottom=True, labeltop=False)
        ax2.tick_params(labeltop=True)
        ax.tick_params(labelleft=False)

    clim_line = plt.Line2D([0], [0], color=auc_col, linestyle='-', linewidth=1.25,
                              label='Climatology')

    # Add legends - only for panel 2
    if legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + [clim_line], labels1 + labels2 + ['climatology'],
                  loc='lower right', frameon=False)

    fig.tight_layout()
    if spec.show_panel:
        plt.show()

    # save figure
    window_str = tuple_to_str(verification_window)
    figure_filename = os.path.join(spec.dir_fig, file_name(None, "panel_bar", window_str))
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    log.debug(f"Figure saved as '{figure_filename}'")

    return fig, (ax, ax2)
