"""Heatmap of forecast-bin Brier skill score and AUC for one model."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from momp.stats.bins import get_target_bins
from momp.utils.printing import tuple_to_str
import logging

log = logging.getLogger(__name__)


def create_heatmap(score_results, *, model, verification_window, day_bins, dir_fig, show_plot=True, **kwargs):
    """Create and save skill score heatmap"""
    auc_forecast =score_results['AUC']
    auc_climatology = score_results['AUC_ref']
    brier_forecast =score_results['BS']
    brier_climatology = score_results['BS_ref']
    skill_results = score_results['skill_results']

    target_bins = get_target_bins(brier_forecast, brier_climatology)

    # Prepare data
    bss_values = [skill_results['bin_fair_brier_skill_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    auc_values = [auc_forecast['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    auc_clim_values = [auc_climatology['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins]

    bin_labels_short = [bin_name.replace('Days ', '') for bin_name in target_bins]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))

    # Plot 1: BSS heatmap
    bss_data = np.array(bss_values).reshape(1, -1)
    sns.heatmap(bss_data*100,
                annot=True,
                fmt='.2g',
                cmap='RdBu',
                vmin=-40, vmax=40,
                center=0,
                xticklabels=bin_labels_short,
                cbar=False,
                ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold'})

    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    ax1.set_ylabel('BSS (%)', fontsize=14)
    ax1.set_yticklabels([])

    # Plot 2: AUC heatmap
    auc_data = np.array(auc_values).reshape(1, -1)
    sns.heatmap(auc_data,
                annot=False,
                cmap='Blues',
                vmin=0.7, vmax=1.0,
                xticklabels=bin_labels_short,
                cbar=False,
                ax=ax2)
    # Add custom annotations
    for i, (auc_val, auc_clim_val) in enumerate(zip(auc_values, auc_clim_values)):
        if not np.isnan(auc_val) and not np.isnan(auc_clim_val):
            ax2.text(i + 0.5, 0.5, f'{auc_val:.2g}',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black')
            ax2.text(i + 0.5, 0.2, f'({auc_clim_val:.2g})',
                    ha='center', va='center',
                    fontsize=8, color='darkblue')
        elif not np.isnan(auc_val):
            ax2.text(i + 0.5, 0.5, f'{auc_val:.2g}',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black')

    ax2.set_xlabel('Forecast Day Bins', fontsize=14)
    ax2.set_ylabel('AUC', fontsize=14)
    ax2.set_yticklabels([])

    plt.tight_layout()

    # Save with model name and forecast days
    window_str = tuple_to_str(verification_window)
    figure_filename = f'skill_scores_heatmap_{model}_{window_str}.png'
    figure_filename = os.path.join(dir_fig, figure_filename)
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close()

    log.debug(f"Figure saved as '{figure_filename}'")
