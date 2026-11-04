"""Reliability diagram of forecast probabilities against observed onset frequency.

`plot_reliability_diagram` bins the forecast-observation pairs of one case into ten probability bins and draws the
observed frequency of each, with the forecast frequency on a log axis.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from hcwf.runs.folder import file_name
from hcwf.naming import tuple_to_str
import logging

__all__ = ["plot_reliability_diagram"]


log = logging.getLogger(__name__)

def plot_reliability_diagram(pairs, case, spec):
    """Plot the reliability diagram of one case from its forecast-observation pairs.

    The pairs are put into ten equal-width probability bins (the first includes 0). The figure is saved to
    ``spec.dir_fig`` if ``spec.save_fig`` is true (300 dpi) and shown if ``spec.show_plot`` is true.

    Parameters
    ----------
    pairs : object
        The ``PairArrays`` of the case window's bins: ``prob`` (forecast probability) and ``observed`` (0 or 1)
        arrays of the same shape.
    case : object
        The case drawn; ``case.model.name`` and ``case.window.span`` are named in the file.
    spec : object
        The run's `~hcwf.config.model.FigureSpec`: ``save_fig``, ``dir_fig`` and ``show_plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The reliability axes (the frequency bars are on a twin axis).
    results_df : pandas.DataFrame
        One row per probability bin: ``Bin_Range``, ``N_Forecasts``, ``Mean_Forecast_Prob``, ``Observed_Frequency``,
        ``Frequency`` and ``Error_Bar``, rounded to three decimals.
    """
    model, verification_window = case.model.name, case.window.span

    predicted, observed = pairs.prob.ravel(), pairs.observed.ravel()

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    reliability_y = np.zeros(n_bins)
    mean_forecast_prob = np.zeros(n_bins)
    frequency = np.zeros(n_bins)
    n_forecasts_array = np.zeros(n_bins)

    log.debug("\nReliability Analysis:")
    log.debug("Bin Range\t\tN_Forecasts\tMean_Forecast_Prob\tReliability\tFrequency\tError_Bar")
    log.debug("-" * 90)

    results_for_csv = []

    for i in range(n_bins):
        if i == 0:
            in_bin = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i+1])
        else:
            in_bin = (predicted > bin_edges[i]) & (predicted <= bin_edges[i+1])

        n_forecasts = in_bin.sum()
        n_forecasts_array[i] = n_forecasts

        if n_forecasts > 0:
            mean_forecast_prob[i] = predicted[in_bin].mean()
            reliability_y[i] = observed[in_bin].mean()
            frequency[i] = n_forecasts / len(predicted)
            error_bar = np.sqrt(reliability_y[i] * (1 - reliability_y[i]) / n_forecasts)
        else:
            mean_forecast_prob[i] = np.nan
            reliability_y[i] = np.nan
            frequency[i] = 0
            error_bar = np.nan

        bin_range = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"

        log.debug(f"{bin_range}\t\t{n_forecasts}\t\t{mean_forecast_prob[i]:.3f}\t\t\t{reliability_y[i]:.3f}\t\t{frequency[i]:.3f}\t\t{error_bar:.3f}")

        results_for_csv.append({
            'Bin_Range': bin_range,
            'N_Forecasts': n_forecasts,
            'Mean_Forecast_Prob': round(mean_forecast_prob[i], 3) if not np.isnan(mean_forecast_prob[i]) else np.nan,
            'Observed_Frequency': round(reliability_y[i], 3) if not np.isnan(reliability_y[i]) else np.nan,
            'Frequency': round(frequency[i], 3),
            'Error_Bar': round(error_bar, 3) if not np.isnan(error_bar) else np.nan
        })

    results_df = pd.DataFrame(results_for_csv)

    error_bars = np.sqrt(reliability_y * (1 - reliability_y) / n_forecasts_array)
    error_bars = np.where(n_forecasts_array > 0, error_bars, 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    valid_bins = ~np.isnan(reliability_y) & ~np.isnan(mean_forecast_prob)
    ax.errorbar(mean_forecast_prob[valid_bins], reliability_y[valid_bins],
                yerr=error_bars[valid_bins], fmt='o-',
                color='blue', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='Reliability')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Reliability')

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.bar(bin_centers, frequency, width=0.08, alpha=0.3, color='gray', label='Frequency')
    max_freq = max(frequency)
    min_freq = min([f for f in frequency if f > 0]) if any(f > 0 for f in frequency) else 1e-4
    ax2.set_ylim(min_freq * 0.5, max_freq * 2)
    ax2.set_ylabel('Forecast frequency', fontsize=12)

    ax.set_xlabel('Forecast Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if spec.save_fig:
        os.makedirs(spec.dir_fig, exist_ok=True)
        window_str = tuple_to_str(verification_window)
        fig_fn = os.path.join(spec.dir_fig, file_name(None, "reliability", model, window_str))
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        log.debug(f"Figure saved to: {fig_fn}")

    plt.tight_layout()
    if spec.show_plot:
        plt.show()

    return fig, ax, results_df
