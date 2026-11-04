"""The tables of one case's scores, without writing anything: the overall and binned skill scores, and the map of back-dated cases."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.metrics.error import year_points
from hcwf.metrics.score import get_target_bins
from hcwf.spatial import axes

log = logging.getLogger(__name__)


def score_tables(score_results):
    """Return the overall and binned skill-score tables of one case, without writing anything.

    Parameters
    ----------
    score_results : dict
        The results of `hcwf.metrics.skill.create_score_results`, with all of ``BS``, ``RPS``, ``AUC``, ``BS_ref``,
        ``AUC_ref`` and ``skill_results`` (so skill scores must have been asked for).

    Returns
    -------
    overall_df : pandas.DataFrame
        One row with the fair Brier score and skill score, the fair RPS and skill score, ``AUC`` and ``AUC_ref``, and
        ``back_dated_member_share`` when the forecast pairs track back-dated members.
    binned_df : pandas.DataFrame
        One row per ``Days a-b`` bin both the forecast and the reference score: the Brier skill score, the AUCs and
        the Brier scores of the forecast and of the climatology.
    binned_data : dict
        The columns of `binned_df` as lists.
    overall_scores_nested : dict of str to float
        The overall scores as ``{metric: value}``.
    """
    # Save overall scores
    overall_scores = {
        'Metric': ['Fair_Brier_Score', 'Fair_Brier_Skill_Score', 'Fair_RPS', 'Fair_RPS_Skill_Score', 'AUC', 'AUC_ref'],
        'Score': [
            score_results['BS']['fair_brier_score'],
            score_results['skill_results']['fair_brier_skill_score'],
            score_results['RPS']['fair_rps'],
            score_results['skill_results']['fair_rps_skill_score'],
            score_results['AUC']['auc'],
            score_results['AUC_ref']['auc'],
        ]
    }

    overall_scores_nested = dict(zip(overall_scores['Metric'], overall_scores['Score']))

    # pd.DataFrame expects lists/arrays as values, or a list of dicts for rows, not a dict of scalars
    # Wrap the dict in a list so pandas treats it as a single row, or make each value wrapped in a list []

    # create dataframe and save as .csv
    overall_df = pd.DataFrame([overall_scores_nested])
    share = back_dated_member_share(score_results.get('forecast_pairs'))
    if share is not None:
        # only with forecast_history "observed"; "none" runs keep their columns
        overall_df['back_dated_member_share'] = share

    # Get target bins, remove bins with "Day Before" and "Day After"
    target_bins = get_target_bins(score_results['BS'], score_results['BS_ref'])
    clean_bins = [b.replace("Days ", "") for b in target_bins]

    # Save binned scores
    binned_data = {
        'Bin': target_bins,
        'clean_bins': clean_bins,
        'Fair_Brier_Skill_Score': [score_results['skill_results']['bin_fair_brier_skill_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'AUC': [score_results['AUC']['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'AUC_ref': [score_results['AUC_ref']['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Forecast': [score_results['BS']['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Climatology': [score_results['BS_ref']['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    }

    # create dataframe and save as .csv
    binned_df = pd.DataFrame(binned_data)
    return overall_df, binned_df, binned_data, overall_scores_nested


def back_dated_member_share(pairs):
    """Return the share of the scored member onsets dated on or before their init, or None when the pairs do not track them.

    Reads the per-case `back_dated` and `onset_members` counts of the forecast pairs (present for definition runs with
    ``forecast_history: observed``).

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays or None
        The forecast pairs, or None.

    Returns
    -------
    float or None
        The share, NaN when no member found an onset, None when `pairs` is None or has no `PairArrays.back_dated`.
    """
    if pairs is None or pairs.back_dated is None:
        return None
    onsets = pairs.onset_members.sum()
    return float(pairs.back_dated.sum() / onsets) if onsets else float('nan')


def add_back_dated_map(spatial_metrics, metrics_df_dict, onset_da_dict):
    """Add the ``n_back_dated`` map (in place) from per-year metrics tables that carry an ``n_back_dated`` column.

    Each cell holds the number of scored init cases, over the years with an observed onset there, whose forecast
    onset was back-dated; a no-op for tables without the column (``forecast_history: none`` runs).

    Parameters
    ----------
    spatial_metrics : dict of str to xarray.DataArray
        The maps of `hcwf.metrics.error.create_spatial_far_mr_mae`; ``n_back_dated`` is added to it.
    metrics_df_dict : dict of int to pandas.DataFrame
        Maps year to the per-grid-point metrics table.
    onset_da_dict : dict of int to xarray.DataArray
        Maps year to the observed onset; the grid of the first defines the map.

    Returns
    -------
    dict of str to xarray.DataArray
        `spatial_metrics`, the same object.
    """
    tables = [df for df in metrics_df_dict.values() if 'n_back_dated' in df.columns]
    if not tables:
        return spatial_metrics
    first = onset_da_dict[list(onset_da_dict)[0]]
    spatial = axes.spatial_dims(first.dims)
    coords = axes.coords_of(first, spatial)
    counts = np.zeros(tuple(len(coords[d]) for d in spatial))
    for year, df in metrics_df_dict.items():
        if 'n_back_dated' not in df.columns or len(df) == 0:
            continue
        points, at = year_points(df, onset_da_dict[year], spatial, coords)
        counts[at] += points['n_back_dated'].to_numpy()
    spatial_metrics['n_back_dated'] = xr.DataArray(
        counts, coords=[(d, coords[d]) for d in spatial], name='n_back_dated',
        attrs={'description': 'Scored init cases whose forecast onset was back-dated (on or before the init, '
                              'scored as the first lead); 0 unless the verification window starts at lead 1'})
    return spatial_metrics
