"""Brier score, RPS, AUC and skill-score calculations for binned onset forecast-observation pairs.

The pair tables (one row per forecast case and bin) are reshaped into (case, bin) matrices and scored by
hcwf.metrics.probabilistic; this module only keeps HCWF's result layout (dicts keyed by bin label) and its guards.
"""

import numpy as np
import pandas as pd
import warnings
from hcwf.metrics import probabilistic
from hcwf.spatial import axes
from hcwf.stats.bins import get_target_bins
import logging

log = logging.getLogger(__name__)


def _pair_matrices(df, members_column=None):
    """(P, Y, members, labels): the probabilities, outcomes and member counts of a pair table as (case, bin) matrices
    (members None without a members_column).

    Rows keep their order, so the table must hold every bin, in the same order, for each case (as the pair
    builders write it); a table in another order is sorted by case and bin first. labels are the bin labels
    in order of appearance.
    """
    codes, labels = pd.factorize(df['bin_label'])
    n_bins = len(labels)
    regular = lambda c: len(c) % n_bins == 0 and bool((c.reshape(-1, n_bins) == np.arange(n_bins)).all())
    order = None
    if not regular(codes):
        case_columns = ['init_time', *axes.spatial_dims(df.columns)]
        if not {*case_columns, 'bin_index'} <= set(df.columns):
            raise ValueError("the pair table does not hold every bin, in the same order, for each case")
        case = df.groupby(case_columns, sort=False).ngroup().to_numpy()
        order = np.lexsort((df['bin_index'].to_numpy(), case))
        codes, labels = pd.factorize(df['bin_label'].to_numpy()[order])
        if not regular(codes):
            raise ValueError("the pair table does not hold the same bins for each case")

    def matrix(column):
        values = df[column].to_numpy()
        return (values if order is None else values[order]).reshape(-1, n_bins)

    members = None if members_column is None else matrix(members_column)
    return matrix('predicted_prob'), matrix('observed_onset'), members, list(labels)


def _brier_results(df, members_column):
    """Brier results of a pair table in HCWF's layout: overall and per-bin Brier and fair Brier scores."""
    P, Y, members, labels = _pair_matrices(df, members_column)
    scores = probabilistic.brier_scores(P, Y, members, scale="element", by_bin=True, bin_mean="compensated", skipna=True)
    by_label = sorted(range(len(labels)), key=lambda j: labels[j])
    return {
        'brier_score': scores['brier'],
        'fair_brier_score': scores['fair_brier'],
        'bin_brier_scores': {labels[j]: float(scores['bin_brier'][j]) for j in by_label},
        'bin_fair_brier_scores': {labels[j]: float(scores['bin_fair_brier'][j]) for j in by_label},
    }


def _require_members(forecast_obs_df, what):
    """Raise unless the first case has more than one ensemble member (the fair scores divide by members - 1)."""
    n_members = forecast_obs_df['total_members'].iloc[0] if len(forecast_obs_df) > 0 else 1
    if n_members <= 1:
        raise ValueError(
            f"Fair {what} requires >1 ensemble member, but found total_members={n_members}. "
            "Check that your model data contains ensemble members and that the 'members' / "
            "'probabilistic' parameters are configured correctly."
        )


def _without_before(forecast_obs_df):
    """The pair table without its "Before day N" bins."""
    return forecast_obs_df[~forecast_obs_df['bin_label'].str.contains('Before', na=False)].copy()


# Function to calculate Brier Score and Fair Brier Score for the model forecasts (both overall and bin-wise)
def calculate_brier_score(forecast_obs_df):
    """
    Calculate Brier Score and Fair Brier Score for probabilistic forecasts.

    Brier Score = (1/n*m) * Σ(Y_ij - p_ij)²
    Fair Brier Score = (1/n*m) * Σ[(Y_ij - p_ij)² - p_ij(1-p_ij)/(ens-1)]

    where:
    - n = number of forecasts
    - m = number of bins per forecast
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - ens = number of ensemble members

    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members', 'bin_label'

    Returns:
    --------
    dict with Brier score metrics
    """
    _require_members(forecast_obs_df, "Brier Score")

    brier_results = _brier_results(forecast_obs_df, 'total_members')

    log.debug(f"Brier Score: {brier_results['brier_score']:.4f}")
    log.debug(f"Fair Brier Score: {brier_results['fair_brier_score']:.4f}")

    return brier_results


def _auc_results(forecast_obs_df):
    """AUC results of a pair table in HCWF's layout: the overall AUC and the AUC of each bin (NaN where a bin has
    only positives or only negatives)."""
    P, Y, _, labels = _pair_matrices(forecast_obs_df)
    p_ij, y_ij = P.ravel(), Y.ravel()
    auc = probabilistic.auc(y_ij, p_ij)
    if np.isnan(auc):
        log.warning("Cannot calculate AUC - all cases are either positive or negative")
        return {
            'auc': np.nan,
            'n_positive': np.sum(y_ij),
            'n_negative': np.sum(1 - y_ij),
            'bin_auc_scores': {},
            'forecast_obs_df_with_ranks': forecast_obs_df
        }
    return {
        'auc': auc,
        'bin_auc_scores': {label: probabilistic.auc(Y[:, j], P[:, j]) for j, label in enumerate(labels)},
    }


# Function to calculate Area Under the Curve (AUC) for the model forecasts (both overall and bin-wise)
def calculate_auc(forecast_obs_df):
    """
    Calculate Area Under the Curve (AUC) for probabilistic forecasts.

    AUC = Σ_{i,j,i',j'} Y_{ij}(1-Y_{i'j'}) · 1[p_{ij} > p_{i'j'}] /
          [(Σ_{i,j} Y_{ij})(Σ_{i,j} (1-Y_{ij}))]

    where:
    - Y_{ij} = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_{ij} = predicted probability for bin j in forecast i
    - 1[p_{ij} > p_{i'j'}] = indicator function (1 if true, 0 if false)

    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'bin_label'

    Returns:
    --------
    dict with AUC metrics
    """
    return _auc_results(forecast_obs_df)


def _rps_matrices(forecast_obs_df, group_id, n_groups):
    """(P, Y, members, valid): the probabilities, outcomes and member count of each forecast case (group) as
    (case, bin) matrices in group order, rows ordered by bin_index within a case.

    Rows whose key is missing (group id -1) are dropped, as groupby does. valid is None when every case has the
    same number of bins; otherwise cases with fewer bins are padded with zeros at the end and valid marks the
    bins they have.
    """
    keep = group_id >= 0
    gid = group_id[keep]
    p = forecast_obs_df['predicted_prob'].to_numpy()[keep]
    y = forecast_obs_df['observed_onset'].to_numpy()[keep]
    members = forecast_obs_df['total_members'].to_numpy()[keep]
    order = np.lexsort((forecast_obs_df['bin_index'].to_numpy()[keep], gid))
    gid, p, y, members = gid[order], p[order], y[order], members[order]
    counts = np.bincount(gid, minlength=n_groups)
    starts = np.concatenate(([0], np.cumsum(counts)))
    if n_groups and (counts == counts[0]).all():
        m = int(counts[0])
        return (np.ascontiguousarray(p.reshape(n_groups, m)), np.ascontiguousarray(y.reshape(n_groups, m)),
                members.reshape(n_groups, m)[:, 0], None)
    m = int(counts.max()) if n_groups else 0
    position = np.arange(len(gid)) - starts[gid]
    P, Y = np.zeros((n_groups, m)), np.zeros((n_groups, m), dtype=y.dtype)
    P[gid, position], Y[gid, position] = p, y
    return P, Y, members[starts[:-1]], np.arange(m)[None, :] < counts[:, None]


# Function to calculate Ranked Probability Score (RPS) and Fair RPS for the model forecasts for (either 15 day or 30 day forecast)
def calculate_rps(forecast_obs_df):
    """
    Calculate Ranked Probability Score (RPS) and Fair RPS for probabilistic forecasts.

    RPS = (1/n*m) * Σ_i Σ_k (Σ_j≤k (Y_ij - p_ij))²
    Fair RPS = (1/n*m) * Σ_i Σ_k [(Σ_j≤k (Y_ij - p_ij))² - (Σ_j≤k p_ij)(1 - Σ_j≤k p_ij)/(ens-1)]

    where:
    - n = number of forecasts
    - m = number of bins per forecast
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - k = cumulative index (1 to m)
    - ens = number of ensemble members

    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members', 'bin_index'

    Returns:
    --------
    dict with RPS metrics
    """
    _require_members(forecast_obs_df, "RPS")

    # Group by forecast (init_time and the location: lat and lon, or unit) to get all bins for each forecast
    forecast_groups = forecast_obs_df.groupby(['init_time', *axes.spatial_dims(forecast_obs_df.columns)])
    group_id = forecast_groups.ngroup().to_numpy()
    n_forecasts = len(forecast_groups)

    P, Y, members, valid = _rps_matrices(forecast_obs_df, group_id, n_forecasts)
    rps_values, fair_rps_values = probabilistic.rps_cases(P, Y, members, summation="prefix", valid=valid)

    # Calculate overall RPS (average over all forecasts)
    rps = np.mean(rps_values)
    fair_rps = np.mean(fair_rps_values)

    rps_results = {
        'rps': rps,
        'fair_rps': fair_rps,
        'n_forecasts': int(n_forecasts),

    }

    log.debug(f"RPS: {rps_results['rps']:.4f}")
    log.debug(f"Fair RPS: {rps_results['fair_rps']:.4f}")
    log.debug(f"Number of forecasts: {rps_results['n_forecasts']}")

    return rps_results


def calculate_brier_score_climatology(forecast_obs_df):
    """
    Calculate Brier Score and Fair Brier Score of a climatology (or other pooled-member) reference.

    The scores of calculate_brier_score, over the bins that are not "Before" bins (e.g. "Before day N"), with the
    fair correction taken from each case's total_members_with_onset instead of total_members.

    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_climatological_forecast_obs_pairs()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members_with_onset', 'bin_label'

    Returns:
    --------
    dict with Brier score metrics
    """

    # Filter out "Before initialization" bin
    filtered_df = _without_before(forecast_obs_df)

    if len(filtered_df) == 0:
        log.warning("No data remaining after filtering out 'Before initialization' bin")
        return {
            'brier_score': np.nan,
            'fair_brier_score': np.nan,
            'bin_brier_scores': {},
            'bin_fair_brier_scores': {},
            'n_samples': 0,
            'filtered_bins': []
        }

    log.debug(f"Calculating Brier Score excluding 'Before initialization' bin")
    log.debug(f"Original samples: {len(forecast_obs_df)}, After filtering: {len(filtered_df)}")

    brier_results = _brier_results(filtered_df, 'total_members_with_onset')
    brier_results.update({
        'n_samples': len(filtered_df),
        'filtered_bins': sorted(filtered_df['bin_label'].unique()),
        'excluded_bins': ['Before initialization']
    })

    log.debug(f"Brier Score (excluding 'Before initialization'): {brier_results['brier_score']:.4f}")
    log.debug(f"Fair Brier Score (excluding 'Before initialization'): {brier_results['fair_brier_score']:.4f}")
    log.debug(f"Bins included in calculation: {brier_results['filtered_bins']}")

    return brier_results


def calculate_auc_climatology(forecast_obs_df):
    """
    Calculate Area Under the Curve (AUC) of a climatology reference: calculate_auc over the bins that are not
    "Before" bins.

    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_climatological_forecast_obs_pairs()
        Must contain columns: 'predicted_prob', 'observed_onset', 'bin_label'

    Returns:
    --------
    dict with AUC metrics
    """
    return _auc_results(_without_before(forecast_obs_df))


def calculate_skill_scores(brier_forecast, rps_forecast,
                          brier_climatology, rps_climatology):
    """
    Calculate skill scores for forecast model relative to climatology.

    Skill Score = 1 - (forecast_score / climatology_score)

    Parameters:
    -----------
    brier_forecast : dict
        Brier score results from forecast model
    rps_forecast : dict
        RPS results from forecast model
    brier_climatology : dict
        Brier score results from climatology
    rps_climatology : dict
        RPS results from climatology

    Returns:
    --------
    dict with skill scores
    """

    skill_scores = {}

    log.debug("="*60)
    log.debug("SKILL SCORE CALCULATIONS")
    log.debug("="*60)

    # Fair Brier Skill Score (1-15 day overall)
    if brier_climatology == 0:
        fair_bss_bin = float("nan")
        warnings.warn("\n brier_climatology is zero → fair BSS set to NaN \n")
    else:
        fair_bss_overall = probabilistic.skill_score(brier_forecast['fair_brier_score'], brier_climatology['fair_brier_score'])

    skill_scores['fair_brier_skill_score'] = fair_bss_overall

    log.debug(f"Fair Brier Skill Score (1-15 day): {fair_bss_overall:.4f}")

    # Fair RPS Skill Score (1-15 day overall)
    fair_rpss_overall = probabilistic.skill_score(rps_forecast['fair_rps'], rps_climatology['fair_rps'])
    skill_scores['fair_rps_skill_score'] = fair_rpss_overall

    log.debug(f"Fair RPS Skill Score (1-15 day): {fair_rpss_overall:.4f}")

    # Automatically extract target bins from the data: the "Days X-Y" bins both have, in day order
    target_bins = get_target_bins(brier_forecast, brier_climatology)
    common_bins = set(brier_forecast['bin_fair_brier_scores']) & set(brier_climatology['bin_fair_brier_scores'])
    excluded_bins = [bin_label for bin_label in common_bins if bin_label not in target_bins]

    log.debug(f"\nAutomatically detected target bins: {target_bins}")
    log.debug(f"Excluded bins: {excluded_bins}")

    # Bin-wise Fair Brier Skill Scores
    bin_fair_bss = {}

    log.debug(f"\nBin-wise Fair Brier Skill Scores:")
    for bin_label in target_bins:
        if bin_label in brier_forecast['bin_fair_brier_scores'] and bin_label in brier_climatology['bin_fair_brier_scores']:
            forecast_fair_brier_bin = brier_forecast['bin_fair_brier_scores'][bin_label]
            clim_fair_brier_bin = brier_climatology['bin_fair_brier_scores'][bin_label]

            fair_bss_bin = probabilistic.skill_score(forecast_fair_brier_bin, clim_fair_brier_bin)
            bin_fair_bss[bin_label] = fair_bss_bin

            log.debug(f"  {bin_label}: Fair BSS = {fair_bss_bin:.4f}")
        else:
            bin_fair_bss[bin_label] = np.nan
            log.debug(f"  {bin_label}: Fair BSS = NaN (missing data)")

    skill_scores['bin_fair_brier_skill_scores'] = bin_fair_bss

    # Dynamic table header based on detected bins
    header = f"{'Metric':<30} {'Overall (1-15 day)':<18}"
    for bin_name in target_bins:
        # Shorten bin names for table display
        short_name = bin_name.replace('Days ', '')
        header += f" {short_name:<12}"

    # Calculate table width
    table_width = 30 + 18 + 12 * len(target_bins)

    # Summary table
    log.debug(f"\n" + "="*table_width)
    log.debug("SKILL SCORE SUMMARY TABLE")
    log.debug("="*table_width)
    log.debug(header)
    log.debug("-"*table_width)

    # Fair Brier Skill Score row
    fair_bss_row = f"{'Fair Brier Skill Score':<30} {fair_bss_overall:<18.4f}"
    for bin_name in target_bins:
        if bin_name in bin_fair_bss and not pd.isna(bin_fair_bss[bin_name]):
            fair_bss_row += f" {bin_fair_bss[bin_name]:<12.4f}"
        else:
            fair_bss_row += f" {'N/A':<12}"
    log.debug(fair_bss_row)

    # Fair RPS Skill Score row
    fair_rpss_row = f"{'Fair RPS Skill Score':<30} {fair_rpss_overall:<18.4f}"
    for bin_name in target_bins:
        fair_rpss_row += f" {'N/A':<12}"  # RPS is overall only
    log.debug(fair_rpss_row)

    log.debug("-"*table_width)

    # Add interpretation guide
    log.debug(f"\nInterpretation Guide:")
    log.debug(f"• Positive skill scores indicate forecast is better than climatology")
    log.debug(f"• Negative skill scores indicate forecast is worse than climatology")
    log.debug(f"• Skill score = 0 means forecast equals climatology")
    log.debug(f"• Perfect score = 1.0")

    # Additional detailed results
    log.debug(f"\n" + "="*60)
    log.debug("DETAILED RESULTS")
    log.debug("="*60)

    log.debug(f"\nForecast Fair Brier Score : {brier_forecast['fair_brier_score']:.4f}")
    log.debug(f"Climatology Fair Brier Score : {brier_climatology['fair_brier_score']:.4f}")
    log.debug(f"Fair Brier Skill Score: {fair_bss_overall:.4f}")

    log.debug(f"\nForecast Fair RPS : {rps_forecast['fair_rps']:.4f}")
    log.debug(f"Climatology Fair RPS : {rps_climatology['fair_rps']:.4f}")
    log.debug(f"Fair RPS Skill Score: {fair_rpss_overall:.4f}")

    log.debug(f"\nBin-wise Fair Brier Score Comparisons:")
    for bin_name in target_bins:
        if bin_name in brier_forecast['bin_fair_brier_scores'] and bin_name in brier_climatology['bin_fair_brier_scores']:
            forecast_val = brier_forecast['bin_fair_brier_scores'][bin_name]
            clim_val = brier_climatology['bin_fair_brier_scores'][bin_name]
            skill_val = bin_fair_bss[bin_name]
            log.debug(f"  {bin_name}:")
            log.debug(f"    Forecast: {forecast_val:.4f}")
            log.debug(f"    Climatology: {clim_val:.4f}")
            log.debug(f"    Skill Score: {skill_val:.4f}")
        else:
            log.debug(f"  {bin_name}: Missing data")

    return skill_scores
