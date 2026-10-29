"""Brier score, RPS, AUC and skill-score calculations for binned onset forecast-observation pairs."""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from momp.stats.bins import extract_day_range
import logging

log = logging.getLogger(__name__)


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
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members'

    Returns:
    --------
    dict with Brier score metrics
    """

    # Guard: fair correction requires >1 member
    n_members = forecast_obs_df['total_members'].iloc[0] if len(forecast_obs_df) > 0 else 1
    if n_members <= 1:
        raise ValueError(
            f"Fair Brier Score requires >1 ensemble member, but found total_members={n_members}. "
            "Check that your model data contains ensemble members and that the 'members' / "
            "'probabilistic' parameters are configured correctly."
        )

    # Calculate squared differences
    squared_diffs = (forecast_obs_df['observed_onset'] - forecast_obs_df['predicted_prob'])**2

    # Calculate overall Brier Score
    brier_score = squared_diffs.mean()

    # Calculate Fair Brier Score correction term
    # ens-1 where ens is the number of ensemble members
    correction_term = (forecast_obs_df['predicted_prob'] * (1 - forecast_obs_df['predicted_prob'])) / (forecast_obs_df['total_members'] - 1)

    # Fair Brier Score
    fair_brier_components = squared_diffs - correction_term
    fair_brier_score = fair_brier_components.mean()
    # Calculate squared differences for bin-wise analysis
    forecast_obs_df['squared_diff'] = squared_diffs
    forecast_obs_df['fair_brier_component'] = fair_brier_components

    # Bin-wise Brier scores
    bin_brier_scores = forecast_obs_df.groupby('bin_label')['squared_diff'].mean()
    bin_fair_brier_scores = forecast_obs_df.groupby('bin_label')['fair_brier_component'].mean()

    brier_results = {
        'brier_score': brier_score,
        'fair_brier_score': fair_brier_score,
        'bin_brier_scores': bin_brier_scores.to_dict(),
        'bin_fair_brier_scores': bin_fair_brier_scores.to_dict(),
    }

    log.debug(f"Brier Score: {brier_results['brier_score']:.4f}")
    log.debug(f"Fair Brier Score: {brier_results['fair_brier_score']:.4f}")

    return brier_results

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
        Must contain columns: 'predicted_prob', 'observed_onset'

    Returns:
    --------
    dict with AUC metrics
    """

    # Extract probabilities and observations
    p_ij = forecast_obs_df['predicted_prob'].values
    y_ij = forecast_obs_df['observed_onset'].values

    # Count total positive and negative cases
    n_positive = np.sum(y_ij)  # Σ Y_{ij}
    n_negative = np.sum(1 - y_ij)  # Σ (1-Y_{ij})

    if n_positive == 0 or n_negative == 0:
        log.warning("Cannot calculate AUC - all cases are either positive or negative")
        return {
            'auc': np.nan,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'bin_auc_scores': {},
            'forecast_obs_df_with_ranks': forecast_obs_df
        }

    # Calculate AUC using the Mann-Whitney U statistic approach
    # This is equivalent to the formula but more computationally efficient

    # Separate positive and negative cases
    positive_probs = p_ij[y_ij == 1]
    negative_probs = p_ij[y_ij == 0]

    # Calculate Mann-Whitney U statistic
    u_statistic, _ = stats.mannwhitneyu(positive_probs, negative_probs, alternative='greater')

    # AUC is U statistic divided by (n_positive * n_negative)
    auc = u_statistic / (n_positive * n_negative)

    # Calculate AUC by bin
    bin_auc_scores = {}
    unique_bins = forecast_obs_df['bin_label'].unique()

    for bin_label in unique_bins:
        bin_data = forecast_obs_df[forecast_obs_df['bin_label'] == bin_label]

        if len(bin_data) > 0:
            bin_p = bin_data['predicted_prob'].values
            bin_y = bin_data['observed_onset'].values

            bin_n_positive = np.sum(bin_y)
            bin_n_negative = np.sum(1 - bin_y)

            if bin_n_positive > 0 and bin_n_negative > 0:
                bin_positive_probs = bin_p[bin_y == 1]
                bin_negative_probs = bin_p[bin_y == 0]

                bin_u_stat, _ = stats.mannwhitneyu(bin_positive_probs, bin_negative_probs, alternative='greater')
                bin_auc = bin_u_stat / (bin_n_positive * bin_n_negative)
            else:
                bin_auc = np.nan

            bin_auc_scores[bin_label] = bin_auc

    auc_results = {
        'auc': auc,
        'bin_auc_scores': bin_auc_scores,
    }

    return auc_results


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

    # Guard: fair correction requires >1 member
    n_members = forecast_obs_df['total_members'].iloc[0] if len(forecast_obs_df) > 0 else 1
    if n_members <= 1:
        raise ValueError(
            f"Fair RPS requires >1 ensemble member, but found total_members={n_members}. "
            "Check that your model data contains ensemble members and that the 'members' / "
            "'probabilistic' parameters are configured correctly."
        )

    # Group by forecast (init_time, lat, lon) to get all bins for each forecast
    forecast_groups = forecast_obs_df.groupby(['init_time', 'lat', 'lon'])

    rps_values = []
    fair_rps_values = []

    for (init_time, lat, lon), group in forecast_groups:
        # Sort by bin_index to ensure proper ordering
        group_sorted = group.sort_values('bin_index')

        # Get predicted probabilities and observations for this forecast
        p_ij = group_sorted['predicted_prob'].values
        y_ij = group_sorted['observed_onset'].values
        total_members = group_sorted['total_members'].iloc[0]  # Same for all bins in forecast

        m = len(p_ij)  # Number of bins

        # Calculate RPS for this forecast
        rps_forecast = 0
        fair_rps_forecast = 0

        for k in range(1, m + 1):  # k from 1 to m
            # Cumulative sum up to bin k
            cum_p = np.sum(p_ij[:k])
            cum_y = np.sum(y_ij[:k])

            # RPS component
            diff_cum = cum_y - cum_p
            rps_component = diff_cum**2
            rps_forecast += rps_component

            # Fair RPS correction term
            fair_correction = (cum_p * (1 - cum_p)) / (total_members - 1)
            fair_rps_component = rps_component - fair_correction
            fair_rps_forecast += fair_rps_component

        rps_values.append(rps_forecast)
        fair_rps_values.append(fair_rps_forecast)

    # Calculate overall RPS (average over all forecasts)
    rps = np.mean(rps_values)
    fair_rps = np.mean(fair_rps_values)

    rps_results = {
        'rps': rps,
        'fair_rps': fair_rps,
        'n_forecasts': len(forecast_groups),

    }

    log.debug(f"RPS: {rps_results['rps']:.4f}")
    log.debug(f"Fair RPS: {rps_results['fair_rps']:.4f}")
    log.debug(f"Number of forecasts: {rps_results['n_forecasts']}")

    return rps_results


def calculate_brier_score_climatology(forecast_obs_df):
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

    Note: Excludes bins whose label contains "Before" (e.g. "Before day N") from calculations

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
    filtered_df = forecast_obs_df[~forecast_obs_df['bin_label'].str.contains('Before', na=False)].copy()

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

    # Calculate squared differences
    squared_diffs = (filtered_df['observed_onset'] - filtered_df['predicted_prob'])**2

    # Calculate overall Brier Score
    brier_score = squared_diffs.mean()

    # Calculate Fair Brier Score correction term
    # ens-1 where ens is the number of ensemble members
    correction_term = (filtered_df['predicted_prob'] * (1 - filtered_df['predicted_prob'])) / (filtered_df['total_members_with_onset'] - 1)

    # Fair Brier Score
    fair_brier_components = squared_diffs - correction_term
    fair_brier_score = fair_brier_components.mean()

    # Calculate squared differences for bin-wise analysis
    filtered_df['squared_diff'] = squared_diffs
    filtered_df['fair_brier_component'] = fair_brier_components

    # Bin-wise Brier scores (excluding "Before initialization")
    bin_brier_scores = filtered_df.groupby('bin_label')['squared_diff'].mean()
    bin_fair_brier_scores = filtered_df.groupby('bin_label')['fair_brier_component'].mean()

    brier_results = {
        'brier_score': brier_score,
        'fair_brier_score': fair_brier_score,
        'bin_brier_scores': bin_brier_scores.to_dict(),
        'bin_fair_brier_scores': bin_fair_brier_scores.to_dict(),
        'n_samples': len(filtered_df),
        'filtered_bins': sorted(filtered_df['bin_label'].unique()),
        'excluded_bins': ['Before initialization']
    }

    log.debug(f"Brier Score (excluding 'Before initialization'): {brier_results['brier_score']:.4f}")
    log.debug(f"Fair Brier Score (excluding 'Before initialization'): {brier_results['fair_brier_score']:.4f}")
    log.debug(f"Bins included in calculation: {brier_results['filtered_bins']}")

    return brier_results

def calculate_auc_climatology(forecast_obs_df):
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
        Must contain columns: 'predicted_prob', 'observed_onset'

    Returns:
    --------
    dict with AUC metrics
    """
    forecast_obs_df = forecast_obs_df[~forecast_obs_df['bin_label'].str.contains('Before', na=False)].copy()
    # Extract probabilities and observations
    p_ij = forecast_obs_df['predicted_prob'].values
    y_ij = forecast_obs_df['observed_onset'].values

    # Count total positive and negative cases
    n_positive = np.sum(y_ij)  # Σ Y_{ij}
    n_negative = np.sum(1 - y_ij)  # Σ (1-Y_{ij})

    if n_positive == 0 or n_negative == 0:
        log.warning("Cannot calculate AUC - all cases are either positive or negative")
        return {
            'auc': np.nan,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'bin_auc_scores': {},
            'forecast_obs_df_with_ranks': forecast_obs_df
        }

    # Calculate AUC using the Mann-Whitney U statistic approach
    # This is equivalent to the formula but more computationally efficient

    # Separate positive and negative cases
    positive_probs = p_ij[y_ij == 1]
    negative_probs = p_ij[y_ij == 0]

    # Calculate Mann-Whitney U statistic
    u_statistic, _ = stats.mannwhitneyu(positive_probs, negative_probs, alternative='greater')

    # AUC is U statistic divided by (n_positive * n_negative)
    auc = u_statistic / (n_positive * n_negative)

    # Calculate AUC by bin
    bin_auc_scores = {}
    unique_bins = forecast_obs_df['bin_label'].unique()

    for bin_label in unique_bins:
        bin_data = forecast_obs_df[forecast_obs_df['bin_label'] == bin_label]

        if len(bin_data) > 0:
            bin_p = bin_data['predicted_prob'].values
            bin_y = bin_data['observed_onset'].values

            bin_n_positive = np.sum(bin_y)
            bin_n_negative = np.sum(1 - bin_y)

            if bin_n_positive > 0 and bin_n_negative > 0:
                bin_positive_probs = bin_p[bin_y == 1]
                bin_negative_probs = bin_p[bin_y == 0]

                bin_u_stat, _ = stats.mannwhitneyu(bin_positive_probs, bin_negative_probs, alternative='greater')
                bin_auc = bin_u_stat / (bin_n_positive * bin_n_negative)
            else:
                bin_auc = np.nan

            bin_auc_scores[bin_label] = bin_auc

    auc_results = {
        'auc': auc,
        'bin_auc_scores': bin_auc_scores,
    }

    return auc_results


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
        fair_bss_overall = 1 - (brier_forecast['fair_brier_score'] / brier_climatology['fair_brier_score'])

    skill_scores['fair_brier_skill_score'] = fair_bss_overall

    log.debug(f"Fair Brier Skill Score (1-15 day): {fair_bss_overall:.4f}")

    # Fair RPS Skill Score (1-15 day overall)
    fair_rpss_overall = 1 - (rps_forecast['fair_rps'] / rps_climatology['fair_rps'])
    skill_scores['fair_rps_skill_score'] = fair_rpss_overall

    log.debug(f"Fair RPS Skill Score (1-15 day): {fair_rpss_overall:.4f}")

    # Automatically extract target bins from the data, excluding unwanted bins
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())

    # Get intersection of bins present in both forecast and climatology
    common_bins = all_forecast_bins.intersection(all_clim_bins)

    # Filter out unwanted bins and keep only "Days X-Y" format bins
    target_bins = []
    excluded_bins = []

    for bin_label in common_bins:
        # Include only bins that start with "Days " and don't contain "After" or "Before"
        if (bin_label.startswith('Days ') and
            not bin_label.startswith('After') and
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)
        else:
            excluded_bins.append(bin_label)

    # Sort bins by their day ranges
    # function extract_day_range already defined in stats.bins, import from there

    target_bins = sorted(target_bins, key=extract_day_range)

    log.debug(f"\nAutomatically detected target bins: {target_bins}")
    log.debug(f"Excluded bins: {excluded_bins}")

    # Bin-wise Fair Brier Skill Scores
    bin_fair_bss = {}

    log.debug(f"\nBin-wise Fair Brier Skill Scores:")
    for bin_label in target_bins:
        if bin_label in brier_forecast['bin_fair_brier_scores'] and bin_label in brier_climatology['bin_fair_brier_scores']:
            forecast_fair_brier_bin = brier_forecast['bin_fair_brier_scores'][bin_label]
            clim_fair_brier_bin = brier_climatology['bin_fair_brier_scores'][bin_label]

            fair_bss_bin = 1 - (forecast_fair_brier_bin / clim_fair_brier_bin)
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
