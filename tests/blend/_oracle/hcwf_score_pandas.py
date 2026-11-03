"""TEST-ONLY ORACLE: HCWF's pandas Brier and AUC calculations as they were before they became adapters of
hcwf.metrics.probabilistic (hcwf/metrics/score.py at 7d1bbd9), kept to prove the new results are byte-identical.
Nothing in hcwf imports this file.
"""

import logging

import numpy as np
from scipy import stats

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
