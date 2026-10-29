"""The vectorised calculate_rps gives bit-identical results to the per-group loop it replaced."""

import numpy as np
import pandas as pd
import pytest

from momp.stats.score import calculate_rps


def _old_calculate_rps(forecast_obs_df):
    """Reference: the original per-group pandas loop."""
    forecast_groups = forecast_obs_df.groupby(['init_time', 'lat', 'lon'])
    rps_values, fair_rps_values = [], []
    for _, group in forecast_groups:
        group_sorted = group.sort_values('bin_index')
        p_ij = group_sorted['predicted_prob'].values
        y_ij = group_sorted['observed_onset'].values
        total_members = group_sorted['total_members'].iloc[0]
        rps_forecast = 0
        fair_rps_forecast = 0
        for k in range(1, len(p_ij) + 1):
            cum_p = np.sum(p_ij[:k])
            cum_y = np.sum(y_ij[:k])
            rps_component = (cum_y - cum_p) ** 2
            rps_forecast += rps_component
            fair_rps_forecast += rps_component - (cum_p * (1 - cum_p)) / (total_members - 1)
        rps_values.append(rps_forecast)
        fair_rps_values.append(fair_rps_forecast)
    return {'rps': np.mean(rps_values), 'fair_rps': np.mean(fair_rps_values), 'n_forecasts': len(forecast_groups)}


def _frame(n_cases, n_bins, members, seed, shuffle=False, ragged=False):
    """Build a random pair table of n_cases cases with n_bins bins each."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cases):
        nb = n_bins - (c % 3 if ragged else 0)
        counts = rng.multinomial(members, np.ones(nb) / nb)
        obs = rng.integers(nb)
        for b in range(nb):
            rows.append((pd.Timestamp("2000-01-01") + pd.Timedelta(days=c % 17), c % 5, c // 5,
                         counts[b] / members, int(b == obs), members, b))
    df = pd.DataFrame(rows, columns=['init_time', 'lat', 'lon', 'predicted_prob', 'observed_onset',
                                     'total_members', 'bin_index'])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True) if shuffle else df


@pytest.mark.parametrize("n_bins", [3, 5, 9, 17, 40])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
def test_rps_identical(n_bins, shuffle, ragged):
    """RPS, fair RPS and the case count equal the loop's exactly."""
    df = _frame(300, n_bins, 51, n_bins, shuffle, ragged)
    new, old = calculate_rps(df), _old_calculate_rps(df)
    assert new['rps'] == old['rps'] and new['fair_rps'] == old['fair_rps']
    assert new['n_forecasts'] == old['n_forecasts']


def test_rps_needs_ensemble():
    """A single member raises as before."""
    with pytest.raises(ValueError):
        calculate_rps(_frame(4, 3, 1, 0))
