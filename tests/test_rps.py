import pandas as pd
import pytest

from momp.stats.score import calculate_rps


def bins(init, lat, probs, observed, members, order):
    return [
        {"init_time": pd.Timestamp(init), "lat": lat, "lon": 70.0, "bin_index": i,
         "predicted_prob": probs[i], "observed_onset": observed[i], "total_members": members}
        for i in order
    ]


def test_rps_accumulates_bins_in_bin_index_order_and_averages_over_forecasts():
    # Forecast A, accumulated as bins 0,1,2: cum_p .6/1/1 vs cum_y 0/0/1 -> RPS 1.36,
    # fair RPS 1.36 - .6*.4/(3-1) = 1.24. Its rows arrive as bins 1,0,2, which would
    # give 1.16 if accumulated as stored. Forecast B is a perfect forecast: 0 and 0.
    rows = (
        bins("2022-06-01", 10.0, [0.6, 0.4, 0.0], [0, 0, 1], members=3, order=[1, 0, 2])
        + bins("2022-06-01", 10.25, [1.0, 0.0, 0.0], [1, 0, 0], members=5, order=[2, 0, 1])
    )

    result = calculate_rps(pd.DataFrame(rows))

    assert result["n_forecasts"] == 2
    assert result["rps"] == pytest.approx(1.36 / 2)
    assert result["fair_rps"] == pytest.approx(1.24 / 2)
