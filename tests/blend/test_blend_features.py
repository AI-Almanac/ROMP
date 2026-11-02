"""Pieces of the weekly features: the members' Kahan mean, lead counts, weeks as LeadBins, the Weekly container."""
import numpy as np
import pandas as pd

from momp.blend.features import _week_sums, member_mean
from momp.blend.samples import Weekly
from momp.onset.engine import lead_counts
from momp.stats.bins import LeadBins


def test_member_mean_is_pandas_grouped_mean_in_the_file_dtype():
    rng = np.random.default_rng(0)
    rain = (rng.gamma(0.5, 8.0, (3, 7, 4, 5))).astype("float32")
    rain[rng.random(rain.shape) < 0.1] = np.nan
    rain[0, :, 0, 0] = np.nan
    got = member_mean(rain)
    assert got.dtype == np.float32
    frame = pd.DataFrame({"g": np.tile(np.arange(3 * 4 * 5), 7), "v": rain.transpose(1, 0, 2, 3).reshape(7, -1).ravel()})
    expected = frame.groupby("g")["v"].mean().to_numpy().reshape(3, 4, 5)
    np.testing.assert_array_equal(got, expected.astype("float32"))


def test_lead_counts_put_prior_leads_at_zero_and_skip_nan():
    lead = np.array([1.0, 3.0, 3.0, np.nan, -2.0, 3.0])
    np.testing.assert_array_equal(lead_counts(lead, 4, axis=0), [1, 1, 0, 3, 0])


def test_weekly_sums_are_the_bins_column_slices():
    bins = LeadBins.uniform(3, 7)
    daily = np.arange(40.0).reshape(2, 20)
    sums = _week_sums(daily, bins)
    for w, (a, b) in enumerate(bins.inner):
        np.testing.assert_array_equal(sums[w], daily[:, a - 1:b].sum(axis=1))
    lead = np.array([np.nan, -1.0, 1.0, 7.0, 8.0, 21.0, 22.0])
    assert bins.codes(lead).tolist() == [-1, -1, 0, 0, 1, 2, 3]


def test_a_legacy_table_is_a_weekly_and_back():
    frame = pd.DataFrame({"id": ["b", "a", "b"], "year": [2001, 2001, 2002], "outcome": ["week1", None, "later"],
                          "x": [1.0, 2.0, np.nan]})
    weekly = Weekly.from_frame(frame)
    assert len(weekly) == 3 and list(weekly.take("id")) == ["b", "a", "b"]
    pd.testing.assert_frame_equal(weekly.frame(), frame)
    np.testing.assert_array_equal(weekly.frame(rows=np.array([2, 0]), columns=["x"])["x"].to_numpy(), [np.nan, 1.0])
