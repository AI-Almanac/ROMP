"""hcwf.metrics.probabilistic and LeadBins: the blending package's numbers, and HCWF's own, to the last bit.

Blending side: the legacy functions of onset_blending@1e6d696 (tests/blend/_oracle/blend_metrics.py) on synthetic
cv_preds frames. HCWF side: the pandas calculations hcwf/metrics/score.py had before (hcwf_score_pandas.py).
"""

import numpy as np
import pandas as pd
import pytest

from hcwf.metrics import probabilistic as prob
from hcwf.metrics import score
from hcwf.metrics.bins import LeadBins
from tests.blend._oracle import blend_metrics as legacy
from tests.blend._oracle import hcwf_score_pandas as hcwf_old
from tests.pair_frames import pairs_of_frame, shuffled_cases


# --- synthetic data -----------------------------------------------------------------------------------------

def cv_frame(n=600, n_weeks=4, seed=0, earlier=False, ties=False, n_units=12, years=(2000, 2001, 2002)):
    """A legacy cv_preds frame: id, year, outcome (a bin label or NaN), cv_week1..N, cv_later (and cv_earlier)."""
    rng = np.random.default_rng(seed)
    bins = legacy.interval_bins(n_weeks)
    raw = rng.dirichlet(np.ones(len(bins) + (1 if earlier else 0)), size=n)
    if ties:
        raw = np.round(raw * 4) / 4
        raw[raw.sum(axis=1) == 0] = 0.25
        raw = raw / raw.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(raw[:, 1:] if earlier else raw, columns=[f"cv_{b}" for b in bins])
    if earlier:
        frame.insert(0, "cv_earlier", raw[:, 0])
    frame["id"] = rng.choice([f"u{k}" for k in range(n_units)], size=n)
    frame["year"] = rng.choice(years, size=n)
    outcome = rng.choice(bins, size=n).astype(object)
    outcome[rng.random(n) < 0.05] = np.nan
    frame["outcome"] = outcome
    frame.loc[rng.random(n) < 0.03, "cv_week1"] = np.nan
    return frame


def arrays(frame, n_weeks=4):
    """(P, outcome codes, ids, labels) of a cv_preds frame, the way predictions hand them to the kernel."""
    bins = legacy.interval_bins(n_weeks)
    P = frame[[f"cv_{b}" for b in bins]].to_numpy(dtype=float)
    codes = np.array([-1 if pd.isna(o) else bins.index(o) for o in frame["outcome"]])
    return P, codes, frame["id"].to_numpy(), bins


def assert_frames_identical(a, b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)


# --- blending parity ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_weeks", [2, 4, 6])
@pytest.mark.parametrize("earlier", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_cell_metrics_equal_the_legacy_ones(n_weeks, earlier, ties):
    frame = cv_frame(seed=n_weeks, n_weeks=n_weeks, earlier=earlier, ties=ties)
    expected = legacy.compute_cell_metrics_fast(frame)
    P, codes, ids, bins = arrays(frame, n_weeks)
    got = prob.cell_metrics(P, codes, ids, bins, earlier=frame["cv_earlier"].to_numpy() if earlier else None)
    assert_frames_identical(got[expected.columns], expected)


def test_cell_metrics_with_allowed_units_equal_the_legacy_ones():
    frame = cv_frame(seed=11)
    allowed = pd.DataFrame({"id": ["u1", "u3", "u4", "u8"]})
    expected = legacy.compute_cell_metrics_fast(frame, allowed_cells=allowed)
    P, codes, ids, bins = arrays(frame)
    got = prob.cell_metrics(P, codes, ids, bins, allowed=np.isin(ids, allowed["id"]))
    assert_frames_identical(got[expected.columns], expected)


def test_cell_metrics_of_nothing_is_empty():
    P = np.full((3, 5), np.nan)
    assert prob.cell_metrics(P, np.array([0, 1, 2]), np.array(["a", "b", "c"]), list("abcde")).empty


def test_scattered_rows_give_the_same_cells():
    """Rows in any order, ids interleaved: per-unit numbers do not depend on the order of the units' rows."""
    frame = cv_frame(seed=5)
    P, codes, ids, bins = arrays(frame)
    order = np.random.default_rng(1).permutation(len(frame))
    a = prob.cell_metrics(P, codes, ids, bins).set_index("id")
    b = prob.cell_metrics(P[order], codes[order], ids[order], bins).set_index("id")
    assert (a.drop(columns=["brier", "rps", "pietra"] + [c for c in a if c.startswith("brier_")])
            .equals(b.drop(columns=["brier", "rps", "pietra"] + [c for c in b if c.startswith("brier_")])))
    np.testing.assert_allclose(a.drop(columns=["lat", "lon"]).astype(float), b.drop(columns=["lat", "lon"]).astype(float),
                               rtol=1e-12)


def test_yearly_metrics_equal_the_legacy_ones_up_to_the_auc_rounding():
    pytest.importorskip("sklearn")
    frame = cv_frame(seed=3, ties=True)
    expected = pd.DataFrame(legacy.compute_yearly_metrics(frame, "m", "global", 4))
    P, codes, _, _ = arrays(frame)
    got = pd.DataFrame(prob.yearly_metrics(P, codes, frame["year"].to_numpy(), "m", "global"))
    for column in ("year", "model", "cv_method", "brier", "rps"):
        assert (got[column] == expected[column]).all()
    np.testing.assert_allclose(got["auc"], expected["auc"], rtol=1e-12)


def test_yearly_metrics_with_sklearns_auc_equal_the_legacy_ones_exactly():
    pytest.importorskip("sklearn")
    frame = cv_frame(seed=3, ties=True)
    expected = pd.DataFrame(legacy.compute_yearly_metrics(frame, "m", "global", 4))
    P, codes, _, _ = arrays(frame)
    got = pd.DataFrame(prob.yearly_metrics(P, codes, frame["year"].to_numpy(), "m", "global", auc_method="sklearn"))
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    with pytest.raises(ValueError, match="auc_method"):
        prob.yearly_metrics(P, codes, frame["year"].to_numpy(), "m", "global", auc_method="x")


def test_yearly_metrics_equal_the_per_year_legacy_pieces():
    """Without sklearn: Brier and RPS of each year equal the legacy formulas on that year's rows, AUC the legacy
    rank-sum."""
    frame = cv_frame(seed=4)
    P, codes, _, bins = arrays(frame)
    got = prob.yearly_metrics(P, codes, frame["year"].to_numpy(), "m", "global")
    sub = frame.dropna(subset=["outcome"] + [f"cv_{b}" for b in bins])
    for row in got:
        g = sub[sub["year"] == row["year"]]
        p = g[[f"cv_{b}" for b in bins]].to_numpy(dtype=float)
        y = np.column_stack([(g["outcome"] == b).astype(float) for b in bins])
        assert row["brier"] == float(np.mean(np.sum((p - y) ** 2, axis=1)))
        assert row["rps"] == legacy.pooled_rps5(p, y)
        assert row["auc"] == legacy._fast_auc(y.flatten(), p.flatten())


@pytest.mark.parametrize("members", [30, 51])
def test_fair_brier_of_samples_equals_the_legacy_fair_brier(members):
    frame = cv_frame(seed=8).dropna(subset=["outcome", "cv_week1"])
    P, codes, _, bins = arrays(frame)
    Y = prob.one_hot(codes, len(bins))
    expected = legacy.compute_fair_brier5(frame, None, n_train=members)
    got = prob.brier_scores(P, Y, members, scale="sample")["fair_brier"]
    assert got == expected


def test_clean_equals_the_legacy_clean():
    rng = np.random.default_rng(2)
    P = rng.random((50, 5))
    P[3, 2] = np.nan
    P[7] = 0.0
    P[9, 1] = -0.5
    P[11, 0] = np.inf
    assert (prob.clean(P) == legacy.clean_probs5(P)).all()


def test_pietra_is_the_ks_statistic_and_nan_without_both_groups():
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    Y = np.array([[1, 0], [0, 1], [1, 0]])
    from scipy.stats import ks_2samp
    assert prob.pietra(P, Y) == ks_2samp([0.7, 0.8, 0.6], [0.3, 0.2, 0.4]).statistic
    assert np.isnan(prob.pietra(P, np.ones_like(Y)))


def test_pooled_skill_is_one_minus_the_ratio_against_the_baseline():
    cells = pd.DataFrame({"id": ["ALL", "ALL", "x"], "model": ["unc_clim_raw", "blend", "blend"],
                          "brier": [0.4, 0.3, 9.0], "rps": [0.5, 0.25, 9.0], "auc": [0.60, 0.75, 9.0]})
    pooled = prob.pooled_skill(cells)
    blend = pooled[pooled["model"] == "blend"].iloc[0]
    assert blend["brier_skill"] == 1.0 - 0.3 / 0.4 and blend["rps_skill"] == 0.5
    assert blend["AUC diff"] == pytest.approx(15.0)
    with pytest.raises(ValueError):
        prob.pooled_skill(cells[cells["model"] == "blend"])


# --- the other conventions ----------------------------------------------------------------------------------

def test_brier_scales_are_related_by_the_bin_count():
    """HCWF's element-wise Brier is blending's sum over bins divided by the number of bins."""
    frame = cv_frame(seed=6).dropna(subset=["outcome", "cv_week1"])
    P, codes, _, bins = arrays(frame)
    Y = prob.one_hot(codes, len(bins))
    element = prob.brier(P, Y, scale="element")
    assert element == pytest.approx(prob.brier(P, Y, scale="sample") / len(bins), rel=1e-14)
    assert prob.brier(P, Y, scale="pooled") == pytest.approx(prob.brier(P, Y, scale="sample"), rel=1e-14)
    with pytest.raises(ValueError):
        prob.brier(P, Y, scale="nope")


def test_brier_scores_by_bin_use_either_mean():
    frame = cv_frame(seed=7).dropna(subset=["outcome", "cv_week1"])
    P, codes, _, bins = arrays(frame)
    Y = prob.one_hot(codes, len(bins))
    plain = prob.brier_scores(P, Y, by_bin=True)["bin_brier"]
    compensated = prob.brier_scores(P, Y, by_bin=True, bin_mean="compensated")["bin_brier"]
    np.testing.assert_allclose(plain, compensated, rtol=1e-14)
    assert plain[0] == np.mean((P[:, 0] - Y[:, 0]) ** 2)


def test_fair_scores_take_off_the_sampling_variance():
    P = np.array([[0.5, 0.5], [1.0, 0.0]])
    Y = np.array([[1, 0], [1, 0]])
    fair = prob.brier_scores(P, Y, members=11, scale="element")["fair_brier"]
    assert fair == pytest.approx(np.mean([(0.25 + 0.25 - 0.25 / 10 * 2) / 2, 0.0]))
    infinite = prob.brier_scores(P, Y, members=np.inf, scale="element")
    assert infinite["fair_brier"] == infinite["brier"]


def test_rps_summations_agree_up_to_rounding_and_match_the_sequential_sums():
    rng = np.random.default_rng(9)
    P = rng.dirichlet(np.ones(17), size=200)
    Y = prob.one_hot(rng.integers(17, size=200), 17)
    cumulative, _ = prob.rps_cases(P, Y)
    prefix, _ = prob.rps_cases(P, Y, summation="prefix")
    np.testing.assert_allclose(cumulative, prefix, rtol=1e-12)
    with pytest.raises(ValueError):
        prob.rps_cases(P, Y, summation="nope")
    with pytest.raises(ValueError):
        prob.rps_cases(P, Y, valid=np.ones(P.shape, bool))


@pytest.mark.parametrize("ties", [False, True])
def test_auc_equals_the_mann_whitney_statistic_and_counts_ties_as_half(ties):
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(10)
    score_ = rng.random(500)
    if ties:
        score_ = np.round(score_ * 5) / 5
    y = (rng.random(500) < 0.3).astype(int)
    u = mannwhitneyu(score_[y == 1], score_[y == 0], alternative="greater").statistic
    assert prob.auc(y, score_) == u / (y.sum() * (1 - y).sum())
    assert prob.auc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert np.isnan(prob.auc(np.ones(3), np.arange(3.0)))


# --- HCWF parity: the pandas calculations before and the adapters now ---------------------------------------

def pair_frame(n_cases, n_bins, members, seed, clim=False):
    """A pair table in the layout the builders write: one row per case and bin, labels like 'Days 1-5'."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = [f"Days {k * 5 + 1}-{k * 5 + 5}" for k in range(n_bins)]
    if clim:
        labels = ["Before day 1"] + labels
    for c in range(n_cases):
        counts = rng.multinomial(members, np.ones(len(labels)) / len(labels))
        observed = rng.integers(len(labels))
        for b, label in enumerate(labels):
            rows.append((pd.Timestamp("2000-01-01") + pd.Timedelta(days=c % 17), 10 + c % 5, 70 + c // 5, label,
                         counts[b] / members, int(b == observed), members, members - c % 3, b))
    return pd.DataFrame(rows, columns=["init_time", "lat", "lon", "bin_label", "predicted_prob", "observed_onset",
                                       "total_members", "total_members_with_onset", "bin_index"])


def same(a, b, path=""):
    if isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            same(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (list, pd.DataFrame)):
        return
    else:
        assert (a == b) or (np.isnan(a) and np.isnan(b)), f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("n_bins", [1, 2, 3, 6, 9, 15, 30])
@pytest.mark.parametrize("function", ["calculate_brier_score", "calculate_auc", "calculate_brier_score_climatology",
                                      "calculate_auc_climatology"])
def test_hcwf_brier_and_auc_are_byte_identical_to_the_pandas_ones(n_bins, function):
    df = pair_frame(300, n_bins, 51, n_bins, clim=function.endswith("climatology"))
    old = getattr(hcwf_old, function)(df.copy())
    old.pop("forecast_obs_df_with_ranks", None)   # the table the pandas AUC handed back when it was undefined; no one read it
    same(getattr(score, function)(pairs_of_frame(df)), old, function)


def test_hcwf_scores_of_shuffled_cases_equal_those_of_the_sorted_ones():
    pairs = pairs_of_frame(pair_frame(120, 4, 31, 3))
    shuffled = shuffled_cases(pairs)
    a, b = score.calculate_brier_score(pairs), score.calculate_brier_score(shuffled)
    for key in ("brier_score", "fair_brier_score"):
        assert a[key] == pytest.approx(b[key], rel=1e-12)
    for key in ("bin_brier_scores", "bin_fair_brier_scores"):
        assert a[key] == {k: pytest.approx(v, rel=1e-12) for k, v in b[key].items()}
    assert score.calculate_auc(pairs)["auc"] == score.calculate_auc(shuffled)["auc"]
    assert score.calculate_rps(pairs) == score.calculate_rps(shuffled)


def test_hcwf_auc_without_both_outcomes_is_nan_with_the_counts():
    df = pair_frame(10, 3, 11, 0)
    df["observed_onset"] = 0
    result = score.calculate_auc(pairs_of_frame(df))
    assert np.isnan(result["auc"]) and result["n_positive"] == 0 and result["n_negative"] == 30
    assert result["bin_auc_scores"] == {}


def test_hcwf_scores_refuse_a_single_member():
    df = pair_frame(5, 3, 1, 0)
    for function in (score.calculate_brier_score, score.calculate_rps):
        with pytest.raises(ValueError):
            function(pairs_of_frame(df))


def test_hcwf_skill_scores_are_one_minus_the_ratio():
    brier = {"fair_brier_score": 0.2, "bin_fair_brier_scores": {"Days 6-10": 0.3, "Days 1-5": 0.1}}
    clim = {"fair_brier_score": 0.4, "bin_fair_brier_scores": {"Days 6-10": 0.6, "Days 1-5": 0.2}}
    skill = score.calculate_skill_scores(brier, {"fair_rps": 0.5}, clim, {"fair_rps": 1.0})
    assert skill["fair_brier_skill_score"] == 0.5 and skill["fair_rps_skill_score"] == 0.5
    assert list(skill["bin_fair_brier_skill_scores"]) == ["Days 1-5", "Days 6-10"]


# --- LeadBins -----------------------------------------------------------------------------------------------

def legacy_outcome_codes(lead, days_per_bin, n_bins):
    """onset_blending core/features.py _outcome_codes@1e6d696, verbatim."""
    labels = [f"week{w}" for w in range(1, n_bins + 1)] + ["later"]
    codes = np.full(len(lead), -1, dtype=np.int8)
    remaining = ~np.isnan(lead) & (lead > 0)
    for w in range(1, n_bins + 1):
        hit = remaining & (lead <= w * days_per_bin)
        codes[hit] = w - 1
        remaining &= ~hit
    codes[remaining] = n_bins
    return codes, labels


@pytest.mark.parametrize("count,days", [(4, 7), (3, 5), (1, 1), (6, 3)])
def test_lead_bin_codes_equal_blendings_outcome_codes(count, days):
    lead = np.concatenate([np.arange(-5, count * days + 10, dtype=float), [np.nan, 0.0]])
    bins = LeadBins.uniform(count, days)
    expected, labels = legacy_outcome_codes(lead, days, count)
    assert (bins.codes(lead) == expected).all()
    assert bins.labels("week", earlier=False) == labels
    assert bins.codes(lead).dtype == expected.dtype


def test_lead_bins_with_the_earlier_bin():
    bins = LeadBins.uniform(2, 7)
    assert bins.labels("week") == ["earlier", "week1", "week2", "later"]
    assert list(bins.codes(np.array([-3.0, 0.0, 1.0, 7.0, 8.0, 15.0, np.nan]), earlier=True)) == [0, 0, 1, 1, 2, 3, -1]


def test_lead_bins_of_hcwf_day_bins_keep_hcwfs_labels_and_edges():
    bins = LeadBins.from_day_bins(((1, 5), (6, 10), (11, 15)))
    assert bins.labels() == ["Before day 1", "Days 1-5", "Days 6-10", "Days 11-15", "After day 15"]
    assert bins.extended == ((-float("inf"), 0), (1, 5), (6, 10), (11, 15), (16, float("inf")))
    assert (bins.first_day, bins.last_day, bins.n_inner, bins.contiguous) == (1, 15, 3, True)
    assert LeadBins.from_day_bins(((3, 5), (6, 8))).extended[0] == (-float("inf"), 2)
    assert LeadBins.uniform(3, 5) == bins


def test_lead_bin_membership_matches_hcwfs_member_and_observed_rules():
    bins = LeadBins.from_day_bins(((1, 5), (6, 10)))
    day = np.array([-1.0, 0.0, 1.0, 5.0, 6.0, 10.0, 11.0, np.nan])
    assert list(bins.in_bin(day, 0)) == [True, True, False, False, False, False, False, False]
    assert list(bins.in_bin(day, 1)) == [False, False, True, True, False, False, False, False]
    assert list(bins.in_bin(day, 3)) == [False, False, False, False, False, False, True, False]
    assert list(bins.in_bin(day, 3, nan_after=True)) == [False] * 6 + [True, True]


def test_lead_bins_reject_what_they_cannot_label():
    with pytest.raises(ValueError):
        LeadBins.from_day_bins(((2, 5), (6, 8))).labels("week")
    with pytest.raises(ValueError):
        LeadBins.from_day_bins(((1, 5), (8, 10))).codes(np.array([1.0]))
    with pytest.raises(ValueError):
        LeadBins(()).labels()
    with pytest.raises(ValueError):
        LeadBins.uniform(2, 3).label(1, "nope")


def test_lead_bins_day_slices_and_sort_key():
    bins = LeadBins.uniform(2, 7)
    daily = np.arange(1.0, 15.0)[None, :]
    assert [daily[:, s].sum() for s in bins.column_slices()] == [sum(range(1, 8)), sum(range(8, 15))]
    assert LeadBins.first_day_of("Days 11-15") == 11 and LeadBins.first_day_of("After day 15") == 999


@pytest.mark.parametrize("n_bins", [3, 9])
def test_hcwf_climatology_brier_skips_the_nan_of_one_member_cases(n_bins):
    """A case whose climatology has one member with onset has 0/0 in the fair term; pandas leaves it out."""
    df = pair_frame(200, n_bins, 51, 5, clim=True)
    df.loc[df["lon"] % 3 == 0, "total_members_with_onset"] = 1
    df.loc[df["lon"] % 3 == 0, "predicted_prob"] = (df["bin_index"] == 2).astype(float)
    assert np.isnan(((df["predicted_prob"] * (1 - df["predicted_prob"])) / (df["total_members_with_onset"] - 1))).any()
    same(score.calculate_brier_score_climatology(pairs_of_frame(df)), hcwf_old.calculate_brier_score_climatology(df.copy()))


def test_grouped_auc_matches_auc_per_group_with_ties_and_empty_classes():
    rng = np.random.default_rng(3)
    n, groups = 4000, 60
    group = rng.integers(0, groups, n)
    score = np.round(rng.random(n), 1)  # many ties
    y = (rng.random(n) < 0.3).astype(float)
    y[group == 5] = 0.0   # a group without positives
    y[group == 6] = 1.0   # and one without negatives
    got = prob.grouped_auc(y, score, group, groups + 1)   # the last group is empty
    for k in range(groups):
        want = prob.auc(y[group == k].astype(int), score[group == k])
        assert (np.isnan(want) and np.isnan(got[k])) or got[k] == want
    assert np.isnan(got[groups]) and np.isnan(got[5]) and np.isnan(got[6])


def test_hcwf_skill_score_against_a_perfect_reference_is_nan_with_a_warning():
    brier = {"fair_brier_score": 0.2, "bin_fair_brier_scores": {"Days 1-5": 0.1}}
    clim = {"fair_brier_score": 0.0, "bin_fair_brier_scores": {"Days 1-5": 0.2}}
    with pytest.warns(UserWarning, match="fair Brier score is zero"):
        skill = score.calculate_skill_scores(brier, {"fair_rps": 0.5}, clim, {"fair_rps": 1.0})
    assert np.isnan(skill["fair_brier_skill_score"]) and skill["fair_rps_skill_score"] == 0.5


def test_every_skill_score_has_a_zero_reference_guard():
    brier = {"fair_brier_score": 0.2, "bin_fair_brier_scores": {"Days 1-5": 0.1}}
    clim = {"fair_brier_score": 0.4, "bin_fair_brier_scores": {"Days 1-5": 0.0}}
    with pytest.warns(UserWarning, match="Days 1-5 is zero"):
        skill = score.calculate_skill_scores(brier, {"fair_rps": 0.5}, clim, {"fair_rps": 0.0})
    assert np.isnan(skill["fair_rps_skill_score"]) and np.isnan(skill["bin_fair_brier_skill_scores"]["Days 1-5"])
    assert skill["fair_brier_skill_score"] == 0.5
    with pytest.warns(UserWarning):
        got = prob.skill_score(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
    assert np.isnan(got[0]) and got[1] == 0.5
    cells = pd.DataFrame({"id": ["ALL", "ALL"], "model": ["a", "b"], "brier": [0.0, 0.1], "rps": [1.0, 0.5],
                          "auc": [0.5, 0.6]})
    with pytest.warns(UserWarning, match="pooled Brier score is zero"):
        pooled = prob.pooled_skill(cells, baseline="a")
    assert np.isnan(pooled["brier_skill"]).all() and list(pooled["rps_skill"]) == [0.0, 0.5]
