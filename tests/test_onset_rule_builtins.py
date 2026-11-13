"""Hand-case semantics for the built-in onset rules.

Each rule gets a few small, fully-worked series so the *meaning* of every
parameter is pinned (the conformance suite pins the contract).
"""

import numpy as np
import pytest
import xarray as xr

from momp.stats.onset_rule import (
    LegacyRule,
    MoronRobertson,
    TwoStageAccumulation,
    first_onset_index,
    has_dry_run,
    lead_all,
    lead_sum,
    min_sub_sum,
)


def series(n=60, **day_mm):
    """1-based day -> mm, e.g. series(d5=6.0)."""
    s = np.zeros(n)
    for d, mm in day_mm.items():
        s[int(d[1:]) - 1] = mm
    return xr.DataArray(s, dims=("step",), coords={"step": np.arange(1, n + 1)})


def onset_day(rule, rain, thresh=None):
    idx = int(first_onset_index(rule(rain, thresh, dim="step"), "step"))
    return None if idx < 0 else idx + 1  # 1-based day


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_lead_sum_alignment_and_start():
    r = xr.DataArray(np.arange(10, dtype=float), dims="t")
    s = lead_sum(r, "t", 3)
    assert s.values[0] == 0 + 1 + 2 and s.values[7] == 7 + 8 + 9
    assert np.isnan(s.values[8:]).all()
    s2 = lead_sum(r, "t", 3, start=2)
    assert s2.values[0] == 2 + 3 + 4
    assert np.isnan(s2.values[6:]).all()


def test_lead_all_and_nan():
    r = xr.DataArray([1, 1, 1, 0, 1, np.nan, 1, 1], dims="t", name="r").astype(float)
    a = lead_all(r >= 1, "t", 3)
    assert a.values.tolist() == [True, False, False, False, False, False, False, False]


def test_has_dry_run_semantics():
    #        idx: 0 1 2 3 4 5 6 7 8 9 10 11
    r = xr.DataArray([5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5], dims="t").astype(float)
    # window of 5 days from d, run of 3 dry days
    h = has_dry_run(r, "t", days=5, run_days=3, dry_below=1)
    # d=0: days 0..4 contain 2,3,4 dry -> 1 ; d=2: days 2..6 -> 1 ; d=3: days 3..7 only 2 dry -> 0
    assert h.values[0] == 1 and h.values[2] == 1 and h.values[3] == 0
    assert np.isnan(h.values[8:]).all()  # window overruns
    r2 = r.copy()
    r2[9] = np.nan
    assert np.isnan(has_dry_run(r2, "t", 5, 3, 1).values[5])  # NaN inside window => not evaluable


def test_min_sub_sum_semantics():
    r = xr.DataArray([10, 10, 0, 0, 10, 10, 10, 10, 10, 10], dims="t").astype(float)
    m = min_sub_sum(r, "t", days=6, sub_days=2)
    # d=0: sub-windows (0,1)=20,(1,2)=10,(2,3)=0,(3,4)=10,(4,5)=20 -> 0
    assert m.values[0] == 0
    # d=4: (4,5),(5,6),(6,7),(7,8),(8,9) all 20 -> 20
    assert m.values[4] == 20
    assert np.isnan(m.values[5:]).all()


# --------------------------------------------------------------------------- #
# LegacyRule
# --------------------------------------------------------------------------- #
def test_legacy_basic_window():
    rule = LegacyRule(wet_init=1, wet_spell=3)
    # days 4,5,6 = 8 mm each: all >= 1, sum 24 > 20
    assert onset_day(rule, series(d4=8, d5=8, d6=8), 20) == 4
    # one day under wet_init breaks the all-wet requirement even though sum is large
    assert onset_day(rule, series(d4=30, d5=0.5, d6=30), 20) is None
    # strict >: exactly 20 is not onset
    assert onset_day(rule, series(d4=10, d5=5, d6=5), 20) is None
    assert onset_day(rule, series(d4=10, d5=5, d6=5.01), 20) == 4


def test_legacy_dry_veto():
    rule = LegacyRule(wet_init=1, wet_spell=3, dry_spell=5, dry_extent=12)
    assert rule.dry_active and rule.lookahead == 11
    wet = dict(d4=8, d5=8, d6=8)
    # days 7..11 dry (5 days) inside [4, 16) => vetoed at d=4
    assert onset_day(rule, series(**wet), 20) is None
    # wet days at 9 and 14 split [7, 16) into runs of 2, 4, 1 => no 5-day run => onset at 4
    assert onset_day(rule, series(**wet, d9=2, d14=2), 20) == 4
    # only day 9 wet leaves days 10..15 (6 dry) => still vetoed
    assert onset_day(rule, series(**wet, d9=2), 20) is None
    # dry_extent <= wet_spell disables the veto entirely (0 is the conventional "off")
    off = LegacyRule(wet_init=1, wet_spell=3, dry_spell=5, dry_extent=0)
    assert not off.dry_active and onset_day(off, series(**wet), 20) == 4
    also_off = LegacyRule(wet_init=1, wet_spell=3, dry_spell=2, dry_extent=3)
    assert not also_off.dry_active and onset_day(also_off, series(**wet), 20) == 4


def test_legacy_resolves_from_flat_config_and_validates():
    with pytest.raises(ValueError):
        LegacyRule(dry_spell=7, dry_extent=5)  # ROMP's existing logic-conflict rule
    with pytest.raises(ValueError):
        LegacyRule(wet_spell=3, dry_spell=0, dry_extent=10)


# --------------------------------------------------------------------------- #
# TwoStageAccumulation
# --------------------------------------------------------------------------- #
def test_two_stage_hand_cases():
    rule = TwoStageAccumulation(stage1_days=10, stage1_mm=10, stage2_days=20, stage2_mm=20)
    assert rule.lookahead == 29

    # Rain on days 5 (6 mm), 12 (4 mm), 20 (20 mm).
    # Stage 1 [d, d+9] needs both day 5 and 12  => d in {3,4,5}
    # Stage 2 [d+10, d+29] needs day 20         => d <= 10
    ok = series(d5=6.0, d12=4.0, d20=20.0)
    m = rule(ok, None, dim="step").values
    assert m[2:5].all() and not m[:2].any() and not m[5:].any()
    assert onset_day(rule, ok) == 3

    # stage 2 short by 1 mm => no onset anywhere
    assert onset_day(rule, series(d5=6.0, d12=4.0, d20=19.0)) is None
    # inclusive=False rejects the exactly-equal case
    assert onset_day(TwoStageAccumulation(inclusive=False), ok) is None
    # tail: day 40 cannot see 29 days ahead in 60 => False even with huge rain
    assert not rule(series(d40=500, d55=500), None, dim="step").values[39]
    # per-cell thresh is ignored
    thresh = xr.DataArray([1000.0], dims="cell")
    st = ok.expand_dims(cell=[0])
    assert first_onset_index(rule(st, thresh, dim="step"), "step").values.tolist() == [2]


# --------------------------------------------------------------------------- #
# MoronRobertson
# --------------------------------------------------------------------------- #
def test_moron_robertson_trigger():
    rule = MoronRobertson(wet_day_mm=1, wet_days=5, threshold=20, follow_days=0)
    assert not rule.veto_active and rule.lookahead == 4
    # first day > 1 mm and 5-day total > 20
    assert onset_day(rule, series(d4=2, d6=10, d8=10)) == 4
    # first day only 1.0 (not > 1) => day 4 is not the first wet day; day 6 is
    assert onset_day(rule, series(d4=1.0, d6=10, d8=10, d10=5)) == 6
    # threshold=None uses the per-cell thresh argument
    r2 = MoronRobertson(follow_days=0)
    assert onset_day(r2, series(d4=2, d6=10, d8=10), thresh=20) == 4
    assert onset_day(r2, series(d4=2, d6=10, d8=10), thresh=25) is None
    with pytest.raises(ValueError):
        r2(series(d4=2), None, dim="step")


def test_moron_robertson_veto():
    rule = MoronRobertson(wet_day_mm=1, wet_days=5, threshold=20, follow_days=20, dry_window=10, dry_min_mm=5)
    assert rule.lookahead == 19
    trigger = dict(d4=2, d6=10, d8=10)
    # nothing after day 8 => a 10-day period with < 5 mm exists in [4, 24) => vetoed
    assert onset_day(rule, series(**trigger)) is None
    # keep every 10-day sub-window >= 5 mm: rain 5 mm every 9 days
    kept = series(**trigger, d13=5, d22=5, d31=5)
    assert onset_day(rule, kept) == 4
    # 4.9 mm is not enough for a sub-window
    assert onset_day(rule, series(**trigger, d13=4.9, d22=5, d31=5)) is None
    with pytest.raises(ValueError):
        MoronRobertson(follow_days=5, dry_window=10)
