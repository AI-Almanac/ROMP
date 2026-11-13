"""Generic contract checks for every registered OnsetRule.

These test *properties*, not semantics, so they apply unchanged to any rule a
contributor (or an LLM) writes. Semantics belong in per-rule hand-case tests
(see test_onset_rule_builtins.py).

Fixture assumption: continuous heavy rain (FAVOURABLE_MM every day) is an onset
under every rule at every evaluable candidate day. A rule for which that is
false should be added to ``NOT_FIRED_BY_HEAVY_RAIN``.
"""

import json

import numpy as np
import pytest
import xarray as xr

from momp.stats import onset_rule as orl
from momp.stats.onset_rule import OnsetRule, first_onset_index, list_rules

FAVOURABLE_MM = 50.0
THRESH = 20.0
NOT_FIRED_BY_HEAVY_RAIN: set[str] = set()


def _instances():
    """Default instance of every registered rule plus variants that switch on optional logic."""
    out = [cls() for cls in list_rules().values()]
    out += [
        orl.LegacyRule(wet_spell=3, dry_spell=7, dry_extent=21),
        orl.LegacyRule(wet_init=2.0, wet_spell=5, dry_spell=10, dry_extent=30),
        orl.TwoStageAccumulation(inclusive=False),
        orl.TwoStageAccumulation(stage1_days=3, stage1_mm=15, stage2_days=7, stage2_mm=10),
        orl.MoronRobertson(threshold=5.0),
        orl.MoronRobertson(follow_days=0),
    ]
    return out


RULES = _instances()
IDS = [r.label() for r in RULES]


def _random_rain(rng, shape, nan_frac=0.02):
    rain = rng.exponential(scale=6.0, size=shape)
    rain[rng.random(shape) < 0.45] = 0.0
    drizzle = rng.random(shape) < 0.1
    rain[drizzle] = rng.uniform(0.0, 0.99, size=drizzle.sum())
    if nan_frac:
        rain[rng.random(shape) < nan_frac] = np.nan
    return rain


def _da(values, dim="time", extra_dims=()):
    dims = tuple(extra_dims) + (dim,)
    return xr.DataArray(values, dims=dims)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_is_frozen_dataclass_with_registered_name(rule):
    assert isinstance(rule, OnsetRule)
    assert type(rule).__dataclass_params__.frozen
    assert list_rules()[rule.name] is type(rule)
    with pytest.raises(Exception):
        setattr(rule, next(iter(rule.params())), 1)


@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_output_contract(rule):
    rng = np.random.default_rng(0)
    rain = _da(_random_rain(rng, (40, 90)), extra_dims=("cell",))
    mask = rule(rain, THRESH, dim="time")
    assert isinstance(mask, xr.DataArray)
    assert mask.dtype == bool
    assert mask.dims == rain.dims and mask.shape == rain.shape
    assert isinstance(rule.lookahead, (int, np.integer)) and rule.lookahead >= 0


@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_causality_bounded_by_lookahead(rule):
    """mask[d] must not change when rain outside [d, d+lookahead] is perturbed."""
    rng = np.random.default_rng(1)
    n, T = 30, 80
    base = _random_rain(rng, (n, T), nan_frac=0.0)
    ref = rule(_da(base, extra_dims=("cell",)), THRESH, dim="time").values
    L = rule.lookahead

    for d in [0, 1, 5, 17, 33, T - L - 2, T - L - 1]:
        if d < 0 or d + L >= T:
            continue
        inside = np.zeros(T, dtype=bool)
        inside[d : d + L + 1] = True
        for fill in (0.0, FAVOURABLE_MM, np.nan):
            pert = base.copy()
            pert[:, ~inside] = fill
            got = rule(_da(pert, extra_dims=("cell",)), THRESH, dim="time").values
            assert (got[:, d] == ref[:, d]).all(), (
                f"{rule.label()}: mask[d={d}] changed when rain outside [d, d+{L}] set to {fill}"
            )


@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_lookahead_is_exact_via_tail(rule):
    """Fewer than `lookahead` days after d => False; exactly `lookahead` => can fire.

    With test_causality this pins `lookahead` exactly: understating it fails
    causality, overstating it fails here.
    """
    if rule.name in NOT_FIRED_BY_HEAVY_RAIN:
        pytest.skip("rule is not fired by uniform heavy rain")
    T = rule.lookahead + 10
    rain = _da(np.full(T, FAVOURABLE_MM))
    mask = rule(rain, THRESH, dim="time").values
    L = rule.lookahead
    assert mask[: T - L].all(), f"{rule.label()}: heavy rain should fire wherever {L} days remain"
    if L > 0:
        assert not mask[T - L :].any(), f"{rule.label()}: fired with fewer than lookahead={L} days remaining"


@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_nan_inside_window_fails_closed(rule):
    if rule.name in NOT_FIRED_BY_HEAVY_RAIN:
        pytest.skip("rule is not fired by uniform heavy rain")
    L = rule.lookahead
    T = L + 20
    d = 5
    for k in range(L + 1):
        rain = np.full(T, FAVOURABLE_MM)
        rain[d + k] = np.nan
        mask = rule(_da(rain), THRESH, dim="time").values
        assert not mask[d], f"{rule.label()}: NaN at offset {k} inside window did not block onset at d={d}"
    # control: NaN just outside the window does not block
    rain = np.full(T, FAVOURABLE_MM)
    rain[d + L + 1] = np.nan
    assert rule(_da(rain), THRESH, dim="time").values[d]


@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_broadcast_invariance(rule):
    """Vectorised over (a, b, time) equals cell-by-cell, with a per-cell threshold."""
    rng = np.random.default_rng(2)
    A, B, T = 4, 5, 70
    rain = xr.DataArray(_random_rain(rng, (A, B, T)), dims=("a", "b", "time"))
    thresh = xr.DataArray(rng.uniform(5, 30, size=(A, B)), dims=("a", "b"))
    full = rule(rain, thresh, dim="time")
    for i in range(A):
        for j in range(B):
            single = rule(rain.isel(a=i, b=j), float(thresh[i, j]), dim="time")
            np.testing.assert_array_equal(full.isel(a=i, b=j).values, single.values)


@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_dim_name_agnostic_and_position_agnostic(rule):
    rng = np.random.default_rng(3)
    vals = _random_rain(rng, (6, 60))
    as_time = rule(xr.DataArray(vals, dims=("cell", "time")), THRESH, dim="time").values
    as_step = rule(xr.DataArray(vals, dims=("cell", "step")), THRESH, dim="step").values
    leading = rule(xr.DataArray(vals.T, dims=("step", "cell")), THRESH, dim="step").values.T
    np.testing.assert_array_equal(as_time, as_step)
    np.testing.assert_array_equal(as_time, leading)


@pytest.mark.parametrize("rule", RULES, ids=IDS)
def test_identity_cache_key_fingerprint_label(rule):
    same = type(rule)(**rule.params())
    assert hash(rule.cache_key()) == hash(same.cache_key())
    assert rule.fingerprint() == same.fingerprint()
    assert rule.label() == same.label()

    # change one numeric parameter within bounds => identity changes
    for f_name, v in rule.params().items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        meta = {f.name: f.metadata for f in orl.fields(rule)}[f_name]
        new_v = v + 1 if ("max" not in meta or v + 1 <= meta["max"]) else v - 1
        other = type(rule)(**{**rule.params(), f_name: new_v})
        assert other.cache_key() != rule.cache_key()
        assert other.fingerprint() != rule.fingerprint()
        assert f_name in other.label() or other.non_default_params().get(f_name) is None
        break

    if not rule.non_default_params():
        assert rule.label() == rule.name


@pytest.mark.parametrize("cls", list(list_rules().values()), ids=list(list_rules()))
def test_schema_and_describe_are_json_serialisable(cls):
    schema = cls.schema()
    assert schema["name"] == cls.name and schema["doc"]
    json.dumps(schema)
    json.dumps(cls().describe(), default=str)
    for p in schema["params"]:
        assert {"name", "type", "default"} <= set(p)


@pytest.mark.parametrize("cls", list(list_rules().values()), ids=list(list_rules()))
def test_bounds_are_enforced(cls):
    for f in orl.fields(cls):
        if "max" in f.metadata:
            with pytest.raises(ValueError):
                cls(**{f.name: f.metadata["max"] + 1})
        if "min" in f.metadata:
            with pytest.raises(ValueError):
                cls(**{f.name: f.metadata["min"] - 1})


def _seasonal_obs(rng, n_cells=200, T=150):
    """Synthetic season: dry start, ramping to wet. Days 0..T."""
    p_wet = np.linspace(0.05, 0.75, T)
    wet = rng.random((n_cells, T)) < p_wet
    amt = rng.exponential(scale=9.0, size=(n_cells, T))
    return np.where(wet, amt, 0.0)


@pytest.mark.parametrize("cls", list(list_rules().values()), ids=list(list_rules()))
def test_defaults_are_not_degenerate_on_seasonal_obs(cls):
    """Health check: at defaults the rule finds onset in most cells and not all on day 0."""
    rng = np.random.default_rng(4)
    rain = xr.DataArray(_seasonal_obs(rng), dims=("cell", "time"))
    rule = cls()
    idx = first_onset_index(rule(rain, THRESH, dim="time"), "time").values
    found = idx >= 0
    assert found.mean() > 0.5, f"{cls.name}: onset found in only {found.mean():.0%} of cells"
    assert (idx[found] == 0).mean() < 0.5, f"{cls.name}: onset is on day 0 in most cells (trivial)"
    assert np.std(idx[found]) > 1.0, f"{cls.name}: onset dates have no spread"


def test_resolve_rule_defaults_to_legacy_from_flat_config():
    r = orl.resolve_rule(None, None, wet_init=2.0, wet_spell=4, dry_spell=6, dry_extent=12, unrelated=1)
    assert r == orl.LegacyRule(wet_init=2.0, wet_spell=4, dry_spell=6, dry_extent=12)
    # explicit params win over flat keys
    r2 = orl.resolve_rule("legacy", {"wet_spell": 7}, wet_spell=3)
    assert r2.wet_spell == 7
    # instance passthrough
    inst = orl.TwoStageAccumulation()
    assert orl.resolve_rule(inst) is inst
    with pytest.raises(ValueError):
        orl.resolve_rule("nope")
    with pytest.raises(ValueError):
        orl.resolve_rule("two_stage", {"bogus": 1})
