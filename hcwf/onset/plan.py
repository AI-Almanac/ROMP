"""Compile an onset definition into a flat, backend-friendly Plan and turn thresholds into integer amounts.

A Plan holds only ints, strings and tuples, so a numba or GPU backend can read it
without touching model.py. Every amount a backend compares against, fixed or
per-cell, is looked up as amounts[key][series].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hcwf.onset import model

GE, GT, LE, LT = 0, 1, 2, 3
OP_CODES = {"ge": GE, "gt": GT, "le": LE, "lt": LT}

TOTAL, WET_DAYS = 0, 1
DRY_RUN, DRY_WINDOW = 0, 1
INSIDE, STARTS_INSIDE, BEGINS = 0, 1, 2
PLACEMENT_CODES = {"inside": INSIDE, "starts_inside": STARTS_INSIDE, "begins": BEGINS}

FIRST_PASSAGE, CURVE = "first_passage", "curve"


@dataclass(frozen=True)
class Cmp:
    """A comparator code (GE/GT/LE/LT) against the integer amount amounts[key][series]."""

    op: int
    key: str


@dataclass(frozen=True)
class CondPlan:
    """A requirement. Its span covers [d + first, d + last) for candidate index d.

    since_start spans instead run from the job's first candidate to d + last.
    kind is TOTAL (compare the span sum in tenths with cmp) or WET_DAYS (compare
    the wet-day count with the plain integer count).
    """

    kind: int
    first: int
    last: int
    since_start: bool
    cmp: Cmp | None = None
    op: int = GE
    count: int = 0


@dataclass(frozen=True)
class VetoPlan:
    """A veto: a dry spell of `length` days (DRY_RUN or DRY_WINDOW) wholly inside or starting inside a span.

    The span covers [d + first, d + last). cmp is only set for DRY_WINDOW.
    """

    kind: int
    length: int
    placement: int
    first: int
    last: int
    cmp: Cmp | None = None


@dataclass(frozen=True)
class AmountSpec:
    """How to build amounts[key]: kind "fixed" (tenths, or mm when finer: tenths None), "threshold" (per-cell mm, rounded for op) or "baseline" (thousandths of mm)."""

    key: str
    kind: str
    tenths: int | None = 0
    name: str = ""
    op: int = GE
    mm: float = 0.0


@dataclass(frozen=True)
class Plan:
    """Everything a backend needs to run one definition.

    family is FIRST_PASSAGE or CURVE. wet is the wet-day comparison. For first
    passage, conditions and vetoes are evaluated per candidate; when
    report_first_wet is set, report_span (first, last, since_start) is the
    window searched for the first wet day. For curve, curve_key names the
    baseline amount (thousandths of mm) and curve_after_min selects the
    day after the minimum rather than the minimum itself.

    dry is the dry-day comparison (None: a dry day is a valid day that is not
    wet). horizon ("censor", "truncate" or "reject") and keep_missing are run
    options of compile(), not part of the definition: see compile().
    """

    family: str
    wet: Cmp
    conditions: tuple[CondPlan, ...]
    vetoes: tuple[VetoPlan, ...]
    report_first_wet: bool
    report_span: tuple[int, int, bool] | None
    curve_key: str | None
    curve_after_min: bool
    amount_specs: tuple[AmountSpec, ...]
    dry: Cmp | None = None
    horizon: str = "censor"
    keep_missing: bool = False


def compile(definition: model.Definition, *, horizon: str = "censor", missing: str = "flag") -> Plan:
    """Turn a Definition into a Plan (a pure function of the definition and the two run options).

    horizon says what a span past the data does. "censor": the candidate is unknown, so
    the search ends CENSORED. "truncate": a veto sees only the data that exists (its
    spell must lie in the data; a veto unknown only for the horizon is False), while
    a requirement past the data stays unknown. "reject": as truncate, and a candidate
    whose veto span runs past the data is rejected. missing "keep" makes missing days
    neither wet nor dry nor unknown (they add 0 to totals), and a missing candidate
    day is rejected; the other policies are settled at ingest.
    """
    if horizon not in model.HORIZON_POLICIES:
        raise ValueError(f"horizon must be one of {model.HORIZON_POLICIES}, got {horizon!r}")
    if missing not in model.MISSING_POLICIES:
        raise ValueError(f"missing must be one of {model.MISSING_POLICIES}, got {missing!r}")
    specs: dict[str, AmountSpec] = {}

    def amount_cmp(cmp: model.Comparison) -> Cmp:
        """Register the amount of a comparison and return its comparator and key."""
        op = OP_CODES[cmp.op]
        if cmp.amount.threshold is None:
            tenths = cmp.amount.tenths
            key = f"fixed:{tenths}" if tenths is not None else f"fixed:{cmp.amount.mm!r}mm"
            specs.setdefault(key, AmountSpec(key, "fixed", tenths=None if tenths is None else int(tenths),
                                             mm=cmp.amount.fixed_mm))
        else:
            key = f"{cmp.amount.threshold}:{cmp.op}"
            specs.setdefault(key, AmountSpec(key, "threshold", name=cmp.amount.threshold, op=op))
        return Cmp(op, key)

    wet = amount_cmp(definition.wet_day)
    dry = None if definition.dry_day is None else amount_cmp(definition.dry_day)
    options = {"dry": dry, "horizon": horizon, "keep_missing": missing == "keep"}
    rule = definition.rule
    if isinstance(rule, model.Curve):
        key = f"{rule.baseline}:baseline"
        specs[key] = AmountSpec(key, "baseline", name=rule.baseline)
        return Plan(CURVE, wet, (), (), False, None, key, rule.report == "day_after_minimum", tuple(specs.values()),
                    **options)

    conditions = []
    for cond in rule.require:
        first, last, since = _offsets(cond.span)
        if isinstance(cond, model.Total):
            conditions.append(CondPlan(TOTAL, first, last, since, cmp=amount_cmp(cond.comparison)))
        else:
            conditions.append(CondPlan(WET_DAYS, first, last, since, op=OP_CODES[cond.op], count=int(cond.count)))
    vetoes = []
    for veto in rule.reject_if:
        first, last, since = _offsets(veto.span)
        if since:
            raise ValueError("a veto span cannot be 'season to date'")
        placement = PLACEMENT_CODES[veto.placement]
        if isinstance(veto.spell, model.DryRun):
            vetoes.append(VetoPlan(DRY_RUN, int(veto.spell.length), placement, first, last))
        else:
            vetoes.append(VetoPlan(DRY_WINDOW, int(veto.spell.days), placement, first, last,
                                   cmp=amount_cmp(veto.spell.comparison)))
    report_span = None
    if rule.report == "first_wet_day":
        if rule.report_span is None:
            raise ValueError("report first_wet_day needs a report_span")
        report_span = _offsets(rule.report_span)
    return Plan(FIRST_PASSAGE, wet, tuple(conditions), tuple(vetoes), rule.report == "first_wet_day",
                report_span, None, True, tuple(specs.values()), **options)


def _offsets(span: model.Span) -> tuple[int, int, bool]:
    """Convert a 1-based inclusive span to (first, last_exclusive, since_search_start) offsets from the candidate index."""
    if span.since_search_start:
        return 0, span.last, True
    return span.first - 1, span.last, False


def resolve_amounts(plan: Plan, thresholds: dict[str, np.ndarray], n_series: int | None = None, *,
                    exact: bool = False) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return (amounts, invalid): int64 amounts for every key the plan reads, and a bool (S,) of series with a NaN amount.

    With exact=True (rain_resolution "exact") the amounts are float64 mm, unrounded, to compare with
    float64 rain; curve rules are not available then.

    A threshold given as (S, W) (one column per verification window, spec
    section 13) gives (S, W) amounts; fixed amounts and (S,) thresholds stay (S,).
    A NaN in any window column marks the whole series invalid.

    Thresholds are per-series values in mm. Named thresholds are rounded for
    their comparator (q = round(10 T, 6); ge and lt take ceil, gt and le take
    floor); curve baselines are thousandths of a mm, half to even. A NaN or
    infinite threshold marks the series invalid and its amount is set to 0.
    n_series is only needed when the plan uses no named threshold.
    """
    sizes = {np.asarray(v).shape[0] for v in thresholds.values() if np.ndim(v) in (1, 2)}
    if n_series is None:
        if not sizes:
            raise ValueError("n_series is required when no threshold array is given")
        n_series = max(sizes)
    amounts: dict[str, np.ndarray] = {}
    invalid = np.zeros(n_series, dtype=bool)
    if exact and plan.family == CURVE:
        raise NotImplementedError("curve rules need rain_resolution 'tenths'")
    for spec in plan.amount_specs:
        if spec.kind == "fixed":
            if exact:
                amounts[spec.key] = np.full(n_series, spec.mm, dtype=np.float64)
            elif spec.tenths is None:
                raise ValueError(f"the amount {spec.mm} mm is finer than 0.1 mm. Fix: round it to a whole tenth "
                                 "of a mm, or use rain_resolution 'exact'")
            else:
                amounts[spec.key] = np.full(n_series, spec.tenths, dtype=np.int64)
            continue
        if spec.name not in thresholds:
            raise KeyError(f"threshold {spec.name!r} is needed by the definition but was not supplied")
        raw = np.asarray(thresholds[spec.name], dtype=np.float64)
        values = np.broadcast_to(raw, (n_series, *raw.shape[1:]) if raw.ndim == 2 else (n_series,))
        bad = ~np.isfinite(values)
        safe = np.where(bad, 0.0, values)
        if exact:
            amounts[spec.key] = safe
            invalid |= bad.any(axis=1) if bad.ndim == 2 else bad
            continue
        if spec.kind == "baseline":
            converted = np.rint(safe * 1000.0)
        else:
            q = np.round(10.0 * safe, 6)
            converted = np.ceil(q) if spec.op in (GE, LT) else np.floor(q)
        amounts[spec.key] = converted.astype(np.int64)
        invalid |= bad.any(axis=1) if bad.ndim == 2 else bad
    return amounts, invalid


def reject_exact_stitched(*tables: np.ndarray) -> None:
    """Stitched searches add history and forecast prefix sums, which float rain cannot reproduce exactly."""
    if any(np.asarray(t).dtype.kind == "f" for t in tables):
        raise NotImplementedError("rain_resolution 'exact' is not available with forecast_history 'observed'")
