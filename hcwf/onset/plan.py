"""Compile an onset definition into a flat, backend-friendly `~hcwf.onset.plan.Plan` and turn thresholds into integer amounts.

A `Plan` holds only ints, strings and tuples, so a numba or GPU backend can read it without touching
`hcwf.onset.model`. Every amount a backend compares against, fixed or per-cell, is looked up as
``amounts[key][series]``. `compile` builds the plan from a `hcwf.onset.model.Definition`; `resolve_amounts` turns the
amounts the plan names into integer (or, for exact rain, float) arrays. The backends in `hcwf.onset.backends` consume
both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hcwf.onset import model

GE, GT, LE, LT = 0, 1, 2, 3
"""Comparator codes: greater or equal, greater, less or equal, less."""
OP_CODES = {"ge": GE, "gt": GT, "le": LE, "lt": LT}
"""The comparator code of each operator name used in a definition."""

TOTAL, WET_DAYS = 0, 1
"""Requirement kinds: compare the span sum in tenths, or compare the wet-day count."""
DRY_RUN, DRY_WINDOW = 0, 1
"""Veto kinds: a run of dry days, or a window whose total is below an amount."""
INSIDE, STARTS_INSIDE, BEGINS = 0, 1, 2
"""Veto placements: the spell lies wholly inside the span, starts inside it, or begins at the span start."""
PLACEMENT_CODES = {"inside": INSIDE, "starts_inside": STARTS_INSIDE, "begins": BEGINS}
"""The placement code of each placement name used in a definition."""

FIRST_PASSAGE, CURVE = "first_passage", "curve"
"""The two plan families: first passage over conditions and vetoes, or a curve (baseline) rule."""


@dataclass(frozen=True)
class Cmp:
    """A comparator code (`GE`/`GT`/`LE`/`LT`) against the integer amount ``amounts[key][series]``."""

    op: int
    """The comparator code."""
    key: str
    """The key of the amount in the amounts dict of `resolve_amounts`."""


@dataclass(frozen=True)
class CondPlan:
    """A requirement.

    Its span covers ``[d + first, d + last)`` for candidate index ``d``. A ``since_start`` span instead runs from the
    job's first candidate to ``d + last``.
    """

    kind: int
    """`TOTAL` (compare the span sum in tenths with `cmp`) or `WET_DAYS` (compare the wet-day count with `count`)."""
    first: int
    """Offset of the first day of the span from the candidate index."""
    last: int
    """Offset one past the last day of the span."""
    since_start: bool
    """Whether the span starts at the job's first candidate instead of ``d + first``."""
    cmp: Cmp | None = None
    """The comparison of the span sum (kind `TOTAL`); None otherwise."""
    op: int = GE
    """The comparator code of the wet-day count (kind `WET_DAYS`)."""
    count: int = 0
    """The plain integer wet-day count to compare with (kind `WET_DAYS`)."""


@dataclass(frozen=True)
class VetoPlan:
    """A veto: a dry spell of ``length`` days (`DRY_RUN` or `DRY_WINDOW`) wholly inside or starting inside a span.

    The span covers ``[d + first, d + last)`` for candidate index ``d``.
    """

    kind: int
    """`DRY_RUN` or `DRY_WINDOW`."""
    length: int
    """The length of the spell in days."""
    placement: int
    """Where the spell lies in the span: `INSIDE`, `STARTS_INSIDE` or `BEGINS`."""
    first: int
    """Offset of the first day of the span from the candidate index."""
    last: int
    """Offset one past the last day of the span."""
    cmp: Cmp | None = None
    """The comparison of the window total; only set for `DRY_WINDOW`."""


@dataclass(frozen=True)
class AmountSpec:
    """How to build ``amounts[key]``.

    The kinds are ``"fixed"`` (tenths, or mm when the amount is finer than a tenth: then ``tenths`` is None),
    ``"threshold"`` (per-cell mm, rounded for the comparator) and ``"baseline"`` (thousandths of mm).
    """

    key: str
    """The key of the amount."""
    kind: str
    """``"fixed"``, ``"threshold"`` or ``"baseline"``."""
    tenths: int | None = 0
    """The fixed amount in tenths of mm; None when it is finer than a tenth (``mm`` is then used)."""
    name: str = ""
    """The name of the threshold or baseline in the thresholds dict (kinds ``"threshold"`` and ``"baseline"``)."""
    op: int = GE
    """The comparator code the threshold is rounded for (kind ``"threshold"``)."""
    mm: float = 0.0
    """The fixed amount in mm (kind ``"fixed"``)."""


@dataclass(frozen=True)
class Plan:
    """Everything a backend needs to run one definition.

    For first passage, conditions and vetoes are evaluated per candidate. For a curve, ``curve_key`` names the baseline
    amount (thousandths of mm). ``horizon`` and ``keep_missing`` are run options of `compile`, not part of the
    definition.
    """

    family: str
    """`FIRST_PASSAGE` or `CURVE`."""
    wet: Cmp
    """The wet-day comparison."""
    conditions: tuple[CondPlan, ...]
    """The requirements of a first-passage rule, all of which must hold."""
    vetoes: tuple[VetoPlan, ...]
    """The vetoes of a first-passage rule, any of which rejects a candidate."""
    report_first_wet: bool
    """Whether the reported day is the first wet day of `report_span` instead of the candidate."""
    report_span: tuple[int, int, bool] | None
    """The window ``(first, last, since_start)`` searched for the first wet day when `report_first_wet` is set."""
    curve_key: str | None
    """The key of the baseline amount (thousandths of mm) for a curve plan."""
    curve_after_min: bool
    """For a curve, report the day after the minimum rather than the minimum itself."""
    amount_specs: tuple[AmountSpec, ...]
    """How to build every amount the plan reads."""
    dry: Cmp | None = None
    """The dry-day comparison; None means a dry day is a valid day that is not wet."""
    horizon: str = model.DEFAULT_HORIZON
    """What a span past the data does: ``"censor"``, ``"truncate"`` or ``"reject"`` (see `compile`)."""
    keep_missing: bool = False
    """Whether missing days count as neither wet nor dry nor unknown (see `compile`)."""


def compile(definition: model.Definition, *, horizon: str = model.DEFAULT_HORIZON, missing: str = "flag") -> Plan:
    """Turn a definition into a plan (a pure function of the definition and the two run options).

    ``horizon`` says what a span past the data does. ``"censor"``: the candidate is unknown, so the search ends
    censored. ``"truncate"``: a veto sees only the data that exists (its spell must lie in the data; a veto unknown
    only for the horizon is False), while a requirement past the data stays unknown. ``"reject"``: as truncate, and a
    candidate whose veto span runs past the data is rejected. ``missing="keep"`` makes missing days neither wet nor
    dry nor unknown (they add 0 to totals), and a missing candidate day is rejected; the other policies are settled at
    ingest.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition to compile.
    horizon : {"censor", "truncate", "reject"}, default "censor"
        What a span that runs past the data does.
    missing : str, default "flag"
        The missing-rain policy; one of `hcwf.onset.model.MISSING_POLICIES`. Only ``"keep"`` changes the plan.

    Returns
    -------
    Plan
        The compiled plan.

    Raises
    ------
    ValueError
        ``horizon`` or ``missing`` is not a known policy, a veto span is ``season to date``, or a rule that reports the
        first wet day has no ``report_span``.
    """
    if horizon not in model.HORIZON_POLICIES:
        raise ValueError(f"horizon must be one of {model.HORIZON_POLICIES}, got {horizon!r}")
    if missing not in model.MISSING_POLICIES:
        raise ValueError(f"missing must be one of {model.MISSING_POLICIES}, got {missing!r}")
    specs: dict[str, AmountSpec] = {}

    def amount_cmp(cmp: model.Comparison) -> Cmp:
        """Register the amount of a comparison in `specs` and return its comparator and key."""
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
    """Convert a 1-based inclusive span to offsets from the candidate index.

    Parameters
    ----------
    span : hcwf.onset.model.Span
        The span, with 1-based inclusive ``first`` and ``last`` days, or one that starts at the search start.

    Returns
    -------
    tuple of (int, int, bool)
        ``(first, last_exclusive, since_search_start)``: the offset of the first day, the offset one past the last day,
        and whether the span starts at the job's first candidate.
    """
    if span.since_search_start:
        return 0, span.last, True
    return span.first - 1, span.last, False


def resolve_amounts(plan: Plan, thresholds: dict[str, np.ndarray], n_series: int | None = None, *,
                    exact: bool = False) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the amounts every key of the plan reads, and the series with a NaN amount.

    With ``exact=True`` (``rain_resolution "exact"``) the amounts are ``float64`` mm, unrounded, to compare with
    ``float64`` rain; curve rules are not available then.

    A threshold given as ``(S, W)`` (one column per verification window, spec section 13) gives ``(S, W)`` amounts;
    fixed amounts and ``(S,)`` thresholds stay ``(S,)``. A NaN in any window column marks the whole series invalid.

    Thresholds are per-series values in mm. Named thresholds are rounded for their comparator
    (``q = round(10 T, 6)``; ``ge`` and ``lt`` take ceil, ``gt`` and ``le`` take floor); curve baselines are thousandths
    of a mm, half to even. A NaN or infinite threshold marks the series invalid and its amount is set to 0.

    Parameters
    ----------
    plan : Plan
        The plan whose amounts to resolve.
    thresholds : dict of str to numpy.ndarray
        The per-series thresholds in mm by name, each ``float64`` of shape ``(series,)`` or ``(series, window)``.
    n_series : int, optional
        The number of series. Only needed when the plan uses no named threshold; otherwise the largest first
        dimension of ``thresholds`` is used.
    exact : bool, default False
        Return unrounded ``float64`` mm instead of integers.

    Returns
    -------
    amounts : dict of str to numpy.ndarray
        The amount of each key of the plan: ``int64`` (tenths, or thousandths of mm for baselines), or ``float64`` mm
        with ``exact``; shape ``(series,)`` or ``(series, window)``.
    invalid : numpy.ndarray of bool, shape (series,)
        True for a series with a NaN or infinite amount.

    Raises
    ------
    ValueError
        ``n_series`` is missing and ``thresholds`` holds no array, or a fixed amount is finer than 0.1 mm and ``exact`` is
        false.
    NotImplementedError
        ``exact`` is set for a curve plan.
    KeyError
        The plan needs a threshold that ``thresholds`` does not hold.
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
    """Refuse exact (float) rain for a stitched search.

    Stitched searches add history and forecast prefix sums, which float rain cannot reproduce exactly.

    Parameters
    ----------
    *tables : numpy.ndarray
        The rain tables of the search; any of a float dtype is refused.

    Raises
    ------
    NotImplementedError
        A table has a float dtype.
    """
    if any(np.asarray(t).dtype.kind == "f" for t in tables):
        raise NotImplementedError("rain_resolution 'exact' is not available with forecast_history 'observed'")
