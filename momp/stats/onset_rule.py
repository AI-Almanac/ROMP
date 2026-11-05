"""Pluggable rainy-season onset definitions.

An :class:`OnsetRule` turns a daily rainfall array into a boolean mask that is
``True`` at index ``d`` when a rainy season *starting on day d* satisfies the
definition. ROMP then takes the first ``True`` along the day axis
(:func:`first_onset_index`). The same rule object is applied to observations
(``dim="time"``) and to forecasts (``dim="step"``), so the two sides of a
benchmark can never drift apart.

Contract
--------
Every rule must satisfy the following; ``tests/test_onset_rule_conformance.py``
checks them generically for every registered rule.

* ``rule(rain, thresh, dim=...)`` returns a **bool** ``xr.DataArray`` with the
  same dims/shape as ``rain`` and no NaN.
* **Causal, bounded lookahead.** ``mask[d]`` depends only on
  ``rain[d : d + lookahead + 1]``; ``rule.lookahead`` reports the bound.
* **Fails closed.** Any NaN inside the inspected window, or fewer than
  ``lookahead`` days remaining after ``d``, yields ``False``.
* **Broadcasts.** Only ``dim`` is touched; every other dimension is carried
  through unchanged.
* ``thresh`` is the per-cell (or scalar) accumulation threshold ROMP loads
  from ``thresh_file`` / ``wet_threshold``. A rule may use it or ignore it.

Writing a rule
--------------
Subclass :class:`OnsetRule` as a frozen dataclass, declare parameters with
:func:`param` (bounds and units drive validation and the UI), implement
``lookahead`` and ``__call__`` using the window helpers in this module, and
register it::

    @register
    @dataclass(frozen=True)
    class MyRule(OnsetRule):
        name = "my_rule"
        doc = "One-line description shown to users."

        wet_days: int = param(5, min=1, max=30, unit="days", doc="Trigger window")
        min_mm: float = param(20.0, min=0, max=200, unit="mm")

        @property
        def lookahead(self) -> int:
            return self.wet_days - 1

        def __call__(self, rain, thresh, *, dim):
            return finalize(lead_sum(rain, dim, self.wet_days) >= self.min_mm)

Then add a hand-case test for the semantics; the conformance suite covers the
contract automatically.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, ClassVar, Hashable

import numpy as np
import xarray as xr

__all__ = [
    "OnsetRule",
    "param",
    "register",
    "get_rule_class",
    "list_rules",
    "resolve_rule",
    "first_onset_index",
    "lead_sum",
    "lead_all",
    "has_dry_run",
    "min_sub_sum",
    "finalize",
    "LegacyRule",
    "TwoStageAccumulation",
    "MoronRobertson",
]


# --------------------------------------------------------------------------- #
# Window helpers (all aligned to the candidate day d; NaN => not evaluable)
# --------------------------------------------------------------------------- #
def _align(rolled: xr.DataArray, dim: str, window: int, start: int) -> xr.DataArray:
    """Rolling results sit at the window *end*; move them to the candidate day.

    ``rolled[t]`` covers ``[t-window+1, t]``. We want ``out[d]`` to cover
    ``[d+start, d+start+window)``, i.e. ``rolled[d+start+window-1]``.
    """
    return rolled.shift({dim: -(window - 1 + start)})


def _lead_window_days(values: np.ndarray, days: int, start: int) -> tuple[list[np.ndarray], int]:
    """Each day of the window ``[d+start, d+start+days)`` for every candidate ``d`` that fits.

    ``values`` has the day axis last. Returns one slice per window day, each
    holding that day for the first ``n_fit`` candidates, and ``n_fit``. Slicing
    instead of an xarray rolling window keeps this a handful of array ops on
    large ensembles.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    n_fit = max(values.shape[-1] - start - days + 1, 0)
    return [values[..., start + k : start + k + n_fit] for k in range(days)], n_fit


def _with_day_axis_last(da: xr.DataArray, dim: str) -> np.ndarray:
    return np.moveaxis(da.values, da.get_axis_num(dim), -1)


def _like(da: xr.DataArray, dim: str, values_day_last: np.ndarray) -> xr.DataArray:
    return da.copy(data=np.moveaxis(values_day_last, -1, da.get_axis_num(dim)))


def lead_sum(rain: xr.DataArray, dim: str, days: int, start: int = 0) -> xr.DataArray:
    """Sum of ``rain[d+start : d+start+days]``. NaN if the window overruns or contains NaN."""
    values = _with_day_axis_last(rain, dim)
    window, n_fit = _lead_window_days(values, days, start)
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
    total = np.full(values.shape, np.nan, dtype=dtype)
    total[..., :n_fit] = sum(window[1:], window[0].astype(dtype))
    return _like(rain, dim, total)


def lead_all(cond: xr.DataArray, dim: str, days: int, start: int = 0) -> xr.DataArray:
    """``cond`` holds on every day of ``[d+start, d+start+days)``. False if the window overruns.

    ``cond`` is normally a comparison on rain (``rain >= x``); NaN rain compares
    False, so missing data fails closed. A NaN in a non-bool ``cond`` also fails.
    """
    holds = _with_day_axis_last(cond, dim) == 1
    window, n_fit = _lead_window_days(holds, days, start)
    every_day = np.zeros(holds.shape, dtype=bool)
    every_day[..., :n_fit] = np.logical_and.reduce(window)
    return _like(cond, dim, every_day)


def has_dry_run(
    rain: xr.DataArray,
    dim: str,
    days: int,
    run_days: int,
    dry_below: float,
    start: int = 0,
) -> xr.DataArray:
    """A run of ``run_days`` consecutive days with ``rain < dry_below`` lies fully
    inside ``[d+start, d+start+days)``.

    Returns a float array: ``1.0`` (run present), ``0.0`` (absent), or NaN where
    the window overruns the series **or contains missing rain** (unknown data
    cannot certify the absence of a dry spell). Compare with ``== 0`` to require
    "no dry run" while letting NaN fail closed.

    Note: ROMP's original scalar path treated NaN as "not dry" and let onset
    through; this helper is deliberately stricter.
    """
    if run_days < 1 or days < run_days:
        raise ValueError(f"need 1 <= run_days <= days, got run_days={run_days}, days={days}")
    dry_end = (rain < dry_below).astype(float).rolling({dim: run_days}, min_periods=run_days).sum() == run_days
    # dry_end[t] : days (t-run_days, t] are all dry. A run fully inside the
    # window ends at t in [d+start+run_days-1, d+start+days).
    n_ends = days - run_days + 1
    n_runs = dry_end.astype(float).rolling({dim: n_ends}, min_periods=n_ends).sum()
    n_runs = _align(n_runs, dim, n_ends, start + run_days - 1)
    evaluable = lead_sum(rain, dim, days, start).notnull()
    return (n_runs > 0).astype(float).where(n_runs.notnull() & evaluable)


def min_sub_sum(rain: xr.DataArray, dim: str, days: int, sub_days: int, start: int = 0) -> xr.DataArray:
    """Minimum over every ``sub_days``-day sub-window of ``[d+start, d+start+days)``
    of the rain total. NaN if the window overruns or contains NaN."""
    if sub_days < 1 or days < sub_days:
        raise ValueError(f"need 1 <= sub_days <= days, got sub_days={sub_days}, days={days}")
    sub = lead_sum(rain, dim, sub_days)  # aligned to sub-window start
    n_starts = days - sub_days + 1
    rolled = sub.rolling({dim: n_starts}, min_periods=n_starts).min()
    return _align(rolled, dim, n_starts, start)


def finalize(mask: xr.DataArray) -> xr.DataArray:
    """Coerce a possibly-NaN condition to the contract's bool output (NaN => False)."""
    return mask.fillna(False).astype(bool)


def first_onset_index(mask: xr.DataArray, dim: str) -> xr.DataArray:
    """Index along ``dim`` of the first True (-1 if none)."""
    cond = finalize(mask)
    return cond.argmax(dim).where(cond.any(dim), -1).astype(int)


# --------------------------------------------------------------------------- #
# Parameter metadata
# --------------------------------------------------------------------------- #
def param(
    default: Any,
    *,
    min: float | None = None,
    max: float | None = None,
    unit: str | None = None,
    doc: str | None = None,
    choices: tuple | None = None,
):
    """Declare a rule parameter with UI/validation metadata."""
    meta = {"min": min, "max": max, "unit": unit, "doc": doc, "choices": choices}
    return field(default=default, metadata={k: v for k, v in meta.items() if v is not None})


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class OnsetRule(ABC):
    """Base for onset definitions. Subclasses must be ``@dataclass(frozen=True)``."""

    name: ClassVar[str]
    doc: ClassVar[str] = ""

    # -- required -----------------------------------------------------------
    @property
    @abstractmethod
    def lookahead(self) -> int:
        """Days after the candidate day that the rule inspects (>= 0)."""

    @abstractmethod
    def __call__(self, rain: xr.DataArray, thresh, *, dim: str) -> xr.DataArray: ...

    # -- optional hooks ------------------------------------------------------
    def validate(self) -> None:
        """Cross-field checks; raise ``ValueError``. Bounds are checked automatically."""

    # -- provided ------------------------------------------------------------
    def __post_init__(self):
        self._check_bounds()
        self.validate()
        if self.lookahead < 0:
            raise ValueError(f"{self.name}: lookahead must be >= 0, got {self.lookahead}")

    def _check_bounds(self):
        for f in fields(self):
            v = getattr(self, f.name)
            m = f.metadata
            if v is None:
                continue
            if "choices" in m and v not in m["choices"]:
                raise ValueError(f"{self.name}.{f.name}={v!r} not in {m['choices']}")
            if "min" in m and v < m["min"]:
                raise ValueError(f"{self.name}.{f.name}={v} below minimum {m['min']}")
            if "max" in m and v > m["max"]:
                raise ValueError(f"{self.name}.{f.name}={v} above maximum {m['max']}")

    def params(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def non_default_params(self) -> dict[str, Any]:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            default = f.default if f.default is not MISSING else None
            if v != default:
                out[f.name] = v
        return out

    def cache_key(self) -> Hashable:
        return (self.name, tuple(sorted(self.params().items())))

    def fingerprint(self) -> str:
        """Short stable hash of the resolved definition, for labelling results."""
        blob = json.dumps({"name": self.name, "params": self.params()}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def label(self) -> str:
        """``name`` at defaults, else ``name (k=v, ...)``."""
        nd = self.non_default_params()
        if not nd:
            return self.name
        return f"{self.name} (" + ", ".join(f"{k}={v}" for k, v in nd.items()) + ")"

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Serialisable description for UIs / API: name, doc, parameter metadata."""
        out = []
        for f in fields(cls):
            entry = {
                "name": f.name,
                "type": getattr(f.type, "__name__", str(f.type)),
                "default": f.default if f.default is not MISSING else None,
            }
            entry.update(f.metadata)
            out.append(entry)
        return {"name": cls.name, "doc": cls.doc, "params": out}

    def describe(self) -> dict[str, Any]:
        d = self.schema()
        d["values"] = self.params()
        d["lookahead"] = self.lookahead
        d["fingerprint"] = self.fingerprint()
        d["label"] = self.label()
        return d

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self.params().items())})"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[OnsetRule]] = {}


def register(cls: type[OnsetRule]) -> type[OnsetRule]:
    name = getattr(cls, "name", None)
    if not name or not isinstance(name, str):
        raise TypeError(f"{cls.__name__} must define a string class attribute `name`")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"onset rule name {name!r} already registered by {_REGISTRY[name].__name__}")
    _REGISTRY[name] = cls
    return cls


def get_rule_class(name: str) -> type[OnsetRule]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown onset rule {name!r}; available: {sorted(_REGISTRY)}") from None


def list_rules() -> dict[str, type[OnsetRule]]:
    return dict(_REGISTRY)


_LEGACY_FIELDS = ("wet_init", "wet_spell", "dry_spell", "dry_extent")


def resolve_rule(onset_rule=None, onset_rule_params=None, **cfg) -> OnsetRule:
    """Build the rule a run should use.

    * ``onset_rule`` may already be an :class:`OnsetRule` instance (library use).
    * Otherwise it is a registry name (default ``"legacy"``) instantiated with
      ``onset_rule_params``. For ``"legacy"``, any parameter not given is taken
      from ROMP's flat ``wet_init/wet_spell/dry_spell/dry_extent`` config keys,
      so existing configs behave exactly as before.
    """
    if isinstance(onset_rule, OnsetRule):
        return onset_rule
    name = onset_rule or "legacy"
    cls = get_rule_class(name)
    params = dict(onset_rule_params or {})
    if cls is LegacyRule:
        for k in _LEGACY_FIELDS:
            if k not in params and cfg.get(k) is not None:
                params[k] = cfg[k]
    valid = {f.name for f in fields(cls)}
    unknown = set(params) - valid
    if unknown:
        raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}; valid: {sorted(valid)}")
    return cls(**params)


# --------------------------------------------------------------------------- #
# Built-in rules
# --------------------------------------------------------------------------- #
@register
@dataclass(frozen=True)
class LegacyRule(OnsetRule):
    """ROMP's original criterion (ICPAC-style)."""

    name: ClassVar[str] = "legacy"
    doc: ClassVar[str] = (
        "Every day of a wet_spell-day window is >= wet_init mm and the window total "
        "exceeds the threshold; optionally rejected if a dry_spell-day run of days "
        "< wet_init occurs within dry_extent days of the candidate (dry_extent > wet_spell "
        "enables the check)."
    )

    wet_init: float = param(1.0, min=0.0, max=50.0, unit="mm", doc="Minimum rain for a wet day")
    wet_spell: int = param(3, min=1, max=30, unit="days", doc="Accumulation window")
    dry_spell: int = param(0, min=0, max=30, unit="days", doc="Dry-run length that vetoes onset")
    dry_extent: int = param(0, min=0, max=90, unit="days", doc="Window searched for the dry run (0 = off)")

    @property
    def dry_active(self) -> bool:
        return self.dry_extent > self.wet_spell

    @property
    def lookahead(self) -> int:
        return (self.dry_extent if self.dry_active else self.wet_spell) - 1

    def validate(self):
        if 0 < self.dry_extent < self.dry_spell:
            raise ValueError(f"legacy: dry_extent ({self.dry_extent}) < dry_spell ({self.dry_spell})")
        if self.dry_active and self.dry_spell < 1:
            raise ValueError("legacy: dry_extent > wet_spell requires dry_spell >= 1")

    def __call__(self, rain, thresh, *, dim):
        if thresh is None:
            raise ValueError("legacy rule requires a threshold (thresh_file or wet_threshold)")
        w = self.wet_spell
        ok = lead_all(rain >= self.wet_init, dim, w) & (lead_sum(rain, dim, w) > thresh)
        if self.dry_active:
            # NaN where the dry window overruns the series => cannot evaluate => block
            ok = ok & (has_dry_run(rain, dim, self.dry_extent, self.dry_spell, self.wet_init) == 0)
        return finalize(ok)


@register
@dataclass(frozen=True)
class TwoStageAccumulation(OnsetRule):
    """Two consecutive accumulation windows (e.g. NiMet: 10 mm/10 d then 20 mm/20 d)."""

    name: ClassVar[str] = "two_stage"
    doc: ClassVar[str] = (
        "At least stage1_mm over the first stage1_days, and at least stage2_mm over the "
        "following stage2_days. Ignores the per-cell threshold file."
    )

    stage1_days: int = param(10, min=1, max=60, unit="days")
    stage1_mm: float = param(10.0, min=0.0, max=500.0, unit="mm")
    stage2_days: int = param(20, min=1, max=90, unit="days")
    stage2_mm: float = param(20.0, min=0.0, max=1000.0, unit="mm")
    inclusive: bool = param(True, doc="Use >= (True) or > (False) for both stages")

    @property
    def lookahead(self) -> int:
        return self.stage1_days + self.stage2_days - 1

    def __call__(self, rain, thresh=None, *, dim):
        cmp = np.greater_equal if self.inclusive else np.greater
        s1 = lead_sum(rain, dim, self.stage1_days)
        s2 = lead_sum(rain, dim, self.stage2_days, start=self.stage1_days)
        return finalize(cmp(s1, self.stage1_mm) & cmp(s2, self.stage2_mm))


@register
@dataclass(frozen=True)
class MoronRobertson(OnsetRule):
    """Modified Moron–Robertson (MOK-style) definition."""

    name: ClassVar[str] = "moron_robertson"
    doc: ClassVar[str] = (
        "First day > wet_day_mm and the wet_days-day total exceeds the threshold, not "
        "followed within follow_days (from the candidate day) by any dry_window-day "
        "period totalling < dry_min_mm. threshold=None uses the per-cell threshold file."
    )

    wet_day_mm: float = param(1.0, min=0.0, max=50.0, unit="mm", doc="First day must exceed this")
    wet_days: int = param(5, min=1, max=30, unit="days", doc="Accumulation window")
    threshold: float | None = param(None, min=0.0, max=500.0, unit="mm", doc="None = per-cell threshold file")
    follow_days: int = param(30, min=0, max=90, unit="days", doc="Veto search window (0 = no veto)")
    dry_window: int = param(10, min=1, max=60, unit="days", doc="Length of the dry period that vetoes")
    dry_min_mm: float = param(5.0, min=0.0, max=200.0, unit="mm", doc="Dry period total below this vetoes")

    @property
    def veto_active(self) -> bool:
        return self.follow_days > 0

    @property
    def lookahead(self) -> int:
        return max(self.wet_days, self.follow_days if self.veto_active else 0) - 1

    def validate(self):
        if self.veto_active and self.follow_days < self.dry_window:
            raise ValueError(
                f"moron_robertson: follow_days ({self.follow_days}) < dry_window ({self.dry_window})"
            )

    def __call__(self, rain, thresh, *, dim):
        thr = self.threshold if self.threshold is not None else thresh
        if thr is None:
            raise ValueError("moron_robertson: set `threshold` or provide thresh_file/wet_threshold")
        ok = lead_all(rain > self.wet_day_mm, dim, 1) & (lead_sum(rain, dim, self.wet_days) > thr)
        if self.veto_active:
            ok = ok & (min_sub_sum(rain, dim, self.follow_days, self.dry_window) >= self.dry_min_mm)
        return finalize(ok)
