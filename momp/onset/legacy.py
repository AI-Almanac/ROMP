"""Translate ROMP's old flat onset keys (wet_init, wet_spell, ...) into a Definition.

The translation reproduces the old detect.py rules, with these deliberate
differences (see docs/onset_in_romp.md):

* dry_threshold was never used by the old code and is ignored;
* the old forecast dry window was one day shorter than the observed one; the
  observed window, days 2 to dry_extent + 1 after the candidate, is used for both;
* the old observed search cut its data off 47 days past end_date, so the last
  candidate whose whole wet spell fit was end_date + 47 - (wet_spell - 1); the
  season ends there.
"""
from __future__ import annotations

import warnings

from momp.onset.model import (Amount, Comparison, Definition, DayRef, DryRun, FirstPassage, Season, Span, Total,
                              Veto, WetDays)

LEGACY_THRESHOLD = "legacy_threshold"
LEGACY_SEASON = "main"
EXTEND_END_DAYS = 47

_REQUIRED = ("wet_init", "wet_spell", "dry_spell", "dry_extent", "end_date")
_warned = False


class LegacyConfigError(ValueError):
    """The flat onset keys cannot be expressed as a Definition; the message says which key and how to fix it."""


def _tenths(cfg: dict, key: str) -> int:
    """The config value of `key` in whole tenths of a mm, or an error if it is finer than 0.1 mm."""
    value = cfg[key]
    try:
        tenths = round(float(value) * 10)
        exact = abs(float(value) * 10 - tenths) < 1e-6
    except (TypeError, ValueError):
        raise LegacyConfigError(f"{key} = {value!r} is not a number of mm. Fix: set {key} to a number") from None
    if not exact:
        raise LegacyConfigError(
            f"{key} = {value!r} is not a whole number of tenths of a mm; the onset component works on 0.1 mm. "
            f"Fix: round {key} to 0.1 mm, e.g. {tenths / 10}")
    return tenths


def _int(cfg: dict, key: str, minimum: int) -> int:
    """The config value of `key` as an int of at least `minimum`."""
    value = cfg[key]
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise LegacyConfigError(f"{key} = {value!r} must be a whole number of at least {minimum}")
    return int(value)


def from_flat_config(cfg: dict) -> Definition:
    """Build the Definition the old flat ROMP keys describe, and warn once per process that they are deprecated.

    Needs wet_init, wet_spell, dry_spell, dry_extent, end_date and (without thresh_file) wet_threshold; reads
    start_date (1 January if absent), earliest_onset_date, thresh_file and thresh_var if present. With a
    thresh_file the total is compared with the named threshold "legacy_threshold",
    which the caller supplies per cell; otherwise with wet_threshold.
    """
    global _warned
    if not _warned:
        _warned = True
        warnings.warn("The flat onset keys (wet_init, wet_threshold, wet_spell, dry_spell, dry_extent, thresh_file, "
                      "earliest_onset_date) are deprecated; set onset_definition instead "
                      "(see docs/onset_in_romp.md).", FutureWarning, stacklevel=2)
    missing = [key for key in _REQUIRED if cfg.get(key) is None]
    if not cfg.get("thresh_file") and cfg.get("wet_threshold") is None:
        missing.append("wet_threshold")
    if missing:
        raise LegacyConfigError(f"the flat onset config lacks {', '.join(missing)}. "
                                "Fix: set them, or set onset_definition to use a definition")
    wet_init = _tenths(cfg, "wet_init")
    window = _int(cfg, "wet_spell", 1)
    dry_spell = _int(cfg, "dry_spell", 1)
    dry_extent = _int(cfg, "dry_extent", 0)
    if cfg.get("thresh_file"):
        amount = Amount(threshold=LEGACY_THRESHOLD)
    else:
        amount = Amount(tenths=_tenths(cfg, "wet_threshold"))
    require = (WetDays(Span(1, window), "ge", window), Total(Span(1, window), Comparison("gt", amount)))
    reject = ()
    if dry_extent > window and dry_extent >= dry_spell:
        reject = (Veto(DryRun(dry_spell), Span(2, dry_extent + 1), "inside"),)
    earliest = cfg.get("earliest_onset_date")
    start = cfg.get("start_date")
    month, day = tuple(earliest) if earliest else tuple(start[1:]) if start else (1, 1)
    end_month, end_day = tuple(cfg["end_date"][1:])
    season = Season(LEGACY_SEASON, DayRef(month=month, day=day),
                    DayRef(month=end_month, day=end_day, offset_days=EXTEND_END_DAYS - (window - 1)))
    return Definition(
        id="legacy_flat_config", version=1, title="ROMP flat onset keys", citation="", wording="", status="draft",
        notes="Translated from the deprecated flat keys; see momp.onset.legacy.",
        wet_day=Comparison("ge", Amount(tenths=wet_init)), seasons=(season,),
        rule=FirstPassage(require=require, reject_if=reject))
