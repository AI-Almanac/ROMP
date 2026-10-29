"""Detect day-boundary and lead-label misalignment between rainfall datasets.

Two datasets that disagree about what a date means are usually shifted against
each other by exactly one day. These checks correlate sqrt-rain anomalies at
small lags and report which lag fits best. Lag L pairs day t of `other` (or
the forecast's date label t) with day t + L of the reference (observations).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from momp.onset.calendar import label_keys
from momp.onset.ingest import parse_day_boundary
from momp.onset.model import DailyRain, ForecastRain

_COORD_TOL = 1e-6
_MIN_R = 0.1
_MIN_MARGIN = 0.02


@dataclass
class AlignmentReport:
    """Lagged correlations between two datasets and the verdict drawn from them.

    correlations maps lag (days) to the pooled sqrt-rain anomaly correlation, with
    n_pairs the number of cell-day pairs behind each. verdict is "aligned",
    "likely shifted by N day(s)" (N is the best lag) or "inconclusive".
    """

    reference: str
    other: str
    correlations: dict[int, float] = field(default_factory=dict)
    n_pairs: dict[int, int] = field(default_factory=dict)
    best_lag: int | None = None
    verdict: str = "inconclusive"
    n_cells: int = 0
    n_overlap_days: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """A short multi-line description of the result."""
        lines = [f"{self.other or 'other'} vs {self.reference or 'reference'}: {self.verdict}"]
        lines.append(f"  shared cells: {self.n_cells}, overlapping days: {self.n_overlap_days}")
        if self.correlations:
            lines.append("  r by lag: " + ", ".join(f"{lag:+d}: {r:.3f}" for lag, r in self.correlations.items()))
            lines.append(f"  best lag: {self.best_lag:+d}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class ForecastAlignmentReport:
    """Per-lead alignment reports and an overall reading of the lead_label."""

    lead_label: str
    per_lead: dict[int, AlignmentReport]
    verdict: str
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """A short multi-line description of the result."""
        lines = [f"forecast (lead_label={self.lead_label!r}): {self.verdict}"]
        lines.extend(f"  note: {note}" for note in self.notes)
        for lead, report in self.per_lead.items():
            lines.append(f"  lead {lead}:")
            lines.extend("  " + line for line in report.summary().splitlines())
        return "\n".join(lines)


def _match_axis(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices (into ref, into other) of coordinate values equal within the tolerance."""
    ref = np.asarray(ref, dtype=float)
    other = np.asarray(other, dtype=float)
    if ref.size == 0 or other.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(other)
    pos = np.clip(np.searchsorted(other[order], ref), 1, len(other) - 1) if len(other) > 1 else np.zeros(
        len(ref), dtype=int)
    cands = np.stack([order[np.maximum(pos - 1, 0)], order[pos]])
    err = np.abs(other[cands] - ref[None, :])
    best = np.argmin(err, axis=0)
    chosen = cands[best, np.arange(len(ref))]
    ok = err[best, np.arange(len(ref))] <= _COORD_TOL
    return np.nonzero(ok)[0], chosen[ok]


def _grid_match(ref_coords: dict, other_coords: dict) -> tuple[tuple, tuple]:
    """Matched (lat, lon) index pairs between two grids."""
    lat = _match_axis(ref_coords["lat"], other_coords["lat"])
    lon = _match_axis(ref_coords["lon"], other_coords["lon"])
    return lat, lon


def _sqrt_field(tenths: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """sqrt of rain in mm, NaN where missing."""
    return np.where(missing, np.nan, np.sqrt(np.maximum(tenths, 0) / 10.0))


def _cells(arr: np.ndarray, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    """Select matched cells of an array with dims (lat, lon, n) and flatten them to (cells, n)."""
    sub = arr[np.ix_(lat_idx, lon_idx)]
    return sub.reshape(-1, arr.shape[-1])


def _rain_cells(rain: DailyRain, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    """sqrt-rain of the selected cells as (cells, time), NaN where missing."""
    order = tuple(rain.dims.index(d) for d in ("lat", "lon", "time"))
    field_ = _sqrt_field(np.transpose(rain.tenths, order), np.transpose(rain.missing, order))
    return _cells(field_, lat_idx, lon_idx)


def _pooled_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Correlation of cell-mean-removed x and y (cells, n), pooled over cells and valid pairs."""
    valid = np.isfinite(x) & np.isfinite(y)
    n_pairs = int(valid.sum())
    if n_pairs < 3:
        return float("nan"), n_pairs
    cnt = valid.sum(axis=1)
    safe = np.maximum(cnt, 1)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    xa = np.where(valid, x - (xv.sum(axis=1) / safe)[:, None], 0.0)
    ya = np.where(valid, y - (yv.sum(axis=1) / safe)[:, None], 0.0)
    denom = np.sqrt((xa ** 2).sum() * (ya ** 2).sum())
    if denom <= 0:
        return float("nan"), n_pairs
    return float((xa * ya).sum() / denom), n_pairs


def _verdict(correlations: dict[int, float]) -> tuple[int | None, str]:
    """Best lag and verdict string from lagged correlations (spec section 8)."""
    finite = {lag: r for lag, r in correlations.items() if np.isfinite(r)}
    if not finite or max(finite.values()) <= _MIN_R:
        return (max(finite, key=finite.get) if finite else None), "inconclusive"
    best = max(finite, key=finite.get)
    r0 = finite.get(0, float("nan"))
    if best != 0 and finite[best] - (r0 if np.isfinite(r0) else -1.0) > _MIN_MARGIN:
        return best, f"likely shifted by {best:+d} day(s)"
    runner_up = max((r for lag, r in finite.items() if lag != 0), default=-1.0)
    if np.isfinite(r0) and r0 > _MIN_R and r0 - runner_up > _MIN_MARGIN:
        return best, "aligned"
    return best, "inconclusive"


def _boundary_shift(ref_boundary: str, other_boundary: str, other_label: str | None = None) -> float | None:
    """Expected lag implied by two declared day boundaries, or None if either cannot be parsed.

    A day labelled D by start covers [D + h/24, D + 1 + h/24); by end it covers
    [D - 1 + h/24, D + h/24). The lag that lines two labellings up is the
    difference of those start offsets.
    """
    try:
        h_ref, label_ref = parse_day_boundary(ref_boundary)
        h_other, label_other = parse_day_boundary(other_boundary)
    except ValueError:
        return None
    if other_label is not None:
        label_other = other_label
    start_ref = h_ref / 24.0 - (label_ref == "end")
    start_other = h_other / 24.0 - (label_other == "end")
    return start_other - start_ref


def _boundary_notes(shift: float | None, ref_boundary: str, other_boundary: str, best_lag: int | None) -> list[str]:
    """Human-readable remarks on the declared day-boundary mismatch."""
    if shift is None:
        return ["could not parse the declared day boundaries"]
    expected = int(np.rint(shift))
    if abs(shift) < 1e-9:
        return []
    note = (f"declared day boundaries differ ({other_boundary!r} vs {ref_boundary!r}); "
            f"that implies a lag of about {shift:+.2f} day(s)")
    if best_lag is not None and expected != 0:
        note += " (matches the best lag)" if best_lag == expected else f" (best lag is {best_lag:+d})"
    return [note]


def _time_notes(label: str, keys: np.ndarray) -> list[str]:
    """Time-axis diagnostics for one series: duplicated, unsorted or gappy day labels."""
    steps = np.diff(keys)
    if np.any(steps <= 0):
        return [f"{label}: time labels are duplicated or not increasing"]
    return []


def _empty_report(reference: str, other: str, note: str, notes: list[str]) -> AlignmentReport:
    """An inconclusive report with a reason."""
    return AlignmentReport(reference=reference, other=other, notes=notes + [note])


def _check_dims(rain: DailyRain, who: str) -> None:
    """Alignment works on (lat, lon, time) data."""
    if set(rain.dims) != {"lat", "lon", "time"}:
        raise ValueError(f"{who} must have dims lat, lon and time; got {rain.dims}")


def check_alignment(reference: DailyRain, other: DailyRain, max_lag: int = 3) -> AlignmentReport:
    """Correlate sqrt-rain anomalies of `other` with `reference` at lags -max_lag..max_lag.

    Lag L compares day t of `other` with day t + L of `reference`. Cells are
    matched by coordinates (tolerance 1e-6, nothing is regridded); anomalies
    remove each cell's mean over the compared days and correlations are pooled
    over cells and days. The verdict is "likely shifted by N day(s)" when the
    best lag N != 0 beats lag 0 by more than 0.02 with r > 0.1, "aligned" when
    lag 0 has r > 0.1 and no such shift exists, else "inconclusive".
    """
    _check_dims(reference, "reference")
    _check_dims(other, "other")
    notes: list[str] = []
    if reference.calendar != other.calendar:
        notes.append(f"calendars differ ({reference.calendar} vs {other.calendar})")
        return _empty_report(reference.source, other.source, "cannot compare different calendars", notes)
    ref_keys = label_keys(reference.year, reference.month, reference.day)
    other_keys = label_keys(other.year, other.month, other.day)
    notes += _time_notes("reference", ref_keys) + _time_notes("other", other_keys)
    common = np.intersect1d(ref_keys, other_keys)
    if common.size == 0:
        return _empty_report(reference.source, other.source, "no overlapping dates", notes)
    (rlat, olat), (rlon, olon) = _grid_match(reference.coords, other.coords)
    if len(rlat) == 0 or len(rlon) == 0:
        return _empty_report(reference.source, other.source, "no shared cells", notes)

    ref_field = _rain_cells(reference, rlat, rlon)
    other_field = _rain_cells(other, olat, olon)
    pos_ref = np.searchsorted(ref_keys, common)
    pos_other = np.searchsorted(other_keys, common)
    correlations, n_pairs = {}, {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = pos_ref + lag
        ok = (shifted >= 0) & (shifted < len(ref_keys))
        if ok.sum() < 3:
            correlations[lag], n_pairs[lag] = float("nan"), 0
            continue
        correlations[lag], n_pairs[lag] = _pooled_corr(ref_field[:, shifted[ok]], other_field[:, pos_other[ok]])
    best, verdict = _verdict(correlations)
    notes += _boundary_notes(_boundary_shift(reference.day_boundary, other.day_boundary), reference.day_boundary,
                             other.day_boundary, best)
    return AlignmentReport(reference=reference.source, other=other.source, correlations=correlations,
                           n_pairs=n_pairs, best_lag=best, verdict=verdict, n_cells=len(rlat) * len(rlon),
                           n_overlap_days=int(common.size), notes=notes)


def check_forecast_alignment(obs: DailyRain, forecast: ForecastRain, leads=(1, 2, 3),
                             max_lag: int = 3) -> ForecastAlignmentReport:
    """Compare forecast leads with observations at lags -max_lag..max_lag, lead by lead.

    Lead k of init i is dated init + k - 1 (lead_label "start") or init + k
    ("end"); lag L pairs it with the observed day L days after that date.
    Members are averaged (after the square root). A best lag that is the same
    non-zero value at every lead suggests the other lead_label.
    """
    _check_dims(obs, "obs")
    if not {"lat", "lon"} <= set(forecast.dims):
        raise ValueError(f"forecast must have lat and lon dims; got {forecast.dims}")
    notes: list[str] = []
    per_lead: dict[int, AlignmentReport] = {}
    if obs.calendar != forecast.calendar:
        notes.append(f"calendars differ ({obs.calendar} vs {forecast.calendar}); cannot compare")
        return ForecastAlignmentReport(forecast.lead_label, per_lead, "inconclusive", notes)
    obs_keys = label_keys(obs.year, obs.month, obs.day)
    notes += _time_notes("obs", obs_keys)
    init_keys = label_keys(forecast.init_year, forecast.init_month, forecast.init_day)
    init_pos = np.searchsorted(obs_keys, init_keys)
    init_in_obs = (init_pos < len(obs_keys)) & (obs_keys[np.minimum(init_pos, len(obs_keys) - 1)] == init_keys)
    (olat, flat), (olon, flon) = _grid_match(obs.coords, forecast.coords)
    obs_field = _rain_cells(obs, olat, olon) if len(olat) and len(olon) else None

    shift = _boundary_shift(obs.day_boundary, forecast.day_boundary, forecast.lead_label)
    lead_axis = forecast.dims.index("lead")
    for k in leads:
        if k not in set(forecast.coords["lead"].tolist()):
            raise ValueError(f"lead {k} is not in the forecast leads")
        report = AlignmentReport(reference=obs.source, other=forecast.source)
        per_lead[k] = report
        if obs_field is None:
            report.notes.append("no shared cells")
            continue
        if not init_in_obs.any():
            report.notes.append("no overlapping dates")
            continue
        sl = np.take(forecast.tenths, int(np.nonzero(forecast.coords["lead"] == k)[0][0]), axis=lead_axis)
        slm = np.take(forecast.missing, int(np.nonzero(forecast.coords["lead"] == k)[0][0]), axis=lead_axis)
        dims = tuple(d for d in forecast.dims if d != "lead")
        f = _sqrt_field(sl, slm)
        if "member" in dims:
            axis = dims.index("member")
            with np.errstate(all="ignore"):
                f = np.nanmean(f, axis=axis) if f.shape[axis] else f
            dims = tuple(d for d in dims if d != "member")
        f = np.transpose(f, [dims.index(d) for d in ("lat", "lon", "init_time")])
        fc_field = _cells(f, flat, flon)
        offset = k - 1 if forecast.lead_label == "start" else k
        base = init_pos + offset
        correlations, n_pairs = {}, {}
        for lag in range(-max_lag, max_lag + 1):
            idx = base + lag
            ok = init_in_obs & (idx >= 0) & (idx < len(obs_keys))
            if ok.sum() < 3:
                correlations[lag], n_pairs[lag] = float("nan"), 0
                continue
            correlations[lag], n_pairs[lag] = _pooled_corr(obs_field[:, idx[ok]], fc_field[:, ok])
        report.correlations, report.n_pairs = correlations, n_pairs
        report.best_lag, report.verdict = _verdict(correlations)
        report.n_cells = len(olat) * len(olon)
        report.n_overlap_days = int(init_in_obs.sum())

    verdict, extra = _forecast_verdict(per_lead, forecast.lead_label)
    notes += extra
    if shift is not None and abs(shift) > 1e-9:
        notes.append(f"declared day boundary and lead_label imply a lag of about {shift:+.2f} day(s)")
    return ForecastAlignmentReport(forecast.lead_label, per_lead, verdict, notes)


def lagged_correlations(obs_days: np.ndarray, forecast: np.ndarray, base: np.ndarray, lags: np.ndarray,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Pooled correlation at every lag and lead in one vectorised step.

    obs_days (init, day, cell) holds sqrt-rain on consecutive days after each init; forecast (init, lead, cell)
    holds the member-mean sqrt-rain of each lead; both are NaN where missing. base (lead,) is the index in the
    day axis of the day each lead's date label names, and lag L pairs lead j with day base[j] + L (NaN outside
    the day axis). The observations are gathered once into a (init, lag, lead, cell) view and every
    (lag, lead) correlation is taken with cell means removed over the valid inits, pooled over cells and inits
    exactly as `_pooled_corr` does. Returns (r, n_pairs), both (lag, lead); r is NaN below three pairs.
    """
    lags = np.asarray(lags, dtype=np.int64)
    day = np.asarray(base, dtype=np.int64)[None, :] + lags[:, None]
    inside = (day >= 0) & (day < obs_days.shape[1])
    stacked = obs_days[:, np.clip(day, 0, obs_days.shape[1] - 1), :].astype(np.float64)
    stacked[:, ~inside, :] = np.nan
    fc = np.broadcast_to(forecast[:, None, :, :].astype(np.float64), stacked.shape)
    valid = np.isfinite(stacked) & np.isfinite(fc)
    count = valid.sum(axis=0)
    safe = np.maximum(count, 1)
    xv = np.where(valid, stacked, 0.0)
    yv = np.where(valid, fc, 0.0)
    xa = np.where(valid, xv - (xv.sum(axis=0) / safe)[None], 0.0)
    ya = np.where(valid, yv - (yv.sum(axis=0) / safe)[None], 0.0)
    with np.errstate(all="ignore"):
        denom = np.sqrt((xa ** 2).sum(axis=(0, 3)) * (ya ** 2).sum(axis=(0, 3)))
        r = (xa * ya).sum(axis=(0, 3)) / denom
    n_pairs = valid.sum(axis=(0, 3))
    r = np.where((n_pairs >= 3) & (denom > 0), r, np.nan)
    return r, n_pairs


def forecast_alignment_from_samples(obs_days: np.ndarray, forecast: np.ndarray, *, leads, pad: int,
                                    lead_label: str, obs: str, source: str, n_inits: int, obs_boundary: str = "",
                                    forecast_boundary: str = "", max_lag: int = 3) -> ForecastAlignmentReport:
    """The per-lead report and overall verdict of `check_forecast_alignment`, from prepared sample arrays.

    obs_days (init, day, cell) holds sqrt-rain for day index j = (days after the init) + pad, so it must reach
    from -pad to the largest lead offset + max_lag; forecast (init, lead, cell) is the member-mean sqrt-rain
    of `leads`. Lead k is dated init + k - 1 ("start") or init + k ("end"). Verdicts are those of
    `check_forecast_alignment` (spec section 8).
    """
    leads = tuple(int(k) for k in leads)
    lags = np.arange(-max_lag, max_lag + 1)
    base = np.array([(k - 1 if lead_label == "start" else k) + pad for k in leads])
    r, n_pairs = lagged_correlations(obs_days, forecast, base, lags)
    n_cells = int(obs_days.shape[2])
    per_lead: dict[int, AlignmentReport] = {}
    for j, k in enumerate(leads):
        report = AlignmentReport(reference=obs, other=source)
        report.correlations = {int(lag): float(r[i, j]) for i, lag in enumerate(lags)}
        report.n_pairs = {int(lag): int(n_pairs[i, j]) for i, lag in enumerate(lags)}
        report.best_lag, report.verdict = _verdict(report.correlations)
        report.n_cells, report.n_overlap_days = n_cells, int(n_inits)
        per_lead[k] = report
    verdict, notes = _forecast_verdict(per_lead, lead_label)
    shift = _boundary_shift(obs_boundary, forecast_boundary, lead_label) if obs_boundary and forecast_boundary \
        else None
    if shift is not None and abs(shift) > 1e-9:
        notes.append(f"declared day boundary and lead_label imply a lag of about {shift:+.2f} day(s)")
    return ForecastAlignmentReport(lead_label, per_lead, verdict, notes)


def _forecast_verdict(per_lead: dict[int, AlignmentReport], lead_label: str) -> tuple[str, list[str]]:
    """Overall verdict across leads, with a lead_label suggestion when the shift is consistent."""
    verdicts = {r.verdict for r in per_lead.values()}
    if not per_lead or verdicts == {"inconclusive"}:
        return "inconclusive", []
    if verdicts == {"aligned"}:
        return "aligned", []
    lags = {r.best_lag for r in per_lead.values() if r.verdict.startswith("likely shifted")}
    if len(lags) == 1 and all(r.verdict.startswith("likely shifted") for r in per_lead.values()):
        lag = lags.pop()
        other = "end" if lead_label == "start" else "start"
        hint = ""
        if (lead_label, lag) in (("start", 1), ("end", -1)):
            hint = f"; the data look labelled like lead_label={other!r}"
        return f"likely shifted by {lag:+d} day(s) at every lead{hint}", []
    return "inconclusive", ["leads disagree about the best lag"]
