"""Detect day-boundary and lead-label misalignment between rainfall datasets.

Two datasets that disagree about what a date means are usually shifted against each other by exactly one day. These
checks correlate sqrt-rain anomalies at small lags and report which lag fits best. Lag ``L`` pairs day ``t`` of
``other`` (or the forecast's date label ``t``) with day ``t + L`` of the reference (observations).

The module holds the two report dataclasses (`AlignmentReport`, `ForecastAlignmentReport`), the checks that fill them
(`check_alignment` for two observation-like datasets, `check_forecast_alignment` for a forecast against
observations) and the pooled-correlation helpers. `hcwf.onset.sample_alignment` computes the same forecast verdict
from sampled arrays through `forecast_alignment_from_samples`; `hcwf.onset.detect` runs the check once per model
and run. The inputs are `hcwf.onset.model.DailyRain` and `hcwf.onset.model.ForecastRain`, ingested by
`hcwf.onset.ingest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hcwf.onset.calendar import label_keys
from hcwf.onset.ingest import parse_day_boundary
from hcwf.onset.model import DailyRain, ForecastRain

__all__ = ["check_forecast_alignment"]


_COORD_TOL = 1e-6
_MIN_R = 0.1
_MIN_MARGIN = 0.02


@dataclass
class AlignmentReport:
    """Lagged correlations between two datasets and the verdict drawn from them.

    `correlations` maps lag (days) to the pooled sqrt-rain anomaly correlation, with `n_pairs` the number of cell-day
    pairs behind each. `verdict` is ``"aligned"``, ``"likely shifted by N day(s)"`` (``N`` is the best lag) or
    ``"inconclusive"``.
    """

    reference: str
    """The source name of the reference dataset (observations)."""
    other: str
    """The source name of the dataset that was checked."""
    correlations: dict[int, float] = field(default_factory=dict)
    """The pooled correlation by lag in days."""
    n_pairs: dict[int, int] = field(default_factory=dict)
    """The number of cell-day pairs behind each correlation, by lag."""
    best_lag: int | None = None
    """The lag with the highest finite correlation, or None when there is none."""
    verdict: str = "inconclusive"
    """``"aligned"``, ``"likely shifted by N day(s)"`` or ``"inconclusive"``."""
    n_cells: int = 0
    """The number of cells the two datasets share."""
    n_overlap_days: int = 0
    """The number of days (or inits, for a forecast) the comparison covers."""
    notes: list[str] = field(default_factory=list)
    """Remarks on the data and on the declared day boundaries."""

    def summary(self) -> str:
        """Describe the result in a few lines.

        Returns
        -------
        str
            The verdict, the shared cell and overlap-day counts, the correlation per lag, the best lag and the notes, one
            per line.
        """
        lines = [f"{self.other or 'other'} vs {self.reference or 'reference'}: {self.verdict}"]
        lines.append(f"  shared cells: {self.n_cells}, overlapping days: {self.n_overlap_days}")
        if self.correlations:
            lines.append("  r by lag: " + ", ".join(f"{lag:+d}: {r:.3f}" for lag, r in self.correlations.items()))
            lines.append(f"  best lag: {self.best_lag:+d}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class ForecastAlignmentReport:
    """Per-lead alignment reports and an overall reading of the ``lead_label``."""

    lead_label: str
    """How the forecast's leads are dated, ``"start"`` or ``"end"``."""
    per_lead: dict[int, AlignmentReport]
    """The report of each lead day."""
    verdict: str
    """The overall verdict across leads."""
    notes: list[str] = field(default_factory=list)
    """Remarks on the data and on the declared day boundaries."""

    def summary(self) -> str:
        """Describe the result in a few lines.

        Returns
        -------
        str
            The overall verdict, the notes, then the summary of each lead's `AlignmentReport` indented, one per line.
        """
        lines = [f"forecast (lead_label={self.lead_label!r}): {self.verdict}"]
        lines.extend(f"  note: {note}" for note in self.notes)
        for lead, report in self.per_lead.items():
            lines.append(f"  lead {lead}:")
            lines.extend("  " + line for line in report.summary().splitlines())
        return "\n".join(lines)


def _match_axis(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the values of two coordinate axes within the tolerance.

    Each value of ``ref`` is paired with the nearest value of ``other``; the pair is kept when they differ by at most
    ``1e-6`` (the tolerance ``_COORD_TOL``).

    Parameters
    ----------
    ref : numpy.ndarray
        The reference coordinate values, converted to ``float64``.
    other : numpy.ndarray
        The coordinate values to match, converted to ``float64``.

    Returns
    -------
    tuple of (numpy.ndarray of int64, numpy.ndarray of int64)
        The indices into ``ref`` and into ``other`` of the matched values, in the order of ``ref``; both empty when either
        axis is empty.
    """
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
    """Match the latitudes and longitudes of two grids.

    Parameters
    ----------
    ref_coords : dict
        The reference grid's coordinates; the keys ``"lat"`` and ``"lon"`` are read.
    other_coords : dict
        The other grid's coordinates, with the same keys.

    Returns
    -------
    tuple of (tuple, tuple)
        ``(lat, lon)``, each the ``(index into ref, index into other)`` pair of `_match_axis`.
    """
    lat = _match_axis(ref_coords["lat"], other_coords["lat"])
    lon = _match_axis(ref_coords["lon"], other_coords["lon"])
    return lat, lon


def _sqrt_field(tenths: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Return the square root of rain in mm, NaN where missing.

    Parameters
    ----------
    tenths : numpy.ndarray of int, any shape
        Rain in tenths of a mm; negative values are treated as 0.
    missing : numpy.ndarray of bool, same shape as `tenths`
        True where there is no observation.

    Returns
    -------
    numpy.ndarray of float64, same shape as `tenths`
        ``sqrt(mm)``, ``NaN`` where ``missing``.
    """
    return np.where(missing, np.nan, np.sqrt(np.maximum(tenths, 0) / 10.0))


def _cells(arr: np.ndarray, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    """Select the matched cells of an array and flatten them to one row per cell.

    Parameters
    ----------
    arr : numpy.ndarray, shape (lat, lon, n)
        The field to select from.
    lat_idx : numpy.ndarray of int64
        The latitude indices of the matched cells.
    lon_idx : numpy.ndarray of int64
        The longitude indices of the matched cells.

    Returns
    -------
    numpy.ndarray, shape (cell, n)
        The ``len(lat_idx) * len(lon_idx)`` selected cells, the latitude varying slowest.
    """
    sub = arr[np.ix_(lat_idx, lon_idx)]
    return sub.reshape(-1, arr.shape[-1])


def _rain_cells(rain: DailyRain, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    """Return the sqrt-rain of the selected cells, NaN where missing.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The rainfall, with dims ``lat``, ``lon`` and ``time`` in any order.
    lat_idx : numpy.ndarray of int64
        The latitude indices of the matched cells.
    lon_idx : numpy.ndarray of int64
        The longitude indices of the matched cells.

    Returns
    -------
    numpy.ndarray of float64, shape (cell, time)
        ``sqrt(mm)`` per selected cell and day.
    """
    order = tuple(rain.dims.index(d) for d in ("lat", "lon", "time"))
    field_ = _sqrt_field(np.transpose(rain.tenths, order), np.transpose(rain.missing, order))
    return _cells(field_, lat_idx, lon_idx)


def _pooled_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Correlate two series after removing each cell's mean, pooled over cells and valid pairs.

    A pair is valid where both values are finite; each cell's mean is taken over its valid pairs.

    Parameters
    ----------
    x : numpy.ndarray of float64, shape (cell, n)
        The first series, NaN where missing.
    y : numpy.ndarray of float64, shape (cell, n)
        The second series, NaN where missing.

    Returns
    -------
    tuple of (float, int)
        The pooled correlation and the number of valid pairs; the correlation is ``NaN`` below three pairs or when
        either series has no variance.
    """
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
    """Derive the best lag and the verdict from the lagged correlations (spec section 8).

    The verdict is ``"likely shifted by N day(s)"`` when the best lag ``N != 0`` beats lag 0 by more than 0.02 and its
    correlation exceeds 0.1, ``"aligned"`` when lag 0 exceeds 0.1 and beats every other lag by more than 0.02, else
    ``"inconclusive"``.

    Parameters
    ----------
    correlations : dict of int to float
        The pooled correlation by lag in days; ``NaN`` entries are ignored.

    Returns
    -------
    tuple of (int or None, str)
        The best lag (None when no correlation is finite) and the verdict.
    """
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
    """Return the lag implied by two declared day boundaries.

    A day labelled ``D`` by start covers ``[D + h/24, D + 1 + h/24)``; by end it covers ``[D - 1 + h/24, D + h/24)``. The
    lag that lines two labellings up is the difference of those start offsets.

    Parameters
    ----------
    ref_boundary : str
        The reference day boundary, as parsed by `hcwf.onset.ingest.parse_day_boundary`.
    other_boundary : str
        The day boundary of the other dataset.
    other_label : str, optional
        Replaces the labelling (``"start"`` or ``"end"``) of ``other_boundary``, for a forecast's ``lead_label``.

    Returns
    -------
    float or None
        The expected lag in days, or None if either boundary cannot be parsed.
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
    """Write the remarks on a declared day-boundary mismatch.

    Parameters
    ----------
    shift : float or None
        The lag implied by the boundaries (`_boundary_shift`).
    ref_boundary : str
        The reference day boundary.
    other_boundary : str
        The other dataset's day boundary.
    best_lag : int or None
        The best lag found from the data.

    Returns
    -------
    list of str
        One note when the boundaries differ (saying whether the implied lag matches ``best_lag``), a note that they could
        not be parsed when ``shift`` is None, else an empty list.
    """
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
    """Check one series' day labels for duplicates or a non-increasing order.

    Parameters
    ----------
    label : str
        The series name used in the note.
    keys : numpy.ndarray of int64
        The day-label keys of the series, in time order (see `hcwf.onset.calendar.label_keys`).

    Returns
    -------
    list of str
        One note when the labels are duplicated or not increasing, else an empty list.
    """
    steps = np.diff(keys)
    if np.any(steps <= 0):
        return [f"{label}: time labels are duplicated or not increasing"]
    return []


def _empty_report(reference: str, other: str, note: str, notes: list[str]) -> AlignmentReport:
    """Build an inconclusive report with a reason.

    Parameters
    ----------
    reference : str
        The source name of the reference dataset.
    other : str
        The source name of the other dataset.
    note : str
        The reason, appended to ``notes``.
    notes : list of str
        The notes collected so far.

    Returns
    -------
    AlignmentReport
        A report with no correlations and the verdict ``"inconclusive"``.
    """
    return AlignmentReport(reference=reference, other=other, notes=notes + [note])


def _check_dims(rain: DailyRain, who: str) -> None:
    """Check that the rainfall has the dims alignment works on.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The rainfall to check.
    who : str
        The name used in the error message.

    Raises
    ------
    ValueError
        The dims of ``rain`` are not exactly ``lat``, ``lon`` and ``time``.
    """
    if set(rain.dims) != {"lat", "lon", "time"}:
        raise ValueError(f"{who} must have dims lat, lon and time; got {rain.dims}")


def check_alignment(reference: DailyRain, other: DailyRain, max_lag: int = 3) -> AlignmentReport:
    """Correlate the sqrt-rain anomalies of ``other`` with ``reference`` at lags ``-max_lag`` to ``max_lag``.

    Lag ``L`` compares day ``t`` of ``other`` with day ``t + L`` of ``reference``. Cells are matched by coordinates
    (tolerance ``1e-6``, nothing is regridded); anomalies remove each cell's mean over the compared days and
    correlations are pooled over cells and days. The verdict is ``"likely shifted by N day(s)"`` when the best lag
    ``N != 0`` beats lag 0 by more than 0.02 with ``r > 0.1``, ``"aligned"`` when lag 0 has ``r > 0.1`` and no such shift
    exists, else ``"inconclusive"``. Datasets with different calendars, no overlapping dates or no shared cells give
    an inconclusive report with a note instead of raising. A dataset whose dims are not exactly ``lat``, ``lon`` and
    ``time`` raises `ValueError` (from `_check_dims`).

    Parameters
    ----------
    reference : hcwf.onset.model.DailyRain
        The reference rainfall (observations), dims ``lat``, ``lon``, ``time``.
    other : hcwf.onset.model.DailyRain
        The rainfall to check, with the same dims.
    max_lag : int, default 3
        The largest lag in days, in both directions.

    Returns
    -------
    AlignmentReport
        The correlation and pair count per lag, the best lag, the verdict and notes.
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
    """Compare forecast leads with observations at lags ``-max_lag`` to ``max_lag``, lead by lead.

    Lead ``k`` of init ``i`` is dated ``init + k - 1`` (``lead_label`` ``"start"``) or ``init + k`` (``"end"``); lag ``L``
    pairs it with the observed day ``L`` days after that date. Members are averaged (after the square root). A best
    lag that is the same non-zero value at every lead suggests the other ``lead_label``.

    Parameters
    ----------
    obs : hcwf.onset.model.DailyRain
        The observations, dims ``lat``, ``lon``, ``time``.
    forecast : hcwf.onset.model.ForecastRain
        The forecast, with at least the dims ``lat``, ``lon``, ``init_time`` and ``lead`` (and optionally ``member``).
    leads : iterable of int, default (1, 2, 3)
        The lead days to check.
    max_lag : int, default 3
        The largest lag in days, in both directions.

    Returns
    -------
    ForecastAlignmentReport
        One `AlignmentReport` per lead and the overall verdict; inconclusive with a note when the calendars differ.

    Raises
    ------
    ValueError
        ``obs`` does not have exactly the dims ``lat``, ``lon`` and ``time``, ``forecast`` lacks ``lat`` or ``lon``, or a
        lead is not among the forecast leads.
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
    """Compute the pooled correlation at every lag and lead in one vectorised step.

    The observations are gathered once into an ``(init, lag, lead, cell)`` view and every ``(lag, lead)`` correlation is
    taken with cell means removed over the valid inits, pooled over cells and inits exactly as `_pooled_corr` does.

    Parameters
    ----------
    obs_days : numpy.ndarray, shape (init, day, cell)
        Sqrt-rain on consecutive days after each init, NaN where missing.
    forecast : numpy.ndarray, shape (init, lead, cell)
        The member-mean sqrt-rain of each lead, NaN where missing.
    base : numpy.ndarray of int, shape (lead,)
        The index in the day axis of the day each lead's date label names.
    lags : numpy.ndarray of int, shape (lag,)
        The lags in days; lag ``L`` pairs lead ``j`` with day ``base[j] + L`` (NaN outside the day axis).

    Returns
    -------
    tuple of (numpy.ndarray of float64, numpy.ndarray of int64)
        ``(r, n_pairs)``, both of shape ``(lag, lead)``; ``r`` is ``NaN`` below three pairs or without variance.
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
    """Build the per-lead report and overall verdict of `check_forecast_alignment` from prepared sample arrays.

    Lead ``k`` is dated ``init + k - 1`` (``"start"``) or ``init + k`` (``"end"``). Verdicts are those of
    `check_forecast_alignment` (spec section 8).

    Parameters
    ----------
    obs_days : numpy.ndarray, shape (init, day, cell)
        Sqrt-rain for day index ``j = (days after the init) + pad``, so it must reach from ``-pad`` to the largest lead
        offset plus ``max_lag``.
    forecast : numpy.ndarray, shape (init, lead, cell)
        The member-mean sqrt-rain of ``leads``.
    leads : iterable of int
        The lead days, in the order of the ``lead`` axis of ``forecast``.
    pad : int
        The offset of day 0 in the day axis of ``obs_days``.
    lead_label : {"start", "end"}
        How a lead is dated.
    obs : str
        The source name of the observations.
    source : str
        The source name of the forecast.
    n_inits : int
        The number of inits, reported as the overlap days of each lead.
    obs_boundary : str, default ""
        The observation day boundary; with ``forecast_boundary`` it adds the implied-lag note.
    forecast_boundary : str, default ""
        The forecast day boundary.
    max_lag : int, default 3
        The largest lag in days, in both directions.

    Returns
    -------
    ForecastAlignmentReport
        One `AlignmentReport` per lead and the overall verdict.
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
    """Combine the per-lead verdicts into one, suggesting a ``lead_label`` when the shift is consistent.

    Parameters
    ----------
    per_lead : dict of int to AlignmentReport
        The report of each lead.
    lead_label : {"start", "end"}
        The lead label in use.

    Returns
    -------
    tuple of (str, list of str)
        The overall verdict (``"aligned"``, ``"inconclusive"`` or ``"likely shifted by N day(s) at every lead"``,
        with a hint at the other label for a lag of +1 with ``"start"`` or -1 with ``"end"``) and notes (that the leads
        disagree, when they do).
    """
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
