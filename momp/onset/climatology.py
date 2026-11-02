"""Reference-climatology products: per-cell thresholds, harmonic season analysis and a threshold cache."""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from momp.cache import NETCDF, Store
from momp.onset.calendar import fixed_bounds, label_keys
from momp.onset.model import Definition, ThresholdSpec, resolution_of
from momp.spatial import ids

MAX_MISSING_FRACTION = 0.10
_CHUNK_VALUES = 1 << 24
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_CUMULATIVE_DAYS = np.concatenate([[0], np.cumsum(_DAYS_IN_MONTH)])
_YEAR_LENGTH = 365


def _year_windows(spec: ThresholdSpec, rain, years: tuple[int, int], partial_years: bool = False) -> list[tuple[int, int]]:
    """Inclusive (first, last) time indices of the `during` window in each reference year.

    A window crossing the year end belongs to the year of its start. Years whose
    window is not fully inside the data are skipped, or with partial_years used for
    the days the data has (a year with none of them is still skipped).
    """
    year_list = np.arange(years[0], years[1] + 1)
    if spec.during is None:
        windows = []
        for y in year_list:
            idx = np.nonzero(rain.year == y)[0]
            if idx.size:
                windows.append((int(idx[0]), int(idx[-1])))
        return windows
    keys = label_keys(rain.year, rain.month, rain.day)
    if partial_years:
        return _partial_windows(keys, year_list, *spec.during)
    first, last = fixed_bounds(keys, year_list, spec.during[0], spec.during[1])
    return [(int(a), int(b)) for a, b in zip(first, last) if a >= 0 and b >= a]


def _partial_windows(keys: np.ndarray, year_list: np.ndarray, start, end) -> list[tuple[int, int]]:
    """The window of each year cut to the days the data holds (first day on or after the start, last on or before the end)."""
    if start.climatological is not None or end.climatological is not None:
        raise ValueError("thresholds take month-day bounds; resolve climatological bounds per cell")
    wraps = (end.month, end.day) < (start.month, start.day)
    low = year_list * 10000 + start.month * 100 + start.day
    high = (year_list + wraps) * 10000 + end.month * 100 + end.day
    first = np.searchsorted(keys, low, side="left") + start.offset_days
    last = np.searchsorted(keys, high, side="right") - 1 + end.offset_days
    first, last = np.maximum(first, 0), np.minimum(last, len(keys) - 1)
    return [(int(a), int(b)) for a, b in zip(first, last) if b >= a]


def _window_sum(block: np.ndarray, sequential: bool) -> np.ndarray:
    """Sum over time of a (series, time) block: numpy's own order, or day by day as a (time, series) array reduces.

    The day-by-day order is the one blending's per-file totals take; it differs from the pairwise sum in the last bits.
    """
    if not sequential:
        return block.sum(axis=1)
    return np.add.reduce(np.ascontiguousarray(block.T), axis=0)


def _add_compensated(total: np.ndarray, compensation: np.ndarray, value: np.ndarray) -> None:
    """total += value with Kahan compensation, in place, as pandas' groupby sum adds (blending's totals over files)."""
    y = value - compensation
    t = total + y
    compensation[:] = (t - total) - y
    total[:] = t


def _threshold_rows(spec: ThresholdSpec, x: np.ndarray, valid: np.ndarray, windows: list[tuple[int, int]],
                    max_missing_fraction: float = MAX_MISSING_FRACTION, sequential: bool = False) -> np.ndarray:
    """Threshold in mm for each row of x (series, time), with NaN where too many reference days are missing."""
    n_days = sum(b - a + 1 for a, b in windows)
    if n_days == 0:
        return np.full(x.shape[0], np.nan)
    if spec.kind in ("mean_total", "mean_daily"):
        total, compensation = np.zeros(x.shape[0]), np.zeros(x.shape[0])
        count = np.zeros(x.shape[0])
        for a, b in windows:
            part = _window_sum(np.where(valid[:, a:b + 1], x[:, a:b + 1], 0.0), sequential)
            if sequential:
                _add_compensated(total, compensation, part)
            else:
                total += part
            count += valid[:, a:b + 1].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
        mean = np.where(count > 0, mean, np.nan)
        out = mean * spec.days if spec.kind == "mean_total" else mean
    else:
        if spec.quantile is None or not 0 <= spec.quantile <= 1:
            raise ValueError("quantile_total needs a quantile between 0 and 1")
        sums, count = [], np.zeros(x.shape[0])
        for a, b in windows:
            xv = np.where(valid[:, a:b + 1], x[:, a:b + 1], 0.0)
            count += valid[:, a:b + 1].sum(axis=1)
            if xv.shape[1] < spec.days:
                continue
            cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(xv, axis=1)], axis=1)
            bad = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(~valid[:, a:b + 1], axis=1)], axis=1)
            totals = cs[:, spec.days:] - cs[:, :-spec.days]
            sums.append(np.where(bad[:, spec.days:] - bad[:, :-spec.days] == 0, totals, np.nan))
        if not sums:
            return np.full(x.shape[0], np.nan)
        with np.errstate(all="ignore"):
            out = np.nanquantile(np.concatenate(sums, axis=1), spec.quantile, axis=1)
    missing_days = n_days - count
    return np.where(missing_days / n_days > max_missing_fraction, np.nan, out)


def compute_threshold(spec: ThresholdSpec, rain, years: tuple[int, int], *,
                      max_missing_fraction: float = MAX_MISSING_FRACTION, partial_years: bool = False) -> np.ndarray:
    """Per-series threshold in mm (spatial shape) from the reference years y0..y1 inclusive.

    mean_total is days x the mean daily rain over the `during` days pooled across
    years; mean_daily is that mean; quantile_total is the quantile (linear
    interpolation) of all `days`-day running totals lying wholly inside `during`,
    pooled across years. Windows crossing the year end belong to the year of
    their start. A series (grid cell or unit: the spatial dims are whatever `rain` has,
    so a unit series gives a unit threshold) is NaN when more than max_missing_fraction
    (default 10 %; 1.0 never) of its reference days are missing.

    Options reproducing onset_blending's computed threshold: max_missing_fraction=1.0, partial_years=True
    (a year whose window is only partly in the data counts its days, where by default it is skipped) and
    float rain (rain_resolution "exact": millimetres are used as they are, summed day by day per year and the
    years added with Kahan compensation, the order of blending's per-file totals and groupby sum). The caller sets `days` (blending defaults it to the trigger window).
    """
    windows = _year_windows(spec, rain, years, partial_years)
    spatial_shape = rain.tenths.shape[:-1]
    n_time = rain.tenths.shape[-1]
    flat = rain.tenths.reshape(-1, n_time)
    flat_missing = rain.missing.reshape(-1, n_time)
    exact = resolution_of(flat) == "exact"
    out = np.empty(flat.shape[0])
    step = max(1, _CHUNK_VALUES // max(n_time, 1))
    for a in range(0, flat.shape[0], step):
        x = flat[a:a + step].astype(np.float64)
        if not exact:
            x = x / 10.0
        out[a:a + step] = _threshold_rows(spec, x, ~flat_missing[a:a + step], windows, max_missing_fraction, exact)
    return out.reshape(spatial_shape)


def _threshold_array(name: str, values: np.ndarray, spec: ThresholdSpec, rain, years: tuple[int, int]) -> xr.DataArray:
    """Wrap a threshold field as a DataArray whose attrs record its provenance."""
    dims = rain.dims[:-1]
    return xr.DataArray(values, dims=dims, coords={d: rain.coords[d] for d in dims}, name=name,
                        attrs={"units": "mm", "source": rain.source, "years": [int(years[0]), int(years[1])],
                               "spec": repr(spec)})


def compute_thresholds(definition: Definition, rain, years: tuple[int, int], **options) -> dict[str, xr.DataArray]:
    """Every threshold the definition names, as DataArrays in mm (attrs: source, years, spec).

    `options` are those of compute_threshold (max_missing_fraction, partial_years)."""
    return {name: _threshold_array(name, compute_threshold(spec, rain, years, **options), spec, rain, years)
            for name, spec in definition.thresholds}


def _during_mask(forecast, spec: ThresholdSpec, years: tuple[int, int], first_lead: int, last_lead: int) -> np.ndarray:
    """Which (init, lead) cells of a forecast count towards a threshold: shape (n_init, n_leads).

    A lead counts when its date label lies in the `during` window (the whole
    year when None) of a year in `years`, and its lead number lies in
    first_lead..last_lead. A window crossing the year end belongs to the year of
    its start.
    """
    from momp.onset.engine import _bound_key, _label_table

    n_leads = forecast.tenths.shape[-1]
    label_y, label_m, label_d = _label_table(forecast, n_leads)
    keys = label_keys(label_y, label_m, label_d)
    leads = np.zeros(n_leads, dtype=bool)
    leads[max(first_lead, 1) - 1:min(last_lead, n_leads)] = True
    if spec.during is None:
        hit = (label_y >= years[0]) & (label_y <= years[1])
        return hit & leads[None, :]
    start_ref, end_ref = spec.during
    wraps = (end_ref.month, end_ref.day) < (start_ref.month, start_ref.day)
    hit = np.zeros(keys.shape, dtype=bool)
    for shift in (-1, 0, 1):
        year = (label_y + shift).ravel()
        low = _bound_key(forecast.calendar, year, start_ref).reshape(keys.shape)
        high = _bound_key(forecast.calendar, year + wraps, end_ref).reshape(keys.shape)
        year = year.reshape(keys.shape)
        hit |= (year >= years[0]) & (year <= years[1]) & (keys >= low) & (keys <= high)
    return hit & leads[None, :]


def compute_thresholds_from_forecasts(definition: Definition, forecasts: Iterable, years: tuple[int, int],
                                      lead_window: tuple[int, int] = (1, 30),
                                      windows: Sequence[tuple[int, int]] | None = None) -> dict[str, xr.DataArray]:
    """The model's own climatology: every threshold of the definition from forecast days, as DataArrays in mm.

    Streams through the forecasts (ForecastRain objects on one grid), summing per
    cell the valid forecast days whose date label (from lead_label) lies in the
    threshold's `during` window of a year in `years` (inclusive), over the leads
    of lead_window, with members pooled. mean_total is days x that mean and
    mean_daily the mean; quantile_total is not supported. A cell is NaN when more
    than 10 % of the expected days are missing. attrs record source
    "model:<source of the first forecast>", years, lead_window and the spec.

    With `windows` (inclusive (first, last) lead pairs, each cut to lead_window)
    every window is pooled separately and each DataArray gains a leading
    "window" dim with coordinates window_first and window_last (spec section 13).
    A window with no forecast days in it is NaN.
    """
    spans = [(max(a, lead_window[0]), min(b, lead_window[1])) for a, b in (windows or [lead_window])]
    n_win = len(spans)
    specs = definition.thresholds
    for name, spec in specs:
        if spec.kind not in ("mean_total", "mean_daily"):
            raise NotImplementedError(
                f"threshold {name!r}: the model climatology supports only mean_total and mean_daily, not {spec.kind}")
    sums = {name: None for name, _ in specs}
    counts = {name: None for name, _ in specs}
    expected = {name: np.zeros(n_win, dtype=np.int64) for name, _ in specs}
    template = None
    for forecast in forecasts:
        if template is None:
            template = forecast
            shape = forecast.tenths.shape[2 if "member" in forecast.dims else 1:-1]
            for name, _ in specs:
                sums[name] = np.zeros((n_win, *shape), dtype=np.int64)
                counts[name] = np.zeros((n_win, *shape), dtype=np.int64)
        elif forecast.tenths.shape[1:-1] != template.tenths.shape[1:-1] or forecast.dims != template.dims:
            raise ValueError("forecasts must share dims and their member and spatial shape")
        valid = (~forecast.missing).astype(np.int64)
        x = np.where(forecast.missing, 0, forecast.tenths).astype(np.int64)
        letters = "abcdefgh"[:forecast.tenths.ndim - 2]
        kept = letters[1:] if "member" in forecast.dims else letters
        n_member = forecast.tenths.shape[1] if "member" in forecast.dims else 1
        for name, spec in specs:
            for w, (first, last) in enumerate(spans):
                mask = _during_mask(forecast, spec, years, first, last).astype(np.int64)
                sums[name][w] += np.einsum(f"i{letters}l,il->{kept}", x, mask)
                counts[name][w] += np.einsum(f"i{letters}l,il->{kept}", valid, mask)
                expected[name][w] += int(mask.sum()) * n_member
    if template is None:
        raise ValueError("no forecasts given")
    dims = tuple(d for d in template.dims if d not in ("init_time", "member", "lead"))
    out = {}
    for name, spec in specs:
        n = expected[name].reshape((-1,) + (1,) * (sums[name].ndim - 1))
        valid_days = counts[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(valid_days > 0, sums[name] / 10.0 / np.maximum(valid_days, 1), np.nan)
            values = mean * spec.days if spec.kind == "mean_total" else mean
            values = np.where(n == 0, np.nan, np.where((n - valid_days) / np.maximum(n, 1) > MAX_MISSING_FRACTION,
                                                       np.nan, values))
        coords = {d: template.coords[d] for d in dims if d in template.coords}
        attrs = {"units": "mm", "source": f"model:{template.source}", "years": [int(years[0]), int(years[1])],
                 "lead_window": [int(lead_window[0]), int(lead_window[1])], "spec": repr(spec)}
        if windows is None:
            out[name] = xr.DataArray(values[0], dims=dims, coords=coords, name=name, attrs=attrs)
        else:
            coords["window_first"] = ("window", np.array([a for a, _ in spans], dtype=np.int64))
            coords["window_last"] = ("window", np.array([b for _, b in spans], dtype=np.int64))
            out[name] = xr.DataArray(values, dims=("window", *dims), coords=coords, name=name, attrs=attrs)
    return out


def _doy_index(month: np.ndarray, day: np.ndarray) -> np.ndarray:
    """0-based day of a 365-day year for each label; -1 for Feb 29 and for days outside the 365-day year."""
    month = np.asarray(month, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    valid = (month >= 1) & (month <= 12) & (day >= 1) & (day <= _DAYS_IN_MONTH[np.clip(month - 1, 0, 11)])
    return np.where(valid, _CUMULATIVE_DAYS[np.clip(month - 1, 0, 11)] + day - 1, -1)


def _mean_cycle(rain, years: tuple[int, int]) -> np.ndarray:
    """Mean annual cycle (cells, 365) of the reference years, NaN for days with no valid data."""
    n_time = rain.tenths.shape[-1]
    flat = rain.tenths.reshape(-1, n_time)
    flat_missing = rain.missing.reshape(-1, n_time)
    doy = _doy_index(rain.month, rain.day)
    in_years = (rain.year >= years[0]) & (rain.year <= years[1])
    cycle = np.full((flat.shape[0], _YEAR_LENGTH), np.nan)
    for d in range(_YEAR_LENGTH):
        cols = np.nonzero((doy == d) & in_years)[0]
        if cols.size == 0:
            raise ValueError(
                f"No data for day {d + 1} of a 365-day year in {years}; "
                "harmonic_seasons needs a calendar with 365/366-day years")
        x = flat[:, cols] / 10.0
        valid = ~flat_missing[:, cols]
        count = valid.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cycle[:, d] = np.where(count > 0, np.where(valid, x, 0.0).sum(axis=1) / count, np.nan)
    return cycle


def _anomaly_extrema(cycle: np.ndarray, start: np.ndarray, length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onset and cessation day of year for one segment of the cycle per cell.

    The segment begins at 0-based day `start` and runs `length` days (both per
    cell). The cumulative sum of its anomaly about its own mean is formed;
    cessation is the day of its maximum and onset the day after its minimum.
    """
    n = cycle.shape[1]
    j = np.arange(n)
    rotated = np.take_along_axis(cycle, (start[:, None] + j[None, :]) % n, axis=1)
    inside = j[None, :] < length[:, None]
    mean = np.where(inside, rotated, 0.0).sum(axis=1) / length
    cum = np.cumsum(np.where(inside, rotated - mean[:, None], 0.0), axis=1)
    j_min = np.argmin(np.where(inside, cum, np.inf), axis=1)
    j_max = np.argmax(np.where(inside, cum, -np.inf), axis=1)
    onset = (start + j_min + 1) % n + 1
    cessation = (start + j_max) % n + 1
    return onset.astype(float), cessation.astype(float)


def harmonic_seasons(rain, years: tuple[int, int], max_seasons: int = 2) -> xr.Dataset:
    """Seasons per cell from harmonic analysis of the mean annual cycle (draft reading of Dunning et al. 2016).

    The mean daily cycle over `years` (Feb 29 dropped, 365 days) is decomposed
    with a Fourier transform. When the amplitude of the second harmonic exceeds
    that of the first, the cell has two seasons, otherwise one. For one season
    the cycle is rotated to start at the minimum of the first harmonic; for two,
    the year is split at the two minima of the second harmonic. Within each
    segment the anomaly about the segment mean is accumulated; onset is the day
    after the minimum of that sum and cessation the day of its maximum. This is
    a working interpretation, to be checked against the paper.

    Returns n_seasons (spatial) and onset_doy / cessation_doy (season, spatial)
    as 1-based days of the 365-day year, NaN where a cell has fewer seasons
    (or no valid data, when n_seasons is 0). Seasons are ordered by onset day of year.
    """
    if max_seasons not in (1, 2):
        raise ValueError("max_seasons must be 1 or 2")
    spatial_shape = rain.tenths.shape[:-1]
    cycle = _mean_cycle(rain, years)
    ok = np.isfinite(cycle).all(axis=1)
    n_cells = cycle.shape[0]
    onset = np.full((max_seasons, n_cells), np.nan)
    cessation = np.full((max_seasons, n_cells), np.nan)
    n_seasons = np.zeros(n_cells, dtype=np.int8)
    if ok.any():
        q = cycle[ok]
        spectrum = np.fft.rfft(q, axis=1)
        a1, a2 = np.abs(spectrum[:, 1]), np.abs(spectrum[:, 2])
        two = (a2 > a1) if max_seasons == 2 else np.zeros(len(q), dtype=bool)
        n = _YEAR_LENGTH
        t_min1 = ((np.pi - np.angle(spectrum[:, 1])) * n / (2 * np.pi)) % n
        t_min2 = ((np.pi - np.angle(spectrum[:, 2])) * n / (4 * np.pi)) % (n / 2)
        start1 = np.rint(t_min1).astype(np.int64) % n
        full = np.full(len(q), n)
        on1, ce1 = _anomaly_extrema(q, start1, full)
        start2 = np.rint(t_min2).astype(np.int64) % n
        mid = np.rint(t_min2 + n / 2).astype(np.int64) % n
        len_a = (mid - start2) % n
        on_a, ce_a = _anomaly_extrema(q, start2, len_a)
        on_b, ce_b = _anomaly_extrema(q, mid, n - len_a)
        first_is_a = on_a <= on_b
        on = np.stack([np.where(first_is_a, on_a, on_b), np.where(first_is_a, on_b, on_a)])
        ce = np.stack([np.where(first_is_a, ce_a, ce_b), np.where(first_is_a, ce_b, ce_a)])
        idx = np.nonzero(ok)[0]
        onset[0, idx] = np.where(two, on[0], on1)
        cessation[0, idx] = np.where(two, ce[0], ce1)
        if max_seasons == 2:
            onset[1, idx] = np.where(two, on[1], np.nan)
            cessation[1, idx] = np.where(two, ce[1], np.nan)
        n_seasons[idx] = np.where(two, 2, 1)
    dims = rain.dims[:-1]
    coords = {d: rain.coords[d] for d in dims}
    coords_season = {**coords, "season": np.arange(1, max_seasons + 1)}
    return xr.Dataset(
        {"n_seasons": (dims, n_seasons.reshape(spatial_shape)),
         "onset_doy": (("season",) + dims, onset.reshape((max_seasons,) + spatial_shape)),
         "cessation_doy": (("season",) + dims, cessation.reshape((max_seasons,) + spatial_shape))},
        coords=coords_season,
        attrs={"source": rain.source, "years": [int(years[0]), int(years[1])], "method": "harmonic (draft)"})


class ThresholdCache:
    """Stores computed threshold fields (a momp.cache.Store of kind "thresholds") so each is computed once.

    A field is keyed by its name, the spec, the reference years, the data source label, the grid coordinates
    (unit ids by value) and `files`, the identities (momp.cache.file_identity) of the files the reference is read
    from, so changed data under the same `source` is recomputed. Give ingested data a source.
    """

    def __init__(self, directory, files=()):
        """Use `directory` for the entries (created on first write); `files` identify the data read."""
        self.store = Store("thresholds", directory, codec=NETCDF, label="Thresholds")
        self.files = list(files)

    def parts(self, spec: ThresholdSpec, rain, years: tuple[int, int], name: str = "", **options) -> dict:
        """The parts that identify one threshold computation (options of compute_threshold only when given)."""
        parts = {"name": name, "source": rain.source, "years": [int(years[0]), int(years[1])], "spec": repr(spec),
                 "grid": {dim: rain.coords[dim] for dim in rain.dims[:-1]}, "files": self.files}
        if options:
            parts["options"] = {key: options[key] for key in sorted(options)}
        if resolution_of(rain.tenths) == "exact":
            parts["resolution"] = "exact"
        return parts

    def get_or_compute(self, spec: ThresholdSpec, rain, years: tuple[int, int], name: str, **options) -> xr.DataArray:
        """Return the stored threshold if present, else compute (with compute_threshold's `options`) and store it."""
        return self.store.get_or_compute(
            self.parts(spec, rain, years, name, **options),
            lambda: _threshold_array(name, compute_threshold(spec, rain, years, **options), spec, rain, years))


# ---------------------------------------------------------------------------
# Reference onset dates: one provider for a fixed month-day, a table, and the computed mean or median
# ---------------------------------------------------------------------------

ONSET_STATISTICS = ("mean", "median")


def reduce_onset_days(doy: xr.DataArray, how: str = "mean") -> xr.DataArray:
    """The climatological onset day of year over the "year" dim of `doy` (NaN for a year without an onset).

    "mean" is the mean over the years that have an onset, rounded half to even (ROMP's climatological onset);
    "median" is the median of those years, NaN where fewer than half of the years have an onset
    (engine.climatological_onset).
    """
    if how == "mean":
        return np.round(doy.mean(dim="year"))
    if how != "median":
        raise ValueError(f"the onset statistic must be one of {ONSET_STATISTICS}, got {how!r}")
    values = doy.transpose("year", ...).values
    n_found = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    median = np.where(n_found * 2 >= values.shape[0], median, np.nan)
    dims = tuple(d for d in doy.dims if d != "year")
    return xr.DataArray(median, dims=dims, coords={d: doy.coords[d] for d in dims if d in doy.coords})


@dataclass(frozen=True)
class ReferenceOnset:
    """The reference (climatological) onset date: one provider for every source.

    Sources: a fixed month-day ("06-02") in every year and unit, a table of dates keyed by unit id and/or year
    (onset_blending's reference_onset), or a computed field: the mean or median onset day of year over many years
    (`from_onsets`), as the climatology reference forecast and the derived seasons use.
    Consumers: `dates` (a date per (year, unit)), `search_start` (the entry for momp.onset.engine's detect_*
    search_start, so an onset search never begins before the reference onset) and `doy_field` (day of year per cell).
    """

    month_day: str | None = None
    table: pd.DataFrame | None = field(default=None, compare=False)
    computed: xr.DataArray | None = field(default=None, compare=False)
    statistic: str | None = None

    def __post_init__(self):
        """Exactly one source."""
        if sum(x is not None for x in (self.month_day, self.table, self.computed)) != 1:
            raise ValueError("a reference onset is exactly one of a month-day, a table or a computed day of year")

    @classmethod
    def from_onsets(cls, doy: xr.DataArray, how: str = "mean") -> "ReferenceOnset":
        """The mean or median of the onset days of year `doy` (dims year, *spatial) over the years (reduce_onset_days)."""
        return cls(computed=reduce_onset_days(doy, how), statistic=how)

    @classmethod
    def read(cls, path, *, date_col: str | None = None, year_col: str | None = None, day_col: str | None = None,
             base_date: str | None = None, unit_col: str | None = None) -> "ReferenceOnset":
        """A table of reference onset dates from a CSV: a date column, or a year and a day offset from `base_date`.

        Keyed by the unit-id column `unit_col` and/or the `year_col` (at least one); dates are datetime.date.
        """
        table = pd.read_csv(path, dtype={unit_col: str} if unit_col else None)
        if date_col and date_col in table.columns:
            table["ref_onset_date"] = pd.to_datetime(table[date_col], errors="coerce").dt.date
        elif year_col and day_col and base_date and year_col in table.columns and day_col in table.columns:
            table["ref_onset_date"] = table.apply(
                lambda row: (pd.to_datetime(f"{int(row[year_col])}-{base_date}")
                             + pd.Timedelta(days=int(row[day_col]))).date(), axis=1)
        else:
            raise ValueError("ref_onset file needs either a parseable 'date_col', or 'year_col'+'day_col'+'base_date'.")
        keys = []
        if unit_col:
            if unit_col not in table.columns:
                raise ValueError(f"ref_onset unit_col '{unit_col}' not in file columns {table.columns.tolist()}")
            table = table.rename(columns={unit_col: "id"})
            table["id"] = ids.normalize(table["id"], context="reference-onset IDs")
            keys.append("id")
        if year_col and year_col in table.columns:
            table = table.rename(columns={year_col: "year"})
            table["year"] = table["year"].astype(int)
            keys.append("year")
        if not keys:
            raise ValueError("ref_onset file needs a 'year_col' and/or 'unit_col' to key on.")
        return cls(table=table[keys + ["ref_onset_date"]].drop_duplicates())

    def dates(self, unit_ids, years) -> np.ndarray:
        """The reference onset date (datetime64[ns], NaT without one) of each (unit, year) pair, row by row.

        A computed field needs `unit_ids` to be its unit coordinate (a `unit` dim) and gives each year's Jan 1 plus
        its day of year.
        """
        years = np.asarray(years, dtype=int)
        none = np.full(len(years), np.datetime64("NaT"), dtype="datetime64[ns]")
        if self.month_day is not None:
            distinct, codes = np.unique(years, return_inverse=True)
            dates = pd.to_datetime(pd.Series([f"{y}-{self.month_day}" for y in distinct]), errors="coerce").to_numpy()
            return dates[codes].astype("datetime64[ns]")
        if self.computed is not None:
            field_ = self.computed
            if field_.ndim != 1 or "unit" not in field_.dims:
                raise ValueError("dates by unit id need a computed reference onset over a 'unit' dim")
            doy = field_.sel(unit=np.asarray(unit_ids)).values
            jan1 = (years - 1970).astype("datetime64[Y]").astype("datetime64[D]")
            with np.errstate(invalid="ignore"):
                offset = np.where(np.isfinite(doy), doy, 1).astype(np.int64) - 1
            out = (jan1 + offset.astype("timedelta64[D]")).astype("datetime64[ns]")
            return np.where(np.isfinite(doy), out, none)
        keys = [k for k in ("id", "year") if k in self.table.columns]
        if not keys:
            return none
        table = self.table.drop_duplicates(keys + ["ref_onset_date"])
        if table.duplicated(keys).any():
            raise ValueError("The reference onset table gives a unit and year more than one date")
        pairs = pd.DataFrame({"id": pd.Series(np.asarray(unit_ids, dtype=str)), "year": years})
        found = pairs[keys].merge(table, on=keys, how="left")["ref_onset_date"]
        return pd.to_datetime(found).to_numpy().astype("datetime64[ns]")

    def search_start(self, years, unit_ids=None, *, dims: tuple[str, ...] = ("unit",)):
        """The `search_start` entry of detect_observed / detect_forecast for these years.

        A month-day is the "MM-DD" string. Otherwise a DataArray of dates (NaT for none) over "year" and, for a table
        keyed by unit, the `dims` of the units (default one "unit" dim holding `unit_ids`); a computed field
        over its own spatial dims gives each year's date.
        """
        years = np.asarray(years, dtype=int)
        if self.month_day is not None:
            return self.month_day
        if self.computed is not None and (unit_ids is None or "unit" not in self.computed.dims):
            doy = self.computed
            jan1 = (years - 1970).astype("datetime64[Y]").astype("datetime64[D]")
            values = np.where(np.isfinite(doy.values)[None], jan1.reshape((-1,) + (1,) * doy.ndim)
                              + (np.nan_to_num(doy.values, nan=1.0).astype(np.int64) - 1)[None].astype("timedelta64[D]"),
                              np.datetime64("NaT")).astype("datetime64[ns]")
            return xr.DataArray(values, dims=("year", *doy.dims), coords={"year": years, **doy.coords})
        if unit_ids is None:
            keyed = [k for k in ("id", "year") if k in self.table.columns]
            if keyed != ["year"]:
                raise ValueError("a reference onset table keyed by unit needs the unit ids")
            unit_ids = [""]
        unit_ids = np.asarray(unit_ids)
        grid_years = np.repeat(years, len(unit_ids))
        grid_units = np.tile(unit_ids, len(years))
        values = self.dates(grid_units, grid_years).reshape(len(years), len(unit_ids))
        return xr.DataArray(values, dims=("year", dims[0]), coords={"year": years, dims[0]: unit_ids})

    def doy_field(self, like: xr.DataArray) -> xr.DataArray:
        """The day of year of the reference onset on the grid of `like` (the climatology reference forecast's input).

        A computed field is returned as it is; a month-day is its day of year in a year of 365 days. A table varies by
        year (and unit), so it has no single day-of-year field.
        """
        if self.computed is not None:
            return self.computed
        if self.month_day is None:
            raise ValueError("a reference onset table has dates per year; it gives no day-of-year field")
        doy = pd.Timestamp(f"2001-{self.month_day}").dayofyear
        return xr.full_like(like.astype(float), float(doy))
