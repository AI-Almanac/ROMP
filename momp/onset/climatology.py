"""Reference-climatology products: per-cell thresholds, harmonic season analysis and a threshold cache."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from momp.onset.calendar import fixed_bounds, label_keys
from momp.onset.model import Definition, ThresholdSpec

MAX_MISSING_FRACTION = 0.10
_CHUNK_VALUES = 1 << 24
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_CUMULATIVE_DAYS = np.concatenate([[0], np.cumsum(_DAYS_IN_MONTH)])
_YEAR_LENGTH = 365


def _year_windows(spec: ThresholdSpec, rain, years: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive (first, last) time indices of the `during` window in each reference year.

    A window crossing the year end belongs to the year of its start. Years whose
    window is not fully inside the data are skipped.
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
    first, last = fixed_bounds(keys, year_list, spec.during[0], spec.during[1])
    return [(int(a), int(b)) for a, b in zip(first, last) if a >= 0 and b >= a]


def _threshold_rows(spec: ThresholdSpec, x: np.ndarray, valid: np.ndarray,
                    windows: list[tuple[int, int]]) -> np.ndarray:
    """Threshold in mm for each row of x (series, time), with NaN where too many reference days are missing."""
    n_days = sum(b - a + 1 for a, b in windows)
    if n_days == 0:
        return np.full(x.shape[0], np.nan)
    if spec.kind in ("mean_total", "mean_daily"):
        total = np.zeros(x.shape[0])
        count = np.zeros(x.shape[0])
        for a, b in windows:
            total += np.where(valid[:, a:b + 1], x[:, a:b + 1], 0.0).sum(axis=1)
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
    return np.where(missing_days / n_days > MAX_MISSING_FRACTION, np.nan, out)


def compute_threshold(spec: ThresholdSpec, rain, years: tuple[int, int]) -> np.ndarray:
    """Per-cell threshold in mm (spatial shape) from the reference years y0..y1 inclusive.

    mean_total is days x the mean daily rain over the `during` days pooled across
    years; mean_daily is that mean; quantile_total is the quantile (linear
    interpolation) of all `days`-day running totals lying wholly inside `during`,
    pooled across years. Windows crossing the year end belong to the year of
    their start. A cell is NaN when more than 10 % of its reference days are missing.
    """
    windows = _year_windows(spec, rain, years)
    spatial_shape = rain.tenths.shape[:-1]
    n_time = rain.tenths.shape[-1]
    flat = rain.tenths.reshape(-1, n_time)
    flat_missing = rain.missing.reshape(-1, n_time)
    out = np.empty(flat.shape[0])
    step = max(1, _CHUNK_VALUES // max(n_time, 1))
    for a in range(0, flat.shape[0], step):
        x = flat[a:a + step].astype(np.float64) / 10.0
        out[a:a + step] = _threshold_rows(spec, x, ~flat_missing[a:a + step], windows)
    return out.reshape(spatial_shape)


def _threshold_array(name: str, values: np.ndarray, spec: ThresholdSpec, rain, years: tuple[int, int]) -> xr.DataArray:
    """Wrap a threshold field as a DataArray whose attrs record its provenance."""
    dims = rain.dims[:-1]
    return xr.DataArray(values, dims=dims, coords={d: rain.coords[d] for d in dims}, name=name,
                        attrs={"units": "mm", "source": rain.source, "years": [int(years[0]), int(years[1])],
                               "spec": repr(spec)})


def compute_thresholds(definition: Definition, rain, years: tuple[int, int]) -> dict[str, xr.DataArray]:
    """Every threshold the definition names, as DataArrays in mm (attrs: source, years, spec)."""
    return {name: _threshold_array(name, compute_threshold(spec, rain, years), spec, rain, years)
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
                                      lead_window: tuple[int, int] = (1, 30)) -> dict[str, xr.DataArray]:
    """The model's own climatology: every threshold of the definition from forecast days, as DataArrays in mm.

    Streams through the forecasts (ForecastRain objects on one grid), summing per
    cell the valid forecast days whose date label (from lead_label) lies in the
    threshold's `during` window of a year in `years` (inclusive), over the leads
    of lead_window, with members pooled. mean_total is days x that mean and
    mean_daily the mean; quantile_total is not supported. A cell is NaN when more
    than 10 % of the expected days are missing. attrs record source
    "model:<source of the first forecast>", years, lead_window and the spec.
    """
    specs = definition.thresholds
    for name, spec in specs:
        if spec.kind not in ("mean_total", "mean_daily"):
            raise NotImplementedError(
                f"threshold {name!r}: the model climatology supports only mean_total and mean_daily, not {spec.kind}")
    sums = {name: None for name, _ in specs}
    counts = {name: None for name, _ in specs}
    expected = {name: 0 for name, _ in specs}
    template = None
    for forecast in forecasts:
        if template is None:
            template = forecast
            shape = forecast.tenths.shape[2 if "member" in forecast.dims else 1:-1]
            for name, _ in specs:
                sums[name], counts[name] = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
        elif forecast.tenths.shape[1:-1] != template.tenths.shape[1:-1] or forecast.dims != template.dims:
            raise ValueError("forecasts must share dims and their member and spatial shape")
        for name, spec in specs:
            mask = _during_mask(forecast, spec, years, lead_window[0], lead_window[1]).astype(np.int64)
            valid = (~forecast.missing).astype(np.int64)
            x = np.where(forecast.missing, 0, forecast.tenths).astype(np.int64)
            letters = "abcdefgh"[:forecast.tenths.ndim - 2]
            kept = letters[1:] if "member" in forecast.dims else letters
            sums[name] += np.einsum(f"i{letters}l,il->{kept}", x, mask)
            counts[name] += np.einsum(f"i{letters}l,il->{kept}", valid, mask)
            n_member = forecast.tenths.shape[1] if "member" in forecast.dims else 1
            expected[name] += int(mask.sum()) * n_member
    if template is None:
        raise ValueError("no forecasts given")
    dims = tuple(d for d in template.dims if d not in ("init_time", "member", "lead"))
    out = {}
    for name, spec in specs:
        n, valid_days = expected[name], counts[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(valid_days > 0, sums[name] / 10.0 / np.maximum(valid_days, 1), np.nan)
        values = mean * spec.days if spec.kind == "mean_total" else mean
        if n == 0:
            values = np.full(values.shape, np.nan)
        else:
            values = np.where((n - valid_days) / n > MAX_MISSING_FRACTION, np.nan, values)
        out[name] = xr.DataArray(
            values, dims=dims, coords={d: template.coords[d] for d in dims if d in template.coords}, name=name,
            attrs={"units": "mm", "source": f"model:{template.source}", "years": [int(years[0]), int(years[1])],
                   "lead_window": [int(lead_window[0]), int(lead_window[1])], "spec": repr(spec)})
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
    """Stores computed threshold fields as NetCDF files so each is computed once.

    The key is the sha256 of the data source, reference years, the spec's repr and
    the grid coordinates, so an empty `source` risks collisions between datasets
    on the same grid; give ingested data a source.
    """

    def __init__(self, directory):
        """Use `directory` (created on first write) for the cache files."""
        self.directory = Path(directory)

    @staticmethod
    def key(spec: ThresholdSpec, rain, years: tuple[int, int]) -> str:
        """Hex digest identifying one threshold computation."""
        grid = hashlib.sha256()
        for dim in rain.dims[:-1]:
            grid.update(dim.encode())
            grid.update(np.ascontiguousarray(rain.coords[dim]).tobytes())
        payload = json.dumps([rain.source, [int(years[0]), int(years[1])], repr(spec), grid.hexdigest()])
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_or_compute(self, spec: ThresholdSpec, rain, years: tuple[int, int], name: str) -> xr.DataArray:
        """Return the stored threshold if present, else compute and store it."""
        path = self.directory / f"{name}_{self.key(spec, rain, years)[:16]}.nc"
        if path.exists():
            with xr.open_dataarray(path) as stored:
                return stored.load()
        result = _threshold_array(name, compute_threshold(spec, rain, years), spec, rain, years)
        self.directory.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(path)
        return result
