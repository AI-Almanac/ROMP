"""Turn gridded daily rainfall (observed or forecast) into the normalized DailyRain / ForecastRain types.

Ingestion is where units, day boundaries, calendars, gaps and missing values are
settled once, so every later stage works on integer tenths of a millimetre, or,
with rain_resolution "exact", on float64 millimetres. The missing-rain policy
("flag", "zero", "drop" or "keep") is applied here ("zero") or recorded for the
engine and backends (the others).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

from hcwf.onset.model import MISSING_POLICIES, RESOLUTIONS, DailyRain, ForecastRain
from hcwf.io.dims import ROLE_ALIASES, find_dim

MM_PER_UNIT = {
    **{name: 1.0 for name in ("mm", "mm/day", "mm/d", "mm day-1", "mm d-1", "mm day**-1", "kg m-2", "kg m**-2",
                              "kg m-2 day-1", "kg m-2 d-1", "kg/m2", "kg/m2/day")},
    **{name: 1000.0 for name in ("m", "metres", "meters", "m/day", "m/d", "m day-1", "m d-1")},
}
"""Millimetres per unit of a rainfall total's declared units, by lower-case name with single spaces (the one table;
the CF spellings of onset_blending included)."""

# Offsets from UTC in hours for the zone names day boundaries may use.
_ZONES = {"UTC": 0.0, "GMT": 0.0, "Z": 0.0, "IST": 5.5, "EAT": 3.0, "CAT": 2.0, "WAT": 1.0,
          "SAST": 2.0, "PKT": 5.0, "NPT": 5.75, "JST": 9.0}

_BOUNDARY_RE = re.compile(
    r"^\s*(\d{1,2})(?::(\d{2}))?\s*([A-Za-z]+)\s*,?\s*labell?ed\s+by\s+(start|end)\s*$", re.IGNORECASE)

_BLOCK_VALUES = 1 << 24
_INT32_MAX = float(np.iinfo(np.int32).max)


def parse_day_boundary(text: str) -> tuple[float, str]:
    """Parse e.g. "03 UTC, labelled by end" or "08:30 IST, labelled by end" into (hour in UTC, "start"|"end").

    The hour is in [0, 24). IST is India Standard Time (UTC+5.5). Unknown zones are an error.
    """
    match = _BOUNDARY_RE.match(text or "")
    if not match:
        raise ValueError(
            f"Cannot parse day boundary {text!r}; expected e.g. '00 UTC, labelled by start' "
            "or '08:30 IST, labelled by end'")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    zone, label = match.group(3).upper(), match.group(4).lower()
    if hour > 23 or minute > 59:
        raise ValueError(f"Day boundary time out of range in {text!r}")
    if zone not in _ZONES:
        raise ValueError(f"Unsupported time zone {zone!r} in {text!r}; supported: {sorted(_ZONES)}")
    return (hour + minute / 60.0 - _ZONES[zone]) % 24.0, label


def observation_shift_days(obs_boundary: str, lead_label: str) -> int:
    """Days that move the observations' dates onto the forecasts' labels: 1 for observations labelled by their start
    against forecast leads labelled by their end (as onset_blending reads them), else 0."""
    return 1 if parse_day_boundary(obs_boundary)[1] == "start" and lead_label == "end" else 0


def unit_scale(units: str | None) -> float:
    """Return the factor taking the given rainfall units to mm (see MM_PER_UNIT), or raise for missing/unknown units."""
    if units is None:
        raise ValueError("No rainfall units: pass units=... or set the 'units' attribute on the data")
    key = " ".join(str(units).lower().split())
    if key in MM_PER_UNIT:
        return MM_PER_UNIT[key]
    raise ValueError(f"Unsupported rainfall units {units!r}; supported: mm, mm/day, kg m-2, m, m/day")


def _find_dim(da: xr.DataArray, role: str, override: str | None, label: str | None = None,
              required: bool = True, exclude: tuple = ()) -> str | None:
    """Pick the dim playing `role` (a key of ROLE_ALIASES): the override if given, else the first alias present."""
    label = label or role
    if override is not None:
        if override not in da.dims:
            raise ValueError(f"{label} dim {override!r} is not in the data dims {da.dims}")
        return override
    found = find_dim(list(da.dims), role, exclude=exclude, coords=da.coords)
    if found is None and required:
        raise ValueError(f"Cannot find the {label} dim among {da.dims}; tried {ROLE_ALIASES[role]}")
    return found


def _rename_spatial(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude style dims to lat and lon."""
    renames = {}
    for target in ("lat", "lon"):
        if target in da.dims:
            continue
        name = find_dim(list(da.dims), target)
        if name is not None:
            renames[name] = target
    return da.rename(renames) if renames else da


def _read_time(values: np.ndarray, constant_time_of_day: bool) -> tuple[str, np.ndarray]:
    """Return (calendar, whole-day ordinals) for datetime64 or cftime time values."""
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("The time axis is empty")
    if np.issubdtype(values.dtype, np.datetime64):
        days = values.astype("datetime64[D]")
        if constant_time_of_day and np.unique(values - days).size > 1:
            raise ValueError("Time stamps are not all at the same time of day; the data is not daily")
        return "standard", days.astype(np.int64)
    if values.dtype == object and hasattr(values.flat[0], "calendar"):
        import cftime
        calendar = values.flat[0].calendar or "standard"
        raw = np.asarray(cftime.date2num(list(values), "days since 0001-01-01", calendar=calendar), dtype=float)
        whole = np.floor(raw + 1e-9)
        if constant_time_of_day and np.ptp(raw - whole) > 1e-6:
            raise ValueError("Time stamps are not all at the same time of day; the data is not daily")
        return calendar, whole.astype(np.int64)
    raise ValueError(f"Unsupported time axis dtype {values.dtype}; need datetime64 or cftime objects")


def _labels(ordinals: np.ndarray, calendar: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (time values, year, month, day) for day ordinals in the calendar."""
    if calendar == "standard":
        days = ordinals.astype("datetime64[D]")
        months = days.astype("datetime64[M]")
        year = days.astype("datetime64[Y]").astype(np.int64) + 1970
        month = months.astype(np.int64) % 12 + 1
        day = (days - months).astype(np.int64) + 1
        return days.astype("datetime64[ns]"), year, month, day
    import cftime
    dates = cftime.num2date(ordinals, "days since 0001-01-01", calendar=calendar,
                            only_use_cftime_datetimes=True)
    dates = np.asarray(dates, dtype=object)
    year = np.array([d.year for d in dates], dtype=np.int64)
    month = np.array([d.month for d in dates], dtype=np.int64)
    day = np.array([d.day for d in dates], dtype=np.int64)
    return dates, year, month, day


def _coords(da: xr.DataArray, dims: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Coordinate values for each dim, using an index range for dims without coordinates."""
    out = {}
    for dim in dims:
        out[dim] = np.asarray(da[dim].values) if dim in da.coords else np.arange(da.sizes[dim])
    return out


NEGATIVES = ("default", "keep")
"""Handling of negative rain: "default" (observations below -0.05 mm are missing, forecasts are set to 0) or
"keep" (negative values stay, as onset_blending reads them)."""
GAPS = ("fill", "refuse")
"""Days missing from the time axis: "fill" (they become missing days, with a warning) or "refuse" (a gap inside a
calendar year is an error, as onset_blending refuses it)."""


def _check_options(rain_resolution: str, missing: str, negatives: str = "default") -> None:
    """Raise unless the resolution, missing-rain policy and negative-value handling are known."""
    if negatives not in NEGATIVES:
        raise ValueError(f"negatives must be one of {NEGATIVES}, got {negatives!r}")
    if rain_resolution not in RESOLUTIONS:
        raise ValueError(f"rain_resolution must be one of {RESOLUTIONS}, got {rain_resolution!r}")
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}, got {missing!r}")


def _to_tenths_numpy(values: np.ndarray, to_tenths: float, clip_negative: bool,
                     exact: bool = False, keep_negative: bool = False) -> tuple[np.ndarray, np.ndarray, int, float]:
    """The numpy reference of _to_tenths (several passes, float64 temporaries); used without numba and by the tests."""
    x = np.asarray(values, dtype=np.float64) * to_tenths
    floor = -0.05 if exact else -0.5
    finite = np.isfinite(x)
    bad = finite & (x < 0) if clip_negative else finite & (x < floor)
    if keep_negative:
        bad = np.zeros(x.shape, dtype=bool)
    lowest = float(x[bad].min()) if bad.any() else 0.0
    if clip_negative:
        x = np.where(bad, 0.0, x)
    too_big = np.zeros(x.shape, dtype=bool) if exact else np.abs(np.where(finite, x, 0.0)) > _INT32_MAX
    missing = ~finite | too_big | (bad & (not clip_negative))
    kept = np.where(missing, 0.0, x)
    tenths = kept if exact else np.rint(kept).astype(np.int32)
    return tenths, missing, int(bad.sum()), lowest


_CHUNKS = 256
"""How many pieces a block is cut into for the parallel kernel (each keeps its own count and lowest value)."""


def _kernel():
    """The numba kernel module (hcwf.onset.tenths_kernel) when numba is installed, else None; imported on first use."""
    from hcwf.onset import tenths_kernel

    return tenths_kernel if tenths_kernel.numba is not None else None


def _to_tenths(values: np.ndarray, to_tenths: float, clip_negative: bool,
               out: tuple[np.ndarray, np.ndarray] | None = None,
               exact: bool = False, keep_negative: bool = False) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Convert one block of rainfall to (int32 tenths, missing flags, count of negatives handled, lowest value in tenths).

    Observations: values below -0.05 mm become missing (they signal bad data).
    Forecasts (clip_negative): every negative value becomes 0, as model rain can dip below zero numerically.
    One parallel numba pass gives exactly what _to_tenths_numpy gives. `out` is an optional pair of C-contiguous
    arrays (int32, bool) of the block's shape to fill instead of allocating new ones. With exact, `to_tenths` is
    the factor to mm, the first array is float64 mm (unrounded) and the lowest value is in mm. With keep_negative
    negative values are neither clipped nor flagged (onset_blending keeps them).
    """
    kernel = _kernel()
    if kernel is None:
        tenths, missing, n_bad, low = _to_tenths_numpy(values, to_tenths, clip_negative, exact, keep_negative)
        if out is not None:
            out[0][...], out[1][...] = tenths, missing
            tenths, missing = out
        return tenths, missing, n_bad, low
    values = np.asarray(values)
    if values.dtype not in (np.float32, np.float64):
        values = values.astype(np.float64)
    flat = np.ascontiguousarray(values).reshape(-1)
    tenths, missing = out if out is not None else (
        np.empty(values.shape, np.float64 if exact else np.int32), np.empty(values.shape, bool))
    n_bad = np.zeros(_CHUNKS, dtype=np.int64)
    lowest = np.zeros(_CHUNKS, dtype=np.float64)
    kernel._tenths_kernel(flat, float(to_tenths), bool(clip_negative), bool(exact), bool(keep_negative), tenths.reshape(-1),
                   missing.reshape(-1), n_bad, lowest)
    return tenths, missing, int(n_bad.sum()), float(lowest.min())


def _convert(da: xr.DataArray, scale: float, out_shape: tuple[int, ...],
             positions: np.ndarray | None, clip_negative: bool = False,
             exact: bool = False, keep_negative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Convert da to (tenths, missing) of shape out_shape, reading blocks along its first axis.

    With `positions`, the last axis of da lands at those indices of the output's
    last axis and every other output day stays missing. With exact, tenths is float64 mm.
    """
    tenths = np.zeros(out_shape, dtype=np.float64 if exact else np.int32)
    per_mm = 1.0 if exact else 10.0
    missing = np.ones(out_shape, dtype=bool)
    if da.ndim == 1:
        blocks = [(slice(None), da.values)]
    else:
        row_size = max(1, int(np.prod(da.shape[1:], dtype=np.int64)))
        step = max(1, _BLOCK_VALUES // row_size)
        blocks = ((slice(a, a + step), da.isel({da.dims[0]: slice(a, a + step)}).values)
                  for a in range(0, da.shape[0], step))
    n_negative, lowest = 0, 0.0
    for rows, block in blocks:
        direct = da.ndim > 1 and positions is None   # the block's rows are contiguous in the output: fill them in place
        t, m, neg, low = _to_tenths(block, scale * per_mm, clip_negative,
                                    out=(tenths[rows], missing[rows]) if direct else None, exact=exact,
                                    keep_negative=keep_negative)
        n_negative += neg
        lowest = min(lowest, low)
        if direct:
            pass
        elif da.ndim == 1:
            index = slice(None) if positions is None else positions
            tenths[index], missing[index] = t, m
        else:
            tenths[rows, ..., positions] = t
            missing[rows, ..., positions] = m
    if n_negative and clip_negative:
        warnings.warn(f"{n_negative} negative values (lowest {lowest / per_mm:.2f} mm) were set to 0",
                      UserWarning, stacklevel=3)
    elif n_negative:
        warnings.warn(f"{n_negative} values below -0.05 mm (lowest {lowest / per_mm:.2f} mm) were set to missing",
                      UserWarning, stacklevel=3)
    return tenths, missing


def _refuse_gaps(ordinals: np.ndarray, calendar: str) -> None:
    """Raise for the first day missing between two days of one calendar year."""
    _, year, _, _ = _labels(ordinals, calendar)
    jump = (np.diff(ordinals) != 1) & (year[1:] == year[:-1])
    if jump.any():
        at = int(np.flatnonzero(jump)[0])
        before, after = (_labels(ordinals[at:at + 2], calendar)[0][i] for i in (0, 1))
        raise ValueError(f"The time axis is not daily-contiguous within a year; first gap is {str(before)[:10]} "
                         f"to {str(after)[:10]}")


def ingest(da: xr.DataArray, *, day_boundary: str, units: str | None = None, source: str = "",
           rain_resolution: str = "tenths", missing: str = "flag", negatives: str = "default",
           gaps: str = "fill") -> DailyRain:
    """Normalize observed daily rainfall to DailyRain (spec section 5).

    Renames the time dim to "time" and the spatial dims to lat/lon, moves time
    last, converts to int32 tenths of mm (half-even), flags NaN and values below
    -0.05 mm as missing, fills time gaps with missing days (with a warning) and
    labels every day in the data's own calendar. Duplicate times are an error.
    The data are read block by block, so subset da before calling for large inputs.

    rain_resolution "exact" keeps float64 millimetres, unrounded, instead of tenths.
    missing is the policy for missing days: "flag" (they make a span unknown),
    "zero" (they become 0 mm, not missing), "drop" (the engine removes every
    cell-year with a missing day) or "keep" (they count 0 in totals, are neither
    wet nor dry, and a missing candidate day is rejected); the last two are
    recorded on the result for the engine.

    negatives "keep" leaves negative values as they are instead of flagging those below -0.05 mm; gaps "refuse"
    raises for days missing inside a calendar year instead of filling them (see NEGATIVES and GAPS).
    """
    _check_options(rain_resolution, missing, negatives)
    if gaps not in GAPS:
        raise ValueError(f"gaps must be one of {GAPS}, got {gaps!r}")
    parse_day_boundary(day_boundary)
    scale = unit_scale(units if units is not None else da.attrs.get("units"))
    time_dim = _find_dim(da, "time", None)
    if time_dim not in da.coords:
        raise ValueError(f"The time dim {time_dim!r} has no coordinate values")
    da = _rename_spatial(da.rename({time_dim: "time"}) if time_dim != "time" else da)
    others = tuple(d for d in da.dims if d != "time")
    da = da.transpose(*others, "time")

    calendar, ordinals = _read_time(da["time"].values, constant_time_of_day=True)
    steps = np.diff(ordinals)
    if np.any(steps == 0):
        raise ValueError(f"Duplicate days on the time axis ({int((steps == 0).sum())} repeats)")
    if np.any(steps < 0):
        raise ValueError("The time axis is not increasing")
    full = np.arange(ordinals[0], ordinals[-1] + 1)
    positions = None
    if len(full) != len(ordinals):
        if gaps == "refuse":
            _refuse_gaps(ordinals, calendar)
        warnings.warn(f"Filled {len(full) - len(ordinals)} missing days in the time axis", UserWarning,
                      stacklevel=2)
        positions = ordinals - ordinals[0]
    tenths, flags = _convert(da, scale, da.shape[:-1] + (len(full),), positions, exact=rain_resolution == "exact",
                             keep_negative=negatives == "keep")
    if missing == "zero":
        flags[...] = False
    time_values, year, month, day = _labels(full, calendar)
    coords = _coords(da, others)
    coords["time"] = time_values
    return DailyRain(tenths=tenths, missing=flags, dims=others + ("time",), coords=coords, year=year,
                     month=month, day=day, calendar=calendar, day_boundary=day_boundary, source=source,
                     missing_policy=missing)


def _lead_days(da: xr.DataArray) -> np.ndarray:
    """Whole-day lead values, checked to be exactly 1, 2, ... n."""
    if "lead" not in da.coords:
        raise ValueError("The lead dim has no coordinate values; cannot check that lead k is the k-th 24 h")
    values = np.asarray(da["lead"].values)
    if np.issubdtype(values.dtype, np.timedelta64):
        days = values / np.timedelta64(1, "D")
    elif np.issubdtype(values.dtype, np.number):
        days = values.astype(np.float64)
    else:
        raise ValueError(f"Unsupported lead dtype {values.dtype}; need integer days or timedeltas")
    if not np.allclose(days, np.rint(days)) or not np.array_equal(np.rint(days).astype(np.int64),
                                                                  np.arange(1, len(days) + 1)):
        raise ValueError(
            "Leads must be consecutive whole days 1, 2, ..., n (lead k is the k-th 24 h after the init); "
            f"got {days[:6].tolist()}{'...' if len(days) > 6 else ''}")
    return np.arange(1, len(days) + 1)


# ---------------------------------------------------------------------------
# Accumulation, sub-daily steps and daily boundaries (spec section 5)
# ---------------------------------------------------------------------------

class IngestInfo(UserWarning):
    """Informational message from ingest_forecast (detected accumulation, step resolution, snapping notes).

    Emitted through `warnings.warn` so it is visible by default and tests can capture or filter it
    with `warnings.simplefilter("ignore", IngestInfo)`.
    """


@dataclass(frozen=True)
class ForecastIngestInfo:
    """How a forecast was turned into daily totals.

    accumulation is "daily" or "since_init"; accumulation_evidence says why (or "given" when the
    caller chose it); step_hours is the detected step length in hours; resolution is "daily" or
    "N-hourly"; daily_boundary is the requested boundary text or None; offsets_hours are the distinct
    hours from the init to the first day boundary (0 = days run from the init); notes lists things
    dropped, such as incomplete final days.
    """

    accumulation: str
    accumulation_evidence: str
    step_hours: float
    resolution: str
    daily_boundary: str | None
    offsets_hours: tuple[float, ...]
    notes: tuple[str, ...]


_CLOCK_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([A-Za-z]+)\s*$")


def _parse_clock(text: str) -> float:
    """Parse "03 UTC" / "08:30 IST" (or a full day-boundary string) into an hour of the UTC day in [0, 24)."""
    match = _CLOCK_RE.match(text or "")
    if not match:
        return parse_day_boundary(text)[0]
    hour, minute, zone = int(match.group(1)), int(match.group(2) or 0), match.group(3).upper()
    if hour > 23 or minute > 59:
        raise ValueError(f"Daily boundary time out of range in {text!r}")
    if zone not in _ZONES:
        raise ValueError(f"Unsupported time zone {zone!r} in {text!r}; supported: {sorted(_ZONES)}")
    return (hour + minute / 60.0 - _ZONES[zone]) % 24.0


def lead_hours(da: xr.DataArray, lead_dim: str) -> np.ndarray:
    """Step lengths since the init in hours (float), from timedeltas or numbers.

    Numbers use the coordinate's units attribute ("hours"/"days") when present; without it,
    integers are whole days if they are 1..n or stay below 24 (the daily contract), else hours.
    """
    if lead_dim not in da.coords:
        raise ValueError("The lead dim has no coordinate values; cannot check that lead k is the k-th 24 h")
    coord = da[lead_dim]
    values = np.asarray(coord.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values / np.timedelta64(1, "h")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Unsupported lead dtype {values.dtype}; need integer days, integer hours or timedeltas")
    num = values.astype(np.float64)
    unit = str(coord.attrs.get("units", "")).lower()
    if unit.startswith("hour") or unit in ("h", "hr", "hrs"):
        return num
    if unit.startswith("day") or unit == "d":
        return num * 24.0
    if np.array_equal(num, np.arange(1, len(num) + 1)) or not np.allclose(num, np.rint(num)) or num.max() < 24:
        return num * 24.0
    return num


def _resolution(hours: np.ndarray) -> tuple[float, str]:
    """Return (step length in hours, "daily" or "N-hourly") as the smallest gap of the positive steps from 0."""
    dt = float(np.min(np.diff(np.concatenate([[0.0], hours]))))
    return dt, ("daily" if abs(dt - 24.0) < 1e-6 else f"{dt:g}-hourly")


def _sample(da: xr.DataArray, lead_dim: str, max_series: int = 20000) -> xr.DataArray:
    """Subsample every non-lead dim with a regular stride so that at most about max_series series remain."""
    others = [d for d in da.dims if d != lead_dim]
    total = int(np.prod([da.sizes[d] for d in others], dtype=np.int64)) if others else 1
    if total <= max_series:
        return da
    stride = int(np.ceil((total / max_series) ** (1.0 / len(others))))
    return da.isel({d: slice(None, None, stride) for d in others})


def _series(da: xr.DataArray, lead_dim: str) -> np.ndarray:
    """Return the data as float64 of shape (series, lead)."""
    others = [d for d in da.dims if d != lead_dim]
    return np.asarray(da.transpose(*others, lead_dim).values, dtype=np.float64).reshape(-1, da.sizes[lead_dim])


def detect_accumulation(da: xr.DataArray, lead_dim: str | None = None) -> tuple[str, str]:
    """Decide whether forecast rainfall is "daily" (per-interval) or "since_init" (accumulated), with the evidence.

    Looks at a regular subsample (at most about 20000 series) along the lead dim, ignoring step 0.
    Since-init needs values non-decreasing along lead (within 0.1 % of the series maximum) in at
    least 99 % of the non-constant series that are fully finite, and the last lead's mean more than
    3x the first lead's. Daily is anything clearly else. Cumulative-looking data without that growth,
    or mostly-cumulative data with growth are ambiguous and raise ValueError asking for an explicit
    accumulation. A single lead, no finite series or all-constant series cannot show accumulation: daily.
    """
    lead_dim = _find_dim(da, "step", lead_dim, "lead")
    hours = lead_hours(da, lead_dim)
    keep = np.flatnonzero(hours > 0)
    keep = keep[np.argsort(hours[keep])]
    if len(keep) < 2:
        return "daily", "fewer than 2 leads, so nothing can accumulate"
    data = _series(_sample(da, lead_dim).isel({lead_dim: keep}), lead_dim)
    data = data[np.isfinite(data).all(axis=1)]
    if len(data) == 0:
        return "daily", "no series with finite data"
    first, last = float(data[:, 0].mean()), float(data[:, -1].mean())
    varying = np.ptp(data, axis=1) > 0
    if not varying.any():
        return "daily", "every series is constant along lead (no accumulation visible)"
    data = data[varying]
    tol = 1e-3 * np.abs(data).max(axis=1)
    frac = float((np.diff(data, axis=1) >= -tol[:, None]).all(axis=1).mean())
    grows = last > 3.0 * first and last > 0
    evidence = (f"non-decreasing along lead in {100 * frac:.1f} % of series; "
                f"mean {first:.3g} -> {last:.3g}")
    if frac >= 0.99 and grows:
        return "since_init", evidence
    if (frac >= 0.99 and not grows) or (frac >= 0.5 and grows):
        raise ValueError(f"Ambiguous accumulation [{evidence}]; set accumulation='daily' or 'since_init'")
    return "daily", evidence


def _to_daily(da: xr.DataArray, lead_dim: str, kind: str,
              offset: float = 0.0) -> tuple[xr.DataArray, float, list[str]]:
    """Turn raw steps into daily totals whose lead dim holds the days 1..n; return (data, step hours, notes).

    Day j covers (offset + 24 (j-1), offset + 24 j] hours after the init; time before the first
    boundary (offset > 0) is dropped, as is an incomplete final day (noted). since_init takes
    acc(day end) - acc(previous day end) with acc(0) = 0; daily sums the per-interval steps of each
    day. Plain daily data with 24 h steps and offset 0 is returned unchanged for the day checks later.
    """
    h_all = lead_hours(da, lead_dim)
    pos = np.flatnonzero(h_all > 0)
    if len(pos) == 0:
        raise ValueError("The lead axis has no step after the init")
    h = h_all[pos]
    dt, _ = _resolution(h)
    if kind == "daily" and abs(dt - 24.0) < 1e-6 and offset == 0:
        return da, dt, []
    if np.any(np.diff(h) <= 0):
        raise ValueError("The lead steps are not strictly increasing")
    if dt > 24.0 + 1e-6 or abs(24.0 / dt - np.rint(24.0 / dt)) > 1e-6:
        raise ValueError(f"The step length {dt:g} h must divide 24 h (supported e.g. 1, 3, 6, 12, 24)")
    per_day = int(np.rint(24.0 / dt))
    notes: list[str] = []
    raw = da.drop_vars(lead_dim) if lead_dim in da.coords else da
    if kind == "since_init":
        n = int(np.floor((h[-1] - offset) / 24.0 + 1e-9))
        if n < 1:
            raise ValueError(f"No complete day: the last step is {h[-1]:g} h and the first day ends at "
                             f"{offset + 24:g} h")
        need = offset + 24.0 * np.arange(0 if offset > 0 else 1, n + 1)
        idx = []
        for t in need:
            hit = np.flatnonzero(np.isclose(h, t, atol=1e-6))
            if len(hit) == 0:
                raise ValueError(f"since_init needs the accumulation at the {t:g} h step (a 24 h boundary), "
                                 f"which is missing from the lead axis")
            idx.append(pos[hit[0]])
        acc = raw.isel({lead_dim: idx})
        daily = acc.diff(lead_dim) if offset > 0 else acc - acc.shift({lead_dim: 1}, fill_value=0.0)
        if h[-1] - need[-1] > 1e-6:
            notes.append(f"dropped the incomplete final day (steps after {need[-1]:g} h)")
    else:
        expected = dt * np.arange(1, len(h) + 1)
        bad = np.flatnonzero(~np.isclose(h, expected, atol=1e-6))
        if len(bad):
            raise ValueError(f"Missing step {expected[bad[0]]:g} h: per-interval steps must be a complete "
                             f"{dt:g} h grid")
        skip = int(np.rint(offset / dt))
        n = (len(h) - skip) // per_day
        if n < 1:
            raise ValueError("Leads do not make a complete day: a day needs "
                             f"{per_day} intervals after the first boundary")
        used = n * per_day
        if len(h) - skip - used:
            notes.append(f"dropped the incomplete final day ({len(h) - skip - used} of {per_day} intervals)")
        daily = raw.isel({lead_dim: pos[skip:skip + used]}).coarsen({lead_dim: per_day}).sum(skipna=False)
    daily = daily.assign_coords({lead_dim: np.arange(1, n + 1)})
    daily.attrs = dict(da.attrs)
    return daily, dt, notes


def _units_stats(daily: xr.DataArray, lead_dim: str) -> float | None:
    """99.9th percentile of the finite daily totals in a sample, or None if there is no positive value."""
    values = _series(_sample(daily, lead_dim), lead_dim).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or not (values > 0).any():
        return None
    return float(np.percentile(values, 99.9))


def detect_units(da: xr.DataArray, accumulation_kind: str, lead_dim: str | None = None) -> tuple[str, str]:
    """Guess "m" or "mm" from the magnitude of daily totals; return (units, evidence).

    The daily totals follow the given accumulation kind ("daily" or "since_init"). A 99.9th
    percentile below 0.5 means metres (mm totals are typically 1-50 on wet days and < 500 always);
    otherwise mm. With no positive values the answer is "mm" and the evidence says it is a guess.
    """
    lead_dim = _find_dim(da, "step", lead_dim, "lead")
    sample = _sample(da, lead_dim)
    daily, _, _ = _to_daily(sample, lead_dim, accumulation_kind)
    q = _units_stats(daily, lead_dim)
    if q is None:
        return "mm", "no positive values; assuming mm"
    if q < 0.5:
        return "m", f"99.9th percentile of daily totals is {q:.3g} (< 0.5, typical of metres)"
    return "mm", f"99.9th percentile of daily totals is {q:.3g} (>= 0.5, typical of mm)"


def _check_units(daily: xr.DataArray, lead_dim: str, scale: float, declared: object) -> None:
    """Raise if the declared units contradict the magnitudes of the daily totals."""
    q = _units_stats(daily, lead_dim)
    if q is None:
        return
    if scale == 1.0 and q < 0.5:
        raise ValueError(
            f"Declared units {declared!r} look wrong: the 99.9th percentile of daily totals is {q:.3g}, "
            "typical of metres (mm totals are usually 1-50 on wet days); try units='m' "
            "(or pass check_units=False)")
    if scale == 1000.0 and q > 1.0:
        raise ValueError(
            f"Declared units {declared!r} look wrong: the 99.9th percentile of daily totals is {q:.3g} m, "
            "far above any real daily rain in metres (< 0.5); the data are probably millimetres, "
            "try units='mm' (or pass check_units=False)")


def _init_hours(values: np.ndarray) -> np.ndarray:
    """Hour of the day (float) of each init time, for datetime64 or cftime values."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return np.round((values - values.astype("datetime64[D]")) / np.timedelta64(1, "h"), 6)
    return np.array([d.hour + d.minute / 60.0 + d.second / 3600.0 for d in values.flat], dtype=np.float64)


def _boundary_offsets(init_hours: np.ndarray, boundary: str, dt: float, tolerance: float) -> np.ndarray:
    """Hours from each init to its first day boundary (0 <= offset <= 24), snapped to the step grid.

    A boundary off the grid is an error naming the nearest reachable UTC hours unless the miss is
    within `tolerance` hours, in which case it snaps (with a warning saying how far).
    """
    hour = _parse_clock(boundary)
    raw = np.round((hour - init_hours) % 24.0, 6)
    off = np.round(raw / dt) * dt
    miss = np.abs(raw - off)
    for ih, r, o, m in sorted({(float(a), float(b), float(c), float(d)) for a, b, c, d in
                               zip(init_hours, raw, off, miss)}):
        if m < 1e-6:
            continue
        if m <= tolerance + 1e-9:
            warnings.warn(f"daily_boundary {boundary!r} is not on the {dt:g}-hourly steps of the {ih:g} UTC "
                          f"init; snapped by {m:g} h to {(ih + o) % 24:g} UTC", UserWarning, stacklevel=4)
            continue
        lo, hi = np.floor(r / dt) * dt, np.ceil(r / dt) * dt
        raise ValueError(
            f"daily_boundary {boundary!r} cannot be hit with {dt:g}-hourly steps from the {ih:02g} UTC init; "
            f"the nearest hours that work are {(ih + lo) % 24:02g} UTC and {(ih + hi) % 24:02g} UTC "
            f"(or set boundary_tolerance_hours >= {m:g} to snap)")
    return off


def ingest_forecast_info(da: xr.DataArray, *, day_boundary: str, lead_label: str, units: str | None = None,
                         source: str = "", init_dim: str | None = None, lead_dim: str | None = None,
                         member_dim: str | None = None, accumulation: str = "auto",
                         daily_boundary: str | None = None, boundary_tolerance_hours: float = 0.0,
                         check_units: bool = True, rain_resolution: str = "tenths",
                         missing: str = "flag", negatives: str = "default") -> tuple[ForecastRain, ForecastIngestInfo]:
    """Like `ingest_forecast`, but also return a ForecastIngestInfo saying how the days were made."""
    _check_options(rain_resolution, missing, negatives)
    parse_day_boundary(day_boundary)
    if lead_label not in ("start", "end"):
        raise ValueError(f"lead_label must be 'start' or 'end', got {lead_label!r}")
    if accumulation not in ("auto", "daily", "since_init"):
        raise ValueError(f"accumulation must be 'auto', 'daily' or 'since_init', got {accumulation!r}")
    declared = units if units is not None else da.attrs.get("units")
    scale = unit_scale(declared)
    lead = _find_dim(da, "step", lead_dim, "lead")
    init = _find_dim(da, "init_time", init_dim, "init", exclude=(lead,))
    member = _find_dim(da, "member", member_dim, "member", required=False, exclude=(lead, init))
    if len({lead, init, member} - {None}) != len([x for x in (lead, init, member) if x is not None]):
        raise ValueError("init, lead and member dims must be distinct")
    if init not in da.coords:
        raise ValueError("The init dim has no coordinate values")

    if accumulation == "auto":
        kind, evidence = detect_accumulation(da, lead)
    else:
        kind, evidence = accumulation, "given"
    hours = lead_hours(da, lead)
    dt, resolution = _resolution(hours[hours > 0]) if (hours > 0).any() else (24.0, "daily")

    n_init = da.sizes[init]
    shift = np.zeros(n_init, dtype=np.int64)
    if daily_boundary is None:
        offsets = np.zeros(n_init)
    else:
        init_hours = _init_hours(da[init].values)
        offsets = _boundary_offsets(init_hours, daily_boundary, dt, boundary_tolerance_hours)
        shift = np.floor((init_hours + offsets) / 24.0 + 1e-9).astype(np.int64)
    parts, order, notes = [], [], []
    for off in np.unique(offsets):
        idx = np.flatnonzero(offsets == off)
        part, _, part_notes = _to_daily(da.isel({init: idx}), lead, kind, float(off))
        parts.append(part)
        order.append(idx)
        notes += [n for n in part_notes if n not in notes]
    if len(parts) == 1:
        daily = parts[0]
    else:
        n_days = min(p.sizes[lead] for p in parts)
        daily = xr.concat([p.isel({lead: slice(0, n_days)}) for p in parts], dim=init)
        daily = daily.isel({init: np.argsort(np.concatenate(order))})
        daily.attrs = dict(da.attrs)
    if check_units:
        _check_units(daily, lead, scale, declared)

    renames = {init: "init_time", lead: "lead"}
    if member is not None:
        renames[member] = "member"
    daily = _rename_spatial(daily.rename({k: v for k, v in renames.items() if k != v}))
    lead_values = _lead_days(daily)
    spatial = tuple(d for d in daily.dims if d not in ("init_time", "member", "lead"))
    dim_order = ("init_time",) + (("member",) if member is not None else ()) + spatial + ("lead",)
    daily = daily.transpose(*dim_order)

    calendar, ordinals = _read_time(daily["init_time"].values, constant_time_of_day=False)
    if len(np.unique(ordinals)) != len(ordinals):
        raise ValueError("Duplicate init dates on the init axis")
    if np.any(np.diff(ordinals) < 0):
        perm = np.argsort(ordinals)
        daily = daily.isel(init_time=perm)
        ordinals, shift = ordinals[perm], shift[perm]
    ordinals = ordinals + shift
    tenths, flags = _convert(daily, scale, daily.shape, None, clip_negative=True, exact=rain_resolution == "exact",
                             keep_negative=negatives == "keep")
    if missing == "zero":
        flags[...] = False
    _, year, month, day = _labels(ordinals, calendar)
    coords = _coords(daily, dim_order[:-1])
    coords["lead"] = lead_values
    forecast = ForecastRain(tenths=tenths, missing=flags, dims=dim_order, coords=coords, init_year=year,
                            init_month=month, init_day=day, lead_label=lead_label, calendar=calendar,
                            day_boundary=day_boundary, source=source, missing_policy=missing)
    info = ForecastIngestInfo(accumulation=kind, accumulation_evidence=evidence, step_hours=dt,
                              resolution=resolution, daily_boundary=daily_boundary,
                              offsets_hours=tuple(float(o) for o in np.unique(offsets)), notes=tuple(notes))
    if accumulation == "auto" or resolution != "daily" or daily_boundary is not None or notes:
        text = f"accumulation: {kind} [{evidence}]; steps: {resolution}"
        if daily_boundary is not None:
            text += f"; days cut at {daily_boundary} (hours from init to first boundary: {info.offsets_hours})"
        if notes:
            text += "; " + "; ".join(notes)
        warnings.warn(text, IngestInfo, stacklevel=3)
    return forecast, info


def ingest_forecast(da: xr.DataArray, *, day_boundary: str, lead_label: str, units: str | None = None,
                    source: str = "", init_dim: str | None = None, lead_dim: str | None = None,
                    member_dim: str | None = None, accumulation: str = "auto",
                    daily_boundary: str | None = None, boundary_tolerance_hours: float = 0.0,
                    check_units: bool = True, rain_resolution: str = "tenths",
                    missing: str = "flag", negatives: str = "default") -> ForecastRain:
    """Normalize forecast rainfall to ForecastRain (spec section 5).

    Dims become init_time, optional member, the remaining (spatial) dims, then lead. Inits are
    sorted if needed and may not repeat (labelled by date). lead_label says whether lead k is
    labelled by the start ("start": init + k - 1) or end ("end": init + k) of its 24 h. Values
    convert as in `ingest`.

    accumulation: "daily" (each step is a per-interval amount; 24 h steps are daily totals, shorter
    steps are summed per day), "since_init" (running total since the init; daily total =
    difference of the totals at 24 h boundaries) or "auto" (see `detect_accumulation`; emits an
    `IngestInfo` warning with the choice and evidence). Steps are timedeltas or numbers (see
    `lead_hours`). daily_boundary (e.g. "03 UTC") cuts days at that UTC hour instead of 24 h from
    the init; the first partial day is dropped and the init labels move to the date the first
    complete day starts, so lead_label keeps its meaning. A boundary off the step grid is an error
    unless within boundary_tolerance_hours (then snapped, with a warning). check_units=False turns
    off the magnitude check of the units. rain_resolution and missing are as for `ingest`
    ("drop" applies to observed rain only). negatives "keep" leaves negative values as they are (the default sets
    them to 0).

    The returned object carries a `ForecastIngestInfo` as the plain attribute `ingest_info`
    (ForecastRain is not changed); `ingest_forecast_info` returns the same pair explicitly.
    """
    forecast, info = ingest_forecast_info(
        da, day_boundary=day_boundary, lead_label=lead_label, units=units, source=source, init_dim=init_dim,
        lead_dim=lead_dim, member_dim=member_dim, accumulation=accumulation, daily_boundary=daily_boundary,
        boundary_tolerance_hours=boundary_tolerance_hours, check_units=check_units,
        rain_resolution=rain_resolution, missing=missing, negatives=negatives)
    forecast.ingest_info = info
    return forecast
