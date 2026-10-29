"""Turn gridded daily rainfall (observed or forecast) into the normalized DailyRain / ForecastRain types.

Ingestion is where units, day boundaries, calendars, gaps and missing values are
settled once, so every later stage works on integer tenths of a millimetre.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import xarray as xr

from momp.onset.model import DailyRain, ForecastRain

_TIME_DIMS = ("time", "TIME", "valid_time", "date")
_LAT_DIMS = ("lat", "latitude", "LATITUDE", "y")
_LON_DIMS = ("lon", "longitude", "LONGITUDE", "x")
_INIT_DIMS = ("init_time", "time", "forecast_reference_time")
_LEAD_DIMS = ("lead", "step", "prediction_timedelta", "prediction_timedelta_daily", "day")
_MEMBER_DIMS = ("member", "number", "ensemble")

_UNITS_MM = {"mm", "mm/day", "mm/d", "mm day-1", "mm d-1", "mm day**-1", "kg m-2", "kg m-2 day-1",
             "kg m-2 d-1", "kg/m2", "kg/m2/day"}
_UNITS_M = {"m", "m/day", "m/d", "m day-1", "m d-1"}

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


def _unit_scale(units: str | None) -> float:
    """Return the factor taking the given rainfall units to mm, or raise for missing/unknown units."""
    if units is None:
        raise ValueError("No rainfall units: pass units=... or set the 'units' attribute on the data")
    key = " ".join(str(units).lower().split())
    if key in _UNITS_MM:
        return 1.0
    if key in _UNITS_M:
        return 1000.0
    raise ValueError(f"Unsupported rainfall units {units!r}; supported: mm, mm/day, kg m-2, m, m/day")


def _find_dim(da: xr.DataArray, candidates: tuple[str, ...], override: str | None, role: str,
              required: bool = True) -> str | None:
    """Pick the dim playing `role`: the override if given, else the first candidate present."""
    if override is not None:
        if override not in da.dims:
            raise ValueError(f"{role} dim {override!r} is not in the data dims {da.dims}")
        return override
    for name in candidates:
        if name in da.dims:
            return name
    if required:
        raise ValueError(f"Cannot find the {role} dim among {da.dims}; tried {candidates}")
    return None


def _rename_spatial(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude style dims to lat and lon."""
    renames = {}
    for target, candidates in (("lat", _LAT_DIMS), ("lon", _LON_DIMS)):
        if target in da.dims:
            continue
        for name in candidates:
            if name in da.dims:
                renames[name] = target
                break
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


def _to_tenths(values: np.ndarray, to_tenths: float,
               clip_negative: bool) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Convert one block of rainfall to (int32 tenths, missing flags, count of negatives handled, lowest value in tenths).

    Observations: values below -0.05 mm become missing (they signal bad data).
    Forecasts (clip_negative): every negative value becomes 0, as model rain can dip below zero numerically.
    """
    x = np.asarray(values, dtype=np.float64) * to_tenths
    finite = np.isfinite(x)
    bad = finite & (x < 0) if clip_negative else finite & (x < -0.5)
    lowest = float(x[bad].min()) if bad.any() else 0.0
    if clip_negative:
        x = np.where(bad, 0.0, x)
        missing = ~finite | (np.abs(np.where(finite, x, 0.0)) > _INT32_MAX)
    else:
        missing = ~finite | bad | (np.abs(np.where(finite, x, 0.0)) > _INT32_MAX)
    tenths = np.rint(np.where(missing, 0.0, x)).astype(np.int32)
    return tenths, missing, int(bad.sum()), lowest


def _convert(da: xr.DataArray, scale: float, out_shape: tuple[int, ...],
             positions: np.ndarray | None, clip_negative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Convert da to (tenths, missing) of shape out_shape, reading blocks along its first axis.

    With `positions`, the last axis of da lands at those indices of the output's
    last axis and every other output day stays missing.
    """
    tenths = np.zeros(out_shape, dtype=np.int32)
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
        t, m, neg, low = _to_tenths(block, scale * 10.0, clip_negative)
        n_negative += neg
        lowest = min(lowest, low)
        if da.ndim == 1:
            index = slice(None) if positions is None else positions
            tenths[index], missing[index] = t, m
        elif positions is None:
            tenths[rows], missing[rows] = t, m
        else:
            tenths[rows, ..., positions] = t
            missing[rows, ..., positions] = m
    if n_negative and clip_negative:
        warnings.warn(f"{n_negative} negative values (lowest {lowest / 10:.2f} mm) were set to 0", UserWarning,
                      stacklevel=3)
    elif n_negative:
        warnings.warn(f"{n_negative} values below -0.05 mm (lowest {lowest / 10:.2f} mm) were set to missing",
                      UserWarning, stacklevel=3)
    return tenths, missing


def ingest(da: xr.DataArray, *, day_boundary: str, units: str | None = None, source: str = "") -> DailyRain:
    """Normalize observed daily rainfall to DailyRain (spec section 5).

    Renames the time dim to "time" and the spatial dims to lat/lon, moves time
    last, converts to int32 tenths of mm (half-even), flags NaN and values below
    -0.05 mm as missing, fills time gaps with missing days (with a warning) and
    labels every day in the data's own calendar. Duplicate times are an error.
    The data are read block by block, so subset da before calling for large inputs.
    """
    parse_day_boundary(day_boundary)
    scale = _unit_scale(units if units is not None else da.attrs.get("units"))
    time_dim = _find_dim(da, _TIME_DIMS, None, "time")
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
        warnings.warn(f"Filled {len(full) - len(ordinals)} missing days in the time axis", UserWarning,
                      stacklevel=2)
        positions = ordinals - ordinals[0]
    tenths, missing = _convert(da, scale, da.shape[:-1] + (len(full),), positions)
    time_values, year, month, day = _labels(full, calendar)
    coords = _coords(da, others)
    coords["time"] = time_values
    return DailyRain(tenths=tenths, missing=missing, dims=others + ("time",), coords=coords, year=year,
                     month=month, day=day, calendar=calendar, day_boundary=day_boundary, source=source)


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


def ingest_forecast(da: xr.DataArray, *, day_boundary: str, lead_label: str, units: str | None = None,
                    source: str = "", init_dim: str | None = None, lead_dim: str | None = None,
                    member_dim: str | None = None) -> ForecastRain:
    """Normalize forecast daily rainfall to ForecastRain (spec section 5).

    Dims become init_time, optional member, the remaining (spatial) dims, then
    lead. Leads may be integer days or timedeltas and must be 1..n. Inits are
    sorted if needed and may not repeat (labelled by date). lead_label says
    whether lead k is labelled by the start ("start": init + k - 1) or end
    ("end": init + k) of its 24 h. Values convert as in `ingest`.
    """
    parse_day_boundary(day_boundary)
    if lead_label not in ("start", "end"):
        raise ValueError(f"lead_label must be 'start' or 'end', got {lead_label!r}")
    scale = _unit_scale(units if units is not None else da.attrs.get("units"))
    lead = _find_dim(da, _LEAD_DIMS, lead_dim, "lead")
    init = _find_dim(da, tuple(n for n in _INIT_DIMS if n != lead), init_dim, "init")
    member = _find_dim(da, tuple(n for n in _MEMBER_DIMS if n not in (lead, init)), member_dim, "member",
                       required=False)
    if len({lead, init, member} - {None}) != len([x for x in (lead, init, member) if x is not None]):
        raise ValueError("init, lead and member dims must be distinct")
    renames = {init: "init_time", lead: "lead"}
    if member is not None:
        renames[member] = "member"
    da = _rename_spatial(da.rename({k: v for k, v in renames.items() if k != v}))
    if "init_time" not in da.coords:
        raise ValueError("The init dim has no coordinate values")
    lead_values = _lead_days(da)
    spatial = tuple(d for d in da.dims if d not in ("init_time", "member", "lead"))
    order = ("init_time",) + (("member",) if member is not None else ()) + spatial + ("lead",)
    da = da.transpose(*order)

    calendar, ordinals = _read_time(da["init_time"].values, constant_time_of_day=False)
    if len(np.unique(ordinals)) != len(ordinals):
        raise ValueError("Duplicate init dates on the init axis")
    if np.any(np.diff(ordinals) < 0):
        da = da.isel(init_time=np.argsort(ordinals))
        ordinals = np.sort(ordinals)
    tenths, missing = _convert(da, scale, da.shape, None, clip_negative=True)
    _, year, month, day = _labels(ordinals, calendar)
    coords = _coords(da, order[:-1])
    coords["lead"] = lead_values
    return ForecastRain(tenths=tenths, missing=missing, dims=order, coords=coords, init_year=year,
                        init_month=month, init_day=day, lead_label=lead_label, calendar=calendar,
                        day_boundary=day_boundary, source=source)
