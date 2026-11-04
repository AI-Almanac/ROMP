"""Unit ids: administrative names as given, grid cells as "<lat>_<lon>".

A grid id formats each coordinate with a number of decimal digits, trimmed
of trailing zeros or fixed. The convention is the configured one, else the one
ids already in use follow ("established"), else the least precise one that
keeps every cell distinct ("inferred"). These are the legacy spatial id rules,
unchanged.

Ids are used at the edges only: reading and writing unit tables, weights,
dissemination lists and trained artefacts. In memory a unit is its position and
its coordinates (see `hcwf.spatial.units`).

The module holds `GridIdConvention`, the formatting and parsing of grid ids (`format_grid_ids`, `parse_grid_id`), the
choice of a convention (`configured`, `established_convention`, `infer_convention`, `resolve`) and the normalising of
id columns (`normalize`, `check_coordinates`, `with_id_column`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_GRID_ID = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))_([+-]?(?:\d+(?:\.\d*)?|\.\d+))$")
"""The pattern of a grid id, ``<lat>_<lon>``, with the two numbers as groups."""


@dataclass(frozen=True)
class GridIdConvention:
    """How the coordinates of a grid cell are written in its id: the number of decimals and trimmed or fixed."""

    decimal_digits: int
    """The number of decimal digits of each coordinate (0 to 15)."""
    number_format: str = "trimmed"
    """``"trimmed"`` (trailing zeros dropped) or ``"fixed"`` (always `decimal_digits` digits)."""
    source: str = "inferred"
    """Where the convention comes from: ``"spec"``, ``"established"`` or ``"inferred"``."""

    def __post_init__(self):
        """Check the number of digits and the format.

        Raises
        ------
        ValueError
            `decimal_digits` is not between 0 and 15, or `number_format` is neither ``"trimmed"`` nor ``"fixed"``.
        """
        if not 0 <= int(self.decimal_digits) <= 15:
            raise ValueError("grid_id_decimal_digits must be between 0 and 15.")
        if self.number_format not in ("trimmed", "fixed"):
            raise ValueError("grid_id_format must be 'trimmed' or 'fixed'.")


def configured(decimal_digits: int | None, number_format: str = "trimmed") -> GridIdConvention | None:
    """Return the configured convention; None (no digits given) means established or inferred per table.

    Parameters
    ----------
    decimal_digits : int or None
        The configured number of decimal digits.
    number_format : {"trimmed", "fixed"}, default "trimmed"
        The configured format.

    Returns
    -------
    GridIdConvention or None
        The convention with source ``"spec"``, or None when ``decimal_digits`` is None.
    """
    if decimal_digits is None:
        return None
    return GridIdConvention(decimal_digits=decimal_digits, number_format=number_format, source="spec")


def parse_grid_id(value) -> tuple[float, float] | None:
    """Parse the coordinates of a ``<lat>_<lon>`` id; None for any other id.

    Parameters
    ----------
    value : object
        The id (surrounding spaces are stripped); None or NaN is not an id.

    Returns
    -------
    tuple of (float, float) or None
        The latitude and longitude, or None when ``value`` is not a grid id.
    """
    if value is None or pd.isna(value):
        return None
    match = _GRID_ID.fullmatch(str(value).strip())
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def format_coord_component(value, convention: GridIdConvention) -> str:
    """Format one coordinate for a grid id.

    Parameters
    ----------
    value : float
        The coordinate, in degrees.
    convention : GridIdConvention
        The digits and the format.

    Returns
    -------
    str
        The coordinate rounded to the digits, trailing zeros dropped for the trimmed format; negative zero is ``"0"``.

    Raises
    ------
    ValueError
        ``value`` is not finite.
    """
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Grid coordinates must be finite.")
    digits = int(convention.decimal_digits)
    rounded = round(value, digits)
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:.{digits}f}"
    if convention.number_format == "trimmed" and "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "-0.0") else text


def format_grid_ids(lat, lon, convention: GridIdConvention) -> list[str]:
    """Format the ids ``<lat>_<lon>`` of grid cells.

    Parameters
    ----------
    lat : array_like of float
        The latitudes, in degrees (flattened).
    lon : array_like of float
        The longitudes, in degrees (flattened); the same length as ``lat``.
    convention : GridIdConvention
        The digits and the format.

    Returns
    -------
    list of str
        One id per (lat, lon) pair.

    Raises
    ------
    ValueError
        ``lat`` and ``lon`` differ in length, or a coordinate is not finite.
    """
    lat = np.asarray(lat).reshape(-1)
    lon = np.asarray(lon).reshape(-1)
    if len(lat) != len(lon):
        raise ValueError("Latitude and longitude arrays must have equal length.")
    return [f"{format_coord_component(a, convention)}_{format_coord_component(b, convention)}" for a, b in zip(lat, lon)]


def _minimum_positive_spacing(values) -> float | None:
    """Find the smallest positive gap between the distinct finite values.

    Parameters
    ----------
    values : array_like of float
        The coordinates.

    Returns
    -------
    float or None
        The gap, or None when there are fewer than two distinct finite values.
    """
    values = np.unique(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    spacing = np.diff(np.sort(values))
    spacing = spacing[spacing > 0]
    return float(spacing.min()) if len(spacing) else None


def infer_convention(lat, lon, max_digits: int = 12, max_error_fraction: float = 1e-4) -> GridIdConvention:
    """Infer the least precise trimmed convention that keeps the cells distinct.

    No coordinate may move by more than a fraction of the grid spacing. With a single cell (no spacing) only
    floating-point noise is allowed.

    Parameters
    ----------
    lat : array_like of float
        The latitudes of the cells, in degrees.
    lon : array_like of float
        The longitudes of the cells, in degrees.
    max_digits : int, default 12
        The most decimal digits to try.
    max_error_fraction : float, default 1e-4
        The largest rounding error allowed, as a fraction of the smallest grid spacing.

    Returns
    -------
    GridIdConvention
        The convention, with source ``"inferred"``.

    Raises
    ------
    ValueError
        The coordinates are empty, unequal in length or not finite, or no precision up to ``max_digits`` is collision
        free.
    """
    lat = np.asarray(lat, dtype=float).reshape(-1)
    lon = np.asarray(lon, dtype=float).reshape(-1)
    if len(lat) != len(lon) or not len(lat):
        raise ValueError("Cannot infer grid IDs from empty or unequal coordinates.")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Cannot infer grid IDs from non-finite coordinates.")
    pairs = np.unique(np.column_stack([lat, lon]), axis=0)
    spacing = [x for x in (_minimum_positive_spacing(pairs[:, 0]), _minimum_positive_spacing(pairs[:, 1])) if x is not None]
    if spacing:
        tolerance = min(spacing) * float(max_error_fraction)
    else:
        # One cell has no spacing: allow only floating-point noise.
        tolerance = max(1.0, float(np.max(np.abs(pairs)))) * 1e-6
    for digits in range(int(max_digits) + 1):
        convention = GridIdConvention(digits, "trimmed", "inferred")
        if len(set(format_grid_ids(pairs[:, 0], pairs[:, 1], convention))) != len(pairs):
            continue
        error = max(
            float(np.max(np.abs(np.round(pairs[:, 0], digits) - pairs[:, 0]))),
            float(np.max(np.abs(np.round(pairs[:, 1], digits) - pairs[:, 1]))),
        )
        if error <= tolerance:
            return convention
    raise ValueError(
        "Could not infer a collision-free grid-ID precision within "
        f"0..{max_digits} decimal digits. Set geometry.grid_id_decimal_digits."
    )


def _distinct_stripped(ids) -> list[str]:
    """List the distinct ids, stripped, in order of first appearance.

    Each distinct value is converted once: id columns repeat a few thousand ids.

    Parameters
    ----------
    ids : array_like
        The ids; missing values are dropped.

    Returns
    -------
    list of str
        The distinct stripped ids.
    """
    ids = pd.Series(ids)
    if ids.dtype == object and pd.api.types.infer_dtype(ids, skipna=True) not in ("string", "empty"):
        return pd.Series(ids, dtype="string").dropna().str.strip().unique().tolist()
    return pd.Series(pd.unique(ids), dtype="string").dropna().str.strip().unique().tolist()


def established_convention(ids) -> GridIdConvention | None:
    """Find the convention that reproduces existing grid ids exactly.

    Parameters
    ----------
    ids : array_like
        The existing ids.

    Returns
    -------
    GridIdConvention or None
        The convention with source ``"established"``; None when there are no ids or they are not all grid ids.

    Raises
    ------
    ValueError
        The ids have more than 15 decimal digits or do not follow one consistent convention.
    """
    values = _distinct_stripped(ids)
    if not values:
        return None
    parsed = [parse_grid_id(value) for value in values]
    if any(value is None for value in parsed):
        return None
    lat, lon = [value[0] for value in parsed], [value[1] for value in parsed]
    digits = max(len(part.partition(".")[2]) for value in values for part in value.split("_", 1))
    if digits > 15:
        raise ValueError("Established grid IDs exceed 15 decimal digits.")
    for digits in range(16):
        for number_format in ("trimmed", "fixed"):
            convention = GridIdConvention(digits, number_format, "established")
            if format_grid_ids(lat, lon, convention) == values:
                return convention
    raise ValueError("Established grid IDs do not follow one consistent decimal convention.")


def _reproduces(convention: GridIdConvention, ids) -> bool:
    """Tell whether ``convention`` formats the existing grid ids exactly as they are.

    Parameters
    ----------
    convention : GridIdConvention
        The convention to test.
    ids : array_like
        The existing ids.

    Returns
    -------
    bool
        True when there are ids, all are grid ids and they are reproduced.
    """
    values = _distinct_stripped(ids)
    parsed = [parse_grid_id(value) for value in values]
    if not values or any(value is None for value in parsed):
        return False
    return format_grid_ids([v[0] for v in parsed], [v[1] for v in parsed], convention) == values


def resolve(explicit: GridIdConvention | None = None, lat=None, lon=None, authoritative_ids=None,
            context: str = "grid") -> GridIdConvention | None:
    """Resolve the convention: established by authoritative ids, else the explicit one, else inferred from coordinates.

    An explicit convention must reproduce authoritative ids when there are any.

    Parameters
    ----------
    explicit : GridIdConvention, optional
        The configured convention.
    lat : array_like of float, optional
        The latitudes of the cells, to infer from.
    lon : array_like of float, optional
        The longitudes of the cells, to infer from.
    authoritative_ids : array_like, optional
        Ids that already exist and must be reproduced.
    context : str, default "grid"
        What the ids belong to, for error messages.

    Returns
    -------
    GridIdConvention or None
        The convention; None when authoritative ids exist but are not grid ids, or when there is nothing to go on.

    Raises
    ------
    ValueError
        The explicit convention conflicts with the established one (or the established ids are inconsistent).
    """
    established = established_convention(authoritative_ids) if authoritative_ids is not None else None
    if established is not None:
        if explicit is not None and not _reproduces(explicit, authoritative_ids):
            raise ValueError(f"{context}: the explicit grid-ID convention conflicts with the established source IDs.")
        return replace(explicit or established, source="established")
    if explicit is not None:
        return explicit
    if authoritative_ids is not None:
        return None
    if lat is not None and lon is not None:
        return infer_convention(lat, lon)
    return None


UNIT_ID_COLUMNS = ("id", "adm3_name")
"""The names of a unit id column in a table, in the order they are preferred."""


def unit_id_column(columns) -> str | None:
    """Return the column of a table that holds its unit ids.

    Parameters
    ----------
    columns : iterable of str
        The column names.

    Returns
    -------
    str or None
        The first of `UNIT_ID_COLUMNS` in ``columns``, or None when there is none.
    """
    names = set(columns)
    return next((c for c in UNIT_ID_COLUMNS if c in names), None)


def normalize(values, context: str = "spatial IDs") -> pd.Series:
    """Normalise ids as stripped strings (never through numbers), refusing missing or empty ones.

    Each distinct id is normalised once.

    Parameters
    ----------
    values : array_like
        The ids.
    context : str, default "spatial IDs"
        What the ids belong to, for the error message.

    Returns
    -------
    pandas.Series of str
        The stripped ids, with the index and name of ``values``.

    Raises
    ------
    ValueError
        An id is missing or empty.
    """
    values = pd.Series(values, copy=True)
    if values.dtype == object and pd.api.types.infer_dtype(values, skipna=True) not in ("string", "empty"):
        # Mixed objects: factorize would merge equal-hashing values of other
        # types (1 and 1.0) that convert to different strings.
        normalized = values.astype("string").str.strip()
        if (values.isna() | normalized.isna() | normalized.eq("")).any():
            raise ValueError(f"{context} contain missing or empty values.")
        return normalized.astype(str)
    codes, distinct = pd.factorize(values, use_na_sentinel=True)
    if (codes < 0).any():
        raise ValueError(f"{context} contain missing or empty values.")
    normalized = pd.Series(distinct).astype("string").str.strip()
    if (normalized.isna() | normalized.eq("")).any():
        raise ValueError(f"{context} contain missing or empty values.")
    out = normalized.astype(str).take(codes)
    out.index, out.name = values.index, values.name
    return out


def check_coordinates(df: pd.DataFrame, id_col: str = "id", context: str = "spatial IDs") -> None:
    """Refuse an id given to more than one coordinate pair.

    Nothing is checked when the table lacks the id column or ``lat`` / ``lon``.

    Parameters
    ----------
    df : pandas.DataFrame
        The table.
    id_col : str, default "id"
        The id column.
    context : str, default "spatial IDs"
        What the table holds, for the error message.

    Raises
    ------
    ValueError
        An id appears with different coordinates.
    """
    if id_col not in df.columns or not {"lat", "lon"} <= set(df.columns):
        return
    pairs = df[[id_col, "lat", "lon"]].copy()
    pairs["lat"] = pd.to_numeric(pairs["lat"], errors="raise")
    pairs["lon"] = pd.to_numeric(pairs["lon"], errors="raise")
    pairs = pairs.drop_duplicates()
    conflicts = pairs[id_col].duplicated(keep=False)
    if conflicts.any():
        sample = ", ".join(pairs.loc[conflicts, id_col].astype(str).unique()[:10])
        raise ValueError(f"{context}: an ID is assigned to multiple coordinate pairs: {sample}")


_COPY_ON_WRITE = int(pd.__version__.split(".")[0]) >= 3 or getattr(pd.options.mode, "copy_on_write", False) is True
"""Whether pandas is copy-on-write: always from pandas 3, so a shallow copy takes new columns without touching the
original; pandas 2 copies unless copy-on-write is on."""


def with_id_column(df: pd.DataFrame, id_col: str = "id", explicit: GridIdConvention | None = None,
                   convention: GridIdConvention | None = None, context: str = "spatial data") -> pd.DataFrame:
    """Return a copy with normalised ids: the id column, else ``adm3_name``, else grid ids from ``lat`` / ``lon``.

    Grid ids use ``convention`` (resolved from ``explicit`` and the coordinates when not given).

    Parameters
    ----------
    df : pandas.DataFrame
        The table.
    id_col : str, default "id"
        The id column to read or create.
    explicit : GridIdConvention, optional
        The configured convention, used to resolve one.
    convention : GridIdConvention, optional
        The convention to format grid ids with; resolved when None.
    context : str, default "spatial data"
        What the table holds, for error messages.

    Returns
    -------
    pandas.DataFrame
        The copy, with the id column as stripped strings.

    Raises
    ------
    ValueError
        There is no id column, ``adm3_name`` or ``lat`` and ``lon``; an id is missing or empty; or two coordinate pairs
        get the same grid id.
    """
    out = df.copy(deep=not _COPY_ON_WRITE)
    if id_col in out.columns:
        out[id_col] = normalize(out[id_col], context=context)
        return out
    source = unit_id_column(out.columns)
    if source is not None:
        out[id_col] = normalize(out[source], context=context)
        return out
    if "lat" not in out.columns or "lon" not in out.columns:
        raise ValueError(f"Cannot create {id_col!r}: need {id_col!r}, 'adm3_name', or both 'lat' and 'lon'.")
    convention = convention or resolve(explicit, lat=out["lat"].values, lon=out["lon"].values, context=context)
    log.info("%s: grid IDs use %d decimal digits, %s format (%s).", context, convention.decimal_digits,
             convention.number_format, convention.source)
    lat = pd.to_numeric(out["lat"], errors="raise").to_numpy(dtype=float)
    lon = pd.to_numeric(out["lon"], errors="raise").to_numpy(dtype=float)
    out[id_col] = format_grid_ids(lat, lon, convention)
    collisions = pd.DataFrame({"lat": lat, "lon": lon, id_col: out[id_col]}).drop_duplicates().groupby(id_col).size()
    collisions = collisions[collisions > 1]
    if len(collisions):
        raise ValueError(
            f"{context}: {len(collisions)} grid IDs represent multiple coordinates after applying "
            "the selected decimal convention."
        )
    return out

