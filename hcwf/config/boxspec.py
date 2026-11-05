"""The lat/lon ``box:`` of a run config, kept light so the schema and the units spec read one parser.

A box is a mapping of ``lat_min``, ``lat_max``, ``lon_min`` and ``lon_max`` in degrees. The top-level ``box:`` of a
run (the area read) takes all four, or the text ``all`` (the whole domain of the files, said explicitly); the units'
``box`` takes any of them (a missing bound is open) or a region name. `parse_box` is the one validator of both.
"""
from __future__ import annotations

__all__ = ["ALL", "BOX_KEYS", "box_bounds", "parse_box", "parse_units_box"]

ALL = "all"
"""The ``box:`` value for the whole domain of the files."""
BOX_KEYS = ("lat_min", "lat_max", "lon_min", "lon_max")
"""The keys of a box mapping."""


def parse_box(value, *, allow_all: bool = False, open_bounds: bool = False):
    """Check a box and return it normalised.

    Parameters
    ----------
    value : dict or str
        The mapping of `BOX_KEYS` in degrees, or ``"all"`` when ``allow_all``.
    allow_all : bool, default False
        Accept the text ``all`` (returned as `ALL`).
    open_bounds : bool, default False
        Accept a mapping with only some of the keys (a missing bound is open); otherwise all four are needed.

    Returns
    -------
    dict or str
        ``{key: float}`` of the keys given, or `ALL`.

    Raises
    ------
    ValueError
        The value is not a mapping with only box keys and numeric values, a latitude is outside -90..90 or the
        southern bound is above the northern one; the message says what is wrong.
    """
    if allow_all and isinstance(value, str) and value.strip().lower() == ALL:
        return ALL
    expected = f"a mapping of {', '.join(BOX_KEYS)} in degrees" + (f", or '{ALL}'" if allow_all else "")
    if not isinstance(value, dict):
        raise ValueError(f"expected {expected}")
    for key in value:
        if key not in BOX_KEYS:
            raise ValueError(f"unknown key {key!r}; keys: {', '.join(BOX_KEYS)}")
    out = {}
    for key in BOX_KEYS:
        item = value.get(key)
        if item is None:
            if not open_bounds:
                raise ValueError(f"{key} is missing; give all of {', '.join(BOX_KEYS)}" + (f" or write {ALL}" if allow_all else ""))
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item != item:
            raise ValueError(f"{key} must be a number of degrees, got {item!r}")
        out[key] = float(item)
    for key in ("lat_min", "lat_max"):
        if key in out and not -90.0 <= out[key] <= 90.0:
            raise ValueError(f"{key} = {out[key]:g} is outside -90 to 90")
    if "lat_min" in out and "lat_max" in out and out["lat_min"] > out["lat_max"]:
        raise ValueError(f"lat_min {out['lat_min']:g} is above lat_max {out['lat_max']:g}")
    return out


def parse_units_box(value):
    """Check the ``units: box`` value: a box with open bounds, or a region name.

    Parameters
    ----------
    value : dict or str
        The mapping (any of `BOX_KEYS`) or a region name or alias.

    Returns
    -------
    dict or str
        The normalised mapping, or the region name as it is. A mapping that `parse_box` refuses raises its
        `ValueError`.
    """
    return value if isinstance(value, str) else parse_box(value, open_bounds=True)


def box_bounds(box: dict) -> tuple[float, float, float, float]:
    """Return a full box as ``(south, north, west, east)``.

    Parameters
    ----------
    box : dict
        A box with the four keys (see `parse_box`).

    Returns
    -------
    tuple of float
        ``(lat_min, lat_max, lon_min, lon_max)``.
    """
    return tuple(float(box[k]) for k in BOX_KEYS)
