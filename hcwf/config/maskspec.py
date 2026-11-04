"""The ``mask:`` spec of a run config, kept light so the config layer can read its defaults.

`MaskSpec` is the one place the defaults of the ``mask:`` keys are written: the parser (`MaskSpec.from_mapping`), the
config documentation and the signature of `hcwf.config.Mask` all read them from its fields. It is re-exported by
`hcwf.spatial.masks`, which builds the masks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

__all__ = ["MaskSpec"]


@dataclass(frozen=True)
class MaskSpec:
    """The ``mask:`` section of a run config.

    The parts given intersect. How a shape decides whether a data cell is kept is the cell rule (``cell_rule`` and
    ``overlap_fraction``), which also turns finer mask files into data cells.

    The last four fields (``land_resolution``, ``boundary``, ``regrid``, ``allow_empty``) are not read from a config;
    ``hcwf.spatial.masks.default_mask`` sets them for the implicit masks.
    """

    land: bool = False
    """Keep land cells only."""
    country: tuple = ()
    """Country names (Natural Earth ``NAME``, ``NAME_LONG`` or ``ADM0_A3``, any case)."""
    file: str | None = None
    """A 0/1 NetCDF mask (see ``variable``, ``keep_value``) or a shapefile / GeoJSON."""
    variable: str | None = None
    """The variable of the mask file; None picks the only one on the (lat, lon) axes."""
    keep_value: float = 1
    """The value of the mask variable that marks a kept cell."""
    condition: dict | None = None
    """A mask from a field: a mapping with ``file``, ``variable``, ``above`` and ``below``."""
    cell_rule: str = "centre"
    """How a shape is turned into data cells: ``"centre"`` or ``"overlap_fraction"``."""
    overlap_fraction: float = 0.5
    """The share of a cell's area that must be inside a shape (or kept mask cells) for the cell to be kept."""
    resolution: str = "50m"
    """The Natural Earth resolution of land and country shapes, one of `hcwf.spatial.masks.RESOLUTIONS`."""
    land_resolution: str | None = None
    """The resolution of the land shape if not ``resolution``."""
    boundary: str = "inclusive"
    """For the centre rule, ``"inclusive"`` (a cell centre on the shape's edge is inside) or ``"strict"``."""
    regrid: str = "area"
    """How a finer mask file is brought to the data grid: ``"area"`` (kept share) or ``"nearest"`` mask cell."""
    allow_empty: bool = False
    """Whether a mask that keeps no cell is not an error."""

    @classmethod
    def from_mapping(cls, raw) -> "MaskSpec":
        """Build the spec from the (normalised) mapping of the config.

        Parameters
        ----------
        raw : MaskSpec or collections.abc.Mapping
            The ``mask:`` section; a `MaskSpec` is returned as it is. A single country name is made a tuple.

        Returns
        -------
        MaskSpec
            The spec; the fields that a config does not set keep their defaults.
        """
        if isinstance(raw, MaskSpec):
            return raw
        raw = dict(raw)
        country = raw.get("country") or ()
        if isinstance(country, str):
            country = (country,)
        return cls(land=bool(raw.get("land", False)), country=tuple(country), file=raw.get("file"),
                   variable=raw.get("variable"), keep_value=raw.get("keep_value", cls.keep_value),
                   condition=dict(raw["condition"]) if raw.get("condition") else None,
                   cell_rule=raw.get("cell_rule", cls.cell_rule),
                   overlap_fraction=float(raw.get("overlap_fraction", cls.overlap_fraction)),
                   resolution=raw.get("resolution", cls.resolution))

    def key(self) -> str:
        """Return a text identifying the spec, for caching.

        Returns
        -------
        str
            The fields as JSON with sorted keys.
        """
        return json.dumps(self.__dict__, sort_keys=True, default=str)
