"""The area of a run, resolved once: the lat/lon box every reader cuts to.

A run's area comes from its ``region`` (the region's domain box), its ``box:`` (a box written in the config, which
replaces the region's box, or is the area of a run without a region) or, without both, its ``mask:`` (the bounds of the
mask's country, shape file or NetCDF mask). ``box: all`` says the files' whole domain, so nothing is cut. `resolve`
is the one place that decides; readers, the evaluation grid, the masks, the maps and the check report all call it
(through `hcwf.config.model.AreaSpec.bounds`) instead of branching on region and box themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

from hcwf.config.boxspec import ALL, box_bounds
from hcwf.config.maskspec import MaskSpec

__all__ = ["Bounds", "describe", "has_area", "needs_area", "resolve"]


@dataclass(frozen=True)
class Bounds:
    """A lat/lon box in degrees and how it is applied to data."""

    south: float
    """The southern bound."""
    north: float
    """The northern bound."""
    west: float
    """The western bound."""
    east: float
    """The eastern bound."""
    wrap: bool
    """Whether the longitudes are put in the data's convention (a written box, a mask's bounds); a region's box is
    left as written."""
    pad: bool
    """Whether the cut is padded by one data cell (a mask's bounds: the cell rules keep cells whose centre is just
    outside the shape)."""
    origin: str
    """What the box is: ``"region box"``, ``"box"`` or ``"mask bounds"``."""

    @property
    def tuple(self) -> tuple[float, float, float, float]:
        """The box as ``(south, north, west, east)``."""
        return self.south, self.north, self.west, self.east

    @property
    def lon_all(self) -> bool:
        """Whether the box spans all longitudes (no longitude cut)."""
        return self.east - self.west >= 360.0 - 1e-6

    def text(self) -> str:
        """Describe the box for a report.

        Returns
        -------
        str
            For example ``"box: latitudes 3 to 15, longitudes 33 to 48"``.
        """
        lon = "all longitudes" if self.lon_all else f"longitudes {self.west:g} to {self.east:g}"
        return f"{self.origin}: latitudes {self.south:g} to {self.north:g}, {lon}"


def needs_area(region, box, mask) -> bool:
    """Tell whether a run without a region has nothing that gives it an area.

    Parameters
    ----------
    region : str or None
        The region name.
    box : dict or str or None
        The config's ``box`` (a mapping or ``"all"``).
    mask : collections.abc.Mapping or MaskSpec or None
        The config's ``mask``.

    Returns
    -------
    bool
        True when there is no region, no box, and no mask with a country or a file.
    """
    return region is None and box is None and not _limits(mask)


def _limits(mask) -> bool:
    """Tell whether a mask names a country, a file or a condition (something with an extent).

    Parameters
    ----------
    mask : collections.abc.Mapping or MaskSpec or None
        The mask.

    Returns
    -------
    bool
        True when it limits the area.
    """
    if not mask:
        return False
    spec = MaskSpec.from_mapping(mask)
    return bool(spec.country or spec.file or spec.condition)


def resolve(region, box, mask) -> Bounds | None:
    """Resolve the area to cut to.

    Parameters
    ----------
    region : str or None
        The region name or alias.
    box : dict or str or None
        The config's ``box``: a mapping of the four bounds, ``"all"`` or None.
    mask : collections.abc.Mapping or MaskSpec or None
        The config's ``mask``.

    Returns
    -------
    Bounds or None
        The box of ``box`` when given (``"all"`` gives None), else the region's, else the mask's bounds; None when the
        whole domain of the files is the area (``box: all``, a land-only mask, or nothing at all). An unknown region
        raises `~hcwf.errors.RegionError`, an unreadable mask `~hcwf.errors.MaskError`.
    """
    if box == ALL:
        return None
    if box is not None:
        return Bounds(*box_bounds(box) if isinstance(box, dict) else box, wrap=True, pad=False, origin="box")
    if region is not None:
        from hcwf.onset.regions import box as region_domain

        return Bounds(*region_domain(region), wrap=False, pad=False, origin="region box")
    if _limits(mask):
        from hcwf.spatial.masks import mask_bounds

        found = mask_bounds(MaskSpec.from_mapping(mask))
        if found is not None:
            return Bounds(*found, wrap=True, pad=True, origin="mask bounds")
    return None


def has_area(region, box, mask) -> bool:
    """Tell whether the area is a box (not the whole domain of the files).

    Parameters
    ----------
    region : str or None
        The region name or alias.
    box : dict or str or None
        The config's ``box``.
    mask : collections.abc.Mapping or MaskSpec or None
        The config's ``mask``.

    Returns
    -------
    bool
        True when `resolve` gives a box.
    """
    return resolve(region, box, mask) is not None


def describe(region, box, mask) -> str:
    """Describe the area for the check report and the run log.

    Parameters
    ----------
    region : str or None
        The region name or alias.
    box : dict or str or None
        The config's ``box``.
    mask : collections.abc.Mapping or MaskSpec or None
        The config's ``mask``.

    Returns
    -------
    str
        The area and its bounds, or the whole domain of the files.
    """
    found = resolve(region, box, mask)
    if found is None:
        return "the whole domain of the files" + (" (box: all)" if box == ALL else "")
    return found.text()
