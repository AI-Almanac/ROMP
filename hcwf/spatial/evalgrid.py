"""The evaluation grid of a run: one grid every source is scored on, and how a source on another grid is put on it.

Observations and models are usually not on one grid. HCWF scores on a single grid per run (one grid for all models,
so that scores stay comparable and the observed onsets, thresholds and climatology are computed once). By default it
is the coarsest grid among the sources (`choose`): a finer source is averaged up onto it. The config key
``evaluation_grid`` names a source (``observations`` or a model's name) whose grid is used instead. The grid is that
source's grid cut to the region's box; the mask (land, country, ``mask:``, ...) is built on it.

A source whose grid in the box already is the evaluation grid is read as it always was (nothing changes for runs whose
grids match). Any other is averaged conservatively: each evaluation cell takes the mean of the source cells weighted by
the area they share with it, on the sphere (``(longitude overlap) x (overlap of the sines of the latitude edges)``,
see `hcwf.spatial.regrid.grid_weights`), renormalised over the source cells that have a value, so that missing cells
(ocean in a land-only product) do not pull a mean towards zero. Nothing is sampled by nearest cell or interpolated.

`region_read` is the reader: it reads a source's box from file through the chunk-aware readers of
`hcwf.spatial.select` (each stored chunk once, only those that feed the evaluation cells) and averages as it goes
(`hcwf.spatial.onto_units.read_on_units`, with the evaluation cells as grid units of `~hcwf.spatial.units.EvaluationUnits`).
The result is gridded on the evaluation grid exactly like a native read: ``(lat, lon)`` dims, or the compact ``cell``
dimension where masks drop cells. `hcwf.io.grids` finds the sources' grids and decides, and the config's area carries the
decision (`hcwf.config.model.AreaSpec.evaluation_grid`).

Polygon or admin units (the ``units:`` section) do not use this: each source is averaged directly from its own grid
onto the polygons (`~hcwf.spatial.units.EvaluationUnits.transform` builds one transform per source grid).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import xarray as xr

from hcwf.cache import memo
from hcwf.spatial.masks import cell_bounds
from hcwf.spatial.onto_units import read_on_units
from hcwf.spatial.select import cut_box, in_memory, mask_region_loaded, region_select
from hcwf.spatial.units import EvaluationUnits

__all__ = ["EvaluationGrid", "build", "choose", "keep_cells", "region_read", "same_grid", "steps"]

TOLERANCE = 1e-3
"""How far (in degrees) two grids' cell centres may differ for the grids to be the same (as HCWF's onset matching)."""
OBSERVATIONS = "observations"
"""The name of the observations among the sources."""
_AVERAGE = "sphere"
"""The cell-area convention of the conservative averaging: exact areas on the sphere."""
_units = memo("spatial.evalgrid.units", maxsize=4)
"""The evaluation cells as grid units, by grid and kept cells (they hold the transform compiled for each source grid)."""
_FINER = 1.0 + 1e-6
"""One grid is coarser than another when its cell area is larger by more than this factor."""


@dataclass(frozen=True, eq=False)
class EvaluationGrid:
    """The grid a run is scored on, and which sources are averaged onto it.

    The grid is the cells of the ``source``'s grid inside the region's box: ``lat`` ascending, ``lon`` as that
    source stores them.
    """

    source: str
    """The source whose grid it is: ``"observations"`` or a model's name."""
    lat: np.ndarray
    """The latitudes of the cell centres, ascending, in degrees."""
    lon: np.ndarray
    """The longitudes of the cell centres, ascending, in degrees."""
    regrid: tuple
    """The sources averaged onto the grid (those whose own grid in the box is another)."""
    steps: tuple
    """``(source, latitude step, longitude step)`` of every source, in degrees."""
    reason: str
    """Why this source's grid was chosen."""

    def regrids(self, source: str) -> bool:
        """Tell whether a source is averaged onto the grid.

        Parameters
        ----------
        source : str
            ``"observations"`` or a model's name.

        Returns
        -------
        bool
            True when the source's own grid is not the evaluation grid.
        """
        return source in self.regrid

    def key(self) -> str:
        """Return text identifying the grid and the sources averaged onto it, for a cache key.

        Returns
        -------
        str
            The source, a digest of the cell centres and the averaged sources.
        """
        digest = hashlib.sha1(np.ascontiguousarray(self.lat).tobytes() + np.ascontiguousarray(self.lon).tobytes())
        return f"{self.source}:{digest.hexdigest()[:12]}:{','.join(sorted(self.regrid))}"

    def lines(self) -> list[str]:
        """Describe the grid for the check report and the run log.

        Returns
        -------
        list of str
            The grid and its source, why it was chosen, and what is averaged onto it.
        """
        dlat, dlon = next((a, b) for name, a, b in self.steps if name == self.source)
        out = [f"evaluation grid: the grid of {self.source} ({len(self.lat)} lats x {len(self.lon)} lons in the region "
               f"box, {dlat:g} x {dlon:g} degrees); {self.reason}"]
        for name, a, b in self.steps:
            if name in self.regrid:
                how = "finer, averaged" if a * b < dlat * dlon else "averaged"
                out.append(f"{name}: {a:g} x {b:g} degrees, {how} conservatively onto it (cell areas on the sphere, "
                           "renormalised over the cells that have a value)")
        return out


def steps(lat, lon) -> tuple[float, float]:
    """Return the cell size of a grid: the median step of each axis, in degrees.

    A one-cell axis takes the step of the other (0 for both).

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes of the cell centres.
    lon : array_like of float, shape (lon,)
        The longitudes of the cell centres.

    Returns
    -------
    tuple of (float, float)
        The latitude and longitude steps, positive.
    """
    out = []
    for axis in (lat, lon):
        axis = np.sort(np.asarray(axis, dtype=float))
        out.append(float(np.median(np.diff(axis))) if axis.size > 1 else 0.0)
    return (out[0] or out[1], out[1] or out[0])


def same_grid(lat, lon, other_lat, other_lon, tolerance: float = TOLERANCE) -> bool:
    """Tell whether two grids have the same cell centres, to a tolerance.

    Longitudes are compared on a common -180 to 180 convention, in ascending order of both axes.

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes of the first grid.
    lon : array_like of float, shape (lon,)
        The longitudes of the first grid.
    other_lat : array_like of float, shape (other_lat,)
        The latitudes of the second grid.
    other_lon : array_like of float, shape (other_lon,)
        The longitudes of the second grid.
    tolerance : float, default `TOLERANCE`
        The most the centres may differ, in degrees.

    Returns
    -------
    bool
        True when both axes have the same length and every centre agrees.
    """
    def wrap(values):
        """Return the longitudes on -180 to 180, ascending."""
        return np.sort(((np.asarray(values, dtype=float) + 180.0) % 360.0) - 180.0)

    if len(lat) != len(other_lat) or len(lon) != len(other_lon):
        return False
    return bool(np.allclose(np.sort(lat), np.sort(other_lat), rtol=0, atol=tolerance)
                and np.allclose(wrap(lon), wrap(other_lon), rtol=0, atol=tolerance))


def choose(grids: dict, override: str | None = None) -> tuple[str, str]:
    """Choose the source whose grid is the evaluation grid.

    The coarsest grid (the largest cell area, in degrees squared) of all sources; where several have the same cell
    size the first of them in the order of ``grids`` (the observations first, then the models in the order of the
    config) wins. ``override`` names the source instead.

    Parameters
    ----------
    grids : dict of str to tuple
        Maps each source name (``"observations"`` or a model's name) to its ``(lat, lon)`` grid in the region's box.
    override : str, optional
        The source the config asks for; None takes the coarsest.

    Returns
    -------
    source : str
        The chosen source.
    reason : str
        Why, in a clause.

    Raises
    ------
    ValueError
        ``override`` is not one of the sources.
    """
    if override is not None:
        if override not in grids:
            raise ValueError(f"evaluation_grid {override!r} is not a source of the run; use one of "
                             f"{', '.join(repr(n) for n in grids)}")
        return override, "chosen by the evaluation_grid setting"
    area = {name: np.prod(steps(*grid)) for name, grid in grids.items()}
    best = next(iter(grids))
    for name in grids:
        if area[name] > area[best] * _FINER:
            best = name
    others = [n for n in grids if n != best]
    if all(area[n] < area[best] / _FINER for n in others) and others:
        return best, "the coarsest of the sources; finer data are averaged up onto it"
    return best, "the coarsest of the sources (of equal cell size, the first listed)"


def build(grids: dict, override: str | None = None) -> EvaluationGrid | None:
    """Build the evaluation grid of the sources' grids, or None when they all are one grid.

    Parameters
    ----------
    grids : dict of str to tuple
        Maps each source name to its ``(lat, lon)`` grid in the region's box (see `choose`).
    override : str, optional
        The source the config asks for; None takes the coarsest.

    Returns
    -------
    EvaluationGrid or None
        The evaluation grid; None when the sources' grids are the same (nothing is averaged, every source is read as it
        is). A `ValueError` is raised (by `choose`) when ``override`` is not one of the sources.
    """
    source, reason = choose(grids, override)
    lat, lon = (np.asarray(a, dtype=float) for a in grids[source])
    regrid = tuple(n for n, g in grids.items() if n != source and not same_grid(*g, lat, lon))
    if not regrid:
        return None
    sizes = tuple((n, *steps(*g)) for n, g in grids.items())
    return EvaluationGrid(source, lat, lon, regrid, sizes, reason)


def keep_cells(grid: EvaluationGrid, area, select, dataset_name, dir_fig) -> np.ndarray:
    """Find the cells of the evaluation grid the run's mask keeps, the mask built on the grid.

    Parameters
    ----------
    grid : EvaluationGrid
        The evaluation grid.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    select : hcwf.config.model.Selection
        The run's selection settings.
    dataset_name : str
        The source the mask is for (it labels the mask's messages, record and figure).
    dir_fig : str or os.PathLike or None
        The folder of the mask's figure.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon)
        True where the cell is kept.
    """
    field = xr.DataArray(np.zeros((len(grid.lat), len(grid.lon)), dtype="float32"), dims=("lat", "lon"),
                         coords={"lat": grid.lat, "lon": grid.lon})
    masked = mask_region_loaded(field, area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)
    return np.isfinite(np.asarray(masked.values))


def _scatter(series, keep, grid, dims, cells):
    """Put averaged unit series back on the evaluation grid, as a native read of it would give.

    Parameters
    ----------
    series : xarray.DataArray
        The averages, dims (the others, ``unit``), the units being the kept cells in latitude-major order.
    keep : numpy.ndarray of bool, shape (lat, lon)
        The cells kept.
    grid : EvaluationGrid
        The evaluation grid.
    dims : tuple of str
        The dims of the source array, whose order the result follows.
    cells : bool
        Return only the kept cells as one ``cell`` dimension when some are dropped.

    Returns
    -------
    xarray.DataArray
        Dims ``(lat, lon)`` in place of the source's, NaN outside ``keep``; or, with ``cells`` and a mask that drops
        cells, the dims without them and then ``cell`` (the cell's flat position in the grid) with ``lat`` and
        ``lon`` over it.
    """
    rows, cols = np.nonzero(keep)
    others = [d for d in series.dims if d != "unit"]
    coords = {k: v for k, v in series.coords.items() if k != "unit"}
    if cells and not keep.all():
        coords.update({"lat": ("cell", grid.lat[rows]), "lon": ("cell", grid.lon[cols]),
                       "cell": rows * len(grid.lon) + cols})
        return xr.DataArray(series.values, dims=tuple(others) + ("cell",), coords=coords, attrs=series.attrs,
                            name=series.name)
    full = np.full(tuple(series.sizes[d] for d in others) + (len(grid.lat), len(grid.lon)), np.nan,
                   dtype=series.dtype)
    full[..., rows, cols] = series.values
    coords.update({"lat": grid.lat, "lon": grid.lon})
    out = xr.DataArray(full, dims=tuple(others) + ("lat", "lon"), coords=coords, attrs=series.attrs, name=series.name)
    return out.transpose(*[d for d in dims if d in out.dims])


def region_read(ds, var, *, area, select, source, read_by, dataset_name, dir_fig, readers=None, skip_masked=False,
                compact=False):
    """Read a source's variable inside the region into memory, on the evaluation grid.

    A source whose grid is the evaluation grid (or a run with no evaluation grid, ``area.evaluation_grid`` None) is
    read by `~hcwf.spatial.select.region_select`, exactly as before. Any other is read from the evaluation grid's extent
    (padded by one of its own cells) by chunk, only the chunks that hold a cell feeding an evaluation cell, and averaged
    conservatively onto the grid's cells (see the module text), the mask being the one built on the evaluation grid.
    The unit conversion is left to the caller: a mean is linear, so it can follow.

    Parameters
    ----------
    ds : xarray.Dataset
        The lazy dataset, in HCWF's dimension names, with ``lat`` and ``lon``.
    var : str
        The data variable to read.
    area : hcwf.config.model.AreaSpec
        The region, mask and evaluation grid of the run.
    select : hcwf.config.model.Selection
        The run's selection settings.
    source : str
        ``"observations"`` or the model's name, which says whether it is averaged onto the evaluation grid.
    read_by : str
        The dim to read along, block by block.
    dataset_name : str
        What the source is called in the mask's messages and figure.
    dir_fig : str or os.PathLike or None
        The folder of the mask's figure.
    readers : int, optional
        The most processes that decode chunks; None reads in this process.
    skip_masked : bool, default False
        Do not read the cells the implicit masks drop (see `~hcwf.spatial.select.region_select`).
    compact : bool, default False
        With ``skip_masked``, return only the kept cells as one ``cell`` dimension.

    Returns
    -------
    xarray.DataArray
        The variable in memory, dims ``(lat, lon)`` (or ``cell``) as `~hcwf.spatial.select.region_select` gives them.

    Raises
    ------
    ValueError
        The source has no cell near the evaluation grid, or the evaluation grid's box crosses the source's longitude
        seam.
    """
    grid = area.evaluation_grid
    if grid is None or not grid.regrids(source):
        return in_memory(region_select(ds[[var]], area=area, select=select, read_by=read_by, readers=readers,
                                       dataset_name=dataset_name, dir_fig=dir_fig, skip_masked=skip_masked,
                                       compact=compact)[var])
    da = ds[var]
    dlat, dlon = steps(da["lat"].values, da["lon"].values)
    lat_edges, lon_edges = cell_bounds(grid.lat, grid.lon)
    cut = cut_box(da, lat_edges[0] - dlat, lat_edges[-1] + dlat, lon_edges[0] - dlon, lon_edges[-1] + dlon,
                  what=f"the evaluation grid ({source})")
    if not cut.sizes["lat"] or not cut.sizes["lon"]:
        raise ValueError(f"{dataset_name} has no cell near the evaluation grid (the grid of {grid.source}, "
                         f"latitudes {grid.lat[0]:g} to {grid.lat[-1]:g}, longitudes {grid.lon[0]:g} to {grid.lon[-1]:g})")
    keep = keep_cells(grid, area, select, dataset_name, dir_fig)
    key = (grid.key(), hashlib.sha1(np.ascontiguousarray(keep).tobytes()).hexdigest())
    units = _units.get(key)
    if units is None:   # one table of cells, and so one compiled transform per source grid, for the run
        units = _units.setdefault(key, EvaluationUnits.from_grid(grid.lat, grid.lon, keep, cell_area=_AVERAGE))
    floating = np.issubdtype(cut.dtype, np.floating)
    series = read_on_units(cut, units, block_dim=read_by, readers=readers, dtype=cut.dtype if floating else None)
    return _scatter(series, keep, grid, tuple(da.dims), compact and skip_masked)
