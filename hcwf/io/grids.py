"""Find the grids of a run's sources and decide the evaluation grid they are scored on.

`attach` looks at the coordinates of the observation file and of each model's file (nothing of the rainfall is read),
and gives the run's `~hcwf.config.model.AreaSpec` the `~hcwf.spatial.evalgrid.EvaluationGrid`: the coarsest grid among
the sources, or the one the ``evaluation_grid`` key names. A run whose sources share one grid gets none and is read as
it always was. The readers (`hcwf.io.observations`, `hcwf.io.forecasts`) then average any source on another grid onto
it through `hcwf.spatial.evalgrid.region_read`.

Runs on evaluation units (``units:``), without a region, or that do not read the grids have no evaluation grid: the
units are built from each source's own grid (see `hcwf.spatial.units`).
"""
from __future__ import annotations

import logging
from dataclasses import replace

from hcwf.config.model import NO_SELECTION
from hcwf.io.forecasts import forecast_grid
from hcwf.io.observations import observation_grid_of
from hcwf.spatial import evalgrid

__all__ = ["attach", "source_grids"]

log = logging.getLogger(__name__)


def source_grids(rc, select=NO_SELECTION) -> dict:
    """Read the grid of every source of the run in the region's box, from the files' coordinates alone.

    The year looked at is the first evaluation year that has files for every source.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration (without an evaluation grid, so that each source's own grid is read).
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.

    Returns
    -------
    dict of str to tuple
        Maps ``"observations"`` and each model's name to its ``(lat, lon)`` in the box; the reference model is
        included when it is a forecast model of its own.

    Raises
    ------
    ValueError
        The files of no evaluation year could be read (a missing file, an unknown region); the message has the last
        error.
    """
    models = list(rc.models)
    if rc.reference.kind == "model" and rc.reference.model not in {m.name for m in models}:
        models.append(rc.reference.as_model())
    ensemble = bool(rc.evaluation.probabilistic)
    error = None
    for year in rc.evaluation.years:
        try:
            grids = {evalgrid.OBSERVATIONS: _pair(observation_grid_of(year, rc, select))}
            for model in models:
                grids[model.name] = _pair(forecast_grid(year, model, rc.area, select, ensemble=ensemble))
            return grids
        except Exception as exc:  # noqa: BLE001 - try the next year; the last error is raised
            error = exc
    raise ValueError(f"the grids of the sources could not be read: {error}") from error


def _pair(grid):
    """Return the ``(lat, lon)`` of a grid given as a dict of coordinate values.

    Parameters
    ----------
    grid : dict of str to numpy.ndarray
        The ``lat`` and ``lon`` coordinate values.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The latitudes and the longitudes.
    """
    return grid["lat"], grid["lon"]


def attach(rc, select=NO_SELECTION):
    """Return the run configuration with its evaluation grid, when its sources are not on one grid.

    Nothing changes (``rc`` itself is returned) for a run without a region, on evaluation units, or whose files cannot
    be looked at (the reads then raise the real error), and for sources that are all on one grid.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run configuration.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.

    Returns
    -------
    hcwf.config.model.RunConfig
        ``rc``, with ``area.evaluation_grid`` set when a source is on another grid than the coarsest (or the named) one.
        A `ValueError` is raised (by `~hcwf.spatial.evalgrid.choose`) when ``evaluation_grid`` names a source the run
        does not have.
    """
    if rc.area.region is None or rc.area.evaluation_grid is not None or _on_units(rc):
        return rc
    try:
        grids = source_grids(rc, select)
    except Exception as exc:  # noqa: BLE001 - the reads raise and name the real problem
        log.debug(f"evaluation grid not looked at: {exc}")
        return rc
    grid = evalgrid.build(grids, rc.area.grid_source)
    if grid is None:
        return rc
    return replace(rc, area=replace(rc.area, evaluation_grid=grid))


def _on_units(rc) -> bool:
    """Tell whether any source of the run is read onto evaluation units.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration.

    Returns
    -------
    bool
        True when the observations, a model or the reference has a ``units:`` specification.
    """
    return any(spec is not None for spec in (rc.observations.units, rc.reference.units, *(m.units for m in rc.models)))
