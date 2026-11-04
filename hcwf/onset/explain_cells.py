"""Write the anatomy plot and ``explain`` log of the observed onset for the cells named in ``explain_cells``.

`write_explain_cells` is the run-level entry point: for each evaluation year it ingests the observations, traces the
observed onset at every configured cell with `hcwf.onset.explain.trace_observed` and writes a PNG
(`hcwf.onset.viz.anatomy.anatomy_plot`) and a text log (`hcwf.onset.explain.explain`) per cell and year.
"""
from __future__ import annotations

from pathlib import Path

from hcwf.io.observations import observations_of
from hcwf.io.thresholds import thresholds_of
from hcwf.onset import explain
from hcwf.onset.ingest import ingest
import logging

__all__ = ["write_explain_cells"]


log = logging.getLogger(__name__)

_GRID_TOLERANCE = 1e-3


def _tag(lat: float, lon: float) -> str:
    """Return a file-name tag for a cell, such as ``"9N_38.75E"``.

    Parameters
    ----------
    lat : float
        Latitude in degrees north (negative is south).
    lon : float
        Longitude in degrees east (negative is west).

    Returns
    -------
    str
        The absolute coordinates in ``%g`` format, each followed by its hemisphere letter, joined by ``_``.
    """
    return f"{abs(lat):g}{'N' if lat >= 0 else 'S'}_{abs(lon):g}{'E' if lon >= 0 else 'W'}"


def _on_grid(thresholds, lat, lon):
    """Return the named thresholds reindexed onto the rainfall grid.

    Thresholds that have both ``lat`` and ``lon`` dims are reindexed to the nearest grid point within
    ``_GRID_TOLERANCE`` degrees, so cells they lack become ``NaN``; other thresholds pass through unchanged.

    Parameters
    ----------
    thresholds : mapping of str to xarray.DataArray, optional
        The named thresholds; ``None`` or empty gives an empty result.
    lat : xarray.DataArray
        The rainfall latitudes to reindex to.
    lon : xarray.DataArray
        The rainfall longitudes to reindex to.

    Returns
    -------
    dict of str to xarray.DataArray
        The thresholds on the rainfall grid, under the same names.
    """
    out = {}
    for name, da in dict(thresholds or {}).items():
        if "lat" in da.dims and "lon" in da.dims:
            da = da.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)
        out[name] = da
    return out


def write_explain_cells(rc) -> list[Path]:
    """Write the explanation PNG and text log of the observed onset for every configured cell and year.

    The files are ``<figure_dir>/explain/<definition>_<cell>_<year>.png`` and ``.txt``. Does nothing when
    ``explain_cells`` is unset. Each explanation uses the same observations, thresholds and settings as the run's
    observed onset. A year without observations, or a cell the trace rejects with `ValueError`, is skipped and logged.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration: ``onset_options.explain_cells`` (list of ``(lat, lon)``), ``onset``, ``output``,
        ``models``, ``evaluation.years``.

    Returns
    -------
    list of pathlib.Path
        The PNG paths written, in year then cell order (empty when ``explain_cells`` is unset).
    """
    from hcwf.onset.viz.anatomy import anatomy_plot

    cells = rc.onset_options.explain_cells
    if not cells:
        return []
    setup = rc.onset
    out_dir = Path(rc.output.figure_dir) / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds_of(rc, rc.models[0])
    clim = getattr(thresholds, "season_climatology", None)
    written = []
    for year in rc.evaluation.years:
        try:
            da = observations_of(year, rc)
        except FileNotFoundError as exc:
            log.info(f"explain_cells: skipping {year} ({exc})")
            continue
        rain = ingest(da, day_boundary=setup.obs_day_boundary, units="mm", source="observations")
        lat, lon = rain.coords["lat"], rain.coords["lon"]
        thr = _on_grid(thresholds, lat, lon) or None
        season_clim = None if clim is None else clim.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)
        for cell_lat, cell_lon in cells:
            try:
                a = explain.trace_observed(setup.definition, rain, lat=cell_lat, lon=cell_lon, year=year,
                                           season=setup.season, thresholds=thr, season_climatology=season_clim)
            except ValueError as exc:
                log.info(f"explain_cells: skipping ({cell_lat:g}, {cell_lon:g}) {year}: {exc}")
                continue
            stem = out_dir / f"{setup.definition.id}_{_tag(cell_lat, cell_lon)}_{year}"
            Path(f"{stem}.txt").write_text(explain.explain(a) + "\n")
            png = Path(f"{stem}.png")
            anatomy_plot(a).savefig(png, dpi=150)
            written.append(png)
            log.info(f"explain_cells: wrote {png}")
    return written
