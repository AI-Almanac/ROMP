"""Write the anatomy plot and explain() log of the observed onset for the cells named in explain_cells."""
from __future__ import annotations

from pathlib import Path


from momp.io.input import load_imd_rainfall, load_thresholds
from momp.lib.onset_config import onset_setup
from momp.onset import explain
from momp.onset.ingest import ingest

_GRID_TOLERANCE = 1e-3


def _tag(lat: float, lon: float) -> str:
    """A file-name tag such as "9N_38.75E"."""
    return f"{abs(lat):g}{'N' if lat >= 0 else 'S'}_{abs(lon):g}{'E' if lon >= 0 else 'W'}"


def _on_grid(thresholds, lat, lon):
    """The named thresholds on the rainfall grid (cells they lack become NaN)."""
    out = {}
    for name, da in dict(thresholds or {}).items():
        if "lat" in da.dims and "lon" in da.dims:
            da = da.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)
        out[name] = da
    return out


def write_explain_cells(cfg) -> list[Path]:
    """Write dir_fig/explain/<definition>_<cell>_<year>.png and .txt for every cell and year; returns the PNG paths.

    Does nothing when explain_cells is unset; in legacy (flat key) mode it prints a note and skips.
    Each explanation uses the same observations, thresholds and settings as the run's observed onset.
    """
    from momp.onset.viz.anatomy import anatomy_plot

    cells = getattr(cfg, "explain_cells", None)
    if not cells:
        return []
    kwargs = vars(cfg)
    setup = onset_setup(kwargs)
    if setup.legacy:
        print("explain_cells: skipped, it needs an onset definition (the old flat onset keys are in use)")
        return []
    out_dir = Path(kwargs["dir_fig"]) / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(**kwargs)
    clim = getattr(thresholds, "season_climatology", None)
    written = []
    for year in range(kwargs["start_date"][0], kwargs["end_date"][0] + 1):
        try:
            da = load_imd_rainfall(year, **kwargs)
        except FileNotFoundError as exc:
            print(f"explain_cells: skipping {year} ({exc})")
            continue
        rain = ingest(da, day_boundary=setup.ingest_boundary, units="mm", source="observations")
        lat, lon = rain.coords["lat"], rain.coords["lon"]
        thr = _on_grid(thresholds, lat, lon) or None
        season_clim = None if clim is None else clim.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)
        for cell_lat, cell_lon in cells:
            try:
                a = explain.trace_observed(setup.definition, rain, lat=cell_lat, lon=cell_lon, year=year,
                                           season=setup.season, thresholds=thr, season_climatology=season_clim)
            except ValueError as exc:
                print(f"explain_cells: skipping ({cell_lat:g}, {cell_lon:g}) {year}: {exc}")
                continue
            stem = out_dir / f"{setup.definition.id}_{_tag(cell_lat, cell_lon)}_{year}"
            Path(f"{stem}.txt").write_text(explain.explain(a) + "\n")
            png = Path(f"{stem}.png")
            anatomy_plot(a).savefig(png, dpi=150)
            written.append(png)
            print(f"explain_cells: wrote {png}")
    return written
