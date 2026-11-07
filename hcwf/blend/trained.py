"""A trained blend: everything operations needs from training, in one file.

`TrainedBlend` holds the run it was trained with (HCWF's resolved run YAML, with the onset definition frozen beside
it), the fitted model (and random forest), the reference data training resolved (per-unit thresholds, reference onsets,
the weights that put each model's grid on the units), the observed onsets the climatology is fitted on, and each
model's forecast signature. `TrainedBlend.forecast` takes only live forecast files and an issue date; nothing else
is read, so operations cannot run with other settings than training did (`hcwf.blend.forecast`).

File formats. A trained blend is a zip of JSON, YAML and NumPy arrays: no pickles, so it loads across library versions
and is safe to read. It is written to a temporary file and renamed into place, and every member is checked against its
digest on load. Two formats load into this one class:

* ``"hcwf/trained-blend"`` (what `hcwf.blend.training.TrainingResult.save` writes): ``manifest.json``, ``run.yaml``
  (the resolved run config, with its sources), ``definition.yaml`` (the onset definition as used, with its hash in
  the manifest), ``recipe.json`` (the recipe's digest and the climatology years training resolved),
  ``model.json``/``.npz``, ``forest.json``/``.npz``, ``units.json``, ``weights.json``/``.npz``,
  ``reference.json``/``.npz``, ``observed.npz`` and ``signatures.json``. Loading rebuilds the run config, the onset
  setup and the recipe from ``run.yaml`` and ``definition.yaml`` and refuses a file whose definition or recipe no
  longer rebuild to what training used.
* ``"onset-blending/trained-blend"`` (onset_blending, features version 1): its BlendConfig ``config.yml`` goes through
  `hcwf.config.convert_specs` in memory (as ``hcwf-run convert`` does) to the run config and recipe; the model,
  forest, reference, units and observed arrays are used as stored.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import logging
import subprocess
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from hcwf.errors import BlendFileError
from hcwf.blend.recipe import blend_season_start
from hcwf.blend.samples import ObservedOnsets
from hcwf.cache import atomic_write, file_identity
from hcwf.config import detect as dataset_detect
from hcwf.config.datasets import is_zarr, resolve_paths
from hcwf.io import stores
from hcwf.spatial import evalgrid

__all__ = ["TrainedBlend"]


log = logging.getLogger(__name__)


FEATURES_VERSION = 1
"""The version of the numbers the features are computed from.

Bump whenever they change: a blend trained under another version is refused rather than fed different features.
"""
FORMAT = "hcwf/trained-blend"
"""The format id of the files `TrainedBlend.save` writes."""
FORMAT_VERSION = 1
"""The version of the `FORMAT` layout."""
OLD_FORMAT = "onset-blending/trained-blend"
"""The format id of onset_blending's trained blends, which still load."""
LEGACY_HCWF_FORMAT = "momp/trained-blend"
"""The id this format had before the package was renamed from momp to hcwf: files saved then still load (same layout)."""
_MANIFEST = "manifest.json"
"""The name of the manifest member of the zip."""
TRAINED_FILENAME = "trained.blend"
"""The file name a trained blend is saved under by default."""


# ---------------------------------------------------------------------------------------------------------------
# Arrays and tables without pickles
# ---------------------------------------------------------------------------------------------------------------


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialise arrays as the bytes of an ``.npz`` file.

    Parameters
    ----------
    arrays : dict of str to numpy.ndarray
        The arrays by name.

    Returns
    -------
    bytes
        The uncompressed ``.npz`` content.
    """
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _npz_load(data: bytes):
    """Read the arrays of ``.npz`` bytes without allowing pickles.

    Parameters
    ----------
    data : bytes
        The content written by `_npz_bytes`.

    Returns
    -------
    numpy.lib.npyio.NpzFile
        The lazily loaded arrays by name.
    """
    return np.load(io.BytesIO(data), allow_pickle=False)


def _put(arrays: dict, prefix: str, table: pd.DataFrame) -> dict:
    """Put a table's columns into arrays (strings as unicode, dates as ISO text) and return its schema.

    A number or bool column is stored as it is under ``<prefix>.<index>``; any other column is stored as text under
    that key with a ``<prefix>.<index>.missing`` mask (``datetime.date`` values as ISO strings).

    Parameters
    ----------
    arrays : dict
        The arrays to add to, by name.
    prefix : str
        The prefix of the array names.
    table : pandas.DataFrame
        The table to store.

    Returns
    -------
    dict
        The schema `_get` needs: ``prefix``, ``rows`` and ``columns`` (a list of ``name``, ``kind`` and ``dtype``).
    """
    columns = []
    for index, name in enumerate(table.columns):
        series, key = table[name], f"{prefix}.{index}"
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            kind, arrays[key] = "number", series.to_numpy()
        else:
            values = series.astype(object)
            is_date = values.map(lambda v: isinstance(v, dt.date) and not pd.isna(v))
            kind = "date" if is_date.any() else "text"
            arrays[key] = np.array(
                [v.isoformat() if kind == "date" and d else ("" if pd.isna(v) else str(v))
                 for v, d in zip(values, is_date)], dtype=str)
            arrays[f"{key}.missing"] = values.isna().to_numpy()
        columns.append({"name": str(name), "kind": kind, "dtype": str(series.dtype)})
    return {"prefix": prefix, "rows": len(table), "columns": columns}


def _get(arrays, schema: dict) -> pd.DataFrame:
    """Rebuild the table that `_put` stored.

    Parameters
    ----------
    arrays : mapping of str to numpy.ndarray
        The arrays, as `_npz_load` returns them.
    schema : dict
        The schema `_put` returned.

    Returns
    -------
    pandas.DataFrame
        The table, on a ``0..rows-1`` index; missing text and dates are None.
    """
    data = {}
    for index, column in enumerate(schema["columns"]):
        key = f"{schema['prefix']}.{index}"
        values = arrays[key]
        if column["kind"] == "number":
            data[column["name"]] = pd.Series(values, dtype=column["dtype"])
            continue
        missing = arrays[f"{key}.missing"]
        if column["kind"] == "date":
            items = [None if m else dt.date.fromisoformat(v) for v, m in zip(values.tolist(), missing.tolist())]
            data[column["name"]] = pd.Series(items, dtype=object)
        else:
            items = [None if m else v for v, m in zip(values.tolist(), missing.tolist())]
            data[column["name"]] = pd.Series(items, dtype=column["dtype"])
    return pd.DataFrame(data, index=pd.RangeIndex(schema["rows"]))


# ---------------------------------------------------------------------------------------------------------------
# The run frozen in the file
# ---------------------------------------------------------------------------------------------------------------


def model_entries(mapping: dict) -> dict[str, dict]:
    """Return the dataset entries of the run's models by name, without their files.

    Parameters
    ----------
    mapping : dict
        The run config as a mapping, with a ``models`` list.

    Returns
    -------
    dict of str to dict
        For each model name its entry with the keys as the run YAML gives them, minus ``paths``, ``path`` and
        ``years``.
    """
    return {m["name"]: {k: v for k, v in m.items() if k not in ("paths", "path", "years")}
            for m in mapping.get("models") or ()}


def frozen_config(mapping: dict, definition_text: str | None = None):
    """Return the run config of a run mapping, without reading any data.

    The config's sections are flattened and typed as a run config's are, the model lists are made from the model
    entries (no files: a forecast names its own), and the onset definition is the frozen one (``definition_text``) when
    given, else the run's own inline rule. Its ``onset`` is the OnsetSetup and its ``blend`` the recipe (the
    invariants of a run that read data are not checked).

    Parameters
    ----------
    mapping : dict
        The run config as a mapping (the content of ``run.yaml``).
    definition_text : str, optional
        The text of the frozen onset definition; None keeps the run's own inline rule.

    Returns
    -------
    hcwf.config.model.RunConfig
        The config, without observation or model files.

    Raises
    ------
    BlendFileError
        The mapping does not load as a run config.
    """
    from hcwf.config.model import RunConfig, complete_flat
    from hcwf.config import schema

    errors: list[str] = []
    flat, origin, special = schema.flatten_sections(mapping, errors)
    cfg = schema.normalise(flat, origin, errors)
    if errors:
        raise BlendFileError("the run config in the file does not load:\n" + "\n".join(f"  - {e}" for e in errors))
    models = special["models"] or []
    cfg["model_list"] = tuple(m["name"] for m in models)
    cfg["model_dir_list"] = tuple("" for _ in models)
    cfg["model_var_list"] = tuple(m.get("variable") for m in models)
    cfg["unit_cvt_list"] = tuple(m.get("unit_cvt") for m in models)
    cfg["file_pattern_list"] = tuple("{}.nc" for _ in models)
    cfg["model_dims_list"] = tuple(m.get("dims") for m in models)
    cfg["accumulation_list"] = tuple(m.get("accumulation", "auto") for m in models)
    cfg["daily_boundary_list"] = tuple(m.get("daily_boundary") for m in models)
    cfg["accumulation_detected"] = tuple(False for _ in models)
    cfg = schema.apply_defaults(cfg, [])      # what only a run's data would complete (observation files) is not needed
    if definition_text is not None:
        cfg["onset_rule"] = yaml.safe_load(definition_text)
        cfg["onset_definition"] = None
        cfg.pop("onset_override", None)
    return RunConfig.from_flat(complete_flat(cfg), check=False)


def _recipe_digest(recipe) -> str:
    """Return a digest of a recipe.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The recipe.

    Returns
    -------
    str
        The SHA-256 hex digest of its ``repr``.
    """
    return hashlib.sha256(repr(recipe).encode()).hexdigest()


def _with_years(recipe, train_years, test_years):
    """Return the recipe with the climatology years training resolved (a forecast must fit the same training window).

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The recipe.
    train_years : iterable of int
        The climatology training years.
    test_years : iterable of int
        The climatology test years.

    Returns
    -------
    hcwf.blend.recipe.Recipe
        A copy with those years in its climatology section.
    """
    return dataclasses.replace(recipe, climatology=dataclasses.replace(
        recipe.climatology, train_years=tuple(train_years), test_years=tuple(test_years)))


# ---------------------------------------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenUnits:
    """One model's units as training resolved them.

    It records how the model's rain reaches the units (`how`, as ``UnitsSpec.how``) and, for units made from a
    shapefile or weights, the weights table that put the training grid on them. It stands in for a ``UnitsSpec`` when
    a live file is read (`FrozenUnits.select`): the weights are matched to the live grid's cells by coordinates, so a
    live file holding more cells than training did is read for the trained ones only. Units that are the file's own
    cells (or its own units) need no weights.
    """

    how: str
    """How the rain reaches the units, as ``UnitsSpec.how`` (for example ``"cells"``)."""
    weights: pd.DataFrame | None = field(default=None, compare=False)
    """The weights table that put the training grid on the units: ``target_id`` with ``lat`` and ``lon`` or
    ``source_id``, and ``weight``; None when the units need no weights. Excluded from comparison."""
    convention: object = None
    """The unit convention (the ``units.convention`` of the run), passed on to the unit builders."""
    grid: object = field(default=None, compare=False, repr=False)
    """The `~hcwf.spatial.evalgrid.EvaluationGrid` training put the model's cells on (``"cells"`` units); a live file
    is averaged onto it. None for a blend saved before the grid was stored (the live file's own cells are used) and for
    units that are not cells."""

    def __post_init__(self):
        """Put a given weights table in its one canonical form (legacy column names renamed)."""
        if self.weights is not None:     # the one canonical form of a weights table (legacy column names renamed)
            from hcwf.spatial.regrid import read_weights

            object.__setattr__(self, "weights", read_weights(self.weights, "the trained blend's weights"))

    def select(self, lat=None, lon=None, ids=None, keep_ids=None):
        """Return the units of a live file and the positions of those used, as ``UnitsSpec.select``.

        Parameters
        ----------
        lat : numpy.ndarray of float64, shape (lat,), optional
            The live file's latitudes, for units that are its own cells.
        lon : numpy.ndarray of float64, shape (lon,), optional
            The live file's longitudes, for units that are its own cells.
        ids : array_like of str, optional
            The unit ids of a file with its own units; these are the units when given.
        keep_ids : iterable of str, optional
            The unit ids to keep; all when None.

        Returns
        -------
        units : hcwf.spatial.units.EvaluationUnits
            The units: the given ``ids``, the targets of the weights, or the grid cells.
        positions : numpy.ndarray of int, shape (kept,)
            The positions in ``units`` of the kept units, ordered by id.

        Raises
        ------
        ValueError
            No unit of the file is kept.
        """
        from hcwf.spatial import evalgrid
        from hcwf.spatial.units import EvaluationUnits

        if ids is not None:
            units = EvaluationUnits(pd.DataFrame({"id": np.asarray(ids).astype(str)}), "polygon")
        elif self.weights is not None:
            targets = sorted(self.weights["target_id"].astype(str).unique())
            units = EvaluationUnits(pd.DataFrame({"id": targets}), "polygon", weights=self.weights)
        elif self.grid is not None and not evalgrid.on_grid(self.grid, lat, lon):   # averaged onto the trained grid
            units = evalgrid.grid_units(self.grid, self.convention)
        else:
            units = EvaluationUnits.from_grid(lat, lon, explicit=self.convention)
        keep = np.ones(len(units), dtype=bool) if keep_ids is None else np.isin(
            units.ids, np.asarray(sorted(keep_ids), dtype=str))
        positions = np.flatnonzero(keep)
        if not len(positions):
            raise ValueError("No unit of the file is kept")
        return units, positions[np.argsort(units.ids[positions], kind="stable")]


@dataclass(frozen=True)
class TrainedGrid:
    """The grid (or the units) the blend's features are indexed by, stored with it.

    For a blend on grid cells it is the evaluation grid training used: the cell centres, which source's grid it is, how
    finer sources were averaged onto it (cell areas on the sphere), the area it was cut to and the cells the blend uses.
    A live forecast is put on this grid, not on the grid of whatever files arrive. For a blend on polygon or admin
    units there is no grid: the units' ids and a digest of the weights (and the shapefile) that put each model's grid on
    them are kept instead, as the identity of the units.
    """

    kind: str
    """``"cells"`` or ``"units"``."""
    ids: tuple[str, ...]
    """The unit ids the blend was trained on (the cells' grid ids, or the polygons' ids)."""
    grid: object = None
    """The `~hcwf.spatial.evalgrid.EvaluationGrid` (cells only)."""
    kept: np.ndarray | None = field(default=None, compare=False)
    """The positions of the used cells in the grid, latitude-major (cells only)."""
    region: str | None = None
    """The region the run was cut to, if any."""
    models: dict = field(default_factory=dict, compare=False)
    """What each model's rain was put on the units with: ``how``, and for weights a ``weights_sha1`` (and the
    ``shapefile``'s name and ``sha1`` when there is one)."""

    @property
    def cell_lat_lon(self) -> tuple[np.ndarray, np.ndarray]:
        """The centres of the cells the blend uses.

        Returns
        -------
        lat : numpy.ndarray of float64, shape (cell,)
            The latitudes.
        lon : numpy.ndarray of float64, shape (cell,)
            The longitudes.
        """
        width = len(self.grid.lon)
        return self.grid.lat[self.kept // width], self.grid.lon[self.kept % width]

    def to_json(self) -> dict:
        """Return the stored form: plain JSON, floats written exactly.

        Returns
        -------
        dict
            The grid, the cells and ids, and the identity of the units.
        """
        out = {"version": 1, "kind": self.kind, "ids": list(self.ids), "region": self.region,
               "models": self.models}
        grid = self.grid
        if grid is None:
            return out

        def axis(values):
            """Describe an axis by its start, step and count."""
            step = float(np.median(np.diff(values))) if len(values) > 1 else 0.0
            return {"start": float(values[0]), "step": step, "count": int(len(values))}

        out.update({
            "source": grid.source, "lat": grid.lat.tolist(), "lon": grid.lon.tolist(),
            "lat_axis": axis(grid.lat), "lon_axis": axis(grid.lon),
            "lon_convention": "0 to 360" if float(grid.lon.max()) > 180.0 else "-180 to 180",
            "cell_area": evalgrid._AVERAGE, "regrid": list(grid.regrid),
            "steps": [[name, a, b] for name, a, b in grid.steps], "reason": grid.reason,
            "box": None if grid.box is None else list(grid.box), "lon_global": bool(grid.lon_global),
            "where": grid.where, "kept": [int(k) for k in self.kept]})
        return out

    @classmethod
    def from_json(cls, stored: dict) -> "TrainedGrid":
        """Rebuild a stored grid.

        Parameters
        ----------
        stored : dict
            What `to_json` returned.

        Returns
        -------
        TrainedGrid
            The stored grid.
        """
        if stored["kind"] != "cells":
            return cls("units", tuple(stored["ids"]), region=stored.get("region"), models=stored.get("models", {}))
        grid = evalgrid.EvaluationGrid(
            stored["source"], np.asarray(stored["lat"], dtype=float), np.asarray(stored["lon"], dtype=float),
            tuple(stored["regrid"]), tuple((n, a, b) for n, a, b in stored["steps"]), stored["reason"],
            None if stored["box"] is None else tuple(stored["box"]), bool(stored["lon_global"]), stored["where"])
        return cls("cells", tuple(stored["ids"]), grid, np.asarray(stored["kept"], dtype=np.int64),
                   stored.get("region"), stored.get("models", {}))

    def problems(self, name: str, lat, lon) -> list[str]:
        """Say how a live file's grid cannot be put on the stored grid.

        Parameters
        ----------
        name : str
            The model the file belongs to.
        lat : array_like of float, shape (lat,)
            The latitudes of the live file's grid.
        lon : array_like of float, shape (lon,)
            The longitudes of the live file's grid.

        Returns
        -------
        list of str
            One message with a Fix naming both grids, or none when the file can be read onto the stored grid.
        """
        if self.grid is None or lat is None or lon is None:
            return []
        why = evalgrid.source_problem(self.grid, *self.cell_lat_lon, lat, lon)
        if why is None:
            return []
        return [f"{name}: the live file's grid ({evalgrid.describe(lat, lon)}) cannot be put on the grid the blend "
                f"was trained on ({evalgrid.describe(self.grid.lat, self.grid.lon)}, the grid of {self.grid.source}): "
                f"{why}. Fix: forecast from files on the trained grid, or on a finer grid that covers it (it is "
                "averaged onto the trained grid conservatively); otherwise retrain the blend on this grid"]


def _digest(table: pd.DataFrame) -> str:
    """Return a digest of a weights table, whatever the order of its rows.

    Parameters
    ----------
    table : pandas.DataFrame
        The weights table.

    Returns
    -------
    str
        The hex SHA-1 of its rows, sorted, with their values.
    """
    ordered = table.sort_values(list(table.columns), kind="stable").reset_index(drop=True)
    return hashlib.sha1(ordered.to_csv(index=False).encode()).hexdigest()


def _freeze_grid(rc, ids, frozen: dict) -> "TrainedGrid | None":
    """Return the grid, or the units, the blend was trained on, as `TrainedGrid`.

    The cells' grid is the run's evaluation grid; when every source is on one grid (the run has none) it is the
    observations' grid in the region's box. A run on polygon or admin units keeps the units' identity instead.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run config of training.
    ids : tuple of str
        The unit ids the blend was trained on.
    frozen : dict of str to FrozenUnits
        Each model's frozen units.

    Returns
    -------
    TrainedGrid or None
        None when the observations' grid cannot be read (the blend is then stored without a grid, as older ones were).
    """
    from hcwf.io import grids as grids_io

    spec = rc.observations.units
    region = rc.area.region
    models = {name: {"how": units.how} for name, units in frozen.items()}
    if spec.how != "cells":
        for name, units in frozen.items():
            if units.weights is not None:
                models[name]["weights_sha1"] = _digest(units.weights)
        for key in ("shapefile", "weights"):
            if getattr(spec, key):
                path = spec.resolve_path(getattr(spec, key))
                if path.is_file():
                    models["observations"] = {"how": spec.how, key: path.name,
                                              "sha1": hashlib.sha1(path.read_bytes()).hexdigest()}
        return TrainedGrid("units", tuple(ids), region=region, models=models)
    grid = rc.area.evaluation_grid
    if grid is None:
        try:
            lat, lon = grids_io.source_grids(rc)[evalgrid.OBSERVATIONS]
        except Exception as exc:  # noqa: BLE001 - a blend without its grid still works as older ones did
            log.warning("the grid of the observations could not be read, so the blend is stored without its grid: %s",
                        exc)
            return None
        lat, lon = np.sort(np.asarray(lat, dtype=float)), np.sort(np.asarray(lon, dtype=float))
        sizes = ((evalgrid.OBSERVATIONS, *evalgrid.steps(lat, lon)),)
        grid = evalgrid.EvaluationGrid(evalgrid.OBSERVATIONS, lat, lon, (), sizes, "every source is on one grid")
    cells = evalgrid.grid_units(grid, spec.convention).ids
    kept = pd.Index(cells).get_indexer(np.asarray(ids, dtype=str))
    if (kept < 0).any():
        log.warning("%d trained units are not cells of the grid, so the blend is stored without its grid",
                    int((kept < 0).sum()))
        return None
    return TrainedGrid("cells", tuple(ids), grid, kept.astype(np.int64), region, models)


def _weights_frame(transform) -> pd.DataFrame:
    """Return a sparse transform as the long weights table it was made from.

    Parameters
    ----------
    transform : hcwf.spatial.onto_units.SparseTransform
        The transform from a source grid (``keys == "coordinate"``) or source ids to the target units.

    Returns
    -------
    pandas.DataFrame
        One row per weight: ``target_id``, ``lat`` and ``lon`` of the source cell (coordinate keys) or
        ``source_id``, and ``weight``.
    """
    matrix = transform.weighted_matrix.tocoo()
    target = np.asarray(transform.target_ids).astype(str)[matrix.row]
    if transform.keys == "coordinate":
        lat, lon = transform.grid
        return pd.DataFrame({"target_id": target, "lat": lat[matrix.col // lon.size], "lon": lon[matrix.col % lon.size],
                             "weight": matrix.data})
    return pd.DataFrame({"target_id": target, "source_id": np.asarray(transform.source_ids).astype(str)[matrix.col],
                         "weight": matrix.data})


def _model_sources(rc, name: str) -> list[str]:
    """Return what a model's training forecasts are read from: its files or its zarr store.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run config of training.
    name : str
        The model's name.

    Returns
    -------
    list of str
        The files, or the one zarr store (see `hcwf.io.stores.model_sources`).
    """
    source = rc.model(name)
    return stores.model_sources(source.dir, source.paths, source.file_pattern)


def _observation_sources(rc) -> list[str]:
    """Return what the training observations are read from: the listed files, the zarr store or the pattern's files.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run config of training.

    Returns
    -------
    list of str
        The files, or the one zarr store.
    """
    obs = rc.observations
    if obs.paths:
        return [str(p) for p in obs.paths]
    if is_zarr(obs.dir):
        return [str(obs.dir)]
    from hcwf.blend import features

    return [stores.observation_source(year, obs.dir, obs.file_pattern)[1] for year in features.observation_years(rc)]


def _training_slices(rc, name: str, entry: dict) -> list[tuple[str, str, int | None]]:
    """Return the parts of a model's training forecasts that must share one structure: its files, or the years of its store.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run config of training.
    name : str
        The model's name.
    entry : dict
        The model's dataset keys, as the signature reads them.

    Returns
    -------
    list of tuple of (str, str, int or None)
        For each part the source (file or store), its name for messages and the year of the store it is (None for a file).
    """
    out = []
    for path in _model_sources(rc, name):
        if not is_zarr(path):
            out.append((path, Path(path).name, None))
            continue
        _, dates = dataset_detect.read_forecast_file(path, entry, selected=False)
        out += [(path, f"{Path(path).name}[{year}]", year) for year in sorted({d.year for d in dates})]
    return out


def _freeze_units(rc, recipe, name: str, ids) -> FrozenUnits:
    """Return the units of model `name` on its (first) training file's grid, restricted to the trained `ids`.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run config of training, with the model's files.
    recipe : hcwf.blend.recipe.Recipe
        The recipe, for the model's lead days.
    name : str
        The model's name.
    ids : iterable of str
        The unit ids the blend was trained on.

    Returns
    -------
    FrozenUnits
        Without weights for the file's own cells or units, with the weights table otherwise.

    Raises
    ------
    BlendFileError
        The model has no ``units`` section.
    """
    from hcwf.config.model import Selection
    from hcwf.io import forecasts as forecasts_io
    from hcwf.spatial import onto_units

    source = rc.model(name)
    spec = source.units
    if spec is None:
        raise BlendFileError(f"model {name}: a trained blend reads its models on evaluation units; give a `units:` "
                             "section (mode cells is the file's own cells)")
    if spec.how == "cells":
        return FrozenUnits("cells", None, spec.convention)
    model = recipe.model(name)
    select = Selection(first_lead=model.lead_days[0], last_lead=model.lead_days[1])
    with forecasts_io.UnitForecast(_model_sources(rc, name)[0], source, spec, keep_ids=ids, select=select) as file:
        if "unit" in file.da.dims:
            return FrozenUnits(spec.how, None, spec.convention)
        transform = onto_units.unit_transform(file.units, np.asarray(file.da["lat"].values),
                                            np.asarray(file.da["lon"].values), file.positions,
                                            convention=spec.convention)
    return FrozenUnits(spec.how, _weights_frame(transform), spec.convention)


def _freeze_thresholds(thresholds, ids, convention) -> dict:
    """Return the run's named thresholds as a constant, or a table of the trained units' values.

    Parameters
    ----------
    thresholds : mapping of str to object
        The thresholds by name: a number (mm) or anything `hcwf.onset.thresholds.unit_thresholds` resolves.
    ids : array_like of str
        The trained unit ids.
    convention : object
        The unit convention, passed on to ``unit_thresholds``.

    Returns
    -------
    dict
        For each name a float (mm), or a `pandas.DataFrame` with ``id`` and ``onset_thresh``.

    Raises
    ------
    BlendFileError
        A threshold cannot be resolved for the trained units.
    """
    from hcwf.onset.thresholds import unit_thresholds

    out = {}
    for name, value in dict(thresholds).items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            out[name] = float(value)
            continue
        try:
            values = unit_thresholds(value, np.asarray(ids).astype(str), convention)
        except (TypeError, ValueError) as exc:
            raise BlendFileError(f"threshold {name!r} cannot be stored for the trained units: {exc}") from None
        out[name] = pd.DataFrame({"id": np.asarray(ids).astype(str), "onset_thresh": values})
    return out


def _freeze_reference(reference) -> dict | pd.DataFrame | None:
    """Return the reference onset in a form that can be stored.

    Parameters
    ----------
    reference : hcwf.onset.climatology.ReferenceOnset or None
        The reference onset of training.

    Returns
    -------
    dict or pandas.DataFrame or None
        None for no reference, ``{"month_day": ...}`` for a fixed day, the table for a table.

    Raises
    ------
    BlendFileError
        The reference is a computed field of days of year.
    """
    if reference is None:
        return None
    if reference.month_day is not None:
        return {"month_day": reference.month_day}
    if reference.table is not None:
        return reference.table
    raise BlendFileError("a computed reference onset (a field of days of year) cannot be stored in a trained blend")


def _thaw_reference(stored):
    """Return the reference onset from its stored form.

    Parameters
    ----------
    stored : dict or pandas.DataFrame or None
        What `_freeze_reference` returned.

    Returns
    -------
    hcwf.onset.climatology.ReferenceOnset or None
        The reference onset; None when nothing is stored.
    """
    from hcwf.onset.climatology import ReferenceOnset

    if stored is None:
        return None
    if isinstance(stored, dict):
        return ReferenceOnset(month_day=stored["month_day"])
    return ReferenceOnset(table=stored)


def _git_commit() -> str | None:
    """Return the commit of the hcwf checkout in use, when it is one.

    Returns
    -------
    str or None
        The ``git rev-parse HEAD`` of the folder of this module; None when it is not a git checkout or git fails.
    """
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _hcwf_version() -> str | None:
    """Return the installed version of hcwf.

    Returns
    -------
    str or None
        The version of the installed distribution; None when it is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("hcwf")
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------------------------------------------
# The trained blend
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class TrainedBlend:
    """Everything operations needs from training, in one object that saves to and loads from one file.

    See the module docstring for what the file holds and its two formats. Build one with
    `TrainedBlend.from_training`, write it with `TrainedBlend.save`, read it with `TrainedBlend.load` and forecast
    with `TrainedBlend.forecast`.
    """

    mapping: dict
    """The run config (``run.yaml``) as a mapping."""
    run_yaml: str
    """The text of the run config: the resolved config with its sources, or the converted one."""
    definition_text: str | None
    """The onset definition as used; None when it is inline in the run config."""
    definition_hash: str
    """The hash of the onset definition, checked on load."""
    recipe: object
    """The `hcwf.blend.recipe.Recipe`, with the climatology years frozen."""
    rc: object
    """The `hcwf.config.model.RunConfig` the mapping makes (no live files)."""
    model: object
    """The fitted `hcwf.blend.logit.FittedModel`."""
    observed: ObservedOnsets
    """The observed onsets the climatology is fitted on."""
    units: tuple[str, ...]
    """The unit ids the blend was trained and scores on."""
    thresholds: dict
    """The named onset thresholds: each is mm (float), or a table with ``id`` and ``onset_thresh``."""
    reference: object
    """The `hcwf.onset.climatology.ReferenceOnset` of training, or None."""
    frozen_units: dict[str, FrozenUnits]
    """Each model's units as training resolved them, by model name."""
    signatures: dict[str, dict]
    """Each model's forecast signature (`hcwf.config.detect`), by model name."""
    unit_table: pd.DataFrame | None = None
    """The table of the trained units (ids with their attributes), or None when the file has none."""
    provenance: dict = field(default_factory=dict)
    """When and from which files the blend was trained."""
    forest: object = None
    """The fitted `hcwf.blend.random_forest.FittedForest`, when one is configured."""
    grid: TrainedGrid | None = None
    """The grid (or the units' identity) the features are indexed by; None for a blend saved before it was stored: its
    live files are checked against the training files' own grid (the signatures) instead."""
    format: str = FORMAT
    """The format the blend was loaded from (`FORMAT`, or `OLD_FORMAT`, which is not written again)."""

    @property
    def setup(self):
        """The resolved onset setup (`hcwf.config.onset.OnsetSetup`)."""
        return self.rc.onset

    @property
    def entries(self) -> dict[str, dict]:
        """The models' dataset entries (variable, dims, units factor, accumulation) the signatures are resolved with."""
        return model_entries(self.mapping)

    # -- building -----------------------------------------------------------------------------------------------

    @classmethod
    def from_training(cls, result) -> "TrainedBlend":
        """Make the blend that a training run produced.

        Freezes the run config (checking that it rebuilds to the same onset definition and recipe), the forecast
        signature and the units of each model, the thresholds and the reference onset.

        Parameters
        ----------
        result : hcwf.blend.training.TrainingResult
            The training result.

        Returns
        -------
        TrainedBlend
            The blend, in the `FORMAT` format.

        Raises
        ------
        BlendFileError
            The run cannot be frozen: the onset definition or recipe does not rebuild from the stored config, or a
            model's training files differ in structure.
        """
        from hcwf.spatial import onto_units
        from hcwf.runs.resolved import resolved_config_text_of
        from hcwf.onset import definition as definition_mod

        rc, recipe, setup, inputs = result.rc, result.recipe, result.setup, result.inputs
        climatology = inputs.climatology
        recipe = _with_years(recipe, climatology.train_years, climatology.test_years)
        run_yaml = resolved_config_text_of(rc)
        mapping = yaml.safe_load(run_yaml)
        definition_text = setup.definition.source_yaml
        definition_hash = definition_mod.definition_hash(setup.definition)
        frozen = frozen_config(mapping, definition_text)
        if definition_mod.definition_hash(frozen.onset.definition) != definition_hash:
            raise BlendFileError("the onset definition of the run does not rebuild from its stored text; "
                                 "cannot freeze this run")
        if _recipe_digest(_with_years(frozen.blend, climatology.train_years,
                                      climatology.test_years)) != _recipe_digest(recipe):
            raise BlendFileError("the blend settings of the run do not rebuild from its stored run config; "
                                 "cannot freeze this run")
        ids = tuple(sorted({str(unit) for unit in inputs.weekly.units}))
        convention = rc.observations.units.convention
        entries = model_entries(mapping)
        signatures, units, problems = {}, {}, []
        for model in recipe.models:
            seen = {}
            for path, name, year in _training_slices(rc, model.name, entries[model.name]):
                signature = dataset_detect.forecast_signature(
                    path, entries[model.name], lead_days=model.lead_days, max_member=model.max_member, year=year)
                seen.setdefault(json.dumps(signature, sort_keys=True), []).append(name)
            if len(seen) > 1:
                groups = "; ".join(f"{', '.join(names[:4])}{'...' if len(names) > 4 else ''}"
                                   for names in seen.values())
                found = [json.loads(text) for text in seen]
                fields = ", ".join(f"{key}: {' vs '.join(dict.fromkeys(str(s.get(key)) for s in found))}"
                                   for key in dict.fromkeys(k for s in found for k in s)
                                   if len({json.dumps(s.get(key), sort_keys=True) for s in found}) > 1)
                problems.append(f"{model.name}: its training files differ in {fields} ({groups}); "
                                f"a blend needs one forecast structure to forecast with")
                continue
            signatures[model.name] = json.loads(next(iter(seen)))
            units[model.name] = _freeze_units(rc, recipe, model.name, ids)
        if problems:
            raise BlendFileError("cannot make a trained blend:\n" + "\n".join(f"  - {p}" for p in problems))
        stored_grid = _freeze_grid(rc, ids, units)
        if stored_grid is not None and stored_grid.grid is not None:   # the models' cells are put on the stored grid
            units = {name: dataclasses.replace(u, grid=stored_grid.grid) if u.how == "cells" else u
                     for name, u in units.items()}
        identities = {"observations": [file_identity(p) for p in _observation_sources(rc)],
                      "models": {m.name: [file_identity(p) for p in _model_sources(rc, m.name)] for m in recipe.models}}
        return cls(
            mapping=mapping, run_yaml=run_yaml, definition_text=definition_text, definition_hash=definition_hash,
            recipe=recipe, rc=frozen, model=result.model, observed=inputs.observed, units=ids,
            thresholds=_freeze_thresholds(inputs.thresholds, ids, convention),
            reference=_thaw_reference(_freeze_reference(inputs.reference)), frozen_units=units,
            signatures=signatures, unit_table=onto_units.unit_table(rc.observations, ids).table, grid=stored_grid,
            provenance={"trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "training_years": list(recipe.evaluation.years.training), "training_files": identities},
            forest=result.forest)

    # -- files --------------------------------------------------------------------------------------------------

    def save(self, path) -> Path:
        """Write the blend to ``path`` (hcwf's format), atomically.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write.

        Returns
        -------
        pathlib.Path
            ``path``.

        Raises
        ------
        BlendFileError
            The blend was loaded from another format.
        """
        if self.format != FORMAT:
            raise BlendFileError(f"a blend loaded from the {self.format} format is not written again; retrain it")
        members: dict[str, bytes] = {"run.yaml": self.run_yaml.encode(),
                                     "definition.yaml": (self.definition_text or "").encode()}
        climatology = self.recipe.climatology
        members["recipe.json"] = json.dumps({
            "digest": _recipe_digest(self.recipe),
            "climatology": {"train_years": list(climatology.train_years), "test_years": list(climatology.test_years)}},
            indent=1).encode()
        meta, arrays = self.model.to_arrays()
        members["model.json"] = json.dumps(meta, indent=1).encode()
        members["model.npz"] = _npz_bytes(arrays)
        if self.forest is not None:
            meta, arrays = self.forest.to_arrays()
            members["forest.json"] = json.dumps(meta, indent=1).encode()
            members["forest.npz"] = _npz_bytes(arrays)
        observed = self.observed.data
        members["observed.npz"] = _npz_bytes({
            "unit": observed["unit"].values.astype(str), "year": observed["year"].values,
            **{name: observed[name].values for name in ("onset_day", "onset_idx", "present")}})
        table = self.unit_table
        members["units.json"] = json.dumps({
            "ids": list(self.units),
            "table": None if table is None else {c: [None if pd.isna(v) else v for v in table[c].tolist()]
                                                 for c in table.columns}}).encode()
        arrays, meta = {}, {}
        for name, units in self.frozen_units.items():
            meta[name] = {"how": units.how, "weights": None if units.weights is None
                          else _put(arrays, f"weights_{name}", units.weights)}
        members["weights.json"] = json.dumps(meta).encode()
        members["weights.npz"] = _npz_bytes(arrays)
        arrays = {}
        meta = {"thresholds": {name: value if isinstance(value, float) else _put(arrays, f"threshold_{name}", value)
                               for name, value in self.thresholds.items()},
                "reference_onset": None if self.reference is None else (
                    {"month_day": self.reference.month_day} if self.reference.month_day is not None
                    else {"table": _put(arrays, "reference_onset", self.reference.table)})}
        members["reference.json"] = json.dumps(meta).encode()
        members["reference.npz"] = _npz_bytes(arrays)
        members["signatures.json"] = json.dumps(self.signatures, indent=1).encode()
        if self.grid is not None:
            members["grid.json"] = json.dumps(self.grid.to_json()).encode()
        manifest = {
            "format": FORMAT, "format_version": FORMAT_VERSION, "features_version": FEATURES_VERSION,
            "hcwf_version": _hcwf_version(), "git_commit": _git_commit(), "provenance": self.provenance,
            "definition_hash": self.definition_hash,
            "members": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}}

        def write(temp: Path) -> None:
            """Write the zip (manifest first, then the members) to the temporary file."""
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(_MANIFEST, json.dumps(manifest, indent=1))
                for name, data in members.items():
                    archive.writestr(name, data)

        atomic_write(path, write)
        return Path(path)

    @classmethod
    def load(cls, path) -> "TrainedBlend":
        """Read a saved blend (either format).

        Refuses a damaged file, another feature version, or a run config, definition or recipe that no longer rebuild
        to what training used.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to read.

        Returns
        -------
        TrainedBlend
            The blend.

        Raises
        ------
        BlendFileError
            The file is unreadable, of an unknown format or version, damaged, or trained with another feature
            version.
        """
        path = Path(path)
        try:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read(_MANIFEST))
                members = {name: archive.read(name) for name in manifest["members"]}
        except (zipfile.BadZipFile, zlib.error, EOFError, KeyError, json.JSONDecodeError, OSError) as exc:
            raise BlendFileError(f"{path} is not a readable trained blend ({exc})") from None
        known = {FORMAT: (FORMAT_VERSION, cls._load_hcwf), LEGACY_HCWF_FORMAT: (FORMAT_VERSION, cls._load_hcwf),
                 OLD_FORMAT: (1, cls._load_old)}
        found = known.get(manifest.get("format"))
        if found is None or manifest.get("format_version") != found[0]:
            raise BlendFileError(f"{path} is not a trained blend this version can read "
                                 f"(format {manifest.get('format')!r} version {manifest.get('format_version')!r})")
        damaged = [n for n, d in members.items() if hashlib.sha256(d).hexdigest() != manifest["members"][n]]
        if damaged:
            raise BlendFileError(f"{path} is damaged: {', '.join(damaged)} do not match their digests")
        if manifest.get("features_version") != FEATURES_VERSION:
            raise BlendFileError(
                f"{path} was trained with feature version {manifest.get('features_version')}; this version computes "
                f"version {FEATURES_VERSION}. Retrain it, or forecast with the version it was trained with.")
        return found[1](manifest, members)

    @classmethod
    def _common(cls, members: dict, *, observed_season: str) -> dict:
        """Read what both formats store alike: the fitted model and forest and the observed onsets.

        Parameters
        ----------
        members : dict of str to bytes
            The members of the file.
        observed_season : str
            The season start of the observed onsets, ``"MM-DD"``.

        Returns
        -------
        dict
            The ``model``, ``observed`` and ``forest`` (None when the file has none) fields of the blend.
        """
        from hcwf.blend.logit import FittedModel

        arrays = _npz_load(members["observed.npz"])
        observed = ObservedOnsets(
            data=xr.Dataset({name: (("unit", "year"), arrays[name]) for name in ("onset_day", "onset_idx", "present")},
                            coords={"unit": arrays["unit"].astype(object), "year": arrays["year"]}),
            season_start=observed_season)
        forest = None
        if "forest.json" in members:
            from hcwf.blend.random_forest import FittedForest

            forest = FittedForest.from_arrays(json.loads(members["forest.json"]),
                                              dict(_npz_load(members["forest.npz"])))
        return {"model": FittedModel.from_arrays(json.loads(members["model.json"]),
                                                 dict(_npz_load(members["model.npz"]))),
                "observed": observed, "forest": forest}

    @classmethod
    def _load_hcwf(cls, manifest: dict, members: dict) -> "TrainedBlend":
        """Read a blend of hcwf's format (also the legacy ``momp`` id).

        Parameters
        ----------
        manifest : dict
            The manifest of the file.
        members : dict of str to bytes
            The members of the file, already checked against their digests.

        Returns
        -------
        TrainedBlend
            The blend.

        Raises
        ------
        BlendFileError
            The onset definition or the recipe no longer rebuilds to what training used.
        """
        from hcwf.onset import definition as definition_mod

        mapping = yaml.safe_load(members["run.yaml"])
        definition_text = members["definition.yaml"].decode() or None
        rc = frozen_config(mapping, definition_text)
        setup = rc.onset
        if definition_mod.definition_hash(setup.definition) != manifest["definition_hash"]:
            raise BlendFileError("the onset definition stored in the file no longer reads as it did when the blend "
                                 "was trained (its hash differs); retrain the blend")
        stored = json.loads(members["recipe.json"])
        recipe = _with_years(rc.blend, **stored["climatology"])
        if _recipe_digest(recipe) != stored["digest"]:
            raise BlendFileError("the blend settings stored in the file no longer rebuild as they were when the "
                                 "blend was trained; retrain the blend")
        season = blend_season_start(setup)
        common = cls._common(members, observed_season=season)
        units = json.loads(members["units.json"])
        weights_arrays = _npz_load(members["weights.npz"])
        from hcwf.onset.detect import unit_convention

        frozen_units = {}
        convention = unit_convention(rc)
        for name, item in json.loads(members["weights.json"]).items():
            frozen_units[name] = FrozenUnits(
                item["how"], None if item["weights"] is None else _get(weights_arrays, item["weights"]), convention)
        stored_grid = TrainedGrid.from_json(json.loads(members["grid.json"])) if "grid.json" in members else None
        if stored_grid is not None and stored_grid.grid is not None:
            frozen_units = {name: dataclasses.replace(u, grid=stored_grid.grid) if u.how == "cells" else u
                            for name, u in frozen_units.items()}
        reference_meta = json.loads(members["reference.json"])
        reference_arrays = _npz_load(members["reference.npz"])
        thresholds = {name: value if not isinstance(value, dict) else _get(reference_arrays, value)
                      for name, value in reference_meta["thresholds"].items()}
        stored_reference = reference_meta["reference_onset"]
        if isinstance(stored_reference, dict) and "table" in stored_reference:
            stored_reference = _get(reference_arrays, stored_reference["table"])
        return cls(mapping=mapping, run_yaml=members["run.yaml"].decode(), definition_text=definition_text,
                   definition_hash=manifest["definition_hash"], recipe=recipe, rc=rc,
                   units=tuple(units["ids"]), thresholds=thresholds, reference=_thaw_reference(stored_reference),
                   frozen_units=frozen_units, signatures=json.loads(members["signatures.json"]),
                   unit_table=None if units["table"] is None else pd.DataFrame(units["table"]),
                   provenance=manifest.get("provenance", {}), format=FORMAT, grid=stored_grid, **common)

    @classmethod
    def _load_old(cls, manifest: dict, members: dict) -> "TrainedBlend":
        """Read an onset_blending trained blend: its ``config.yml`` converted in memory, its arrays as stored.

        The conversion is the one ``hcwf-run convert`` does.

        Parameters
        ----------
        manifest : dict
            The manifest of the file.
        members : dict of str to bytes
            The members of the file, already checked against their digests.

        Returns
        -------
        TrainedBlend
            The blend, in the `OLD_FORMAT` format.

        Raises
        ------
        BlendFileError
            The stored config cannot be converted.
        """
        from hcwf.config.convert_specs import convert_blend
        from hcwf.errors import ConvertError
        from hcwf.onset import definition as definition_mod

        try:
            conversion = convert_blend(yaml.safe_load(members["config.yml"]))
        except ConvertError as exc:
            raise BlendFileError(f"the config stored in the trained blend cannot be converted: {exc}") from None
        mapping = conversion.mapping
        rc = frozen_config(mapping)
        setup, recipe = rc.onset, rc.blend
        season = blend_season_start(setup)
        common = cls._common(members, observed_season=season)
        reference_meta = json.loads(members["reference.json"])
        reference_arrays = _npz_load(members["reference.npz"])
        thresholds = reference_meta["thresholds"]
        if isinstance(thresholds, dict):
            thresholds = _get(reference_arrays, thresholds)
        elif thresholds is not None:
            thresholds = float(thresholds)
        names = [name for name, _ in setup.definition.thresholds]
        reference = reference_meta["reference_onset"]
        if isinstance(reference, dict) and "table" in reference:
            reference = _get(reference_arrays, reference["table"])
        elif isinstance(reference, dict):
            reference = {"month_day": reference["month_day"]}
        frozen_units = {}
        for model in rc.models:
            name, spec = model.name, model.units
            weights = None
            for key in ("regrid_weights", "cell_transforms"):
                if name in reference_meta[key]:
                    weights = _get(reference_arrays, reference_meta[key][name])
            frozen_units[name] = FrozenUnits(spec.how, weights, spec.convention)
        return cls(mapping=mapping, run_yaml=conversion.to_yaml({"config": "the trained blend's config.yml"}),
                   definition_text=None, definition_hash=definition_mod.definition_hash(setup.definition),
                   recipe=recipe, rc=rc, units=tuple(json.loads(members["units.json"])),
                   thresholds={name: thresholds for name in names}, reference=_thaw_reference(reference),
                   frozen_units=frozen_units, signatures=json.loads(members["signatures.json"]),
                   provenance=manifest.get("provenance", {}), format=OLD_FORMAT, **common)

    # -- operations ---------------------------------------------------------------------------------------------

    def forecast(self, forecasts: dict, *, issue_date=None, fewer_members: bool = False, workers: int | None = None):
        """Return blended onset probabilities from live forecast files (see `hcwf.blend.forecast.run`).

        Parameters
        ----------
        forecasts : dict
            Maps each trained model to a file, a glob or a list of them.
        issue_date : object, optional
            The issue date to forecast; passed on to `hcwf.blend.forecast.run`.
        fewer_members : bool, default False
            Passed on to `hcwf.blend.forecast.run`.
        workers : int, optional
            The number of worker processes, passed on to `hcwf.blend.forecast.run`.

        Returns
        -------
        object
            What `hcwf.blend.forecast.run` returns.
        """
        from hcwf.blend import forecast as forecast_module

        return forecast_module.run(self, forecasts, issue_date=issue_date, fewer_members=fewer_members,
                                   workers=workers)


def expand_files(value) -> list[Path]:
    """Return the files a forecast names.

    Parameters
    ----------
    value : str or pathlib.Path or list
        A path, a glob, or a list of them.

    Returns
    -------
    list of pathlib.Path
        The files, sorted by year in the name, then path.
    """
    return resolve_paths(value, origin="forecast files")
