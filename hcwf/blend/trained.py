"""TrainedBlend: everything operations needs from training, in one file.

A trained blend holds the run it was trained with (HCWF's resolved run YAML, with the onset definition frozen beside
it), the fitted model (and random forest), the reference data training resolved (per-unit thresholds, reference onsets,
the weights that put each model's grid on the units), the observed onsets the climatology is fitted on, and each
model's forecast signature. `forecast` takes only live forecast files and an issue date; nothing else is read, so
operations cannot run with other settings than training did (`hcwf.blend.forecast`).

File formats. A trained blend is a zip of JSON, YAML and NumPy arrays: no pickles, so it loads across library versions
and is safe to read. It is written to a temporary file and renamed into place, and every member is checked against its
digest on load. Two formats load into this one class:

* "hcwf/trained-blend" (what `TrainingResult.save` writes): manifest.json, run.yaml (the resolved run config, with its
  sources), definition.yaml (the onset definition as used, with its hash in the manifest), recipe.json (the recipe's
  digest and the climatology years training resolved), model.json/.npz, forest.json/.npz, units.json, weights.json/.npz,
  reference.json/.npz, observed.npz and signatures.json. Loading rebuilds the run config, the onset setup and the
  recipe from run.yaml and definition.yaml and refuses a file whose definition or recipe no longer rebuild to what
  training used.
* "onset-blending/trained-blend" (onset_blending, features version 1): its BlendConfig config.yml goes through
  `hcwf.config.convert_specs` in memory (as `hcwf-run convert` does) to the run config and recipe; the model, forest,
  reference, units and observed arrays are used as stored.
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
from hcwf.blend.samples import ObservedOnsets
from hcwf.cache import atomic_write, file_identity
from hcwf.config import detect as dataset_detect
from hcwf.config.datasets import resolve_paths

log = logging.getLogger(__name__)

__all__ = ["FEATURES_VERSION", "FORMAT", "FORMAT_VERSION", "FrozenUnits", "LEGACY_HCWF_FORMAT", "OLD_FORMAT", "TrainedBlend", "frozen_config"]

# Bump whenever the numbers the features are computed from change: a blend trained under another version is refused
# rather than fed different features.
FEATURES_VERSION = 1
FORMAT = "hcwf/trained-blend"
FORMAT_VERSION = 1
OLD_FORMAT = "onset-blending/trained-blend"
# The id this format had before the package was renamed from momp to hcwf: files saved then still load (same layout).
LEGACY_HCWF_FORMAT = "momp/trained-blend"
_MANIFEST = "manifest.json"
TRAINED_FILENAME = "trained.blend"


# ---------------------------------------------------------------------------------------------------------------
# Arrays and tables without pickles
# ---------------------------------------------------------------------------------------------------------------


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _npz_load(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)


def _put(arrays: dict, prefix: str, table: pd.DataFrame) -> dict:
    """A table's columns into arrays (strings as unicode, dates as ISO text); returns its schema."""
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
    """The table `_put` described."""
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
    """The dataset entries of the run's models by name (their keys as the run YAML gives them), without their files."""
    return {m["name"]: {k: v for k, v in m.items() if k not in ("paths", "path", "years")}
            for m in mapping.get("models") or ()}


def frozen_config(mapping: dict, definition_text: str | None = None):
    """The RunConfig of a run mapping, without reading any data: the config's sections are flattened and typed as a run
    config's are, the model lists are made from the model entries (no files: a forecast names its own), and the onset
    definition is the frozen one (`definition_text`) when given, else the run's own inline rule. Its `onset` is the
    OnsetSetup and its `blend` the recipe (the invariants of a run that read data are not checked)."""
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
    return hashlib.sha256(repr(recipe).encode()).hexdigest()


def _with_years(recipe, train_years, test_years):
    """The recipe with the climatology years training resolved (a forecast must fit the same training window)."""
    return dataclasses.replace(recipe, climatology=dataclasses.replace(
        recipe.climatology, train_years=tuple(train_years), test_years=tuple(test_years)))


# ---------------------------------------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenUnits:
    """One model's units as training resolved them: how its rain reaches them (`how`, as `UnitsSpec.how`) and, for
    units made from a shapefile or weights, the weights table that put the training grid on them (target_id with
    lat/lon or source_id, and weight). It stands in for a `UnitsSpec` when a live file is read (`select`): the weights
    are matched to the live grid's cells by coordinates, so a live file holding more cells than training did is read
    for the trained ones only. Units that are the file's own cells (or its own units) need no weights."""

    how: str
    weights: pd.DataFrame | None = field(default=None, compare=False)
    convention: object = None

    def __post_init__(self):
        if self.weights is not None:     # the one canonical form of a weights table (legacy column names renamed)
            from hcwf.spatial.regrid import read_weights

            object.__setattr__(self, "weights", read_weights(self.weights, "the trained blend's weights"))

    def select(self, lat=None, lon=None, ids=None, keep_ids=None):
        """(units, positions of those used, ordered by id), as `UnitsSpec.select`."""
        from hcwf.spatial.units import EvaluationUnits

        if ids is not None:
            units = EvaluationUnits(pd.DataFrame({"id": np.asarray(ids).astype(str)}), "polygon")
        elif self.weights is not None:
            targets = sorted(self.weights["target_id"].astype(str).unique())
            units = EvaluationUnits(pd.DataFrame({"id": targets}), "polygon", weights=self.weights)
        else:
            units = EvaluationUnits.from_grid(lat, lon, explicit=self.convention)
        keep = np.ones(len(units), dtype=bool) if keep_ids is None else np.isin(
            units.ids, np.asarray(sorted(keep_ids), dtype=str))
        positions = np.flatnonzero(keep)
        if not len(positions):
            raise ValueError("No unit of the file is kept")
        return units, positions[np.argsort(units.ids[positions], kind="stable")]


def _weights_frame(transform) -> pd.DataFrame:
    """A SparseTransform as the long weights table it was made from (target_id, source cell or id, weight)."""
    matrix = transform.weighted_matrix.tocoo()
    target = np.asarray(transform.target_ids).astype(str)[matrix.row]
    if transform.keys == "coordinate":
        lat, lon = transform.grid
        return pd.DataFrame({"target_id": target, "lat": lat[matrix.col // lon.size], "lon": lon[matrix.col % lon.size],
                             "weight": matrix.data})
    return pd.DataFrame({"target_id": target, "source_id": np.asarray(transform.source_ids).astype(str)[matrix.col],
                         "weight": matrix.data})


def _freeze_units(rc, recipe, name: str, ids) -> FrozenUnits:
    """The units of model `name` on its (first) training file's grid, restricted to the trained `ids`."""
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
    with forecasts_io.UnitForecast(source.paths[0], source, spec, keep_ids=ids, select=select) as file:
        if "unit" in file.da.dims:
            return FrozenUnits(spec.how, None, spec.convention)
        transform = onto_units.unit_transform(file.units, np.asarray(file.da["lat"].values),
                                            np.asarray(file.da["lon"].values), file.positions,
                                            convention=spec.convention)
    return FrozenUnits(spec.how, _weights_frame(transform), spec.convention)


def _freeze_thresholds(thresholds, ids, convention) -> dict:
    """The run's named thresholds as a constant, or a table of the trained units' values."""
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
    if reference is None:
        return None
    if reference.month_day is not None:
        return {"month_day": reference.month_day}
    if reference.table is not None:
        return reference.table
    raise BlendFileError("a computed reference onset (a field of days of year) cannot be stored in a trained blend")


def _thaw_reference(stored):
    from hcwf.onset.climatology import ReferenceOnset

    if stored is None:
        return None
    if isinstance(stored, dict):
        return ReferenceOnset(month_day=stored["month_day"])
    return ReferenceOnset(table=stored)


def _git_commit() -> str | None:
    """The commit of the hcwf checkout in use, when it is one."""
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _hcwf_version() -> str | None:
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
    mapping: dict                       # the run config (run.yaml) as a mapping
    run_yaml: str                       # its text: the resolved config with its sources, or the converted one
    definition_text: str | None         # the onset definition as used (None: inline in the run config)
    definition_hash: str
    recipe: object                      # hcwf.blend.recipe.Recipe, climatology years frozen
    rc: object                          # hcwf.config.model.RunConfig the mapping makes (no live files)
    model: object                       # hcwf.blend.logit.FittedModel
    observed: ObservedOnsets            # what the climatology is fitted on
    units: tuple[str, ...]              # the units the blend was trained and scores on
    thresholds: dict                    # name -> mm, or a table (id, onset_thresh)
    reference: object                   # hcwf.onset.climatology.ReferenceOnset | None
    frozen_units: dict[str, FrozenUnits]
    signatures: dict[str, dict]         # model name -> forecast signature (hcwf.config.detect)
    unit_table: pd.DataFrame | None = None
    provenance: dict = field(default_factory=dict)
    forest: object = None               # hcwf.blend.random_forest.FittedForest, when configured
    format: str = FORMAT

    @property
    def setup(self):
        """The resolved onset setup (hcwf.config.onset.OnsetSetup)."""
        return self.rc.onset

    @property
    def entries(self) -> dict[str, dict]:
        """The models' dataset entries (variable, dims, units factor, accumulation) the signatures are resolved with."""
        return model_entries(self.mapping)

    # -- building -----------------------------------------------------------------------------------------------

    @classmethod
    def from_training(cls, result) -> "TrainedBlend":
        """The blend a training run produced (`hcwf.blend.training.TrainingResult`)."""
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
            for path in rc.model(model.name).paths:
                signature = dataset_detect.forecast_signature(
                    path, entries[model.name], lead_days=model.lead_days, max_member=model.max_member)
                seen.setdefault(json.dumps(signature, sort_keys=True), []).append(Path(path).name)
            if len(seen) > 1:
                groups = "; ".join(f"{', '.join(names[:4])}{'...' if len(names) > 4 else ''}"
                                   for names in seen.values())
                problems.append(f"{model.name}: its training files differ in dimensions, members, lead days or grid "
                                f"({groups}); a blend needs one forecast structure to forecast with")
                continue
            signatures[model.name] = json.loads(next(iter(seen)))
            units[model.name] = _freeze_units(rc, recipe, model.name, ids)
        if problems:
            raise BlendFileError("cannot make a trained blend:\n" + "\n".join(f"  - {p}" for p in problems))
        identities = {"observations": [file_identity(p) for p in rc.observations.paths or ()],
                      "models": {m.name: [file_identity(p) for p in rc.model(m.name).paths] for m in recipe.models}}
        return cls(
            mapping=mapping, run_yaml=run_yaml, definition_text=definition_text, definition_hash=definition_hash,
            recipe=recipe, rc=frozen, model=result.model, observed=inputs.observed, units=ids,
            thresholds=_freeze_thresholds(inputs.thresholds, ids, convention),
            reference=_thaw_reference(_freeze_reference(inputs.reference)), frozen_units=units,
            signatures=signatures, unit_table=onto_units.unit_table(rc.observations, ids).table,
            provenance={"trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "training_years": list(recipe.evaluation.years.training), "training_files": identities},
            forest=result.forest)

    # -- files --------------------------------------------------------------------------------------------------

    def save(self, path) -> Path:
        """Write the blend to `path` (hcwf's format), atomically."""
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
        manifest = {
            "format": FORMAT, "format_version": FORMAT_VERSION, "features_version": FEATURES_VERSION,
            "hcwf_version": _hcwf_version(), "git_commit": _git_commit(), "provenance": self.provenance,
            "definition_hash": self.definition_hash,
            "members": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}}

        def write(temp: Path) -> None:
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(_MANIFEST, json.dumps(manifest, indent=1))
                for name, data in members.items():
                    archive.writestr(name, data)

        atomic_write(path, write)
        return Path(path)

    @classmethod
    def load(cls, path) -> "TrainedBlend":
        """Read a saved blend (either format); refuses a damaged file, another feature version, or a run config,
        definition or recipe that no longer rebuild to what training used."""
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
        """What both formats store alike: the fitted model and forest, the observed onsets, the units."""
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
        season = f"{setup.definition.seasons[0].start.month:02d}-{setup.definition.seasons[0].start.day:02d}"
        common = cls._common(members, observed_season=season)
        units = json.loads(members["units.json"])
        weights_arrays = _npz_load(members["weights.npz"])
        frozen_units = {}
        convention = _convention(rc)
        for name, item in json.loads(members["weights.json"]).items():
            frozen_units[name] = FrozenUnits(
                item["how"], None if item["weights"] is None else _get(weights_arrays, item["weights"]), convention)
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
                   provenance=manifest.get("provenance", {}), format=FORMAT, **common)

    @classmethod
    def _load_old(cls, manifest: dict, members: dict) -> "TrainedBlend":
        """An onset_blending trained blend: its config.yml converted in memory (as `hcwf-run convert` does), its
        arrays as stored."""
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
        season = f"{setup.definition.seasons[0].start.month:02d}-{setup.definition.seasons[0].start.day:02d}"
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
        """Blended onset probabilities from live forecast files (see `hcwf.blend.forecast.run`): `forecasts` maps each
        trained model to a file, a glob or a list of them."""
        from hcwf.blend import forecast as forecast_module

        return forecast_module.run(self, forecasts, issue_date=issue_date, fewer_members=fewer_members,
                                   workers=workers)


def _convention(rc):
    spec = rc.observations.units
    return None if spec is None else spec.convention


def expand_files(value) -> list[Path]:
    """The files a forecast names: a path, a glob, or a list of them (sorted by year in the name, then path)."""
    return resolve_paths(value, origin="forecast files")
