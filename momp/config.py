"""The ROMP run config as Python objects, for notebooks and scripts; the YAML run file is the same thing on disk.

    from momp.config import Config, Dataset, Onset, Evaluation
    cfg = Config(
        region="ethiopia",
        observations=Dataset(path="/data/chirps", variable="precip", units="mm"),
        models=[Dataset(name="AIFS", path="/data/aifs", units="m")],
        onset=Onset(definition="icpac", override={"season": {"from": "05-15"}}),
        evaluation=Evaluation(start_date=(2019, 5, 1), end_date=(2024, 7, 31)),
    )
    flat = cfg.to_flat()          # validated and resolved flat dict, as the pipeline uses it
    cfg.to_yaml("run.yaml")       # the same config as a YAML file; Config.from_yaml reads it back

Only what you set is written to YAML: region defaults, schema defaults and values detected from your
files are resolved by ``to_flat`` at run time (every detection is reported in ``flat["dataset_report"]``).
Sections (Onset, Evaluation, Metrics, Output) have one attribute per key of the YAML section of the
same name; see momp.lib.config_schema for each key's type, default and help. Unset keys are the
``UNSET`` sentinel.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from momp.lib import config_schema as schema
from momp.lib.config_schema import ConfigSchemaError
from momp.lib.config_yaml import build_config
from momp.lib.convert import plain, render_yaml


class _Unset:
    """Marker for a key that was not set (distinct from None, which is an explicit null)."""

    def __repr__(self):
        """Print as UNSET."""
        return "UNSET"

    def __bool__(self):
        """UNSET is falsy."""
        return False


UNSET = _Unset()


@dataclasses.dataclass
class Dataset:
    """An observation or forecast dataset: where it is and how to read it.

    Set only what you know: path (required); file_pattern, variable, units (mm or m), dims, members, accumulation and
    lead_label (forecasts) or day_boundary (observations) are detected from the files when left UNSET, with
    the evidence reported. ``dataset`` names an entry of a dataset file (dataset_files in the config).
    name is the model name used in output files (models only).
    """

    path: Any = UNSET
    name: Any = UNSET
    dataset: Any = UNSET
    file_pattern: Any = UNSET
    variable: Any = UNSET
    units: Any = UNSET
    dims: Any = UNSET
    day_boundary: Any = UNSET
    lead_label: Any = UNSET
    accumulation: Any = UNSET
    members: Any = UNSET

    def to_mapping(self) -> dict:
        """The keys that are set, as a plain mapping."""
        return {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                if getattr(self, f.name) is not UNSET}


@dataclasses.dataclass
class Mask:
    """The `mask:` of a run config: the cells to evaluate, built on each dataset's own grid (rainfall is never regridded).

    Parts given intersect: land (Natural Earth land), country (a name or list of names), file (0/1 NetCDF with
    variable and keep_value, or a shapefile / GeoJSON) and condition ({file, variable, above, below}).
    cell_rule is 'centre' or 'overlap_fraction' (with overlap_fraction, default 0.5); resolution is the Natural
    Earth resolution ('10m', '50m' (default) or '110m'). Unset keys are UNSET. See momp.utils.mask.
    """

    land: Any = UNSET
    country: Any = UNSET
    file: Any = UNSET
    variable: Any = UNSET
    keep_value: Any = UNSET
    condition: Any = UNSET
    cell_rule: Any = UNSET
    overlap_fraction: Any = UNSET
    resolution: Any = UNSET

    def to_mapping(self) -> dict:
        """The keys that are set, as a plain mapping."""
        return {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                if getattr(self, f.name) is not UNSET}


def _section(name: str, extra=()):
    """Build the dataclass of one config section from the schema: an attribute per key, all UNSET by default."""
    names = [f.nested_name for f in schema.FIELDS if f.section == name] + list(extra)
    cls = dataclasses.make_dataclass(
        name.capitalize(), [(n, Any, dataclasses.field(default=UNSET)) for n in names],
        namespace={"__doc__": f"The `{name}:` section of a run config ({schema.SECTIONS[name]}). Attributes: "
                              + ", ".join(names) + ". Unset keys are UNSET.",
                   "to_mapping": lambda self: {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                                               if getattr(self, f.name) is not UNSET}})
    cls.__module__ = __name__
    return cls


Reference = _section("reference", extra=("dataset", "units"))
Onset = _section("onset")
Evaluation = _section("evaluation")
Metrics = _section("metrics")
Output = _section("output")
_SECTION_CLASSES = {"reference": Reference, "onset": Onset, "evaluation": Evaluation, "metrics": Metrics,
                    "output": Output}
_TOP_KEYS = [f.key for f in schema.FIELDS if f.section is None and not f.key.endswith("_list")]


class _ConfigMethods:
    """Conversions of a Config: to and from mappings, YAML files and the flat pipeline dict."""

    def to_mapping(self) -> dict:
        """The nested run-config mapping holding only what is set."""
        out: dict = {}
        for key in _TOP_KEYS:
            if getattr(self, key) is not UNSET:
                out[key] = plain(getattr(self, key))
        if self.observations is not UNSET:
            out["observations"] = self.observations.to_mapping()
        for name in _SECTION_CLASSES:
            section = getattr(self, name)
            if section is not UNSET and section.to_mapping():
                out[name] = section.to_mapping()
        if self.models is not UNSET:
            out["models"] = [m.to_mapping() for m in self.models]
        order = {k: i for i, k in enumerate(list(_TOP_KEYS) + list(schema.SECTIONS))}
        return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], 99)))

    def to_flat(self, source: str = "Config") -> dict:
        """Validate and resolve into the flat config dict the pipeline uses (see momp.config.flat_config)."""
        return build_config(self.to_mapping(), source=source)

    def to_yaml(self, path=None) -> str:
        """The config as YAML text; also written to path if given."""
        text = render_yaml(self.to_mapping())
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_mapping(cls, raw: dict):
        """Build a Config from a nested mapping (as read from YAML); unknown keys are errors with suggestions."""
        errors: list[str] = []
        if not isinstance(raw, dict):
            raise ConfigSchemaError(f"Invalid config: expected a mapping at the top level, got {type(raw).__name__}")
        schema.flatten_sections(raw, errors)
        kwargs: dict = {}
        for key, value in raw.items():
            if key == "observations" and isinstance(value, dict):
                known = {f.name for f in dataclasses.fields(Dataset)}
                aliases = {"dir": "path", "dataset": "dataset"}
                items = {aliases.get(k, k): v for k, v in value.items()}
                kwargs[key] = Dataset(**{k: v for k, v in items.items() if k in known})
            elif key == "models" and isinstance(value, list):
                known = {f.name for f in dataclasses.fields(Dataset)}
                kwargs[key] = [Dataset(**{k: v for k, v in m.items() if k in known}) if isinstance(m, dict) else m
                               for m in value]
            elif key in _SECTION_CLASSES and isinstance(value, dict):
                cls_ = _SECTION_CLASSES[key]
                known = {f.name for f in dataclasses.fields(cls_)}
                kwargs[key] = cls_(**{k: v for k, v in value.items() if k in known})
            elif key in _TOP_KEYS:
                kwargs[key] = value
        schema.raise_errors(errors, "<mapping>")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path):
        """Read a run-config YAML file; the values are checked when to_flat() is called."""
        try:
            raw = yaml.safe_load(Path(path).read_text())
        except yaml.YAMLError as exc:
            raise ConfigSchemaError(f"Invalid config {path}: not valid YAML: {exc}") from None
        return cls.from_mapping(raw or {})


Config = dataclasses.make_dataclass(
    "Config",
    [(key, Any, dataclasses.field(default=UNSET)) for key in _TOP_KEYS]
    + [("observations", Any, dataclasses.field(default=UNSET)), ("reference", Any, dataclasses.field(default=UNSET)),
       ("models", Any, dataclasses.field(default=UNSET)), ("onset", Any, dataclasses.field(default=UNSET)),
       ("evaluation", Any, dataclasses.field(default=UNSET)), ("metrics", Any, dataclasses.field(default=UNSET)),
       ("output", Any, dataclasses.field(default=UNSET))],
    bases=(_ConfigMethods,),
    namespace={"__doc__": "A ROMP run config. Attributes: " + ", ".join(_TOP_KEYS) +
               ", and the sections observations (a Dataset), reference, models (a list of Dataset), onset, "
               "evaluation, metrics, output. Unset keys are UNSET and take their default (or the region's, or "
               "are detected from the files) when resolved by to_flat()."})
Config.__module__ = __name__


def flat_config(cfg: "Config") -> dict:
    """The flat config dict the pipeline uses, built from a Config object (validated, resolved, detections reported).

    This is the single conversion from the Python objects to the pipeline's dict.
    """
    return cfg.to_flat()
