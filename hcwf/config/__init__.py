"""The HCWF run config as Python objects, for notebooks and scripts; the YAML run file is the same thing on disk.

This package holds the config layer: the Python objects defined here (`Config`, `Dataset`, `Mask` and one class per
section), the schema of every key (`hcwf.config.schema`), the conversion of a raw mapping into the flat pipeline dict
(`hcwf.config.mapping`, `hcwf.config.build`) and the typed, frozen `hcwf.config.model.RunConfig` that the pipeline
reads::

    from hcwf.config import Config, Dataset, Onset, Evaluation
    cfg = Config(
        region="<region>",
        observations=Dataset(path="/path/to/observations", variable="<rain variable>", units="mm"),
        models=[Dataset(name="<model>", path="/path/to/forecasts", units="m")],
        onset=Onset(definition="<onset definition>", override={"season": {"from": "<MM-DD>"}}),
        evaluation=Evaluation(start_date="<YYYY-MM-DD>", end_date="<YYYY-MM-DD>"),
    )
    flat = cfg.to_flat()          # validated and resolved flat dict, as the pipeline uses it
    cfg.to_yaml("run.yaml")       # the same config as a YAML file; Config.from_yaml reads it back

Only what you set is written to YAML: region defaults, schema defaults and values detected from your
files are resolved by ``to_flat`` at run time (every detection is reported in ``flat["dataset_report"]``).
Sections (`Onset`, `Evaluation`, `Metrics`, `Output`) have one attribute per key of the YAML section of the
same name; see `hcwf.config.schema` for each key's type, default and help. Unset keys are the
`UNSET` sentinel.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import yaml

from hcwf.config import schema
from hcwf.config.convert import plain, render_yaml
from hcwf.errors import ConfigSchemaError


class _Unset:
    """The type of `UNSET`: the marker for a key that was not set (distinct from ``None``, an explicit null)."""

    def __repr__(self) -> str:
        """Return the text ``UNSET``.

        Returns
        -------
        str
            ``"UNSET"``.
        """
        return "UNSET"

    def __bool__(self) -> bool:
        """Return false: `UNSET` is falsy.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False


UNSET = _Unset()
"""The sentinel of a config key that was not set."""


def _prose(text: str) -> str:
    """Return schema help text with its single backticks made literals.

    The schema writes key names in single backticks; in a docstring they would be object references.

    Parameters
    ----------
    text : str
        The help text.

    Returns
    -------
    str
        The text with every single-backtick span in double backticks.
    """
    return re.sub(r"(?<!`)`([^`]+)`(?!`)", r"``\1``", text)


def _field_entry(field: schema.Field) -> tuple[str, str]:
    """Describe a schema field as the type line and the description of a config attribute.

    Parameters
    ----------
    field : hcwf.config.schema.Field
        The schema field of the attribute.

    Returns
    -------
    tuple of (str, str)
        The type line (the field's expected form, with its choices and ``null``) and the description (the help, then the
        default, ``Required`` or the note that HCWF sets the key itself).
    """
    text = _prose(field.help)
    if field.internal:
        text += "; set by HCWF, refused in a YAML file"
    if field.required:
        text += ". Required."
    elif field.fill and field.default is not None and not callable(field.default):
        text += f". Default {field.default!r}."
    elif not text.endswith("."):
        text += "."
    return field.expected, text


def _attributes_section(entries: dict[str, tuple[str, str]]) -> str:
    """Return a NumPy ``Attributes`` section for the given attributes.

    Parameters
    ----------
    entries : dict of str to tuple of (str, str)
        Attribute name to its type line and description.

    Returns
    -------
    str
        The section text, one ``name : type`` line and an indented description per attribute.
    """
    lines = ["Attributes", "----------"]
    for name, (kind, text) in entries.items():
        lines += [f"{name} : ``{kind}``", f"    {text}"]
    return "\n".join(lines)


def _document(cls: type, entries: dict[str, tuple[str, str]]) -> None:
    """Append the generated ``Attributes`` section to a config class's docstring.

    Parameters
    ----------
    cls : type
        The config class; every dataclass field must be in ``entries``.
    entries : dict of str to tuple of (str, str)
        Attribute name to its type line and description, taken from the schema where it has the key.
    """
    names = [f.name for f in dataclasses.fields(cls)]
    cls.__doc__ = (cls.__doc__ or "").rstrip() + "\n\n" + _attributes_section({n: entries[n] for n in names})


@dataclasses.dataclass
class Dataset:
    """An observation or forecast dataset: where it is and how to read it.

    Set only what you know: ``path`` (required, unless ``paths`` lists the files: names or globs, relative to ``path``,
    any number per year, ``years`` keeping those of some years); ``unit_cvt`` is the multiplier to mm of files that
    declare no units. ``file_pattern``, ``variable``, ``units`` (``mm`` or ``m``), ``dims``, ``members``,
    ``accumulation``, ``daily_boundary`` and ``lead_label`` (forecasts) or ``day_boundary`` (observations) are detected
    from the files when left `UNSET`, with the evidence reported. ``dataset`` names an entry of a dataset file
    (``dataset_files`` in the config). ``name`` is the model name used in output files (models only). Every field
    is `UNSET` unless given.
    """

    path: Any = UNSET
    paths: Any = UNSET
    years: Any = UNSET
    name: Any = UNSET
    dataset: Any = UNSET
    file_pattern: Any = UNSET
    variable: Any = UNSET
    units: Any = UNSET
    unit_cvt: Any = UNSET
    dims: Any = UNSET
    day_boundary: Any = UNSET
    lead_label: Any = UNSET
    accumulation: Any = UNSET
    daily_boundary: Any = UNSET
    members: Any = UNSET

    def to_mapping(self) -> dict:
        """Return the keys that are set, as a plain mapping.

        Returns
        -------
        dict
            Field name to its value converted by `hcwf.config.convert.plain`, only for the fields that are not
            `UNSET`.
        """
        return {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                if getattr(self, f.name) is not UNSET}


@dataclasses.dataclass
class Mask:
    """The ``mask:`` of a run config: the cells to evaluate, built on each dataset's own grid.

    Rainfall is never regridded. Parts given intersect: ``land`` (Natural Earth land), ``country`` (a name or list of
    names), ``file`` (0/1 NetCDF with ``variable`` and ``keep_value``, or a shapefile / GeoJSON) and ``condition``
    (``{file, variable, above, below}``). ``cell_rule`` is ``"centre"`` or ``"overlap_fraction"`` (with
    ``overlap_fraction``, default 0.5); ``resolution`` is the Natural Earth resolution (``"10m"``, ``"50m"`` (default)
    or ``"110m"``). Unset keys are `UNSET`. See `hcwf.spatial.masks`.
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
        """Return the keys that are set, as a plain mapping.

        Returns
        -------
        dict
            Field name to its value converted by `hcwf.config.convert.plain`, only for the fields that are not
            `UNSET`.
        """
        return {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                if getattr(self, f.name) is not UNSET}


def _dataset_entry(kind: str, text: str, choices=None) -> tuple[str, str]:
    """Describe a `Dataset` attribute: the schema's description of its kind, and a hand-written description.

    The dataset keys are shared by observations and models, so the schema's per-key help (written for one of them) does
    not fit; the type line still comes from the schema's `hcwf.config.schema.KINDS`.

    Parameters
    ----------
    kind : str
        A key of `hcwf.config.schema.KINDS`.
    text : str
        The description of the attribute.
    choices : tuple, optional
        The allowed values, if the kind is narrowed to some.

    Returns
    -------
    tuple of (str, str)
        The type line and the description.
    """
    return schema.Field("_", kind, "", choices=choices).expected, text


_DATASET_ENTRIES = {
    "path": _dataset_entry("path", "The folder of the files (required unless ``paths`` lists them)."),
    "paths": _dataset_entry("paths", "The files to read: names or globs relative to ``path``, any number per year (a "
                                     "list of lists, one per year, for several models)."),
    "years": _dataset_entry("years", "The years whose files to keep (the others are left out)."),
    "name": _dataset_entry("str", "The model name used in output files (models only)."),
    "dataset": _dataset_entry("str", "The name of an entry of a dataset file (``dataset_files`` in the config)."),
    "file_pattern": _dataset_entry("patterns", "The file name pattern of the yearly files; detected when `UNSET`."),
    "variable": _dataset_entry("str", "The name of the rainfall variable in the files; detected when `UNSET`."),
    "units": _dataset_entry("str", "The units of the rainfall; detected when `UNSET`.", choices=("mm", "m")),
    "unit_cvt": _dataset_entry("number", "The multiplier to mm of files that declare no units."),
    "dims": _dataset_entry("dims", "The names of the dimensions in the files, by canonical name; detected when "
                                   "`UNSET`."),
    "day_boundary": _dataset_entry("str", "The boundary of the observed day (observations only); detected when "
                                          "`UNSET`."),
    "lead_label": _dataset_entry("str", "How the lead dimension is labelled (forecasts only); detected when `UNSET`.",
                                 choices=schema.BY_PATH[("onset", "lead_label")].choices),
    "accumulation": _dataset_entry("str", "Whether the forecast rain is accumulated over the lead (forecasts only); "
                                          "detected when `UNSET`."),
    "daily_boundary": _dataset_entry("str", "The boundary of the forecast day (forecasts only); detected when "
                                            "`UNSET`."),
    "members": _dataset_entry("members", "The ensemble members to use (forecasts only); detected when `UNSET`."),
}
"""The type line and description of each `Dataset` attribute."""
_document(Dataset, _DATASET_ENTRIES)
_document(Mask, {k: (t, _prose(h) + ".") for k, (t, h) in schema.MASK_KEYS.items()})


def _section(name: str, extra=()):
    """Build the dataclass of one config section from the schema.

    Parameters
    ----------
    name : str
        The section name, a key of `hcwf.config.schema.SECTIONS` (``"onset"``, ``"output"``, ...).
    extra : tuple of str, default ()
        Attribute names to add to the keys the schema holds for the section.

    Returns
    -------
    type
        A dataclass named after the section (``Onset``), with one attribute per key, all `UNSET` by default, and a
        ``to_mapping`` method that returns the keys that are set.
    """
    fields = [f for f in schema.FIELDS if f.section == name]
    names = [f.nested_name for f in fields] + list(extra)
    cls = dataclasses.make_dataclass(
        name.capitalize(), [(n, Any, dataclasses.field(default=UNSET)) for n in names],
        namespace={"__doc__": f"The ``{name}:`` section of a run config ({schema.SECTIONS[name]}).\n\n"
                              "One attribute per key of the section, described by the schema. Unset keys are "
                              "`UNSET`.",
                   "to_mapping": lambda self: {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                                               if getattr(self, f.name) is not UNSET}})
    cls.__module__ = __name__
    _document(cls, {**{n: _DATASET_ENTRIES[n] for n in extra}, **{f.nested_name: _field_entry(f) for f in fields}})
    return cls


Reference = _section("reference", extra=("dataset", "units"))
Onset = _section("onset")
Units = _section("units")
Blend = _section("blend")
Evaluation = _section("evaluation")
Metrics = _section("metrics")
Output = _section("output")
_SECTION_CLASSES = {"reference": Reference, "onset": Onset, "units": Units, "blend": Blend,
                    "evaluation": Evaluation, "metrics": Metrics, "output": Output}
"""The section classes by section name."""
_TOP_KEYS = [f.key for f in schema.FIELDS if f.section is None and not f.key.endswith("_list")]
"""The config keys at the top level of the run file (not in a section, not a ``*_list`` key)."""


class _ConfigMethods:
    """The conversions of a `Config`: to and from mappings, YAML files and the flat pipeline dict.

    The mixin base of the generated `Config` dataclass.
    """

    def to_mapping(self) -> dict:
        """Return the nested run-config mapping holding only what is set.

        Returns
        -------
        dict
            The top-level keys that are set, then ``observations``, the sections and ``models``, in the order of
            the schema; the values are plain Python objects.
        """
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
        """Validate and resolve the config into the flat config dict the pipeline uses.

        Reads the datasets' files to detect what is left unset.

        Parameters
        ----------
        source : str, default "Config"
            What the config is called in error messages.

        Returns
        -------
        dict
            The flat config, see `hcwf.config.mapping.build_config`.
        """
        from hcwf.config.mapping import build_config   # reads the datasets' files (xarray, pandas): load on use

        return build_config(self.to_mapping(), source=source)

    def to_yaml(self, path=None) -> str:
        """Return the config as YAML text, and write it to ``path`` if given.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            The file to write; ``None`` writes nothing.

        Returns
        -------
        str
            The YAML text of `to_mapping`.
        """
        text = render_yaml(self.to_mapping())
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_mapping(cls, raw: dict):
        """Build a Config from a nested mapping (as read from YAML).

        Unknown keys are errors, reported with suggestions.

        Parameters
        ----------
        raw : dict
            The nested run-config mapping.

        Returns
        -------
        Config
            The config, with `Dataset` and section objects for ``observations``, ``models`` and the sections.

        Raises
        ------
        ConfigSchemaError
            ``raw`` is not a mapping, or has an unknown key.
        """
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
        """Read a run-config YAML file; the values are checked when `to_flat` is called.

        Parameters
        ----------
        path : str or pathlib.Path
            The YAML file.

        Returns
        -------
        Config
            The config of the file (an empty file gives an empty config).

        Raises
        ------
        ConfigSchemaError
            The file is not valid YAML, or has an unknown key.
        """
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
       ("units", Any, dataclasses.field(default=UNSET)), ("blend", Any, dataclasses.field(default=UNSET)),
       ("evaluation", Any, dataclasses.field(default=UNSET)), ("metrics", Any, dataclasses.field(default=UNSET)),
       ("output", Any, dataclasses.field(default=UNSET))],
    bases=(_ConfigMethods,),
    namespace={"__doc__": "An HCWF run config.\n\nOne attribute per top-level key and per section, described by the "
               "schema. Unset keys are `UNSET` and take their default (or the region's, or are detected from the "
               "files) when resolved by ``to_flat()``."})
Config.__module__ = __name__
_document(Config, {
    **{f.key: _field_entry(f) for f in schema.FIELDS if f.section is None},
    "observations": ("a Dataset", f"The ``observations:`` section: {schema.SECTIONS['observations']}."),
    "reference": ("a Reference", f"The ``reference:`` section: {schema.SECTIONS['reference']}."),
    "models": ("a list of Dataset", f"The ``models:`` list: {schema.SECTIONS['models']}, one `Dataset` each."),
    **{name: (f"an {name.capitalize()}" if name[0] in "oe" else f"a {name.capitalize()}",
              f"The ``{name}:`` section: {schema.SECTIONS[name]}.")
       for name in ("onset", "units", "blend", "evaluation", "metrics", "output")},
})
