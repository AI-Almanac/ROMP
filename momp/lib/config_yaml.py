"""Turn a nested run-config mapping (the YAML file) into the flat config dict the rest of ROMP uses.

Steps: flatten the sections to flat keys and check types (momp.lib.config_schema); describe the
observations and each model as a dataset, completing what the user left out from the files
(momp.lib.dataset_detect, every detection reported) and checking what the user wrote against
them; detect the region from the data if it is not given; derive what follows from other values
(climatology years from the region, reference data from the observations); fill defaults.
Defaults, region values and detections are resolved here at run time, not copied into the file.
Every problem found is reported together in one ConfigSchemaError.

Besides the flat keys the result carries ``dataset_report`` (printable lines saying where each
dataset value came from) and ``config_sources`` (flat key -> where its value came from).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from momp.lib import config_schema as schema
from momp.lib import dataset_detect as detect
from momp.lib.config_schema import ConfigSchemaError
from momp.lib.dataset_schema import Dataset, DatasetError, expand_path, get_dataset, load_datasets, suggest

YAML_SUFFIXES = (".yaml", ".yml")
"""File suffixes loaded as YAML."""
_OBS_KEYS = {"obs": "name", "obs_dir": "path", "obs_file_pattern": "file_pattern", "obs_var": "variable",
             "obs_day_boundary": "day_boundary"}
_MODEL_LISTS = ("model_list", "model_dir_list", "model_var_list", "unit_cvt_list", "file_pattern_list",
                "model_dims_list")
_REPORT_ONLY = ("dataset_report", "config_sources")


def is_yaml_path(path) -> bool:
    """Whether a config path names a YAML config (by its suffix)."""
    return Path(str(path)).suffix.lower() in YAML_SUFFIXES


def load_yaml_config(path) -> dict:
    """Read, resolve and validate a YAML config file; returns the flat config dict."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ConfigSchemaError(f"Invalid config {path}: not valid YAML: {exc}") from None
    return build_config(raw, source=str(path))


def _entry_from_dataset(ds: Dataset) -> tuple[dict, dict]:
    """A dataset-file entry as raw entry keys, with its source text per key."""
    entry = {"name": ds.name, "path": ds.path, "file_pattern": ds.file_pattern, "variable": ds.variable,
             "dims": ds.dims, "day_boundary": ds.day_boundary, "lead_label": ds.lead_label, "members": ds.members}
    if ds.units:
        entry["units"] = ds.units
    if ds.kind == "forecast":
        entry["accumulation"] = ds.accumulation
    entry = {k: v for k, v in entry.items() if v is not None}
    return entry, {k: f"dataset '{ds.id}'" for k in entry}


def _explicit_entry(given: dict, ref, datasets, kind, written, errors) -> tuple[dict, dict]:
    """Entry keys for one dataset: values of the referenced dataset file, overridden by the inline keys given."""
    entry, sources = {}, {}
    if ref is not None:
        try:
            if not isinstance(ref, str):
                raise DatasetError(f"expected a dataset id (text), got {ref!r}")
            entry, sources = _entry_from_dataset(get_dataset(ref, datasets, kind))
        except DatasetError as exc:
            errors.append(f"'{written}.dataset': {exc}")
    for key, value in given.items():
        entry[key] = value
        sources[key] = "run config"
    return entry, sources


def _problems(res: detect.Resolution, written: str, errors: list[str]) -> None:
    """Move a resolution's problems to errors, naming the key as written in the config."""
    for key, message in res.problems.items():
        errors.append(f"'{written}.{key}': {message}")


def _region_chain(config):
    """The region named by config['region'] and its ancestors, or [] if it is not a known region."""
    from momp.onset import regions

    try:
        region = regions.load_region(str(config.get("region")))
    except regions.RegionError:
        return []
    return [region, *regions.ancestors(region)]


def _implied_label(config) -> str | None:
    """The lead label implied by the observation day boundary (the config's, else the region's), or None."""
    from momp.onset.ingest import parse_day_boundary

    boundary = config.get("obs_day_boundary")
    if boundary is None:
        boundary = next((r.day_boundary for r in _region_chain(config) if r.day_boundary), None)
    try:
        return parse_day_boundary(boundary)[1] if boundary else None
    except ValueError:
        return None


def _resolve_observations(config, sources, given_block, datasets, report, errors):
    """Resolve the observations dataset; returns its resolution (or None) and writes the flat keys."""
    given = {entry_key: config[flat] for flat, entry_key in _OBS_KEYS.items() if flat in config}
    if "file_pattern" in given:
        given["file_pattern"] = given["file_pattern"][0]
    for key in ("units", "dims"):
        if key in given_block:
            given[key] = given_block[key]
    if not given and "dataset" not in given_block:
        return None
    entry, entry_sources = _explicit_entry(given, given_block.get("dataset"), datasets, "observations",
                                           "observations", errors)
    res = detect.resolve_entry(entry, "observations", sources=entry_sources)
    return res


def _finish_observations(res, config, sources, errors):
    """Write the resolved observation values to the flat keys (after the region fallback for day_boundary)."""
    values = detect.finalize_units(res.values)
    flat = {"obs": values.get("name"), "obs_dir": expand_path(values["path"]) if "path" in values else None,
            "obs_var": values.get("variable"), "obs_unit_cvt": values.get("unit_cvt"),
            "obs_day_boundary": values.get("day_boundary")}
    pattern = config.get("obs_file_pattern") or ((values["file_pattern"],) if "file_pattern" in values else None)
    if pattern:
        flat["obs_file_pattern"] = tuple(pattern)
    for key, value in flat.items():
        if key in ("obs", "obs_dir", "obs_var", "obs_file_pattern") and value is None:
            continue
        if key == "obs_unit_cvt" and "unit_cvt" not in values:
            continue
        config[key] = value
        sources.setdefault(key, "dataset values")
    for line in res.lines:
        sources[{v: k for k, v in _OBS_KEYS.items()}.get(line.key, line.key)] = (
            f"{line.how}" + (f": {line.evidence}" if line.evidence else ""))


def _region_default_boundary(config):
    """The observation day boundary declared by the region (nearest in its chain), with the region id."""
    for region in _region_chain(config):
        if region.day_boundary:
            return region.day_boundary, region.id
    return None, None


def _resolve_models(config, sources, models, obs_values, datasets, report, errors):
    """Resolve the `models:` list into the flat per-model lists."""
    if not isinstance(models, list) or not models:
        errors.append(f"'models': expected a non-empty list of models, got {models!r}. Fix: write one "
                      "'- name: AIFS' item per model, with path, variable and units")
        return
    given_lists = [k for k in _MODEL_LISTS if k in config]
    if given_lists:
        errors.append(f"'models' cannot be combined with {', '.join(given_lists)}. Fix: use one or the other")
        return
    rows, labels = [], []
    for i, item in enumerate(models):
        written = f"models[{i}]"
        if not isinstance(item, dict):
            errors.append(f"'{written}': expected a mapping with {', '.join(schema.MODEL_ENTRY_KEYS)}, got {item!r}")
            continue
        for key in item:
            if key not in schema.MODEL_ENTRY_KEYS:
                errors.append(f"unknown key '{written}.{key}'.{suggest(key, schema.MODEL_ENTRY_KEYS)} "
                              f"Keys: {', '.join(schema.MODEL_ENTRY_KEYS)}")
        given = {k: v for k, v in item.items() if k in schema.MODEL_ENTRY_KEYS and k != "dataset"}
        entry, entry_sources = _explicit_entry(given, item.get("dataset"), datasets, "forecast", written, errors)
        if "lead_label" not in entry and "model_lead_label" in config:
            entry["lead_label"], entry_sources["lead_label"] = config["model_lead_label"], "onset.lead_label"
        if "name" not in entry:
            errors.append(f"'{written}.name': missing. Fix: give the model a name (used in output file names, "
                          "no spaces)")
        res = detect.resolve_entry(entry, "forecast", sources=entry_sources, obs=obs_values)
        _problems(res, written, errors)
        detect.warn_all(res)
        report.append(f"{written} {entry.get('name', '')}:")
        report.extend("  " + t for t in res.report())
        values = detect.finalize_units(res.values)
        for key in ("name", "path", "variable"):
            if key not in values:
                values[key] = None
        rows.append(values)
        if values.get("lead_label"):
            labels.append((values.get("name"), values["lead_label"]))
    names = [r.get("name") for r in rows]
    for name in dict.fromkeys(n for n in names if n and names.count(n) > 1):
        errors.append(f"'models': two models are named '{name}'. Fix: give each a different name")
    if len(rows) != len(models) or any(r["name"] is None or r["path"] is None for r in rows):
        return
    config["model_list"] = tuple(r["name"] for r in rows)
    config["model_dir_list"] = tuple(expand_path(r["path"]) for r in rows)
    config["model_var_list"] = tuple(r["variable"] for r in rows)
    config["unit_cvt_list"] = tuple(r.get("unit_cvt") for r in rows)
    config["file_pattern_list"] = tuple(r.get("file_pattern", "{}.nc") for r in rows)
    config["accumulation_list"] = tuple(r.get("accumulation", "auto") for r in rows)
    if any(r.get("dims") for r in rows):
        config["model_dims_list"] = tuple(r.get("dims") for r in rows)
    for key in _MODEL_LISTS + ("accumulation_list",):
        sources[key] = "models"
    if "model_lead_label" not in config:
        distinct = {label for _, label in labels}
        if len(distinct) > 1:
            errors.append("the models disagree on the lead label (" + ", ".join(f"{n}: {v}" for n, v in labels) +
                          "). Fix: evaluate them in separate runs")
        elif distinct and next(iter(distinct)) != _implied_label(config):
            config["model_lead_label"] = next(iter(distinct))
            sources["model_lead_label"] = "models' lead_label"


def _resolve_reference(config, sources, block, datasets, errors):
    """Fill the reference keys from a dataset file entry and `units`; whatever is not given is derived later."""
    if "dataset" in block:
        try:
            ds = get_dataset(block["dataset"], datasets)
            entry, _ = _entry_from_dataset(ds)
            for flat, key in (("ref_model", "name"), ("ref_model_dir", "path"), ("ref_model_file_pattern", "file_pattern"),
                              ("ref_model_var", "variable"), ("ref_model_unit_cvt", "unit_cvt"),
                              ("ref_model_dims", "dims")):
                if flat not in config and (key in entry or key == "unit_cvt"):
                    config[flat] = ds.name or ds.id if key == "name" else (ds.unit_cvt if key == "unit_cvt" else entry[key])
                    sources[flat] = f"dataset '{ds.id}'"
        except (DatasetError, TypeError) as exc:
            errors.append(f"'reference.dataset': {exc}")
    if "units" in block and "ref_model_unit_cvt" not in config:
        try:
            from momp.lib.dataset_schema import unit_factor

            config["ref_model_unit_cvt"] = unit_factor(block["units"], "reference.units")
        except DatasetError as exc:
            errors.append(str(exc))


def _derive(config, sources, errors):
    """Fill what follows from other values: climatology years, date_filter_year, reference data."""
    if "start_year_clim" not in config and "end_year_clim" not in config:
        years = next((r.climatology_years for r in _region_chain(config) if r.climatology_years), None)
        origin = "region default"
        if not years and config.get("threshold_years"):
            years, origin = config["threshold_years"], "onset.threshold_years"
        if years:
            config["start_year_clim"], config["end_year_clim"] = years
            sources["start_year_clim"] = sources["end_year_clim"] = origin
    if config.get("date_filter_year") is None and config.get("start_date"):
        config["date_filter_year"] = config["start_date"][0]
        sources["date_filter_year"] = "year of start_date"
    if str(config.get("ref_model", "climatology")).endswith("climatology"):
        pairs = (("ref_model_dir", config.get("obs_dir")),
                 ("ref_model_file_pattern", (config.get("obs_file_pattern") or (None,))[0]),
                 ("ref_model_var", config.get("obs_var")), ("ref_model_unit_cvt", config.get("obs_unit_cvt")))
        for key, value in pairs:
            if key not in config:
                config[key] = value
                sources[key] = "the observations"
    else:
        for key, written in (("ref_model_dir", "reference.path"), ("ref_model_var", "reference.variable")):
            if config.get(key) is None:
                errors.append(f"missing required key '{written}': the reference model "
                              f"'{config['ref_model']}' needs it (or name a reference.dataset)")
        config.setdefault("ref_model_file_pattern", "{}.nc")
        config.setdefault("ref_model_unit_cvt", None)


def _detect_region(config, sources, res, report, errors):
    """Take the region from the data extent when it is not given; warn if a given region disagrees with it."""
    if res is None or res.extent is None:
        return
    found, evidence = detect.detect_region(res.extent)
    if "region" not in config:
        if found == "unknown":
            errors.append(f"'region': could not be detected ({evidence}). Fix: set region")
        else:
            config["region"], sources["region"] = found, f"detected: {evidence}"
            report.append(f"region: {found}  [detected: {evidence}]")
    elif found != "unknown":
        covered = {r.id for r in _region_chain(config)}
        if covered and found not in covered:
            warnings.warn(f"region is {config['region']!r} but the observations look like {found} ({evidence})",
                          UserWarning, stacklevel=3)
            report.append(f"warning: region is {config['region']!r} but the observations look like {found} "
                          f"({evidence})")


def build_config(raw, source="<config>") -> dict:
    """Resolve and validate a nested run config given as a Python mapping; returns the flat config dict.

    Raises ConfigSchemaError listing every problem. source names the file in the message.
    """
    errors: list[str] = []
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigSchemaError(f"Invalid config {source}: expected a mapping of keys at the top level, "
                                f"got {type(raw).__name__}. Fix: write 'key: value' lines")
    flat, origin, special = schema.flatten_sections(raw, errors)
    models_given = special["models"]
    config = schema.normalise(flat, origin, errors)
    sources = {key: "run config" for key in config}
    report: list[str] = []
    for key, kind in (("dataset", "str"), ("units", "str"), ("dims", "dims")):
        for block in ("observations", "reference"):
            if key in special[block]:
                field = schema.Field(key, kind, key)
                special[block][key] = schema.normalise_value(field, special[block][key], f"{block}.{key}", errors)
    try:
        datasets = load_datasets(config.get("dataset_files", ()), config.get("catalog_dirs", ()))
    except DatasetError as exc:
        errors.append(str(exc))
        datasets = {}

    obs_res = _resolve_observations(config, sources, special["observations"], datasets, report, errors)
    if obs_res is not None:
        _problems_obs = dict(obs_res.problems)
        _detect_region(config, sources, obs_res, report, errors)
        if "day_boundary" in _problems_obs:
            boundary, region_id = _region_default_boundary(config)
            if boundary and "day_boundary" not in obs_res.values:
                obs_res.values["day_boundary"] = None
                obs_res.problems.pop("day_boundary")
                obs_res.lines.append(detect.Line("day_boundary", None, "from the region",
                                                 f"{boundary!r}, the declared default of region {region_id}"))
        detect.warn_all(obs_res)
        _problems(obs_res, "observations", errors)
        report.append("observations:")
        report.extend("  " + t for t in obs_res.report())
        _finish_observations(obs_res, config, sources, errors)
    obs_values = dict(obs_res.values) if obs_res is not None else None
    if obs_values is not None and not obs_values.get("day_boundary"):
        boundary, _ = _region_default_boundary(config)
        obs_values["day_boundary"] = boundary

    _resolve_reference(config, sources, special["reference"], datasets, errors)
    if models_given is not None:
        _resolve_models(config, sources, models_given, obs_values, datasets, report, errors)
    elif "model_list" not in config:
        errors.append("missing required key 'models': at least one forecast model. Fix: write 'models:' with "
                      "'- name: AIFS', path, variable and units")
    _derive(config, sources, errors)
    skip = _MODEL_LISTS if any(e.startswith(("'models", "missing required key 'models'")) for e in errors) else ()
    filled = schema.apply_defaults(config, errors, skip=skip)
    for key in filled:
        sources.setdefault(key, "default")
    try:
        schema.raise_errors(errors, source)
    except ConfigSchemaError as exc:
        exc.report = tuple(report)
        raise
    filled["dataset_report"] = tuple(report)
    filled["config_sources"] = sources
    return filled
