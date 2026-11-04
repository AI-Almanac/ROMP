"""Turn a nested run-config mapping (the YAML file) into the flat config dict the rest of HCWF uses.

`build_config` is the entry. Steps: flatten the sections to flat keys and check types (`hcwf.config.schema`); describe
the observations and each model as a dataset, completing what the user left out from the files
(`hcwf.config.detect`, every detection reported) and checking what the user wrote against
them; detect the region from the data if it is not given; derive what follows from other values
(climatology years from the region, reference data from the observations); fill defaults.
Defaults, region values and detections are resolved here at run time, not copied into the file.
Every problem found is reported together in one ``ConfigSchemaError``. `hcwf.config.build` calls it, and
``hcwf.config.Config.to_flat`` is a thin wrapper.

Besides the flat keys the result carries ``dataset_report`` (printable lines saying where each
dataset value came from) and ``config_sources`` (flat key -> where its value came from).
"""
from __future__ import annotations

import warnings
from pathlib import Path


from hcwf.config import schema
from hcwf.config import detect
from hcwf.config.datasets import Dataset, expand_path, file_year, get_dataset, load_datasets, suggest
from hcwf.errors import ConfigSchemaError, DatasetError, RegionError

YAML_SUFFIXES = (".yaml", ".yml")
"""File suffixes loaded as YAML."""
_OBS_KEYS = {"obs": "name", "obs_dir": "path", "obs_file_pattern": "file_pattern", "obs_var": "variable",
             "obs_day_boundary": "day_boundary"}
"""The flat keys of the observations, by the key of their dataset entry."""
_MODEL_LISTS = ("model_list", "model_dir_list", "model_var_list", "unit_cvt_list", "file_pattern_list",
                "model_dims_list")
"""The per-model list keys that ``models:`` fills (and that cannot be given beside it)."""


def is_yaml_path(path) -> bool:
    """Return whether a config path names a YAML config (by its suffix).

    Parameters
    ----------
    path : str or pathlib.Path
        The config path.

    Returns
    -------
    bool
        True for a ``.yaml`` or ``.yml`` suffix, in any case.
    """
    return Path(str(path)).suffix.lower() in YAML_SUFFIXES


def _listed_files(values: dict, blend: bool) -> tuple[str, ...]:
    """Return the files of a dataset given as ``paths``.

    For a blend (which reads every file of a source) also the yearly files of ``path`` and ``file_pattern``.

    Parameters
    ----------
    values : dict
        The dataset's resolved values: ``paths``, ``path`` and ``file_pattern`` are read.
    blend : bool
        Whether the config has a ``blend:`` section.

    Returns
    -------
    tuple of str
        The files; for the yearly files sorted by year. Empty when there is nothing to list.
    """
    if values.get("paths"):
        return tuple(str(f) for f in detect.dataset_files(values))
    if blend and values.get("path") and values.get("file_pattern") and "{}" in values["file_pattern"]:
        import glob

        found = glob.glob(str(Path(expand_path(values["path"])) / values["file_pattern"].replace("{}", "[0-9][0-9][0-9][0-9]")))
        return tuple(sorted(found, key=lambda f: (file_year(f) or 0, f)))
    return ()


def _entry_from_dataset(ds: Dataset) -> tuple[dict, dict]:
    """Return a dataset-file entry as raw entry keys, with its source text per key.

    Parameters
    ----------
    ds : hcwf.config.datasets.Dataset
        The dataset of a dataset file.

    Returns
    -------
    tuple of (dict, dict)
        The entry keys that the dataset sets (not None), and the text ``dataset '<id>'`` for each of them.
    """
    entry = {"name": ds.name, "path": ds.path, "file_pattern": ds.file_pattern, "variable": ds.variable,
             "dims": ds.dims, "day_boundary": ds.day_boundary, "lead_label": ds.lead_label, "members": ds.members}
    if ds.units:
        entry["units"] = ds.units
    if ds.kind == "forecast":
        entry["accumulation"] = ds.accumulation
        if ds.daily_boundary:
            entry["daily_boundary"] = ds.daily_boundary
    entry = {k: v for k, v in entry.items() if v is not None}
    return entry, {k: f"dataset '{ds.id}'" for k in entry}


def _explicit_entry(given: dict, ref, datasets, kind, written, errors) -> tuple[dict, dict]:
    """Return the entry keys for one dataset: values of the referenced dataset file, overridden by the inline keys given.

    Parameters
    ----------
    given : dict
        The keys written inline in the config.
    ref : object
        The ``dataset`` key (an id in a dataset file), or None.
    datasets : dict
        The datasets loaded from the dataset files, by id.
    kind : str
        ``"observations"`` or ``"forecast"``.
    written : str
        How the entry is written in the config (``observations``, ``models[0]``), for messages.
    errors : list of str
        Takes the message when the reference is not a known dataset.

    Returns
    -------
    tuple of (dict, dict)
        The entry keys and the source text of each (``"run config"`` for the inline ones).

    Raises
    ------
    DatasetError
        ``ref`` is not a string; raised inside the function and recorded in ``errors``.
    """
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
    """Move a resolution's problems to ``errors``, naming the key as written in the config.

    Parameters
    ----------
    res : hcwf.config.detect.Resolution
        The resolution of a dataset.
    written : str
        How the dataset is written in the config.
    errors : list of str
        Takes one message per problem.
    """
    for key, message in res.problems.items():
        errors.append(f"'{written}.{key}': {message}")


def _region_chain(config):
    """Return the region named by ``config["region"]`` and its ancestors.

    Parameters
    ----------
    config : collections.abc.Mapping
        The flat config being built.

    Returns
    -------
    list
        The region and its ancestors, nearest first; empty if it is not a known region.
    """
    from hcwf.onset import regions

    try:
        region = regions.load_region(str(config.get("region")))
    except RegionError:
        return []
    return [region, *regions.ancestors(region)]


def _implied_label(config) -> str | None:
    """Return the lead label implied by the observation day boundary (the config's, else the region's).

    Parameters
    ----------
    config : collections.abc.Mapping
        The flat config being built.

    Returns
    -------
    str or None
        ``"start"`` or ``"end"``, or None when there is no boundary or it cannot be parsed.
    """
    from hcwf.onset.ingest import parse_day_boundary

    boundary = config.get("obs_day_boundary")
    if boundary is None:
        boundary = next((r.day_boundary for r in _region_chain(config) if r.day_boundary), None)
    try:
        return parse_day_boundary(boundary)[1] if boundary else None
    except ValueError:
        return None


def _resolve_observations(config, given_block, datasets, errors):
    """Resolve the observations dataset.

    Parameters
    ----------
    config : collections.abc.Mapping
        The flat config being built: the ``obs_*`` keys are read.
    given_block : dict
        The ``observations`` section as written (``units``, ``unit_cvt``, ``dims``, ``paths``, ``years``, ``dataset``).
    datasets : dict
        The datasets loaded from the dataset files, by id.
    errors : list of str
        Takes the messages of a bad dataset reference.

    Returns
    -------
    hcwf.config.detect.Resolution or None
        The resolution; None when the config gives no observations.
    """
    given = {entry_key: config[flat] for flat, entry_key in _OBS_KEYS.items() if flat in config}
    if "file_pattern" in given:
        given["file_pattern"] = given["file_pattern"][0]
    for key in ("units", "unit_cvt", "dims", "paths", "years"):
        if key in given_block:
            given[key] = given_block[key]
    if not given and "dataset" not in given_block:
        return None
    entry, entry_sources = _explicit_entry(given, given_block.get("dataset"), datasets, "observations",
                                           "observations", errors)
    res = detect.resolve_entry(entry, "observations", sources=entry_sources)
    return res


def _finish_observations(res, config, sources):
    """Write the resolved observation values to the flat keys (after the region fallback for ``day_boundary``).

    Parameters
    ----------
    res : hcwf.config.detect.Resolution
        The resolution of the observations.
    config : dict
        The flat config being built, changed in place.
    sources : dict
        The flat key to its source text, changed in place.
    """
    values = detect.finalize_units(res.values)
    flat = {"obs": values.get("name"), "obs_dir": expand_path(values["path"]) if values.get("path") else None,
            "obs_var": values.get("variable"), "obs_unit_cvt": values.get("unit_cvt"),
            "obs_day_boundary": values.get("day_boundary")}
    listed = _listed_files(values, schema.has_blend(config))
    if listed:   # listed files: any number per year, the years filter applied
        flat["obs_paths"] = listed
        flat["obs_dir"] = flat["obs_dir"] or ""
    if values.get("dims") and any(line.key == "dims" and not line.how.startswith("detected") for line in res.lines):
        flat["obs_dims"] = dict(values["dims"])   # only dims the user (or a dataset file) gave rename anything
    pattern = config.get("obs_file_pattern") or ((values["file_pattern"],) if "file_pattern" in values else None)
    if pattern:
        flat["obs_file_pattern"] = tuple(pattern)
    for key, value in flat.items():
        if key in ("obs", "obs_var", "obs_file_pattern") and value is None:
            continue
        if key == "obs_dir" and value is None:
            continue
        if key == "obs_unit_cvt" and "unit_cvt" not in values:
            continue
        config[key] = value
        sources.setdefault(key, "dataset values")
    for line in res.lines:
        sources[{v: k for k, v in _OBS_KEYS.items()}.get(line.key, line.key)] = (
            f"{line.how}" + (f": {line.evidence}" if line.evidence else ""))


def _region_default_boundary(config):
    """Return the observation day boundary declared by the region (nearest in its chain), with the region id.

    Parameters
    ----------
    config : collections.abc.Mapping
        The flat config being built.

    Returns
    -------
    tuple of (str or None, str or None)
        The boundary and the id of the region that declares it; ``(None, None)`` when none does.
    """
    for region in _region_chain(config):
        if region.day_boundary:
            return region.day_boundary, region.id
    return None, None


def _resolve_models(config, sources, models, obs_values, datasets, report, errors):
    """Resolve the ``models:`` list into the flat per-model lists.

    Parameters
    ----------
    config : dict
        The flat config being built, changed in place (the ``*_list`` keys).
    sources : dict
        The flat key to its source text, changed in place.
    models : object
        The ``models`` section as written: a non-empty list of mappings.
    obs_values : dict or None
        The observations' resolved values, used to detect the lead label.
    datasets : dict
        The datasets loaded from the dataset files, by id.
    report : list of str
        Takes the detection report lines.
    errors : list of str
        Takes the messages of invalid entries; the lists are not written when there are entry errors.

    Returns
    -------
    None
        Nothing; the lists are written into ``config``.
    """
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
        align = bool(config.get("check_alignment", False))
        res = detect.resolve_entry(entry, "forecast", sources=entry_sources, obs=obs_values,
                                   detect_lead_label=align, lead_label_default=not align)
        _problems(res, written, errors)
        detect.warn_all(res)
        report.append(f"{written} {entry.get('name', '')}:")
        report.extend("  " + t for t in res.report())
        values = detect.finalize_units(res.values)
        values["_accumulation_detected"] = any(line.key == "accumulation" and line.how.startswith("detected") for line in res.lines)
        for key in ("name", "path", "variable"):
            if key not in values:
                values[key] = "" if key == "path" and values.get("paths") else None
        rows.append(values)
        if values.get("lead_label"):
            labels.append((values.get("name"), values["lead_label"]))
    names = [r.get("name") for r in rows]
    for name in dict.fromkeys(n for n in names if n and names.count(n) > 1):
        errors.append(f"'models': two models are named '{name}'. Fix: give each a different name")
    if len(rows) != len(models) or any(r["name"] is None or r["path"] is None for r in rows):
        return
    config["model_list"] = tuple(r["name"] for r in rows)
    config["model_dir_list"] = tuple(expand_path(r["path"]) if r["path"] else "" for r in rows)
    config["model_var_list"] = tuple(r["variable"] for r in rows)
    config["unit_cvt_list"] = tuple(r.get("unit_cvt") for r in rows)
    config["file_pattern_list"] = tuple(r.get("file_pattern", "{}.nc") for r in rows)
    config["accumulation_list"] = tuple(r.get("accumulation", "auto") for r in rows)
    config["daily_boundary_list"] = tuple(r.get("daily_boundary") for r in rows)
    config["accumulation_detected"] = tuple(bool(r.get("_accumulation_detected")) for r in rows)
    if any(r.get("dims") for r in rows) or schema.has_blend(config):
        config["model_dims_list"] = tuple(r.get("dims") for r in rows)
    listed = [_listed_files(r, schema.has_blend(config)) for r in rows]
    if any(listed):   # listed files, any number per year (a blend reads every file of a model)
        config["model_paths_list"] = tuple(listed)
        sources["model_paths_list"] = "models"
    for key in _MODEL_LISTS + ("accumulation_list", "daily_boundary_list", "accumulation_detected"):
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
    """Fill the reference keys from a dataset file entry and ``units``; whatever is not given is derived later.

    Parameters
    ----------
    config : dict
        The flat config being built, changed in place (the ``ref_model_*`` keys).
    sources : dict
        The flat key to its source text, changed in place.
    block : dict
        The ``reference`` section as written.
    datasets : dict
        The datasets loaded from the dataset files, by id.
    errors : list of str
        Takes the messages of a bad dataset reference or units.
    """
    if "dataset" in block:
        try:
            ds = get_dataset(block["dataset"], datasets)
            entry, _ = _entry_from_dataset(ds)
            for flat, key in (("ref_model", "name"), ("ref_model_dir", "path"), ("ref_model_file_pattern", "file_pattern"),
                              ("ref_model_var", "variable"), ("ref_model_unit_cvt", "unit_cvt"),
                              ("ref_model_dims", "dims"), ("ref_model_accumulation", "accumulation"),
                              ("ref_model_daily_boundary", "daily_boundary")):
                if flat not in config and (key in entry or key == "unit_cvt"):
                    config[flat] = ds.name or ds.id if key == "name" else (ds.unit_cvt if key == "unit_cvt" else entry[key])
                    sources[flat] = f"dataset '{ds.id}'"
        except (DatasetError, TypeError) as exc:
            errors.append(f"'reference.dataset': {exc}")
    if "units" in block and "ref_model_unit_cvt" not in config:
        try:
            from hcwf.config.datasets import unit_factor

            config["ref_model_unit_cvt"] = unit_factor(block["units"], "reference.units")
        except DatasetError as exc:
            errors.append(str(exc))


def _resolve_reference_settings(config, sources, given_units, report):
    """Resolve the accumulation of a reference model's files (and its daily boundary, given only) as a forecast model's are.

    A given ``reference.accumulation`` is kept, otherwise it is detected from the first file with the same detection
    (`hcwf.config.detect.resolve_entry`), and ``"auto"`` (worked out when read) if the files say nothing. A climatology
    reference has no files of its own and no settings.

    Parameters
    ----------
    config : dict
        The flat config being built, changed in place (``ref_model_accumulation`` and its ``_detected`` flag).
    sources : dict
        The flat key to its source text, changed in place.
    given_units : str or None
        The ``reference.units`` as written, used when no multiplier is set.
    report : list of str
        Takes one line about the reference's accumulation.

    Returns
    -------
    None
        Nothing; a climatology reference is left alone.
    """
    if str(config.get("ref_model", "climatology")).endswith("climatology") or not config.get("ref_model_dir"):
        return
    entry = {"name": config["ref_model"], "path": expand_path(config["ref_model_dir"]),
             "variable": config.get("ref_model_var"), "file_pattern": config.get("ref_model_file_pattern")}
    if config.get("ref_model_dims"):
        entry["dims"] = config["ref_model_dims"]
    if config.get("ref_model_unit_cvt") is not None:
        entry["unit_cvt"] = config["ref_model_unit_cvt"]
    elif given_units:
        entry["units"] = given_units
    for key, flat in (("accumulation", "ref_model_accumulation"), ("daily_boundary", "ref_model_daily_boundary")):
        if config.get(flat):
            entry[key] = config[flat]
    entry = {k: v for k, v in entry.items() if v is not None}
    res = detect.resolve_entry(entry, "forecast", sources={}, detect_lead_label=False, lead_label_default=True)
    if not config.get("ref_model_accumulation"):
        config["ref_model_accumulation"] = res.values.get("accumulation") or "auto"
        config["ref_model_accumulation_detected"] = any(
            line.key == "accumulation" and line.how.startswith("detected") for line in res.lines)
        sources["ref_model_accumulation"] = "detected" if config["ref_model_accumulation_detected"] else "default"
    report.append(f"reference {config['ref_model']}: accumulation {config['ref_model_accumulation']}"
                  + (f", days cut at {config['ref_model_daily_boundary']}" if config.get("ref_model_daily_boundary") else ""))


def _check_blend(config, errors):
    """Build the blend ``Recipe`` from the section as far as the config says, so its problems are reported with the rest.

    Parameters
    ----------
    config : collections.abc.Mapping
        The flat config being built.
    errors : list of str
        Takes one ``blend: ...`` message per recipe problem.

    Returns
    -------
    None
        Nothing; without a ``blend:`` section or models there is nothing to check.
    """
    if not schema.has_blend(config) or "model_list" not in config:
        return
    from hcwf.blend.recipe import recipe_from_config
    from hcwf.errors import RecipeError

    try:
        recipe_from_config(config)
    except RecipeError as exc:
        errors.extend(f"blend: {problem}" for problem in exc.problems)


def _derive_blend_years(config, sources):
    """Derive the benchmark's dates and climatology years of a blend run from ``blend.years``.

    A blend run names its years in ``blend.years``, so the benchmark's dates and climatology years it does not use are
    taken from them when not given: the dates span the first to the last year, and so do the climatology years.

    Parameters
    ----------
    config : dict
        The flat config being built, changed in place (``start_date``, ``end_date``, ``start_year_clim``,
        ``end_year_clim`` when not given).
    sources : dict
        The flat key to its source text, changed in place.

    Returns
    -------
    None
        Nothing; a config without a ``blend:`` section or without years is left alone.
    """
    if not schema.has_blend(config):
        return
    from hcwf.blend.recipe import years_of

    given = config.get("blend_years")
    values = given.values() if isinstance(given, dict) else [given]
    years = sorted({y for value in values if value is not None for y in years_of(value, "blend.years", [])})
    if not years:
        return
    for key, value in (("start_date", (years[0], 1, 1)), ("end_date", (years[-1], 12, 31)),
                       ("start_year_clim", years[0]), ("end_year_clim", years[-1])):
        if key not in config:
            config[key] = value
            sources[key] = "blend.years"


def _derive(config, sources, errors):
    """Fill what follows from other values: climatology years, ``date_filter_year``, reference data.

    Parameters
    ----------
    config : dict
        The flat config being built, changed in place.
    sources : dict
        The flat key to its source text, changed in place.
    errors : list of str
        Takes the message of a model reference without its required keys.
    """
    if "start_year_clim" not in config and "end_year_clim" not in config:
        years = next((r.climatology_years for r in _region_chain(config) if r.climatology_years), None)
        origin = "region default"
        if not years and config.get("threshold_years"):
            years, origin = config["threshold_years"], "onset.threshold_years"
        if years:
            config["start_year_clim"], config["end_year_clim"] = years
            sources["start_year_clim"] = sources["end_year_clim"] = origin
    _derive_blend_years(config, sources)
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
    """Take the region from the data extent when it is not given; warn if a given region disagrees with it.

    Parameters
    ----------
    config : dict
        The flat config being built, changed in place (``region``).
    sources : dict
        The flat key to its source text, changed in place.
    res : hcwf.config.detect.Resolution or None
        The resolution of the observations; its ``extent`` is the data's extent.
    report : list of str
        Takes the line about the detected region or the warning.
    errors : list of str
        Takes the message when the region is not given and cannot be detected.

    Returns
    -------
    None
        Nothing; without observations or an extent there is nothing to detect.
    """
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
    """Resolve and validate a nested run config given as a Python mapping.

    Every problem found is listed in one ``ConfigSchemaError``, whose ``report`` holds the detection report lines.

    Parameters
    ----------
    raw : dict or None
        The nested run config (as the YAML); None is an empty config.
    source : str, default "<config>"
        What the config is called in error messages (the file's path).

    Returns
    -------
    dict
        The flat config, with ``dataset_report`` (the detection report lines) and ``config_sources`` (flat key to where
        its value came from) added.

    Raises
    ------
    ConfigSchemaError
        ``raw`` is not a mapping, or any value is invalid or missing.
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
    for key, kind in (("dataset", "str"), ("units", "str"), ("unit_cvt", "number"), ("dims", "dims")):
        for block in ("observations", "reference"):
            if key in special[block]:
                field = schema.Field(key, kind, key)
                special[block][key] = schema.normalise_value(field, special[block][key], f"{block}.{key}", errors)
    try:
        datasets = load_datasets(config.get("dataset_files", ()), config.get("catalog_dirs", ()))
    except DatasetError as exc:
        errors.append(str(exc))
        datasets = {}

    obs_res = _resolve_observations(config, special["observations"], datasets, errors)
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
        _finish_observations(obs_res, config, sources)
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
    if config.get("mask") and config.get("nc_mask"):
        errors.append("'mask' and 'nc_mask' are both set. Fix: use one; `nc_mask: file.nc` is the old flat key "
                      "(it also applies the land and country masks implicitly), `mask: {file: file.nc}` the new section")
    _derive(config, sources, errors)
    _resolve_reference_settings(config, sources, special["reference"].get("units"), report)
    _check_blend(config, errors)
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
