"""Write a starting run config for ``hcwf-run init``.

The file holds what only the user knows (region, years, datasets). Values HCWF can work out itself (the
region's onset definition, day boundary and climatology years, schema defaults) are NOT copied in; they are
resolved at run time. Dataset values detected from paths you pass with ``--obs``/``--model`` are written, each with
a comment showing the evidence, so you can review and edit them. `render` builds the text, using `hcwf.config.detect`
to read the datasets and `hcwf.onset.regions` for the region; `write_init` checks it loads as a `hcwf.config.Config`
and writes it.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from hcwf.config import detect
from hcwf.config.convert import _flow
from hcwf.config.datasets import expand_path
from hcwf.onset import regions
from hcwf.onset.ingest import UNITS_HELP
from hcwf.errors import RegionError

__all__ = ["write_init"]


_OBS_PLACEHOLDERS = (("file_pattern", '"{}.nc"', "the year placeholder; detected from the files if left out"),
                     ("variable", "rainfall", "rainfall variable; detected if it is the only one"),
                     ("units", "mm", f"{UNITS_HELP}; read from the file's attribute if left out"),
                     ("day_boundary", '"00 UTC, labelled by start"',
                      "when the day starts and which date labels it; the region's if left out"))
"""The ``(key, example value, comment)`` lines of an observations entry that detection could not fill."""
_MODEL_PLACEHOLDERS = (("file_pattern", '"{}.nc"', "the year placeholder; detected from the files if left out"),
                       ("variable", "tp", "rainfall variable; detected if it is the only one"),
                       ("units", "m", f"{UNITS_HELP}; read from the file's attribute if left out"),
                       ("lead_label", "start", "start: lead k is dated init + k - 1; end: init + k. "
                                               "Detected against the observations if left out"))
"""The ``(key, example value, comment)`` lines of a model entry that detection could not fill."""


def _season_bounds(region_name: str | None):
    """Return the season of a region's default onset definition, or the demo's May to July.

    Any failure to load the region or its definition falls back to the demo window. A season that wraps into the next
    year, or a derived one, also falls back.

    Parameters
    ----------
    region_name : str or None
        The region, or None for the demo window.

    Returns
    -------
    tuple of (tuple of (int, int), tuple of (int, int), str or None)
        The ``(month, day)`` of the season's start and of its end, and the id of the region's default definition
        (None for the demo window).
    """
    from hcwf.onset import definition as definition_mod
    from hcwf.onset.model import DerivedSeasons

    try:
        default = regions.definitions_for(region_name)[0]
        seasons = definition_mod.load(default).seasons
        if not isinstance(seasons, DerivedSeasons):
            s = seasons[0]
            if (s.end.month, s.end.day) > (s.start.month, s.start.day):
                return (s.start.month, s.start.day), (s.end.month, s.end.day), default
    except Exception:  # noqa: BLE001 - any failure falls back to the default window
        pass
    return (5, 1), (7, 31), None


def _box_lines(extent) -> list[str]:
    """Return the ``box:`` line of a config whose files match no registered region.

    The files' extent is written as the box, except for files that cover all longitudes: reading the globe is not what a
    run should do by default, so those get a placeholder to fill in (``box: all`` is the explicit whole domain).

    Parameters
    ----------
    extent : tuple of (float, float, float, float)
        The observations' extent ``(lat0, lat1, lon0, lon1)`` in degrees.

    Returns
    -------
    list of str
        The YAML lines.
    """
    la0, la1, lo0, lo1 = (round(float(v), 4) for v in extent)
    if lo1 - lo0 >= 359.0:
        return ["box: {lat_min: <south>, lat_max: <north>, lon_min: <west>, lon_max: <east>}  # TODO: the area to read; "
                f"the files are global (lat {la0:g}..{la1:g}); `box: all` reads the whole files"]
    return [f"box: {{lat_min: {la0:g}, lat_max: {la1:g}, lon_min: {lo0:g}, lon_max: {lo1:g}}}  # [the files' extent]; "
            "narrow it to the area you want"]


def _entry_lines(res: detect.Resolution, placeholders, indent: str) -> list[str]:
    """Return the YAML lines of one dataset: detected or specified values with evidence, TODOs for what cannot be detected.

    Parameters
    ----------
    res : hcwf.config.detect.Resolution
        The detection result of the dataset.
    placeholders : tuple of tuple of (str, str, str)
        The ``(key, example, comment)`` lines of the dataset kind, see `_OBS_PLACEHOLDERS`.
    indent : str
        The indentation of each line.

    Returns
    -------
    list of str
        One line per resolved key (with a ``# [how: evidence]`` comment), then a commented ``TODO`` line per problem.
    """
    lines = []
    shown = set()
    for line in res.lines:
        if line.key == "path" or line.key in shown:
            continue
        shown.add(line.key)
        note = f"{line.how}" + (f": {line.evidence}" if line.evidence else "")
        lines.append(f"{indent}{line.key}: {_flow(line.value)}  # [{note}]")
    for key, message in res.problems.items():
        name = key.split(".")[0]
        example = next((e for k, e, _ in placeholders if k == name), "...")
        lines.append(f"{indent}# {name}: {example}  # TODO: {message}")
    return lines


def _blend_lines(names, years) -> list[str]:
    """Return the ``blend:`` skeleton of a starting config: years, one formula and the models' rain features.

    These are the lines ``hcwf-run convert`` writes from an ``onset_blending`` config.

    Parameters
    ----------
    names : list of str
        The model names (the first one names the formula's feature; ``MODEL`` when empty).
    years : tuple of (int, int)
        The first and last year; the training and cross-validation years end the year before the last.

    Returns
    -------
    list of str
        The YAML lines.
    """
    lead = names[0] if names else "MODEL"
    lines = ["blend:  # trains an onset blend instead of benchmarking; `hcwf-run convert blend.yml` writes one from an "
             "onset_blending config",
             f"  years: {{training: '{years[0]}:{years[1] - 1}', cv_holdout: '{years[0]}:{years[1] - 1}', "
             f"true_holdout: {years[1]}}}  # TODO: the years to fit, to cross-validate and to hold out",
             f"  formulas: {{blended_model: 'outcome ~ prob_clim_qx + diff_{lead}_qx'}}  # features: prob_clim_qx, "
             "<model>_p_onset_qx, diff_<model>_qx, max_<model>_<n>day_qx ...",
             "  models:  # per model: lead days read, member limit, rain horizon, rain predictors",
             ] + [f"    {name}: {{lead_days: [1, 40], rain_predictors: [{{agg: diff}}]}}" for name in names or [lead]]
    return lines


def _anchored(path, base) -> str:
    """Return a folder path as the written config must spell it: relative to the config file's folder.

    A config resolves its relative paths from its own folder, so a path given relative to the working directory is
    rewritten when the config is written elsewhere. Absolute paths, ``~`` and ``${ENV}`` paths stay as they are.

    Parameters
    ----------
    path : str or os.PathLike
        The path as the user gave it (relative to the working directory).
    base : str or os.PathLike or None
        The folder the config is written to; None leaves the path as it is.

    Returns
    -------
    str
        The path, relative to ``base`` when it was relative to the working directory.
    """
    text = str(path)
    if base is None or "$" in text or text.startswith("~") or os.path.isabs(text):
        return text
    return os.path.relpath(os.path.abspath(text), base)


def render(region=None, obs=None, models=(), years=(2019, 2024), definition=None, blend=False, base=None) -> str:
    """Return the text of a starting run config.

    Parameters
    ----------
    region : str, optional
        The region; detected from the observations' extent if None and they are given.
    obs : str, optional
        The folder of the observations; None writes placeholders.
    models : iterable of str, default ()
        The folders of the forecast models; an empty one (or a None entry) writes a placeholder model.
    years : tuple of (int, int), default (2019, 2024)
        The first and last year evaluated.
    definition : str, optional
        An onset definition (registry id or path) to write; None writes a comment saying the region's is used.
    blend : bool, default False
        Add a ``blend:`` section skeleton.
    base : str or os.PathLike, optional
        The folder the config is written to. Folders given relative to the working directory are written relative to
        it (a config resolves its relative paths from its own folder); None writes them as given.

    Returns
    -------
    str
        The YAML text, with a comment on each detected value and a ``TODO`` on what is left to fill in.
    """
    obs_res = None
    obs_lines: list[str]
    if obs:
        obs_res = detect.resolve_entry({"path": obs}, "observations")
        obs_lines = [f"  path: {_flow(_anchored(obs, base))}"] + _entry_lines(obs_res, _OBS_PLACEHOLDERS, "  ")
    else:
        obs_lines = ["  path: /path/to/observations  # TODO: folder of the yearly observation files"] + [
            f"  # {k}: {example}  # {text}" for k, example, text in _OBS_PLACEHOLDERS]
    region_line = None
    area_lines: list[str] = []
    if region:
        region_line = f"region: {regions.load_region(region).title}"
    elif obs_res is not None and obs_res.extent:
        found, evidence = detect.detect_region(obs_res.extent)
        if found != "unknown":
            region_line = f"region: {regions.load_region(found).title}  # [detected: {evidence}]"
        else:   # no registered region fits the files: no region, and the area stated by a box
            region_line = f"region: null  # [no registered region matches: {evidence}]"
            area_lines = _box_lines(obs_res.extent)
    region_line = region_line or "region: <region>  # TODO: your region, see `hcwf-run onset regions`"
    no_region = region_line.startswith("region: null")
    region_name = None if no_region else region_line.split(":")[1].split("#")[0].strip()
    try:
        regions.load_region(region_name) if region_name else None
    except RegionError:
        region_name = None
    start, end, default = _season_bounds(region_name)
    obs_for_labels = dict(obs_res.values) if obs_res is not None else None
    if obs_for_labels is not None and not obs_for_labels.get("day_boundary"):
        boundary = next((r.day_boundary for r in ([regions.load_region(region_name), *regions.ancestors(
            regions.load_region(region_name))] if region_name else []) if r.day_boundary), None)
        obs_for_labels["day_boundary"] = boundary
        if boundary and obs_res is not None:
            obs_res.problems.pop("day_boundary", None)
            obs_lines = [line for line in obs_lines if "day_boundary" not in line]
            obs_lines.append(f"  # day_boundary: from the region ({boundary}); set it here to override")
    model_blocks, probabilistic = [], False
    for i, path in enumerate(models or [None]):
        if path:
            res = detect.resolve_entry({"path": path}, "forecast", obs=obs_for_labels)
            name = Path(expand_path(path)).name or f"MODEL{i + 1}"
            block = [f"  - name: {_flow(name)}", f"    path: {_flow(_anchored(path, base))}"] + _entry_lines(res, _MODEL_PLACEHOLDERS, "    ")
            probabilistic = probabilistic or (res.values.get("members") or 1) > 1
        else:
            block = ["  - name: MODEL  # TODO: model name used in output files (no spaces)",
                     "    path: /path/to/forecasts  # TODO: folder of the yearly forecast files"] + [
                f"    # {k}: {example}  # {text}" for k, example, text in _MODEL_PLACEHOLDERS]
        model_blocks.append("\n".join(block))
    out = ["# HCWF run config. Edit, then run:  hcwf-run -p run.yaml",
           "# Only what you must decide is here. The region supplies the onset definition, day boundary and",
           "# climatology years; unset keys take their defaults. Each run writes resolved_config.yaml (every value",
           "# and where it came from) next to its outputs. Keys: see hcwf/config/schema.py.",
           "", region_line] + area_lines + [""]
    if no_region:
        out[1:4] = ["# Only what you must decide is here. Without a region nothing is defaulted: set the area (box), the",
                    "# onset definition, the season and the climatology years yourself (the TODO lines). Each run writes",
                    "# resolved_config.yaml (every value and where it came from) next to its outputs. Keys: see hcwf/config/schema.py."]
    if definition:
        out += ["onset:", f"  definition: {definition}", ""]
    elif no_region:
        out += ["onset:",
                "  # definition: <registry id or path>  # TODO: no region, so no default definition; see `hcwf-run onset list`",
                "  # season: <season name>  # TODO: needed when the definition has several seasons",
                ""]
    else:
        out += [f"# onset:  # the region's definition is used (default here: {default or 'none'});",
                "#   definition: <registry id or path>  # choose another, see `hcwf-run onset list`",
                "#   override: {season: {from: 05-15}}  # change part of a definition",
                ""]
    out += ["observations:"] + obs_lines + ["", "models:"] + model_blocks + [
        "", "evaluation:",
        f"  start_date: [{years[0]}, {start[0]}, {start[1]}]  # the years evaluated; month/day bound the init dates",
        f"  end_date: [{years[1]}, {end[0]}, {end[1]}]"]
    if no_region:
        out += [f"  # start_year_clim: {years[0]}  # TODO: the climatology years (a region would supply them)",
                f"  # end_year_clim: {years[1]}"]
    if probabilistic:
        out.append("  probabilistic: true  # ensemble forecasts detected")
    if blend:
        out += [""] + _blend_lines([Path(expand_path(m)).name if m else "MODEL" for m in models] or ["MODEL"], years)
    return "\n".join(out) + "\n"


def write_init(target, force=False, **kwargs) -> Path:
    """Render the starting config and write it to ``target`` (or print it for ``"-"``).

    The text is loaded as a `hcwf.config.Config` first, so a starting config that does not load is never written.

    Parameters
    ----------
    target : str or pathlib.Path
        The file to write; ``"-"`` prints the text to the standard output.
    force : bool, default False
        Overwrite ``target`` when it exists.
    **kwargs : dict
        Passed on to `render`.

    Returns
    -------
    pathlib.Path
        ``target``, or ``Path("-")`` when the text was printed.

    Raises
    ------
    FileExistsError
        ``target`` exists and ``force`` is false.
    """
    from hcwf.config import Config

    base = None if str(target) == "-" else os.path.dirname(os.path.abspath(target))
    text = render(**kwargs, base=base)
    Config.from_mapping(yaml.safe_load(text))
    if str(target) == "-":
        print(text, end="")
        return Path("-")
    target = Path(target)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists. Pass --force to overwrite it")
    target.write_text(text)
    return target
