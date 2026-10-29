"""Where a run writes: the run folder, its fingerprint and name, the shared cache, and file naming.

Two modes, decided by the config:

* **explicit** (``output.data_dir`` is written in the run config): files go to that folder (figures to
  ``figure_dir``) with the historical file names, exactly as before; a re-run overwrites. The run's log is
  ``run.log`` there and the onset cache is ``<dir_out>/.onset_cache``.
* **run folder** (``data_dir`` left out): ``runs/`` next to the config file holds one folder per run, named
  ``<region>_<definition>[+mod]_<years>_<fingerprint>`` (at most 40 characters). The 4-character fingerprint
  is a hash of everything that defines the results (not of output folders, plotting or the backend), so a
  changed config gets a new folder and the same config is refused (``RunExistsError``) unless ``overwrite``,
  which clears that folder first. Inside: ``run.yaml`` (the resolved config), ``definition.yaml``, ``run.log``
  and the subfolders ``onsets/``, ``scores/`` and ``figures/``, each only when it would hold more than one
  file (else its files sit in the run folder); file names are short (``spatial_AIFS_1-15.nc``,
  ``onsets/observed.nc``). ``runs/.cache/`` is shared by all runs (thresholds, observed onsets).

``plan`` builds a ``RunPaths`` without touching the disk; ``prepare`` creates (or clears) the folder.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

MAX_NAME = 40
RUN_FILE = "run.yaml"
DEFINITION_FILE = "definition.yaml"
LOG_FILE = "run.log"
_NOT_IN_FINGERPRINT = {
    "dataset_report", "config_sources", "onset_lead_lines", "run_paths", "config_file", "cache_dir", "short_names",
    "run_overrides", "work_dir", "pkg_dir", "dir_out", "dir_fig", "project_name", "debug", "parallel", "onset_backend",
    "show_plot", "show_panel", "save_fig", "save_nc_spatial_far_mr_mae", "save_csv_score", "save_nc_climatology",
    "save_onsets", "plot_spatial_far_mr_mae", "plot_heatmap_bss_auc", "plot_reliability", "plot_climatology_onset",
    "plot_panel_heatmap_error", "plot_panel_heatmap_skill", "plot_bar_bss_rpss_auc", "plot_portrait", "explain_cells",
    "lead_window"}
_CLIMATOLOGY_REFERENCES = ("climatology", "conditional_climatology")


class RunExistsError(FileExistsError):
    """The run folder of this config already holds a run; the message says how to re-run."""


@dataclass(frozen=True)
class RunPaths:
    """The places a run reads and writes. explicit is True when the config named dir_out (historical layout)."""

    explicit: bool
    root: Path
    name: str | None
    fingerprint: str
    dir_out: Path
    dir_fig: Path
    onsets: Path
    cache_dir: Path | None
    runs_dir: Path | None
    short_names: bool
    write: bool = True

    @property
    def log_file(self) -> Path:
        """The run's log file."""
        return self.root / LOG_FILE

    @property
    def config_name(self) -> str:
        """File name of the resolved config."""
        return "resolved_config.yaml" if self.explicit else RUN_FILE

    @property
    def definition_name(self) -> str:
        """File name of the resolved onset definition."""
        return "resolved_definition.yaml" if self.explicit else DEFINITION_FILE

    @property
    def exists(self) -> bool:
        """Whether a run has already been started in this folder (its resolved config is there)."""
        return (self.root / self.config_name).exists()


def _plain(value):
    """A JSON-able form of a config value (tuples to lists, paths and others to text)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def fingerprint(dic: dict) -> str:
    """A 16-hex-character hash of everything in the flat config that defines the results.

    Left out: output folders and switches, plotting, project name, parallel, the backend and the reports the
    loader adds. The onset definition's own hash is included, so editing a definition file changes it.
    """
    from momp.lib.onset_config import onset_setup
    from momp.onset import definition as definition_mod

    body = {k: _plain(v) for k, v in dic.items() if k not in _NOT_IN_FINGERPRINT}
    try:
        body["_definition_hash"] = definition_mod.definition_hash(onset_setup(dic).definition)
    except Exception:  # noqa: BLE001 - a broken config is reported elsewhere; the hash just omits the definition
        body["_definition_hash"] = None
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _clean(text: str) -> str:
    """A name part with only letters, digits, '.', '-' and '_' in lower case."""
    return re.sub(r"[^a-z0-9.\-]+", "-", str(text).lower()).strip("-") or "x"


def run_name(dic: dict, digest: str) -> str:
    """The folder name ``<region>_<definition>[+mod]_<years>_<4 characters of digest>``, at most 40 characters."""
    from momp.lib.onset_config import onset_setup
    from momp.onset import regions

    try:
        region = regions.load_region(str(dic.get("region"))).id
    except Exception:  # noqa: BLE001 - an unknown region is reported by the config checks
        region = _clean(dic.get("region"))
    try:
        setup = onset_setup(dic)
        definition = "legacy" if setup.legacy else setup.definition.id
        modified = bool(setup.overrides) or bool(dic.get("onset_rule"))
    except Exception:  # noqa: BLE001
        definition, modified = "onset", False
    start, end = dic["start_date"][0], dic["end_date"][0]
    years = str(start) if start == end else f"{start}-{end}"
    tail = f"_{years}_{digest[:4]}"
    mod = "+mod" if modified else ""
    room = MAX_NAME - len(tail) - len(mod) - 1
    region, definition = _clean(region), _clean(definition)
    if len(region) + len(definition) > room:
        keep_region = min(len(region), max(room // 2, room - len(definition)))
        region = region[:keep_region]
        definition = definition[:max(room - len(region), 1)]
    return f"{region}_{definition}{mod}{tail}"


def _count_scores(dic: dict) -> int:
    """How many files the run will write to dir_out (an estimate from the switches and the cases)."""
    windows = len(dic.get("verification_window_list") or (None,))
    models = len(dic.get("model_list") or ())
    reference = 1 if dic.get("ref_model") else 0
    if dic.get("probabilistic"):
        return models * windows * 2 * bool(dic.get("save_csv_score"))
    count = (models + reference) * windows * bool(dic.get("save_nc_spatial_far_mr_mae"))
    return count + int(bool(dic.get("save_nc_climatology")) and dic.get("ref_model") == "climatology")


def _count_figures(dic: dict) -> int:
    """How many files the run will write to dir_fig (an estimate: one per enabled plot and case, plus mask maps)."""
    windows = len(dic.get("verification_window_list") or (None,))
    models = len(dic.get("model_list") or ())
    on = lambda key: int(bool(dic.get(key)))   # noqa: E731 - a switch as 0 or 1
    if dic.get("probabilistic"):
        count = (on("plot_heatmap_bss_auc") + on("plot_reliability")) * models * windows \
            + (on("plot_panel_heatmap_skill") + on("plot_bar_bss_rpss_auc")) * windows
    else:
        count = on("plot_spatial_far_mr_mae") * (models + 1) * windows + on("plot_panel_heatmap_error") \
            + int(bool(dic.get("plot_climatology_onset")) and dic.get("ref_model") == "climatology")
    return count + (1 + models if dic.get("mask") else 0)


def count_onsets(dic: dict) -> int:
    """How many onset files a full export writes: observed, each model and a climatology reference."""
    return 1 + len(dic.get("model_list") or ()) + int(dic.get("ref_model") in _CLIMATOLOGY_REFERENCES)


def plan(dic: dict, config_file=None, write: bool = True) -> RunPaths:
    """Decide the run's folders from the flat config (after overrides); nothing is created.

    config_file is the path of the run YAML (``runs/`` goes next to it; the current folder when None).
    write=False (notebook runs that write nothing) puts every output in the returned paths' temporary
    stand-ins, which the caller creates; the cache is then in memory only.
    """
    sources = dic.get("config_sources") or {}
    digest = fingerprint(dic)
    explicit_out = sources.get("dir_out") == "run config" or "config_sources" not in dic   # a bare flat dict names its dir_out
    explicit_fig = sources.get("dir_fig") == "run config"
    if explicit_out:
        root = Path(dic["dir_out"])
        return RunPaths(True, root, None, digest, root, Path(dic["dir_fig"]), root / "onsets", None, None, False, write)
    base = Path(config_file).resolve().parent if config_file else Path.cwd()
    runs_dir = base / "runs"
    name = run_name(dic, digest)
    root = runs_dir / name
    scores = root / "scores" if _count_scores(dic) > 1 else root
    figures = root / "figures" if _count_figures(dic) > 1 else root
    onsets = root / "onsets" if count_onsets(dic) > 1 else root
    return RunPaths(False, root, name, digest, scores, Path(dic["dir_fig"]) if explicit_fig else figures, onsets,
                    runs_dir / ".cache", runs_dir, True, write)


def apply(dic: dict, paths: RunPaths) -> dict:
    """Put the planned folders into the flat config (dir_out, dir_fig, cache_dir, short_names, run_paths)."""
    dic["dir_out"], dic["dir_fig"] = str(paths.dir_out), str(paths.dir_fig)
    dic["cache_dir"] = str(paths.cache_dir) if paths.cache_dir is not None else None
    dic["short_names"] = paths.short_names
    dic["run_paths"] = paths
    return dic


def prepare(paths: RunPaths, overwrite: bool = False) -> None:
    """Create the run's folders. A run folder that already holds a run is refused, or cleared with overwrite.

    Only the run folder itself is cleared (never ``runs/.cache`` or an explicit dir_out).
    """
    if not paths.explicit and paths.exists:
        if not overwrite:
            raise RunExistsError(
                f"this config already ran: {paths.root}. Fix: pass --overwrite to clear that folder and run "
                "again, or change the config (a changed config gets its own folder)")
        if paths.root.parent != paths.runs_dir:
            raise RunExistsError(f"refusing to clear {paths.root}: it is not inside {paths.runs_dir}")
        shutil.rmtree(paths.root)
    for folder in {paths.root, paths.dir_out, paths.dir_fig}:
        folder.mkdir(parents=True, exist_ok=True)
    if paths.cache_dir is not None:
        paths.cache_dir.mkdir(parents=True, exist_ok=True)


def file_name(dic: dict, kind: str, *parts) -> str:
    """The name of an output file of the given kind: short in run folders, historical for an explicit dir_out.

    kinds: ``overall`` and ``binned`` (score CSVs; parts model, window), ``spatial`` (spatial metrics NetCDF;
    case name), ``climatology`` (climatological onset NetCDF; years text), ``onset`` (onset export; source).
    """
    short = bool(dic.get("short_names"))
    if kind == "overall":
        model, window = parts
        return f"overall_{model}_{window}.csv" if short else f"overall_skill_scores_{model}_{window}.csv"
    if kind == "binned":
        model, window = parts
        return f"binned_{model}_{window}.csv" if short else f"binned_skill_scores_{model}_{window}.csv"
    if kind == "spatial":
        return f"spatial_{parts[0]}.nc" if short else f"spatial_metrics_{parts[0]}.nc"
    if kind == "climatology":
        return "climatology_onset.nc" if short else f"climatology_onset_doy_{parts[0]}.nc"
    if kind == "onset":
        return f"{parts[0]}.nc" if short else f"onsets_{parts[0]}.nc"
    raise ValueError(f"unknown file kind {kind!r}")
