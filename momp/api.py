"""Run ROMP from Python: ``momp.run(cfg)`` returns the results as xarray and pandas objects.

    import momp
    result = momp.run("run.yaml", max_forecast_day=15)       # a YAML path, a momp.config.Config or a mapping
    result.observed          # xarray.Dataset: observed onset dates (year, lat, lon), from the observed-onset cache
    result.scores            # {"spatial": {(model, window): Dataset}, "summary": DataFrame}  (deterministic)
                             # {"overall": DataFrame, "binned": DataFrame}                    (ensemble)
    result.forecasts         # {source: Dataset}: gridded onset dates of each model and climatology (computed on use)
    result.folder            # the run folder (None when nothing was written)

    plan = momp.check("run.yaml")                              # the same report as `momp-run -p run.yaml --check`

``write`` chooses whether files are written. The default is ``write=False``: scores, figures and onset files are
not written and nothing is created on disk (temporary folders hold what the pipeline needs and are deleted;
the observed-onset cache lives in memory), which suits notebooks. ``write=True`` writes the run exactly as
``momp-run`` does (run folder, or the config's data_dir); ``overwrite=True`` re-runs a run folder that exists.
Further keyword arguments override config values by their flat names (the names of momp.lib.config_schema,
for example ``verification_window_list``, ``max_forecast_day``, ``start_date``, ``probabilistic``).
"""
from __future__ import annotations

import contextlib
import functools
import logging
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import xarray as xr

from momp.utils import log as logs

log = logging.getLogger(__name__)
_NO_FILES = {"save_fig": False, "save_nc_spatial_far_mr_mae": False, "save_csv_score": False,
             "save_nc_climatology": False, "save_onsets": False, "plot_spatial_far_mr_mae": False,
             "plot_heatmap_bss_auc": False, "plot_reliability": False, "plot_climatology_onset": False,
             "plot_panel_heatmap_error": False, "plot_panel_heatmap_skill": False, "plot_bar_bss_rpss_auc": False,
             "plot_portrait": False, "show_plot": False, "show_panel": False, "explain_cells": None}
"""Config values that turn every file-writing step off (write=False)."""


@dataclass
class RunResult:
    """What a run returns: observed onsets, scores, the run folder, and the onset forecasts on request.

    observed is an xarray.Dataset (variable onset_date over year, lat, lon; attribute fingerprint) or None.
    scores holds pandas / xarray objects (see the module text). folder is the run folder, or None when
    write=False. config is the resolved flat config dict. forecasts is computed the first time it is read.
    """

    observed: xr.Dataset | None
    scores: dict
    folder: Path | None
    config: dict = field(repr=False, default_factory=dict)
    _forecasts: dict | None = field(default=None, repr=False)

    @property
    def forecasts(self) -> dict:
        """{source: xarray.Dataset} of the gridded onset dates of every model and climatology reference.

        Computed on first use with momp.export (the scoring's own detection, repeated) and kept.
        """
        if self._forecasts is None:
            from momp import export

            dic = self.config
            self._forecasts = {source: export._collect(dic, source) for source in export.default_sources(dic)[1:]}
        return self._forecasts


def _flat(source, overrides: dict) -> dict:
    """The flat config of a path, Config, mapping or flat dict, with the keyword overrides applied and checked."""
    from momp.config import Config
    from momp.lib import config_schema as schema
    from momp.lib.config_yaml import build_config

    config_file = None
    if isinstance(source, (str, Path)):
        config_file = str(Path(source).expanduser().resolve())
        flat = Config.from_yaml(config_file).to_flat(source=config_file)
    elif isinstance(source, Config):
        flat = source.to_flat()
    elif isinstance(source, SimpleNamespace):
        flat = dict(vars(source))
    elif isinstance(source, dict) and ("model_list" in source or "obs_dir" in source):
        flat = dict(source)
    elif isinstance(source, dict):
        flat = build_config(source, source="<mapping>")
    else:
        raise TypeError(f"momp.run needs a YAML path, a momp.config.Config or a mapping, got {type(source).__name__}")
    errors: list[str] = []
    for key, value in overrides.items():
        field_ = schema.BY_KEY.get(key)
        if field_ is None or field_.internal:
            raise TypeError(f"unknown override '{key}'.{schema.suggest(key, [k for k in schema.BY_KEY])} Overrides "
                            "are flat config keys, e.g. max_forecast_day, verification_window_list, start_date")
        flat[key] = schema.normalise_value(field_, value, key, errors)
        flat.setdefault("config_sources", {})[key] = "override"
    schema.raise_errors(errors, "<overrides>")
    flat["config_file"] = config_file
    return flat


def _prepare(source, overrides: dict, write: bool, tmp: Path | None) -> dict:
    """The flat config of a run, finished (mode, paths, lead window, run folder plan) and validated."""
    from momp.lib import loader
    from momp.lib.assertion import ROMPValidator

    dic = _flat(source, overrides)
    dic = loader._rename_deprecated_keys(dic)
    dic = loader._set_onset_mode(dic)
    dic = loader._resolve_paths(dic)
    loader.plan_run(dic, dic.get("config_file"), write)
    ROMPValidator(dic).validate()
    if not write:
        for key, value in _NO_FILES.items():
            if key in dic or key == "explain_cells":
                dic[key] = value
        if tmp is not None:
            dic["dir_out"] = dic["dir_fig"] = str(tmp)
            dic["cache_dir"] = None
    return dic


def check(cfg, overwrite: bool = False, **overrides):
    """The plan of a run as a momp.lib.check.CheckReport (``.text()``, ``.errors``, ``.warnings``, ``.ok``); runs nothing."""
    from momp.lib.check import build_report

    with logs.session(logs.QUIET):
        return build_report(_prepare(cfg, overrides, True, None), overwrite=overwrite)


def run(cfg, *, write: bool = False, overwrite: bool = False, quiet: bool = False, verbose: bool = False,
        **overrides) -> RunResult:
    """Run ROMP on a config and return a RunResult (see the module text).

    cfg is the path of a run YAML, a momp.config.Config, a nested mapping (as the YAML) or a flat config dict.
    write=False (default) writes no files; write=True writes the run as the command line does and refuses a run
    folder that already holds a run unless overwrite=True. quiet and verbose set how much is printed.
    Other keyword arguments override flat config keys.
    """
    tmp = None if write else Path(tempfile.mkdtemp(prefix="romp_"))
    try:
        dic = _prepare(cfg, overrides, write, tmp)
        return execute(dic, write=write, overwrite=overwrite, verbosity=logs.verbosity_from(quiet, verbose))
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def _observed_dataset() -> xr.Dataset | None:
    """The observed onsets of this run (from the observed-onset cache) as one Dataset, or None."""
    from momp.stats import observed_cache

    tables = [(fp, t) for fp, t in observed_cache.current().items() if t["years"]]
    if not tables:
        return None
    fp, table = max(tables, key=lambda item: len(item[1]["years"]))
    years = sorted(table["years"])
    import numpy as np

    ds = xr.Dataset({"onset_date": (("year", "lat", "lon"), np.stack([table["years"][y] for y in years]))},
                    coords={"year": years, "lat": table["lat"], "lon": table["lon"]})
    ds["onset_date"].attrs["long_name"] = "observed rainy-season onset date"
    ds.attrs["fingerprint"] = fp
    return ds


def _scores(results, probabilistic_tables) -> dict:
    """The scores of a run as xarray / pandas objects."""
    if probabilistic_tables is not None:
        return dict(probabilistic_tables)
    out: dict = {"spatial": {}, "summary": None}
    rows = []
    for model, windows in (results or {}).items():
        for window, metrics in windows.items():
            name = "-".join(str(w) for w in window) if isinstance(window, tuple) else str(window)
            out["spatial"][(model, name)] = xr.Dataset(dict(metrics))
            rows.append({"model": model, "window": name,
                         **{key: float(metrics[key].mean()) for key in ("mean_mae", "false_alarm_rate", "miss_rate")
                            if key in metrics}})
    out["summary"] = pd.DataFrame(rows)
    return out


def execute(dic: dict, *, write: bool, overwrite: bool, verbosity: int = logs.NORMAL) -> RunResult:
    """Run the benchmark for a finished flat config (see _prepare) and return the RunResult.

    This is what ``momp-run`` and ``momp.run`` share: prepare the run folder (refusing or clearing an existing
    one), attach run.log, print the settings, write the resolved config, optionally explain cells and export the
    onsets, activate the observed-onset cache, then score.
    """
    from momp.app.bin_skill_score import skill_score_in_bins
    from momp.app.spatial_far_mr_mae import spatial_far_mr_mae_map
    from momp.lib import run_folder
    from momp.lib.control import init_dataclass
    from momp.lib.convention import Setting
    from momp.lib.explain_cells import write_explain_cells
    from momp.lib.loader import use_config
    from momp.lib.onset_config import onset_setup
    from momp.lib.resolved import definition_lines, write_resolved
    from momp.stats import detect, observed_cache
    from momp.io import input as io_input
    from momp.utils import mask as mask_mod
    from momp.utils.printing import print_momp_banner

    paths = dic["run_paths"]
    if write:
        run_folder.prepare(paths, overwrite)
    with logs.session(verbosity, paths.log_file if write else None):
        mask_mod.reset()
        detect._alignment_done.clear()
        io_input._loaded_thresholds.clear()
        io_input._reported_ingest.clear()
        cfg = SimpleNamespace(**dic)
        print_momp_banner(cfg)
        log.info("Onset settings:")
        for line in onset_setup(dic).lines():
            log.info(f"  {line}")
        for line in dic.get("onset_lead_lines", ()):
            log.info(f"  {line}")
        for line in dic.get("dataset_report", ()):
            log.info(f"  {line}")
        if not dic.get("onset_legacy", False):
            for line in definition_lines(dic):
                log.info(f"  {line}")
        if write:
            log.info(f"  Run folder       : {paths.root}")
            try:
                config_path = write_resolved(dic)[0]
                log.info(f"  Resolved config  : {config_path}")
            except OSError as exc:
                log.warning(f"could not write {paths.config_name}: {exc}")
        observed_cache.activate(dic.get("cache_dir") or (Path(dic["dir_out"]) / ".onset_cache" if write else None))
        results = tables = None
        try:
            with use_config(dic):
                setting = init_dataclass(Setting, dic)
                if write:
                    write_explain_cells(cfg)
                    if dic.get("save_onsets"):
                        try:
                            from momp.export import export_onsets

                            export_onsets(dic)
                        except Exception as exc:  # noqa: BLE001 - the scores matter more than the export
                            log.error(f"output.save_onsets: could not write the onset files: {exc}")
                            log.debug(traceback.format_exc())
                log.info("Starting ROMP Workflow...")
                tables = skill_score_in_bins(cfg, setting)
                results = spatial_far_mr_mae_map(cfg, setting)
                if write and dic.get("mask"):
                    try:
                        write_resolved(dic)   # again, now that the masks of each dataset's grid are known
                    except OSError as exc:
                        log.warning(f"could not update {paths.config_name} with the masks: {exc}")
                log.info("ROMP Workflow completed successfully!")
            observed = _observed_dataset()
        finally:
            observed_cache.reset()
    return RunResult(observed=observed, scores=_scores(results, tables), folder=paths.root if write else None,
                     config=dic)
