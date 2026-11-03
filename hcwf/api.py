"""Run HCWF from Python: ``hcwf.run(cfg)`` returns the results as xarray and pandas objects.

    import hcwf
    result = hcwf.run("run.yaml", max_forecast_day=15)       # a YAML path, a hcwf.config.Config or a mapping
    result.observed          # xarray.Dataset: observed onset dates (year, lat, lon; year, unit on evaluation units)
    result.scores            # {"spatial": {(model, window): Dataset}, "summary": DataFrame}  (deterministic)
                             # {"overall": DataFrame, "binned": DataFrame}                    (ensemble)
    result.forecasts         # {source: Dataset}: gridded onset dates of each model and climatology (computed on use)
    result.folder            # the run folder (None when nothing was written)

    plan = hcwf.check("run.yaml")                              # the same report as `hcwf-run -p run.yaml --check`

    fc = hcwf.api.forecast("run/blend/trained.blend", {"aifs": "live/aifs.nc"})   # a trained blend on live forecasts

A config with a ``blend:`` section trains the blend instead of benchmarking: ``result.blend`` is the
``hcwf.blend.training.TrainingResult`` (evaluation tables and the trained blend; ``write=True`` saves its artifacts under
their own names into ``<run folder>/blend/``), and ``observed`` and ``scores`` are empty.

``write`` chooses whether files are written. The default is ``write=False``: scores, figures and onset files are
not written and nothing is created on disk (temporary folders hold what the pipeline needs and are deleted;
the observed-onset cache lives in memory), which suits notebooks. ``write=True`` writes the run exactly as
``hcwf-run`` does (run folder, or the config's data_dir); ``overwrite=True`` re-runs a run folder that exists.
Further keyword arguments override config values by their flat names (the names of hcwf.lib.config_schema,
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

import pandas as pd
import xarray as xr

from hcwf.config.model import RunConfig
from hcwf.utils import log as logs

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """What a run returns: observed onsets, scores, the run folder, and the onset forecasts on request.

    observed is an xarray.Dataset (variable onset_date over year, lat, lon; attribute fingerprint) or None.
    scores holds pandas / xarray objects (see the module text). folder is the run folder, or None when
    write=False. config is the resolved flat config dict and run_config the RunConfig it belongs to. forecasts is
    computed the first time it is read.
    blend is the TrainingResult of a run with a `blend:` section (None otherwise).
    """

    observed: xr.Dataset | None
    scores: dict
    folder: Path | None
    config: dict = field(repr=False, default_factory=dict)
    _forecasts: dict | None = field(default=None, repr=False)
    blend: object | None = None
    run_config: RunConfig | None = field(repr=False, default=None)

    @property
    def forecasts(self) -> dict:
        """{source: xarray.Dataset} of the gridded onset dates of every model and climatology reference.

        Computed on first use with hcwf.export (the scoring's own detection, repeated) and kept.
        """
        if self._forecasts is None:
            from hcwf import export

            rc = self.run_config
            self._forecasts = {source: export._collect(rc, source) for source in export.default_sources(self.config)[1:]}
        return self._forecasts


def check(cfg, overwrite: bool = False, **overrides):
    """The plan of a run as a hcwf.lib.check.CheckReport (``.text()``, ``.errors``, ``.warnings``, ``.ok``); runs nothing."""
    from hcwf.config.build import resolve
    from hcwf.lib.assertion import HCWFConfigError
    from hcwf.lib.check import build_report

    rc, found = resolve(cfg, overrides, check=True)
    if rc is None:
        raise HCWFConfigError(found.message())
    with logs.session(logs.QUIET):
        return build_report(rc, found, overwrite=overwrite)


def run(cfg, *, write: bool = False, overwrite: bool = False, quiet: bool = False, verbose: bool = False,
        **overrides) -> RunResult:
    """Run HCWF on a config and return a RunResult (see the module text).

    cfg is the path of a run YAML, a hcwf.config.Config, a nested mapping (as the YAML) or a flat config dict.
    write=False (default) writes no files; write=True writes the run as the command line does and refuses a run
    folder that already holds a run unless overwrite=True. quiet and verbose set how much is printed.
    Other keyword arguments override flat config keys.
    """
    from hcwf.config.build import resolve

    tmp = None if write else Path(tempfile.mkdtemp(prefix="hcwf_"))
    try:
        rc = resolve(cfg, overrides, write=write, tmp=tmp)
        return execute(rc, write=write, overwrite=overwrite, verbosity=logs.verbosity_from(quiet, verbose))
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def _observed_dataset(dic: dict) -> xr.Dataset | None:
    """The observed onsets of this run (from the observed-onset cache) as one Dataset, or None."""
    from hcwf.io.output import with_unit_labels
    from hcwf.stats import observed_cache

    tables = [(fp, t) for fp, t in observed_cache.current().items() if t["years"]]
    if not tables:
        return None
    fp, table = max(tables, key=lambda item: len(item[1]["years"]))
    years = sorted(table["years"])
    import numpy as np

    dims = tuple(table["dims"])
    ds = xr.Dataset({"onset_date": (("year", *dims), np.stack([table["years"][y] for y in years]))},
                    coords={"year": years, **{d: table["coords"][d] for d in dims}})
    ds["onset_date"].attrs["long_name"] = "observed rainy-season onset date"
    ds.attrs["fingerprint"] = fp
    return with_unit_labels(ds, dic)


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


def _trainer():
    """`hcwf.blend.training.train`, or a clear error when the training module is not in this checkout."""
    from hcwf.lib.assertion import HCWFConfigError

    try:
        from hcwf.blend.training import train
    except ImportError as exc:
        raise HCWFConfigError(f"this config has a `blend:` section, but blend training is not available "
                              f"(hcwf.blend.training could not be imported: {exc}). Fix: install the packages of the "
                              "blend extras, or update hcwf") from None
    return train


def _execute_blend(rc: RunConfig, *, write: bool, overwrite: bool, verbosity: int) -> RunResult:
    """Train the blend a config's `blend:` section describes (`hcwf.blend.training.train(cfg, recipe, setup)`).

    The run folder is prepared as for a benchmark (refused when it exists unless overwrite), run.log and the resolved config
    are written, and the TrainingResult is saved into the run folder's blend/ subfolder (artifact names are kept).
    """
    from hcwf.blend.recipe import recipe_from_config
    from hcwf.config.flat_bridge import to_flat
    from hcwf.lib import run_folder
    from hcwf.lib.onset_config import onset_setup
    from hcwf.lib.resolved import definition_lines, write_resolved
    from hcwf.utils.printing import print_hcwf_banner

    train = _trainer()
    dic = to_flat(rc)
    paths = dic["run_paths"]
    setup = onset_setup(dic)
    recipe = recipe_from_config(dic, setup)
    if write:
        run_folder.prepare(paths, overwrite)
    with logs.session(verbosity, paths.log_file if write else None):
        print_hcwf_banner(rc.project_name)
        log.info("Blend: %s (%s); models %s", recipe.name, "writes into " + str(paths.blend) if write else "nothing is written",
                 ", ".join(recipe.model_names))
        for line in (*onset_setup(dic).lines(), *dic.get("dataset_report", ()), *definition_lines(dic)):
            log.info(f"  {line}")
        if write:
            try:
                log.info(f"  Resolved config  : {write_resolved(rc)[0]}")
            except OSError as exc:
                log.warning(f"could not write {paths.config_name}: {exc}")
        try:
            result = train(dic, recipe, setup)
            if write:
                chosen = {"observations": "observed", "features": "weekly"}   # the recipe's debug outputs that exist
                extra = {"intermediates": tuple(chosen[n] for n in recipe.outputs.intermediates if n in chosen)} \
                    if recipe.outputs.intermediates else {}
                result.save(paths.blend, **extra)
                log.info(f"  Blend artifacts  : {paths.blend}")
        finally:
            from hcwf.stats import observed_cache, parallel

            parallel.shutdown()
            observed_cache.reset()
        log.info("HCWF blend completed successfully!")
    return RunResult(observed=None, scores={}, folder=paths.root if write else None, config=dic, blend=result,
                     run_config=rc)


def execute(rc: RunConfig, *, write: bool, overwrite: bool, verbosity: int = logs.NORMAL) -> RunResult:
    """Run the benchmark for a resolved RunConfig (see hcwf.config.build.resolve) and return the RunResult.

    This is what ``hcwf-run`` and ``hcwf.run`` share: prepare the run folder (refusing or clearing an existing
    one), attach run.log, print the settings, write the resolved config, optionally explain cells and export the
    onsets, activate the observed-onset cache, then score.
    """
    if rc.blend is not None:
        return _execute_blend(rc, write=write, overwrite=overwrite, verbosity=verbosity)
    from hcwf.config.flat_bridge import to_flat
    from hcwf.app.bin_skill_score import skill_score_in_bins
    from hcwf.app.spatial_far_mr_mae import spatial_far_mr_mae_map
    from hcwf.lib import run_folder
    from hcwf.lib.explain_cells import write_explain_cells
    from hcwf.lib.onset_config import onset_setup
    from hcwf.lib.resolved import definition_lines, write_resolved
    from hcwf.stats import benchmark, climatology, detect, observed_cache, parallel
    from hcwf.io import input as io_input
    from hcwf.utils import mask as mask_mod
    from hcwf.utils.printing import print_hcwf_banner

    dic = to_flat(rc)
    paths = dic["run_paths"]
    if write:
        run_folder.prepare(paths, overwrite)
    with logs.session(verbosity, paths.log_file if write else None):
        mask_mod.reset()
        benchmark.reset_onset_tables()
        climatology.reset_memo()
        detect._alignment_done.clear()
        io_input._loaded_thresholds.clear()
        io_input._reported_ingest.clear()
        print_hcwf_banner(rc.project_name)
        log.info("Onset settings:")
        for line in onset_setup(dic).lines():
            log.info(f"  {line}")
        for line in dic.get("onset_lead_lines", ()):
            log.info(f"  {line}")
        for line in dic.get("dataset_report", ()):
            log.info(f"  {line}")
        for line in definition_lines(dic):
            log.info(f"  {line}")
        if write:
            log.info(f"  Run folder       : {paths.root}")
            try:
                config_path = write_resolved(rc)[0]
                log.info(f"  Resolved config  : {config_path}")
            except OSError as exc:
                log.warning(f"could not write {paths.config_name}: {exc}")
        observed_cache.activate(dic.get("cache_dir") or (Path(dic["dir_out"]) / ".onset_cache" if write else None))
        results = tables = None
        try:
            if write:
                write_explain_cells(rc)
                if dic.get("save_onsets"):
                    try:
                        from hcwf.export import export_onsets

                        export_onsets(rc)
                    except Exception as exc:  # noqa: BLE001 - the scores matter more than the export
                        log.error(f"output.save_onsets: could not write the onset files: {exc}")
                        log.debug(traceback.format_exc())
            log.info("Starting HCWF Workflow...")
            tables = skill_score_in_bins(rc)
            results = spatial_far_mr_mae_map(rc)
            if write and dic.get("mask"):
                try:
                    write_resolved(rc)   # again, now that the masks of each dataset's grid are known
                except OSError as exc:
                    log.warning(f"could not update {paths.config_name} with the masks: {exc}")
            log.info("HCWF Workflow completed successfully!")
            observed = _observed_dataset(dic)
        finally:
            parallel.shutdown()
            observed_cache.reset()
            benchmark.reset_onset_tables()
            climatology.reset_memo()
    return RunResult(observed=observed, scores=_scores(results, tables), folder=paths.root if write else None,
                     config=dic, run_config=rc)


def forecast(blend, forecasts, *, issue_date=None, fewer_members=False, output=None, write=False,
             netcdf=False):
    """Apply a trained blend to live forecast files and return the `hcwf.blend.forecast.Forecast`.

        fc = hcwf.api.forecast("run/blend/trained.blend", {"aifs": "live/aifs_2026-06-09.nc"})
        fc.weekly           # xarray.Dataset of blended week probabilities over (time, id)
        fc.weekly_table     # the same as blending's contract table (id, time, onset_threshold, lat, lon, cv_*, clim_*)

    `blend` is a trained.blend path (hcwf's format, or onset_blending's) or a `hcwf.blend.TrainedBlend`; `forecasts`
    maps each trained model to a file, a glob or a list of them. The live files may hold more than the blend was
    trained on (lead days, members, a larger domain), never less, and must have its units, accumulation, grid spacing and
    issue hour (`hcwf.blend.forecast.IncompatibleForecast` lists every difference); `issue_date` picks one issue of files
    holding several, `fewer_members` lets an ensemble hold only some of the trained members. With `write` the forecast
    is saved into `output`, default `forecasts/<issue date>/` beside the blend file (`Forecast.folder`); the CSVs
    are the product, `netcdf` adds a NetCDF twin of the probabilities.
    """
    from hcwf.blend.trained import TrainedBlend

    path = None if isinstance(blend, TrainedBlend) else Path(blend)
    trained = blend if path is None else TrainedBlend.load(path)
    result = trained.forecast(forecasts, issue_date=issue_date, fewer_members=fewer_members)
    if write:
        if output is None:
            if path is None:
                raise ValueError("give output: a TrainedBlend that was not read from a file has no folder to write beside")
            output = path.resolve().parent / "forecasts" / result.label
        result.save(output, netcdf=netcdf)
    return result
