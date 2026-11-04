"""Run HCWF from Python: `run` returns the results as xarray and pandas objects.

This module implements the functions that `hcwf` re-exports (`run`, `check`) and the application of a trained blend
(`forecast`)::

    import hcwf
    result = hcwf.run("run.yaml", max_forecast_day=15)       # a YAML path, a hcwf.config.Config or a mapping
    result.observed          # xarray.Dataset: observed onset dates (year, lat, lon; year, unit on evaluation units)
    result.scores            # {"spatial": {(model, window): Dataset}, "summary": DataFrame}  (deterministic)
                             # {"overall": DataFrame, "binned": DataFrame}                    (ensemble)
    result.forecasts         # {source: Dataset}: gridded onset dates of each model and climatology (computed on use)
    result.folder            # the run folder (None when nothing was written)

    plan = hcwf.check("run.yaml")                              # the same report as `hcwf-run -p run.yaml --check`

    from hcwf.api import forecast
    fc = forecast("run/blend/trained.blend", {"aifs": "live/aifs.nc"})   # a trained blend on live forecasts

A config with a ``blend:`` section trains the blend instead of benchmarking: ``result.blend`` is the
`hcwf.blend.training.TrainingResult` (evaluation tables and the trained blend; ``write=True`` saves its artifacts under
their own names into ``<run folder>/blend/``), and ``observed`` and ``scores`` are empty.

``write`` chooses whether files are written. The default is ``write=False``: scores, figures and onset files are
not written and nothing is created on disk (temporary folders hold what the pipeline needs and are deleted;
the observed-onset cache lives in memory), which suits notebooks. ``write=True`` writes the run exactly as
``hcwf-run`` does (run folder, or the config's ``output.data_dir``); ``overwrite=True`` re-runs a run folder that exists.
Further keyword arguments override config values by their flat names (the names of `hcwf.config.schema`,
for example ``verification_window_list``, ``max_forecast_day``, ``start_date``, ``probabilistic``).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import xarray as xr

from hcwf.cache import run_scope
from hcwf.config.model import RunConfig
from hcwf import log as logs

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """What a run returns: observed onsets, scores, the run folder, and the onset forecasts on request.

    The `forecasts` property is computed the first time it is read.
    """

    observed: xr.Dataset | None
    """Observed onsets: variable ``onset_date`` over ``year, lat, lon`` (attribute ``fingerprint``), or None."""
    scores: dict
    """The scores as pandas and xarray objects (see the module text)."""
    folder: Path | None
    """The run folder, or None when ``write=False``."""
    config: dict = field(repr=False, default_factory=dict)
    """The resolved flat config dict."""
    _forecasts: dict | None = field(default=None, repr=False)
    blend: object | None = None
    """The `hcwf.blend.training.TrainingResult` of a run with a ``blend:`` section, else None."""
    run_config: RunConfig | None = field(repr=False, default=None)
    """The `hcwf.config.model.RunConfig` the run was made from."""

    @property
    def forecasts(self) -> dict:
        """The gridded onset dates of every model and climatology reference, as ``{source: xarray.Dataset}``.

        Computed on first use with `hcwf.runs.export` (the scoring's own detection, repeated) and kept.
        """
        if self._forecasts is None:
            from hcwf.runs import export

            rc = self.run_config
            with run_scope():
                self._forecasts = {source: export._collect(rc, source) for source in export.default_sources(rc)[1:]}
        return self._forecasts


def check(cfg, overwrite: bool = False, **overrides):
    """Return the plan of a run as a `hcwf.runs.check.CheckReport` (``.text()``, ``.errors``, ``.warnings``, ``.ok``); run nothing.

    Parameters
    ----------
    cfg : str or pathlib.Path or hcwf.config.Config or dict
        A run YAML path, a `hcwf.config.Config`, a nested mapping (as the YAML) or a flat config dict.
    overwrite : bool, default False
        Plan as if ``--overwrite`` were given (an existing run folder is then not an error).
    **overrides : dict
        Flat config keys that override values of ``cfg``.

    Returns
    -------
    hcwf.runs.check.CheckReport
        The settings, datasets, hints, errors and warnings of the run.

    Raises
    ------
    HCWFConfigError
        The config cannot be resolved; the message lists every problem.
    """
    from hcwf.config.build import resolve
    from hcwf.errors import HCWFConfigError
    from hcwf.runs.check import build_report

    rc, found = resolve(cfg, overrides, check=True)
    if rc is None:
        raise HCWFConfigError(found.message())
    with run_scope(), logs.session(logs.QUIET):
        return build_report(rc, found, overwrite=overwrite)


def run(cfg, *, write: bool = False, overwrite: bool = False, quiet: bool = False, verbose: bool = False,
        **overrides) -> RunResult:
    """Run HCWF on a config and return a `RunResult` (see the module text).

    Parameters
    ----------
    cfg : str or pathlib.Path or hcwf.config.Config or dict
        A run YAML path, a `hcwf.config.Config`, a nested mapping (as the YAML) or a flat config dict.
    write : bool, default False
        False writes no files; True writes the run as the command line does and refuses a run folder that already holds
        a run unless ``overwrite`` is True.
    overwrite : bool, default False
        Re-run a run folder that exists, clearing it first (with ``write=True``).
    quiet : bool, default False
        Print only warnings.
    verbose : bool, default False
        Also print debug lines.
    **overrides : dict
        Flat config keys that override values of ``cfg``.

    Returns
    -------
    RunResult
        The observed onsets, scores and run folder (or the blend, for a config with a ``blend:`` section).
    """
    from hcwf.config.build import resolve

    tmp = None if write else Path(tempfile.mkdtemp(prefix="hcwf_"))
    try:
        with run_scope():   # the config's detections are part of the run's state too
            rc = resolve(cfg, overrides, write=write, tmp=tmp)
            return execute(rc, write=write, overwrite=overwrite, verbosity=logs.verbosity_from(quiet, verbose))
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def _observed_dataset(rc: RunConfig) -> xr.Dataset | None:
    """Return the observed onsets of this run (from the observed-onset cache) as one Dataset.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.

    Returns
    -------
    xarray.Dataset or None
        Variable ``onset_date`` over ``year`` and the cache's dims; None when nothing was cached.
    """
    from hcwf.io.output import with_unit_labels
    from hcwf.onset import observed_cache

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
    return with_unit_labels(ds, rc.observations.units)


def _scores(results, probabilistic_tables) -> dict:
    """Collect the scores of a run as xarray and pandas objects.

    Parameters
    ----------
    results : dict or None
        The deterministic metrics by model and window, from `hcwf.scoring.deterministic.spatial_far_mr_mae_map`.
    probabilistic_tables : dict or None
        The ensemble tables (``overall`` and ``binned``), when the run is probabilistic.

    Returns
    -------
    dict
        The ensemble tables as given, else ``{"spatial": {(model, window): Dataset}, "summary": DataFrame}``.
    """
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


def _execute_blend(rc: RunConfig, *, write: bool, overwrite: bool, verbosity: int) -> RunResult:
    """Train the blend a config's ``blend:`` section describes (`hcwf.blend.training.train`).

    The run folder is prepared as for a benchmark (refused when it exists unless ``overwrite``), run.log and the resolved
    config are written, and the TrainingResult is saved into the run folder's ``blend/`` subfolder (artifact names are
    kept).

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config, with ``blend`` set.
    write : bool
        Write the run folder, run.log and the artifacts.
    overwrite : bool
        Clear an existing run folder first.
    verbosity : int
        The console verbosity (see `hcwf.log.session`).

    Returns
    -------
    RunResult
        With ``blend`` set to the training result and empty ``scores``.
    """
    from hcwf.runs import folder as run_folder
    from hcwf.runs.resolved import definition_lines, write_resolved
    from hcwf.log import print_hcwf_banner

    from hcwf.blend.training import train

    paths, recipe = rc.paths, rc.blend
    if write:
        run_folder.prepare(paths, overwrite)
    with logs.session(verbosity, paths.log_file if write else None):
        print_hcwf_banner(rc.project_name)
        log.info("Blend: %s (%s); models %s", recipe.name, "writes into " + str(paths.blend) if write else "nothing is written",
                 ", ".join(recipe.model_names))
        for line in (*rc.onset.lines(), *rc.report, *definition_lines(rc.onset)):
            log.info(f"  {line}")
        if write:
            try:
                log.info(f"  Resolved config  : {write_resolved(rc)[0]}")
            except OSError as exc:
                log.warning(f"could not write {paths.config_name}: {exc}")
        result = train(rc)
        if write:
            chosen = {"observations": "observed", "features": "weekly"}   # the recipe's debug outputs that exist
            extra = {"intermediates": tuple(chosen[n] for n in recipe.outputs.intermediates if n in chosen)} \
                if recipe.outputs.intermediates else {}
            result.save(paths.blend, **extra)
            log.info(f"  Blend artifacts  : {paths.blend}")
        log.info("HCWF blend completed successfully!")
    return RunResult(observed=None, scores={}, folder=paths.root if write else None, config=dict(rc.values),
                     blend=result, run_config=rc)


def execute(rc: RunConfig, *, write: bool, overwrite: bool, verbosity: int = logs.NORMAL) -> RunResult:
    """Run the benchmark (or train the blend) for a resolved RunConfig and return the `RunResult`.

    This is what ``hcwf-run`` and `hcwf.run` share: prepare the run folder (refusing or clearing an existing
    one), attach run.log, print the settings, write the resolved config, optionally explain cells and export the
    onsets, activate the observed-onset cache, then score.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config, from `hcwf.config.build.resolve`.
    write : bool
        Write the run folder and its files.
    overwrite : bool
        Clear an existing run folder first.
    verbosity : int, default ``NORMAL``
        The console verbosity (see `hcwf.log.session`).

    Returns
    -------
    RunResult
        The result of the benchmark or blend training.
    """
    with run_scope():   # the run's state starts empty and is cleared at the end, whichever flow runs
        if rc.blend is not None:
            return _execute_blend(rc, write=write, overwrite=overwrite, verbosity=verbosity)
        return _execute_benchmark(rc, write=write, overwrite=overwrite, verbosity=verbosity)


def _execute_benchmark(rc: RunConfig, *, write: bool, overwrite: bool, verbosity: int) -> RunResult:
    """Run the benchmark flow of `execute`.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    write : bool
        Write the run folder and its files.
    overwrite : bool
        Clear an existing run folder first.
    verbosity : int
        The console verbosity (see `hcwf.log.session`).

    Returns
    -------
    RunResult
        The observed onsets and scores.
    """
    from hcwf.scoring.probabilistic import skill_score_in_bins
    from hcwf.scoring.deterministic import spatial_far_mr_mae_map
    from hcwf.runs import folder as run_folder
    from hcwf.onset.explain_cells import write_explain_cells
    from hcwf.runs.resolved import definition_lines, write_resolved
    from hcwf.onset import observed_cache
    from hcwf.log import print_hcwf_banner

    paths = rc.paths
    if write:
        run_folder.prepare(paths, overwrite)
    with logs.session(verbosity, paths.log_file if write else None):
        print_hcwf_banner(rc.project_name)
        log.info("Onset settings:")
        for line in rc.onset.lines():
            log.info(f"  {line}")
        for line in rc.lead_lines:
            log.info(f"  {line}")
        for line in rc.report:
            log.info(f"  {line}")
        for line in definition_lines(rc.onset):
            log.info(f"  {line}")
        if write:
            log.info(f"  Run folder       : {paths.root}")
            try:
                config_path = write_resolved(rc)[0]
                log.info(f"  Resolved config  : {config_path}")
            except OSError as exc:
                log.warning(f"could not write {paths.config_name}: {exc}")
        observed_cache.activate(rc.cache_dir or (Path(rc.output.data_dir) / ".onset_cache" if write else None))
        if write:
            write_explain_cells(rc)
            if rc.output.save_onsets:
                try:
                    from hcwf.runs.export import export_onsets

                    export_onsets(rc)
                except Exception as exc:  # noqa: BLE001 - the scores matter more than the export
                    log.error(f"output.save_onsets: could not write the onset files: {exc}")
                    log.debug(traceback.format_exc())
        log.info("Starting HCWF Workflow...")
        tables = skill_score_in_bins(rc)
        results = spatial_far_mr_mae_map(rc)
        if write and rc.area.mask is not None:
            try:
                write_resolved(rc)   # again, now that the masks of each dataset's grid are known
            except OSError as exc:
                log.warning(f"could not update {paths.config_name} with the masks: {exc}")
        log.info("HCWF Workflow completed successfully!")
        observed = _observed_dataset(rc)
    return RunResult(observed=observed, scores=_scores(results, tables), folder=paths.root if write else None,
                     config=dict(rc.values), run_config=rc)


def forecast(blend, forecasts, *, issue_date=None, fewer_members=False, output=None, write=False,
             netcdf=False):
    """Apply a trained blend to live forecast files and return the `hcwf.blend.forecast.Forecast`.

    ::

        fc = forecast("run/blend/trained.blend", {"aifs": "live/aifs_2026-06-09.nc"})
        fc.weekly           # xarray.Dataset of blended week probabilities over (time, id)
        fc.weekly_table     # the same as blending's contract table (id, time, onset_threshold, lat, lon, cv_*, clim_*)

    The live files may hold more than the blend was trained on (lead days, members, a larger domain), never less, and
    must have its units, accumulation, grid spacing and issue hour (`hcwf.errors.IncompatibleForecast` lists every
    difference).

    Parameters
    ----------
    blend : str or pathlib.Path or hcwf.blend.trained.TrainedBlend
        A ``trained.blend`` path (hcwf's format, or onset_blending's) or a trained blend.
    forecasts : dict of str to (str or list of str)
        Maps each trained model to a file, a glob or a list of them.
    issue_date : str or datetime.date, optional
        Picks one issue of files holding several.
    fewer_members : bool, default False
        Let an ensemble hold only some of the trained members.
    output : str or pathlib.Path, optional
        The folder the forecast is saved into with ``write``; default ``forecasts/<issue date>/`` beside the blend file
        (``Forecast.folder``).
    write : bool, default False
        Save the forecast; the CSVs are the product.
    netcdf : bool, default False
        With ``write``, add a NetCDF twin of the probabilities.

    Returns
    -------
    hcwf.blend.forecast.Forecast
        The blended week probabilities for the issue date.

    Raises
    ------
    ValueError
        ``write`` is True without ``output`` for a blend that was not read from a file.
    """
    from hcwf.blend.trained import TrainedBlend

    path = None if isinstance(blend, TrainedBlend) else Path(blend)
    with run_scope():
        trained = blend if path is None else TrainedBlend.load(path)
        result = trained.forecast(forecasts, issue_date=issue_date, fewer_members=fewer_members)
        if write:
            if output is None:
                if path is None:
                    raise ValueError("give output: a TrainedBlend that was not read from a file has no folder to write beside")
                output = path.resolve().parent / "forecasts" / result.label
            result.save(output, netcdf=netcdf)
        return result
