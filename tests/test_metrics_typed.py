"""The typed metrics path (step S8 of docs/review/runconfig_design.md): compute_metrics, prepare_pairs and
create_score_results on a RunConfig and a case.

Serial and parallel runs give the same tables; what a worker is sent is small."""
import typed_io
import pickle

import pandas as pd
import pytest
from hcwf.cache import reset_run_state
from hcwf.config.model import Selection
from hcwf.metrics.skill import create_score_results
from hcwf.scoring.pairs import prepare_pairs
from hcwf.scoring import onset_tables as benchmark
from hcwf import parallel
from hcwf.scoring.onset_tables import compute_metrics
from hcwf.scoring.pairs import _pairs_of_years
from old_flat_rule import old_rule
from test_skill_reference_model import YEARS, DAY_BINS, _write_model, _write_obs
from tests.config_world import configs, world  # noqa: F401  (fixtures: the fixture world and its configs)

LIMIT = 32 * 1024


@pytest.fixture
def config(tmp_path):
    """A flat config (3 years, ensemble model A, model reference B) of synthetic data, without a region (read as is, so
    worker processes need no patched reader)."""
    _write_obs(tmp_path / "obs")
    _write_model(tmp_path / "a", seed=1)
    _write_model(tmp_path / "b", seed=2)
    return dict(
        BS=True, RPS=True, AUC=True, skill_score=True, probabilistic=True, parallel=False,
        model="A", model_dir=str(tmp_path / "a"), model_var="tp", file_pattern="m_{}.nc", unit_cvt=None,
        model_dims={"init_time": "time", "step": "step", "member": "number"},
        ref_model="B", ref_model_dir=str(tmp_path / "b"), ref_model_var="tp",
        ref_model_file_pattern="m_{}.nc", ref_model_unit_cvt=None,
        years=YEARS, years_clim=YEARS, obs_dir=str(tmp_path / "obs"), obs_file_pattern=["obs_{}.nc"],
        obs_var="rain", obs_unit_cvt=None,
        date_filter_year=2000, init_days=(0, 2, 4), dir_out=str(tmp_path / "out"),
        **old_rule(wet_init=1.0, wet_spell=2, dry_spell=1, dry_extent=1, wet_threshold=20.0,
                   start_date=(2000, 5, 1), end_date=(2000, 5, 31)),
        **{"members": None, "onset_percentage_threshold": 0.3, "max_forecast_day": 10},
        day_bins=DAY_BINS, region=None, nc_mask=None, check_alignment=False, grid_point=False,
        tolerance_days=3, verification_window=(1, 10),
    )


def _frames_equal(a, b):
    """Two tables (or dicts of tables, or onset arrays) equal."""
    if isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            _frames_equal(a[key], b[key])
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif a is None:
        assert b is None
    else:
        assert a.equals(b)


@pytest.mark.parametrize("source", ["A", "climatology"])
def test_serial_equals_parallel_on_the_typed_path(config, source):
    """The onset tables (and so the metrics) and the pair tables of worker processes are the serial ones, in year order."""
    probabilistic = source == "A"
    config = dict(config, probabilistic=probabilistic, model=source)
    results = {}
    try:
        for flag in (False, True):
            rc, case, select = typed_io.from_case_kwargs(dict(config, parallel=flag, parallel_workers=2))
            assert rc.parallel.enabled is flag
            reset_run_state()
            metrics, onsets = compute_metrics(rc, case, select)
            pairs = prepare_pairs(rc, case.model, select) if probabilistic else None
            results[flag] = (metrics, onsets, pairs)
    finally:
        parallel.shutdown()
        reset_run_state()
    serial, par = results[False], results[True]
    assert list(serial[0]) == list(par[0]) == list(YEARS)
    _frames_equal(serial[0], par[0])
    _frames_equal(serial[1], par[1])
    if probabilistic:
        for name in ("forecast", "reference"):
            _frames_equal(getattr(serial[2], name), getattr(par[2], name))
        assert len(serial[2].forecast["year"].unique()) == len(YEARS)


@pytest.mark.parametrize("workers", [False, True])
def test_the_skill_pass_hands_its_ensemble_tables_to_the_spatial_metrics(config, monkeypatch, workers):
    """prepare_pairs keeps the ensemble onset tables its detection made (from worker processes too); onset_tables takes them
    instead of reading and detecting the years again, and they equal the tables it would have made."""
    rc, case, select = typed_io.from_case_kwargs(dict(config, parallel=workers, parallel_workers=2))
    reads, original = [], benchmark.forecasts_io.forecast_of
    try:
        reset_run_state()
        prepare_pairs(rc, case.model, select)
        assert len(benchmark._SHARED) == 2   # the model's and its reference model's
        monkeypatch.setattr(benchmark.forecasts_io, "forecast_of", lambda *a, **k: reads.append(a) or original(*a, **k))
        shared, _ = benchmark.onset_tables(rc, case, select, ens=True)
        assert len(benchmark._SHARED) == 1 and not reads   # the model's are taken out, the reference's wait
        benchmark.onset_tables(rc, rc.reference_cases()[0], select, ens=True)
        assert len(benchmark._SHARED) == 0 and not reads   # the reference model's too: the key is the case builder's
        reset_run_state()
        fresh, _ = benchmark.onset_tables(rc, case, select, ens=True)
    finally:
        parallel.shutdown()
        reset_run_state()
    assert list(shared) == list(fresh) == list(YEARS)
    for year in YEARS:
        pd.testing.assert_frame_equal(shared[year][0].reset_index(drop=True), fresh[year][0].reset_index(drop=True))
        assert shared[year][1].equals(fresh[year][1])


def test_scored_bins_are_the_evaluations(config):
    """create_score_results scores the bins of the evaluation spec it is given (a window's bins, in the ensemble flow)."""
    from dataclasses import replace

    rc, case, select = typed_io.from_case_kwargs(config)
    prepared = prepare_pairs(rc, case.model, select)
    narrowed = create_score_results(case, prepared, rc.metrics, replace(rc.evaluation, day_bins=DAY_BINS[:1]))
    assert set(narrowed["forecast_obs_df"]["bin_label"]) == {"Days 1-5"}
    assert len(create_score_results(case, prepared, rc.metrics, rc.evaluation)["forecast_obs_df"]) > \
        len(narrowed["forecast_obs_df"])


def test_worker_payload_is_small(config, configs):  # noqa: F811
    """What a worker is sent for a year (the function, the year, and the run, source and selection) is under 32 KB, for the
    synthetic config and for the fixture world's configs; and the run round-trips."""
    from typed_io import flat_config
    from hcwf.config.model import RunConfig

    runs = [typed_io.from_case_kwargs(config)] + [
        (rc, rc.cases()[0], Selection()) for rc in (RunConfig.from_flat(flat_config(configs[name]), check=False)
                                                    for name in ("two_models_windows", "model_reference"))]
    for rc, case, select in runs:
        pairs_args = (rc, case.model, select)
        tables_args = (rc, case, select, True, {})
        for func, args in ((_pairs_of_years, pairs_args), (benchmark._tables_of_years, tables_args)):
            payload = pickle.dumps((parallel._years_call, 2001, (func, args, None)))
            assert len(payload) < LIMIT, len(payload)
        assert pickle.loads(pickle.dumps(rc)) == rc
