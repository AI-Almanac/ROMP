"""Test helpers: the typed readers of hcwf.io.input called with the specs of a flat test config.

The tests keep their configs as flat keyword dicts (what a resolved run holds, or a part of it); these helpers build the
typed inputs once with `flat_bridge.reader_inputs` and call the typed API (`load_observations`, `load_forecast`,
`load_thresholds`, ...), not the `# R2-bridge` wrappers of the old names.
"""
from hcwf.config import flat_bridge
from hcwf.io import input as io_input


def inputs(kwargs):
    """The typed inputs of a flat config (see flat_bridge.ReaderInputs)."""
    return flat_bridge.reader_inputs(kwargs)


def observations(year, kwargs):
    """`load_observations` of a flat config."""
    got = inputs(kwargs)
    return io_input.load_observations(year, got.obs, got.area, got.options, got.select, onset=got.onset_if_units(),
                                      run_years=got.ev.years, dir_fig=got.dir_fig)


def obs_years(years, kwargs, hold=True):
    """`read_obs_years` of a flat config."""
    got = inputs(kwargs)
    return io_input.read_obs_years(years, got.obs, got.area, got.options, got.select, onset=got.onset_if_units(),
                                   run_years=got.ev.years, dir_fig=got.dir_fig, parallel=got.parallel, hold=hold)


def forecast(year, kwargs, ensemble):
    """`load_forecast` of a flat config."""
    got = inputs(kwargs)
    return io_input.load_forecast(year, got.model, got.ev, got.area, got.options, got.onset_if_steps(),
                                  ensemble=ensemble, select=got.select, dir_fig=got.dir_fig)


def thresholds(kwargs):
    """`load_thresholds` of a flat config."""
    got = inputs(kwargs)
    return io_input.load_thresholds(got.onset(), got.obs, got.area, got.options, got.ev, got.output, model=got.model,
                                    select=got.select, parallel=got.parallel, cache_dir=got.cache_dir)


def history(year, current, kwargs):
    """`observed_history` of a flat config."""
    got = inputs(kwargs)
    return io_input.observed_history(year, current, got.onset(), got.obs, got.area, got.options, got.select,
                                     run_years=got.ev.years, dir_fig=got.dir_fig)


def steps_needed(kwargs):
    """`steps_needed` of a flat config."""
    got = inputs(kwargs)
    return io_input.steps_needed(kwargs.get("max_forecast_day"), got.onset_if_steps())


def initialization_dates(year, kwargs):
    """`initialization_dates` of a flat config."""
    return io_input.initialization_dates(year, inputs(kwargs).ev)
