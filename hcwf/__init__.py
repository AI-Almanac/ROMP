"""Evaluate rainy-season onset forecasts against observations (HCWF, Human-Centered Weather Forecasts).

The package root exposes the two entry points of the Python interface; everything else lives in the subpackages
(`hcwf.config`, `hcwf.onset`, `hcwf.scoring`, `hcwf.blend`, ...) and in `hcwf.api`, which implements both functions::

    import hcwf
    result = hcwf.run("run.yaml")      # RunResult with .observed, .scores, .forecasts, .folder (see hcwf.api)
    plan = hcwf.check("run.yaml")      # the report of ``hcwf-run -p run.yaml --check``

Both import `hcwf.api` lazily, so ``import hcwf`` itself is cheap.
"""


def run(cfg, **kwargs):
    """Run HCWF on a config and return its result; see `hcwf.api.run`.

    Parameters
    ----------
    cfg : str or pathlib.Path or hcwf.config.Config or dict
        A YAML path, a `hcwf.config.Config` or a mapping.
    **kwargs : dict
        Passed on to `hcwf.api.run` (``write``, ``overwrite``, ``quiet``, ``verbose`` and flat config overrides).

    Returns
    -------
    hcwf.api.RunResult
        The observed onsets, scores and run folder of the run.
    """
    from hcwf.api import run as _run

    return _run(cfg, **kwargs)


def check(cfg, **kwargs):
    """Validate and resolve a config and return the plan; runs nothing. See `hcwf.api.check`.

    Parameters
    ----------
    cfg : str or pathlib.Path or hcwf.config.Config or dict
        A YAML path, a `hcwf.config.Config` or a mapping.
    **kwargs : dict
        Passed on to `hcwf.api.check` (``overwrite`` and flat config overrides).

    Returns
    -------
    hcwf.runs.check.CheckReport
        The plan: settings, datasets, hints, errors and warnings.
    """
    from hcwf.api import check as _check

    return _check(cfg, **kwargs)
