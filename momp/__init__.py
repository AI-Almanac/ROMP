"""ROMP (momp): evaluation of rainy-season onset forecasts against observations.

    import momp
    result = momp.run("run.yaml")      # RunResult with .observed, .scores, .forecasts, .folder (see momp.api)
    plan = momp.check("run.yaml")      # the report of `momp-run -p run.yaml --check`
"""


def run(cfg, **kwargs):
    """Run ROMP on a YAML path, momp.config.Config or mapping and return a momp.api.RunResult (see momp.api.run)."""
    from momp.api import run as _run

    return _run(cfg, **kwargs)


def check(cfg, **kwargs):
    """Validate and resolve a config and return the plan as a CheckReport; runs nothing (see momp.api.check)."""
    from momp.api import check as _check

    return _check(cfg, **kwargs)
