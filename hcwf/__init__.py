"""HCWF (hcwf): evaluation of rainy-season onset forecasts against observations.

    import hcwf
    result = hcwf.run("run.yaml")      # RunResult with .observed, .scores, .forecasts, .folder (see hcwf.api)
    plan = hcwf.check("run.yaml")      # the report of `hcwf-run -p run.yaml --check`
"""


def run(cfg, **kwargs):
    """Run HCWF on a YAML path, hcwf.config.Config or mapping and return a hcwf.api.RunResult (see hcwf.api.run)."""
    from hcwf.api import run as _run

    return _run(cfg, **kwargs)


def check(cfg, **kwargs):
    """Validate and resolve a config and return the plan as a CheckReport; runs nothing (see hcwf.api.check)."""
    from hcwf.api import check as _check

    return _check(cfg, **kwargs)
