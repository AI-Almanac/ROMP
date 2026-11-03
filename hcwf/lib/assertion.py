"""The configuration error every validation raises (the checks themselves are in hcwf.config.validate)."""


class HCWFConfigError(ValueError):
    """Base of every configuration or validation error (a ValueError); the CLI reports these without a traceback."""
    pass
