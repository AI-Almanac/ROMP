"""Define every error HCWF raises on purpose, in one hierarchy.

This is a dependency-free leaf: any module may import it, so there are no import cycles. The hierarchy is::

    ValueError
      HCWFConfigError                  what the CLI reports without a traceback
        ConfigSchemaError              a YAML config: unknown keys, wrong types, missing keys
        OnsetConfigError               onset settings (`hcwf.config.onset`)
        DefinitionError                an onset definition (`hcwf.onset.definition`)
        UnitsError                     onset-grammar text (`hcwf.onset.units`)
        RegionError                    a region entry (`hcwf.onset.regions`)
        DatasetError                   a dataset description (`hcwf.config.datasets`)
        ConvertError                   a blending config that cannot be converted (`hcwf.config.convert_specs`)
        AreaError, UnitsSpecError      evaluation units and areas (`hcwf.spatial.units`)
        MaskError                      a region mask (`hcwf.spatial.masks`)
        RecipeError                    a blend recipe (`hcwf.blend.recipe`)
        BlendFileError                 a trained-blend file (`hcwf.blend.trained`)
      IncompatibleForecast             live forecasts a trained blend cannot be applied to (`hcwf.blend.forecast`)
    FileExistsError
      RunExistsError                   the run folder already holds a run (`hcwf.runs.folder`)

`IncompatibleForecast` and `RunExistsError` keep their own bases (a ``ValueError`` of the data, a ``FileExistsError`` of
the folder): they are not configuration errors, so the CLI handles them on their own.
"""


class HCWFConfigError(ValueError):
    """Base of every configuration or validation error; the CLI reports these without a traceback."""


class ConfigSchemaError(HCWFConfigError):
    """A YAML config has unknown keys, wrongly typed values or missing required keys; all are listed."""


class OnsetConfigError(HCWFConfigError):
    """The onset settings are missing, inconsistent or unknown; the message says how to fix them."""


class DefinitionError(HCWFConfigError):
    """An onset definition that is malformed or ambiguous; the message names the YAML path and the fix."""


class UnitsError(HCWFConfigError):
    """Text that does not follow the onset grammar; the message ends with how to fix it."""


class RegionError(HCWFConfigError):
    """A malformed region entry or an unknown region; the message names the YAML path and the fix."""


class DatasetError(HCWFConfigError):
    """A malformed dataset entry or an unknown dataset; the message names the file or key and the fix."""


class ConvertError(HCWFConfigError):
    """A blending config or spec set that cannot be converted; lists every problem found.

    Parameters
    ----------
    problems : iterable of str
        One line per problem found; all are listed in the message.

    Attributes
    ----------
    problems : list of str
        The problem lines, in the order given.
    """

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("Cannot convert:\n" + "\n".join(f"  - {p}" for p in self.problems))


class AreaError(HCWFConfigError):
    """No unit falls inside the area of interest."""


class UnitsSpecError(HCWFConfigError):
    """A units specification that cannot be used (an unset environment variable in a path, no source for a mode)."""


class MaskError(HCWFConfigError):
    """A region mask cannot be built: bad specification, unreadable source, grid not covered, or no cell kept."""


class RecipeError(HCWFConfigError):
    """A recipe that cannot be used; lists every problem found.

    Parameters
    ----------
    problems : list of str
        One line per problem found; all are listed in the message.

    Attributes
    ----------
    problems : list of str
        The problem lines, in the order given.
    """

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Invalid blend recipe:\n" + "\n".join(f"  - {p}" for p in self.problems))


class BlendFileError(HCWFConfigError):
    """A trained-blend file that cannot be used (damaged, another format or version, no longer rebuilds)."""


class IncompatibleForecast(ValueError):
    """Live forecasts the trained blend cannot be applied to; lists every reason.

    Parameters
    ----------
    problems : list of str
        One line per incompatibility found; all are listed in the message.

    Attributes
    ----------
    problems : list of str
        The incompatibility lines, in the order given.
    """

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Cannot forecast with this blend:\n" + "\n".join(f"  - {p}" for p in self.problems))


class RunExistsError(FileExistsError):
    """The run folder of this config already holds a run; the message says how to re-run."""
