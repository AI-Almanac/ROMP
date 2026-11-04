"""The one place a resolved flat config is validated (design: ``docs/review/runconfig_design.md`` 2.5).

This module holds `problems`, which runs every check on the flat config that `hcwf.config.build` finishes, and the
`Problems` it returns. The checks that need other modules (`hcwf.config.onset`, ``hcwf.spatial.units``,
``hcwf.blend.recipe``) import them inside the functions::

    found = problems(values)       # every check, once: nothing raised, nothing printed
    if found: raise HCWFConfigError(found.message())

`Problems` has two tiers, which ``hcwf-run --check`` prints in different places:

* ``fatal``: cross-key problems (dates, list lengths, model names, windows, the forecast limit, probabilistic bins,
  ``explain_cells``, the mask section) and the onset problems (removed keys, the onset definition). ``--check`` reports
  these as "the config cannot be resolved".
* ``units`` and ``blend``: problems of the ``units:`` and ``blend:`` sections. ``--check`` lists these in its Units and
  Blend sections. They are looked for only when the fatal tier is empty, as they need the models and the onset settings.

A config with a units problem therefore fails when it is resolved, before any run folder exists (it used to fail half way
through a run, when the first file was read). The units' specs and the blend ``Recipe`` are kept in the result so that
``--check`` prints them without building them again.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Mapping

__all__ = ["Problems", "problems"]


@dataclass(frozen=True)
class Problems:
    """What is wrong with a config: the fatal problems, the units' and the blend's (each a message, in the order found)."""

    fatal: tuple[str, ...] = ()
    """The cross-key and onset problems; when there are any, the other tiers are not looked for."""
    units: tuple[str, ...] = ()
    """The problems of the ``units:`` section."""
    blend: tuple[str, ...] = ()
    """The problems of the ``blend:`` section."""
    specs: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    """``{source: UnitsSpec or None (grid cells)}`` of the observations and every model; empty when the units are invalid."""
    recipe: Any = field(default=None, compare=False, repr=False)
    """The blend ``Recipe`` of a ``blend:`` section, None without one or when it is invalid."""

    def __bool__(self) -> bool:
        """Return true when there is any problem.

        Returns
        -------
        bool
            Whether any tier holds a message.
        """
        return bool(self.fatal or self.units or self.blend)

    def all(self) -> tuple[str, ...]:
        """Return every message: fatal, then units, then blend.

        Returns
        -------
        tuple of str
            The messages.
        """
        return (*self.fatal, *self.units, *self.blend)

    def message(self) -> str:
        """Return the text of the ``HCWFConfigError`` a run raises.

        Returns
        -------
        str
            A header line, then one bulleted line per problem.
        """
        return "HCWF Configuration is invalid:\n" + "\n".join(f"  • {text}" for text in self.all())

    def raise_if_any(self) -> None:
        """Raise one ``HCWFConfigError`` holding every problem.

        Raises
        ------
        HCWFConfigError
            There is any problem; its message is `message`.
        """
        if self:
            from hcwf.errors import HCWFConfigError

            raise HCWFConfigError(self.message())


def problems(values: Mapping[str, Any]) -> Problems:
    """Run every check of a resolved flat config: the fatal tier first, then (when it is empty) the units and the blend.

    Nothing is raised or printed.

    Parameters
    ----------
    values : collections.abc.Mapping
        The resolved flat config.

    Returns
    -------
    Problems
        What is wrong, in tiers; falsy when the config is valid.
    """
    cfg = SimpleNamespace(**values)
    fatal = [*_temporal(cfg), *_list_lengths(cfg), *_model_names(cfg), *_tuple_ranges(cfg), *_forecast_limits(cfg),
             *_probabilistic(cfg), *_onset(cfg), *_explain_cells(cfg), *_mask(cfg)]
    if fatal:
        return Problems(tuple(fatal))
    specs, units = _units(values)
    recipe, blend = _blend(values)
    return Problems((), units, blend, specs, recipe)


def _wrapped(first: str, second: str, gap: int) -> str:
    """Join two halves of a message by `gap` spaces.

    The messages that were continued with a backslash in a string keep the indentation of the continued line, and the
    messages are kept as they were.

    Parameters
    ----------
    first : str
        The first half.
    second : str
        The second half.
    gap : int
        The number of spaces between them.

    Returns
    -------
    str
        The message.
    """
    return first + " " * gap + second


def _temporal(cfg) -> list[str]:
    """Check the ``start_date``/``end_date`` ordering, the climatology year range and the explicit years.

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        The problems found; a date that cannot be formed is reported as ``Date formatting error``.
    """
    errors = []
    try:
        sd = datetime(*cfg.start_date)
        ed = datetime(*cfg.end_date)
        if sd >= ed:
            errors.append(f"start_date ({sd.date()}) is after end_date ({ed.date()})")

        sy = cfg.start_year_clim
        ey = cfg.end_year_clim
        if sd > ed:
            errors.append(f"start_year_clim ({sy}) is after end_year_clim ({ey})")

        for key, name, first, last, bounds in (
                ("years", "evaluation.years", sd.year, ed.year, "start_date and end_date"),
                ("years_clim", "evaluation.climatology_years", sy, ey, "start_year_clim and end_year_clim")):
            outside = [y for y in getattr(cfg, key, None) or () if not first <= y <= last]
            if outside:
                errors.append(f"{name} {outside} lie outside {first}-{last}, the range of {bounds}. "
                              f"Fix: list only years in that range, or widen the range")
    except Exception as e:
        errors.append(f"Date formatting error: {e}")
    return errors


def _list_lengths(cfg) -> list[str]:
    """Check that the per-model ``*_list`` entries match ``model_list`` (``tolerance_days_list``: the verification windows).

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        One ``Length Mismatch`` message per list of the wrong length.
    """
    errors = []
    master_len = len(cfg.model_list)
    for key, value in vars(cfg).items():
        if key.endswith('_list') and isinstance(value, (list, tuple)):
            if key not in ('verification_window_list', 'tolerance_days_list'):
                if len(value) != master_len:
                    errors.append(_wrapped(f"Length Mismatch: '{key}' has {len(value)} items, ",
                                           f"expected {master_len}", 40))
            if key in ('tolerance_days_list',):
                if len(value) != len(cfg.verification_window_list):
                    errors.append(_wrapped(f"Length Mismatch: '{key}' has {len(value)} items, ",
                                           f"expected {len(cfg.verification_window_list)}", 40))
    return errors


def _model_names(cfg) -> list[str]:
    """Check that model names contain no spaces (they name files).

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        One message per model name with a space.
    """
    return [f"Invalid model name: '{model}' (Spaces are not allowed)." for model in cfg.model_list if " " in model]


def _tuple_ranges(cfg) -> list[str]:
    """Check that each ``(start, end)`` pair in the verification windows and ``day_bins`` has ``start < end``.

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        One message per invalid pair.
    """
    errors = []
    for attr_name in ['verification_window_list', 'day_bins']:
        for window in getattr(cfg, attr_name):
            if window[0] >= window[1]:
                errors.append(_wrapped(f"Invalid range in {attr_name}: {window}. ",
                                       "First number must be smaller than the second.", 28))
    return errors


def _forecast_limits(cfg) -> list[str]:
    """Check that no verification window extends beyond ``max_forecast_day``.

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        One message when the largest window end exceeds ``max_forecast_day``, else empty.
    """
    max_v_day = max([pair[1] for pair in cfg.verification_window_list])
    if max_v_day > cfg.max_forecast_day:
        return [_wrapped(f"Verification window limit ({max_v_day}) ",
                         f"exceeds max_forecast_day ({cfg.max_forecast_day}).", 20)]
    return []


def _probabilistic(cfg) -> list[str]:
    """Check that, in probabilistic mode, the largest window end covers the largest ``day_bins`` end.

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        One message when it does not, else empty (always empty when not probabilistic).
    """
    if getattr(cfg, 'probabilistic', False):
        max_v_day = max([pair[1] for pair in cfg.verification_window_list])
        max_bin_day = max([pair[1] for pair in cfg.day_bins])
        if max_v_day < max_bin_day:
            return [_wrapped("Window mismatch: Largest verification day ",
                             f"({max_v_day}) is smaller than largest day_bin ({max_bin_day}).", 24)]
    return []


def _onset(cfg) -> list[str]:
    """Check the onset part of the config.

    Keys earlier versions read (the flat onset keys and the other removed ones) are refused with the message a YAML
    config gets, then the onset part is fully resolved: every needed setting is known.

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        The message of the ``OnsetConfigError`` when a removed key is set or the onset setup does not resolve, else
        empty.

    Raises
    ------
    OnsetConfigError
        A removed key is set; raised inside the check and returned as a message.
    """
    from hcwf.config import schema
    from hcwf.config.onset import onset_setup
    from hcwf.errors import OnsetConfigError

    dic = vars(cfg)
    try:
        removed = [key for key in schema.REMOVED_KEYS if dic.get(key) is not None]
        if removed:
            raise OnsetConfigError("\n".join(schema.unknown_key_message(key) for key in removed))
        onset_setup({"onset_definition": None, **dic})
    except OnsetConfigError as exc:
        return [str(exc)]
    return []


def _explain_cells(cfg) -> list[str]:
    """Check that ``explain_cells`` is None or a list of ``(lat, lon)`` pairs of numbers.

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        One message when it is not, else empty.
    """
    cells = getattr(cfg, "explain_cells", None)
    if cells is None:
        return []
    ok = isinstance(cells, (list, tuple)) and all(
        isinstance(c, (list, tuple)) and len(c) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in c) for c in cells)
    if not ok:
        return ["explain_cells must be None or a list of (lat, lon) pairs of numbers, e.g. [(9.0, 38.75), (8.5, 39.0)]"]
    return []


def _mask(cfg) -> list[str]:
    """Check that the ``mask`` section builds a ``MaskSpec`` (a run builds it when it reads the first file).

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The flat config as attributes.

    Returns
    -------
    list of str
        One ``mask: ...`` message when it does not, else empty.
    """
    from hcwf.errors import HCWFConfigError
    from hcwf.spatial.masks import MaskSpec

    mask = getattr(cfg, "mask", None)
    if mask is None:
        return []
    try:
        MaskSpec.from_mapping(mask)
    except (HCWFConfigError, TypeError, ValueError) as exc:
        return [f"mask: {exc}"]
    return []


def _units(values) -> tuple[dict, tuple[str, ...]]:
    """Resolve the ``units:`` section as each source reads it.

    Parameters
    ----------
    values : collections.abc.Mapping
        The flat config.

    Returns
    -------
    tuple of (dict, tuple of str)
        ``{source: UnitsSpec or None}`` of the observations and every model (empty when the units are invalid), and
        the problems (one ``units: ...`` message, or none).
    """
    from hcwf.spatial.units import UnitsSpec
    from hcwf.errors import UnitsSpecError

    try:
        return {name: UnitsSpec.from_flat(values, name) for name in ["observations", *values["model_list"]]}, ()
    except UnitsSpecError as exc:
        return {}, (f"units: {exc}",)


def _blend(values) -> tuple[Any, tuple[str, ...]]:
    """Build the recipe of the ``blend:`` section as the run reads it (with the onset settings).

    Parameters
    ----------
    values : collections.abc.Mapping
        The flat config.

    Returns
    -------
    tuple of (object, tuple of str)
        The blend ``Recipe`` and the problems (``blend: ...`` messages); ``(None, ())`` for a config without a
        ``blend:`` section, and ``None`` with the problems when the recipe is invalid or the onset settings fail.
    """
    from hcwf.config.schema import has_blend

    if not has_blend(values):
        return None, ()
    from hcwf.blend.recipe import recipe_from_config
    from hcwf.errors import RecipeError
    from hcwf.config.onset import onset_setup

    try:
        return recipe_from_config(values, onset_setup(values)), ()
    except RecipeError as exc:
        return None, tuple(f"blend: {problem}" for problem in exc.problems)
    except Exception as exc:  # noqa: BLE001 - the onset settings failing is reported with the rest of the check
        return None, (f"blend: {exc}",)
