"""Dataclasses holding per-case (Case) and global (Setting) configuration."""
from dataclasses import dataclass, field, fields as dataclass_fields, make_dataclass
from typing import Optional, Union


@dataclass
class Case:
    """One model/verification-window combination, built from the config."""

    case_name: str = field(default=None)

    model: str = field(default=None)
    model_var: str = field(default=None)
    model_dir: str = field(default="data")

    obs: str = field(default=None)
    obs_var: str = field(default=None)

    ref_model: Optional[str] = field(default=None)
    ref_model_var: Optional[str] = field(default=None)

    thresh_file: Optional[str] = field(default=None)
    thresh_var: Optional[str] = field(default=None)

    wet_init: float = field(default=1.0)
    wet_threshold: float = field(default=20.0)
    wet_spell: int = field(default=3)
    dry_threshold: float = field(default=1.0)
    dry_spell: int = field(default=0)
    dry_extent: int = field(default=0)

    probabilistic: bool = field(default=False)
    members: Optional[tuple[int, ...]] = field(default=None)

    onset_percentage_threshold: float = field(default=0.5)

    verification_window: tuple[int, int] = field(default=(1,15))
    tolerance_days: int = field(default=3)
    max_forecast_day: int = field(default=30)
    day_bins: Optional[tuple[tuple[int, int], ...]] = field(default=((1, 5), (6, 10), (11, 15)))

    years: Optional[Union[tuple[int, ...], str]] = field(default=None)
    years_clim: Optional[Union[tuple[int, ...], str]] = field(default=None)

    earliest_onset_date: Optional[tuple[int, int]] = field(default=None)

    region: str = field(default=None)

    file_pattern: str = field(default=None)
    unit_cvt: Optional[float] = field(default=None)
    accumulation: str = field(default="auto")
    # UTC clock time ("03 UTC") at which the model's days are cut, or None for 24 h from the init.
    daily_boundary: Optional[str] = field(default=None)
    # Canonical name -> name in the model files, e.g. {"init_time": "time"}.
    model_dims: Optional[dict] = field(default=None)

    def update(self, updates):
        """Set the fields named in updates; keys that are not fields are ignored."""
        for key, value in updates.items():
            if key in self.__annotations__:
                setattr(self, key, value)


_RUNTIME_SETTINGS = (
    # Set by the run, not config keys: the shared cache folder (None: next to dir_out), whether output file names are
    # short, the (first, last) leads searched (set by the loader in definition mode) and whether the config uses the old
    # flat onset keys (set by the loader).
    ("cache_dir", None), ("short_names", False), ("lead_window", None), ("onset_legacy", False))


def _setting_fields() -> list:
    """The fields of Setting: every config key of the schema (momp.lib.config_schema, the one list of keys) that is not
    a Case field, and the run's own settings. Case fields are left out because the scoring merges a Setting over a
    Case, so a Setting must not overwrite the case's own values."""
    import copy
    from dataclasses import fields as dataclass_fields
    from typing import Any

    from momp.lib import config_schema as schema

    case_names = {f.name for f in dataclass_fields(Case)}
    out = []
    for item in schema.FIELDS:
        if item.key in case_names:
            continue
        default = None if callable(item.default) or item.default is schema.REQUIRED else item.default
        out.append((item.key, Any, field(default_factory=lambda d=default: copy.deepcopy(d))
                    if isinstance(default, (list, dict, set)) else field(default=default)))
    out.extend((name, Any, field(default=default)) for name, default in _RUNTIME_SETTINGS)
    return out


def _update(self, updates):
    """Set the fields named in updates; keys that are not fields are ignored."""
    names = {f.name for f in dataclass_fields(self)}
    for key, value in updates.items():
        if key in names:
            setattr(self, key, value)


Setting = make_dataclass(
    "Setting", _setting_fields(), namespace={"update": _update, "__module__": __name__,
                                             "__doc__": "Run-wide settings shared by all cases: one field per config "
                                                        "key of the schema (paths, dates, metrics, output switches, "
                                                        "units, onset) except the Case fields, and the run's own "
                                                        "settings."})
