"""FROZEN, TEST-ONLY copy of the configuration code as it was before wave R2 (design: docs/review/runconfig_design.md,
section 3 "Oracle"). Nothing here may be imported by hcwf; nothing here may be edited to follow the live code.

It holds, verbatim (only imports and the `loader.` prefixes of calls between the copied functions changed):

* `Case`, `Setting` (hcwf/lib/convention.py) and `make_case`, `iter_list`, `init_dataclass`, ... (hcwf/lib/control.py);
* the two reference-case constructions of the scoring flows, written out as `deterministic_reference_kwargs`
  (app/spatial_far_mr_mae.py, reference loop) and `ensemble_reference_kwargs` (metrics/skill.py:reference_model_kwargs
  applied to the case dict of app/bin_skill_score.py), and `case_kwargs` (export.py);
* the four finishing variants of a flat config: the loader's `build_cfg`/`plan_run`/`load_run_config`, `api._flat`/
  `_prepare`, `export.load_cfg`/`_complete`/`_as_dict`, and the dict branch of `driver._run` (`driver_finish`),
  with the argparse parser (hcwf/lib/parser.py) they use.

Not frozen (live code both sides call, which R2 does not replace): the schema, `Config`, `build_config`,
`run_folder.plan/apply`, `apply_lead_window`. The validators (`HCWFValidator`, `check_config`, the units and blend problems
of `--check`) are frozen at the end of the file (step S2). `get_cfg`/`get_setting`/`use_config` (process-wide
caches) are not copied. Delete this file when R3 is done (design section 4, question 7).
"""
# ruff: noqa
import argparse
import ast
import copy
import os
from dataclasses import asdict, dataclass, field, fields, fields as dataclass_fields, make_dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

from hcwf.lib.assertion import HCWFConfigError
from hcwf.utils.practical import set_dir
from hcwf.utils.printing import combi_to_str




# ======================================================================= hcwf/lib/convention.py

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

    probabilistic: bool = field(default=False)
    members: Optional[tuple[int, ...]] = field(default=None)

    onset_percentage_threshold: float = field(default=0.5)

    verification_window: tuple[int, int] = field(default=(1,15))
    tolerance_days: int = field(default=3)
    max_forecast_day: int = field(default=30)
    day_bins: Optional[tuple[tuple[int, int], ...]] = field(default=((1, 5), (6, 10), (11, 15)))

    years: Optional[Union[tuple[int, ...], str]] = field(default=None)
    years_clim: Optional[Union[tuple[int, ...], str]] = field(default=None)

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
    # short and the (first, last) leads searched (set by the loader).
    ("cache_dir", None), ("short_names", False), ("lead_window", None))


def _setting_fields() -> list:
    """The fields of Setting: every config key of the schema (hcwf.lib.config_schema, the one list of keys) that is not
    a Case field, and the run's own settings. Case fields are left out because the scoring merges a Setting over a
    Case, so a Setting must not overwrite the case's own values."""
    import copy
    from dataclasses import fields as dataclass_fields
    from typing import Any

    from hcwf.lib import config_schema as schema

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


# ======================================================================= hcwf/lib/control.py

LAYOUT = ("model", "verification_window")
"""The dimensions of the processed cases: every model with every verification window."""


def init_dataclass(dc, dic):
    """Instantiate dataclass dc from the entries of dic that match its annotated field names."""
    keys = [f.name for f in fields(dc)]
    subset = {key: dic[key] for key in keys if key in dic}
    return dc(**subset)


def modify_list_keys(dictionary):
    """Return the keys of dictionary ending in "_list", with that suffix removed."""
    modified_keys = []
    for key, value in dictionary.items():
        if key.endswith("_list"):
            modified_key = key[:-5]  # Remove the last 5 characters "_list"
            modified_keys.append(modified_key)
    return modified_keys


def case_across_list(item, list1, list2):
    """find the corresponding item in list2 for a given item in list1"""
    if item in list1:
        index = list1.index(item)
        if index < len(list2):
            return list2[index]
    return None


def iter_list(dic, ext="_list"):
    """Return the dic[field + ext] list for each field of LAYOUT."""
    layout_pool = []
    for field in LAYOUT:
        lst = dic.get(field + ext)
        layout_pool.append(lst)
    return layout_pool


def years_tuple_clim(year_start: int, year_end: int) -> tuple[int, ...]:
    """
    Create a tuple of integers from year_start to year_end inclusive.
    """
    return tuple(range(year_start, year_end + 1))


def years_tuple_model(start_date: tuple[int,int,int], end_date: tuple[int,int,int]) -> tuple[int, ...]:
    """
    Create a tuple of years from start_date to end_date inclusive.
    Each date is a tuple like (year, month, day)
    """
    start_year = start_date[0]
    end_year = end_date[0]
    return tuple(range(start_year, end_year + 1))


def evaluation_years(dic: dict) -> tuple[int, ...]:
    """The years evaluated: `years` (evaluation.years) if given, else every year from start_date to end_date."""
    return tuple(dic["years"]) if dic.get("years") else years_tuple_model(dic["start_date"], dic["end_date"])


def climatology_years(dic: dict) -> tuple[int, ...]:
    """The climatology years: `years_clim` (evaluation.climatology_years) if given, else start_year_clim to end_year_clim."""
    return tuple(dic["years_clim"]) if dic.get("years_clim") else years_tuple_clim(dic["start_year_clim"],
                                                                                   dic["end_year_clim"])


def take_ensemble_members(
    members: Optional[Union[tuple[int, ...], list[int], str]]
) -> list[int]:
    """
    Normalize members into a tuple of ints (or None).

    Accepts:
    - list[int] or tuple[int, ...] → returned as a tuple
    - string 'start-end' → expanded to a tuple
    - None, empty or 'All' → None
    """
    # Case 0: None, empty or 'All'
    if not members or members == 'All':
        return None

    # Case 1: list or tuple of ints
    if isinstance(members, (list, tuple)):
        return tuple(members)  # normalize to tuple

    # Case 2: string range "1-5"
    if isinstance(members, str) and "-" in members:
        start, end = map(int, members.split("-", 1))
        return tuple(range(start, end + 1))

    raise TypeError(
        "members must be list[int], tuple[int, ...], 'start-end' string, or None"
    )


def make_case(dataclass, combi, dic):
    """Build one Case: combi gives the values of the layout fields (in LAYOUT order)
    and the remaining per-model _list entries are looked up by model name.
    """
    layout = LAYOUT

    layout_dict = {key: None for key in layout}
    layout_dict.update(zip(layout_dict.keys(), combi))

    case_keys = [
        field.name for field in fields(dataclass)
    ]  # get defined keys in dataclass

    dic_list_keys = modify_list_keys(dic)  # return keys endwith _list, removing _list
    dic_list_keys = [
        item for item in dic_list_keys if item not in list(layout_dict.keys())
    ]
    list_keys = list(set(case_keys).intersection(set(dic_list_keys)))
    list_keys.remove("tolerance_days") # tolerance_days is paired with verification_window

    case = init_dataclass(dataclass, dic)
    case.update(layout_dict)

    value_list = []

    for key in list_keys:
        value = case_across_list(case.model, dic["model_list"], dic[key + "_list"])

        if key == "model_dir":
            if not Path(value).is_absolute():
                value = set_dir(value)

        value_list.append(value)

    case_name = combi_to_str(combi)
    case.case_name = case_name.replace(" ", "_")

    case.years = evaluation_years(dic)
    case.years_clim = climatology_years(dic)

    case.members = take_ensemble_members(dic['members'])

    value_dic = dict(zip(list_keys, value_list))
    case.update(value_dic)

    case.tolerance_days = case_across_list(case.verification_window,
                                           dic["verification_window_list"], dic["tolerance_days_list"])

    return case


def filter_bins_in_window(day_bins, verification_window):
    """
    Returns only the day bins that lie completely within the verification window.

    Args:
        day_bins: tuple of tuples, e.g. ((1,5), (6,10), ...)
        verification_window: tuple of (start, end), e.g. (6, 20)

    Returns:
        tuple of tuples containing only the bins inside the window
    """
    start_win, end_win = verification_window

    return tuple(
        bin_range
        for bin_range in day_bins
        if start_win <= bin_range[0] and bin_range[1] <= end_win
    )


# ======================================================================= hcwf/metrics/skill.py

def reference_model_kwargs(kwargs):
    """Copy kwargs with the reference model's name, directory, variable, file pattern, unit conversion and dims.

    The forecast model's own model_dims are dropped (the reference archive is
    read by name inference) unless kwargs carries ref_model_dims.
    """
    return {**kwargs,
            'model': kwargs['ref_model'],
            'model_dir': kwargs['ref_model_dir'],
            'model_var': kwargs['ref_model_var'],
            'file_pattern': kwargs['ref_model_file_pattern'],
            'unit_cvt': kwargs['ref_model_unit_cvt'],
            'model_dims': kwargs.get('ref_model_dims')}


# ======================================================================= hcwf/export.py

def case_kwargs(dic: dict, model: str, *, reference: bool = False) -> dict:
    """The keyword arguments the scoring builds for one model: Setting and Case values of its first window.

    The detection does not depend on the verification window, so the first one stands for all. With
    reference=True the case is the climatology reference's, built as spatial_far_mr_mae_map builds it:
    the model name is the reference's and the data location the reference's.
    """
    dic = dict(dic)
    if reference:
        dic["model_list"] = (model,)
    pool = iter_list(dic)
    combi = tuple(model if field == "model" else values[0] for field, values in zip(LAYOUT, pool))
    case = make_case(Case, combi, dic)
    if reference:
        case.update({"model_dir": dic["ref_model_dir"], "model_var": case.ref_model_var,
                     "file_pattern": dic["ref_model_file_pattern"], "unit_cvt": dic["ref_model_unit_cvt"],
                     "model_dims": None})
    return {**asdict(init_dataclass(Setting, dic)), **asdict(case)}


# ======================================================================= hcwf/lib/parser.py

def create_parser(config, cli_args=None):
    """Parse CLI overrides, using the config dict values as defaults; unknown arguments are ignored."""

    parser = argparse.ArgumentParser(description="HCWF Package Parameter Loader")

    parser.add_argument(
        "--model_list",
        nargs="+",
        default=config["model_list"],
        help=f"Model list (default: {config['model_list']})"
    )

    parser.add_argument(
        "--verification_window_list",
        type=parse_window_list,
        default=config['verification_window_list'],
        help="Verification window as start end day (default: {config['verification_window_list']}), \
                (e.g., '1,15 16,20')"
    )

    parser.add_argument(
        "--max_forecast_day",
        type=int,
        default=config['max_forecast_day'],
        help="Max forecast day (default: {config['max_forecast_day']})"
    )

    if "dry_spell_check" in config:
        parser.add_argument(
            "--dry_spell_check",
            type=str2bool,
            default=config["dry_spell_check"],
            help=f"Keep the definition's dry-spell condition, True or False (default: {config['dry_spell_check']})"
        )

    for name, choices in (("forecast_history", ("observed", "none")), ("threshold_source", ("observations", "model"))):
        if name in config:
            parser.add_argument(f"--{name}", choices=choices, default=config[name],
                                help=f"{name}: {' or '.join(choices)} (default: {config[name]})")

    parser.add_argument(
        "--onset_percentage_threshold",
        type=float,
        default=config['onset_percentage_threshold'],
        help=f"Probability threshold (0.0-1.0) for ensemble onset (default: {config['onset_percentage_threshold']})"
    )

    # Allowed error margin (in days)
    parser.add_argument(
        "--tolerance_days_list",
        nargs='+',
        type=int,
        action=ForceTupleAction,
        default=config['tolerance_days_list'],
        help=f"Allowed error margins in days e.g., 3 5 (default: {config['tolerance_days_list']})"
    )

    # Time windows for grouping statistics
    parser.add_argument(
        "--day_bins",
        type=parse_window_list,
        default=config['day_bins'],
        help=f"Day bins for statistics e.g., '1,15 16,30' (default: {config['day_bins']})"
    )

    # Date range for evaluation
    parser.add_argument(
        "--start_date",
        nargs=3,
        type=int,
        action=ForceTupleAction,
        default=config['start_date'],
        help="Start date as: YYYY M D (e.g., 2024 5 1)"
    )

    parser.add_argument(
        "--end_date",
        nargs=3,
        type=int,
        action=ForceTupleAction,
        default=config['end_date'],
        help=f"Evaluation end date as YYYY M D (e.g., 2024 10 31) (default: {config['end_date']})"
    )

    parser.add_argument(
        "--show_plot",
        type=str2bool,
        default=config['show_plot'],
        help="Toggle individual model plot display, True of False (default: {config['show_plot']})"
    )

    parser.add_argument(
        "--show_panel",
        type=str2bool,
        default=config['show_panel'],
        help="Toggle panel plot display, True of False (default: {config['show_panel']})"
    )

    parser.add_argument(
        "--probabilistic",
        type=str2bool,
        default=config['probabilistic'],
        help=f"Ensemble evaluation toggle, True or False (default: {config['probabilistic']})"
    )

    parser.add_argument(
        "--parallel",
        type=str2bool,
        default=config['parallel'],
        help="parallelization, True of False (default: {config['parallel']})"
    )

    parser.add_argument(
        "--region",
        type=str,
        default=config['region'],
        help="region as defined in params.region_def (default: {config['region']})"
    )

    # run control (handled by the driver, not config values)
    parser.add_argument("--check", action="store_true",
                        help="validate and resolve the config, print the plan (settings, datasets, files, leads, "
                             "hints, run folder, cache) and run nothing; exit 1 on errors")
    parser.add_argument("--alignment", action="store_true",
                        help="with --check: also run the observation/forecast alignment check (off by default)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run a config whose run folder already exists (clears that folder first)")
    parser.add_argument("--quiet", action="store_true", help="show only warnings and errors")
    parser.add_argument("--verbose", action="store_true", help="show debug detail (run.log always has it)")

    # cli_args can be passed explicitly; if None, default to sys.argv
    if cli_args is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(cli_args)

    # ---- Convert list → tuple if needed ----
    # Only convert if user supplied; default already tuple
    if isinstance(args.model_list, list):
        args.model_list = tuple(args.model_list)

    return args


class ForceTupleAction(argparse.Action):
    """Custom action to convert nargs list into a tuple automatically."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Store values on the namespace as a tuple."""
        setattr(namespace, self.dest, tuple(values))


def parse_window_list(string):
    """
    Converts a string input like '1,15 16,20' into ((1, 15), (16, 20))
    """
    try:
        # Split by space to get individual pairs, then split by comma
        pairs = [tuple(map(int, p.split(','))) for p in string.split()]
        return tuple(pairs)
    except Exception:
        raise argparse.ArgumentTypeError("Window list must be in format 'start,end start,end' (e.g., '1,15 16,20')")


def str2bool(v):
    """Convert a yes/no/true/false-style string (or a bool) to bool for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_tuple(value):
    """
    Parses a string into its native Python type.
    Example: "(2019, 5, 1)" -> tuple
    Example: "((1, 5), (6, 10))" -> nested tuple
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Fallback for simple strings if eval fails
        return value


def parse_num_to_tuple(value):
    """
    Converts a space-separated string into a Python tuple of integers.
    CLI: "2024 5 1" -> Python: (2024, 5, 1)
    """
    try:
        # Split string by spaces, convert each part to int, then to tuple
        return tuple(map(int, value.split()))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{value}'. Expected space-separated integers like '2024 5 1'."
        )


def ensure_config_exists(param_path: str) -> Path:
    """Return param_path expanded and resolved, raising FileNotFoundError (with a hint) if it doesn't exist."""
    path = Path(param_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. Start one with `hcwf-run init`.")
    return path


# ======================================================================= hcwf/lib/loader.py

def _get_config_path_pre_parse(cli_args=None):
    """Return the config path from -p/--param in cli_args, else the HCWF_CONFIG environment variable.

    A run must always say which config it uses; with neither set this raises,
    explaining how to give one (start one with `hcwf-run init`, or see examples/config.yaml).
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-p", "--param", default=None)
    args, _ = pre_parser.parse_known_args(cli_args)
    path = args.param or os.environ.get("HCWF_CONFIG")
    if not path:
        raise HCWFConfigError(
            "No config file given. Pass one with `hcwf-run -p path/to/run.yaml` (or set the "
            "HCWF_CONFIG environment variable). Start one with `hcwf-run init`; see `hcwf-run -h`.")
    return path


def _read_config(config_path):
    """Read a YAML run config into the flat config dict (validated, datasets described, defaults resolved)."""
    from hcwf.config import Config
    from hcwf.lib.config_yaml import is_yaml_path

    config_item = set_dir(config_path)
    if not config_item.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    if not is_yaml_path(config_item):
        raise HCWFConfigError(
            f"{config_path} is not a YAML config. hcwf-run reads YAML only. Start a new one with `hcwf-run init`.")
    return Config.from_yaml(config_item).to_flat(source=str(config_item))


def _resolve_threshold_ref(ref):
    """Make the path part of a "path.nc:variable" threshold reference absolute."""
    path, sep, var = str(ref).rpartition(":")
    if not sep or not path or Path(path).is_absolute():
        return ref
    return f"{set_dir(path)}:{var}"


def _resolve_paths(dic):
    """Resolve relative paths in config dict to absolute paths."""
    for key in ["ref_model_dir", "dir_out", "dir_fig", "obs_dir"]:
        if key in dic and not Path(str(dic[key])).is_absolute():
            dic[key] = set_dir(dic[key])

    for key in ["shpfile_dir", "nc_mask"]:
        if dic.get(key) is not None:
            if not Path(str(dic[key])).is_absolute():
                dic[key] = set_dir(dic[key])

    _anchor_units_paths(dic)
    _anchor_mme_weights(dic)

    if isinstance(dic.get("mask"), dict):
        mask = dict(dic["mask"])
        if mask.get("file") and not Path(str(mask["file"])).is_absolute():
            mask["file"] = str(set_dir(mask["file"]))
        condition = mask.get("condition")
        if condition and not Path(str(condition["file"])).is_absolute():
            mask["condition"] = {**condition, "file": str(set_dir(condition["file"]))}
        dic["mask"] = mask

    if isinstance(dic.get("thresholds"), dict):
        dic["thresholds"] = {
            name: _resolve_threshold_ref(ref) for name, ref in dic["thresholds"].items()
        }

    return dic


_UNITS_PATH_KEYS = ("units_shapefile", "units_weights", "units_centroids", "units_dissemination", "units_outline")
"""The `units:` keys that name files (units_weights may also map a source name to its file)."""


def _anchor_units_paths(dic):
    """Make the relative file paths of the `units:` section start at the folder of the config file (the working directory
    when the config is not from a file), as a region file's units start at the region file."""
    config_file = dic.get("config_file")
    if not config_file:
        return
    base = Path(str(config_file)).expanduser().resolve().parent

    def anchor(path):
        path = Path(str(path)).expanduser()
        return str(path if path.is_absolute() else (base / path).resolve())

    for key in _UNITS_PATH_KEYS:
        value = dic.get(key)
        if isinstance(value, dict):
            dic[key] = {source: anchor(path) for source, path in value.items()}
        elif value is not None:
            dic[key] = anchor(value)


def _anchor_mme_weights(dic):
    """Make the relative path of `blend.mme.weights` (a weights file; a mapping is inline weights) start at the folder
    of the config file, the working directory when the config is not from a file."""
    mme = dic.get("blend_mme")
    if not isinstance(mme, dict) or not isinstance(mme.get("weights"), (str, Path)):
        return
    base = Path(str(dic["config_file"])).expanduser().resolve().parent if dic.get("config_file") else Path.cwd()
    path = Path(str(mme["weights"])).expanduser()
    dic["blend_mme"] = {**mme, "weights": str(path if path.is_absolute() else (base / path).resolve())}


def build_cfg(cli_args=None):
    """Read the YAML run config, resolve paths, apply CLI overrides and plan the run folders; return the config dict.

    Nothing is created on disk: dic["run_paths"] (hcwf.lib.run_folder.RunPaths) says where the run will write, and
    dic["config_file"] is the config's path. The folders are made by run_folder.prepare when the run starts.
    """
    config_path = ensure_config_exists(_get_config_path_pre_parse(cli_args))
    dic = _read_config(config_path)
    dic["config_file"] = str(config_path)
    dic = _resolve_paths(dic)

    args = create_parser(dic, cli_args=cli_args)
    args_dict = vars(args)
    overrides = {k: v for k, v in args_dict.items() if k in dic and v is not None}
    dic.update(overrides)
    dic["config_file"] = str(config_path)
    return plan_run(dic, config_path)


def plan_run(dic, config_file=None, write=True):
    """Finish a flat config for a run: the lead window and the run folders planned from the final values."""
    from hcwf.lib import run_folder

    if "lead_window" not in dic:
        from hcwf.lib import config_schema as schema

        if schema.has_blend(dic):   # a blend reads each model's leads itself (lead_days), not HCWF's benchmark window
            dic["lead_window"], dic["onset_lead_lines"] = (1, dic["max_forecast_day"]), ()
        else:
            from hcwf.lib.lead_window import apply_lead_window

            apply_lead_window(dic)
    return run_folder.apply(dic, run_folder.plan(dic, config_file, write))


def load_run_config(cli_args=None):
    """The validated flat config of a run (lead window applied, folders planned); raises HCWFConfigError if invalid."""
    dic = build_cfg(cli_args)
    HCWFValidator(dic).validate()
    return dic


# ======================================================================= hcwf/api.py

_NO_FILES = {"save_fig": False, "save_nc_spatial_far_mr_mae": False, "save_csv_score": False,
             "save_nc_climatology": False, "save_onsets": False, "plot_spatial_far_mr_mae": False,
             "plot_heatmap_bss_auc": False, "plot_reliability": False, "plot_climatology_onset": False,
             "plot_panel_heatmap_error": False, "plot_panel_heatmap_skill": False, "plot_bar_bss_rpss_auc": False,
             "plot_portrait": False, "show_plot": False, "show_panel": False, "explain_cells": None}
"""Config values that turn every file-writing step off (write=False)."""


def _flat(source, overrides: dict) -> dict:
    """The flat config of a path, Config, mapping or flat dict, with the keyword overrides applied and checked."""
    from hcwf.config import Config
    from hcwf.lib import config_schema as schema
    from hcwf.lib.config_yaml import build_config

    config_file = None
    if isinstance(source, (str, Path)):
        config_file = str(Path(source).expanduser().resolve())
        flat = Config.from_yaml(config_file).to_flat(source=config_file)
    elif isinstance(source, Config):
        flat = source.to_flat()
    elif isinstance(source, SimpleNamespace):
        flat = dict(vars(source))
    elif isinstance(source, dict) and ("model_list" in source or "obs_dir" in source):
        flat = dict(source)
    elif isinstance(source, dict):
        flat = build_config(source, source="<mapping>")
    else:
        raise TypeError(f"hcwf.run needs a YAML path, a hcwf.config.Config or a mapping, got {type(source).__name__}")
    errors: list[str] = []
    for key, value in overrides.items():
        field_ = schema.BY_KEY.get(key)
        if field_ is None or field_.internal:
            raise TypeError(f"unknown override '{key}'.{schema.suggest(key, [k for k in schema.BY_KEY])} Overrides "
                            "are flat config keys, e.g. max_forecast_day, verification_window_list, start_date")
        flat[key] = schema.normalise_value(field_, value, key, errors)
        flat.setdefault("config_sources", {})[key] = "override"
    schema.raise_errors(errors, "<overrides>")
    flat["config_file"] = config_file
    return flat


def _prepare(source, overrides: dict, write: bool, tmp: Path | None) -> dict:
    """The flat config of a run, finished (paths, lead window, run folder plan) and validated."""

    dic = _flat(source, overrides)
    dic = _resolve_paths(dic)
    plan_run(dic, dic.get("config_file"), write)
    HCWFValidator(dic).validate()
    if not write:
        for key, value in _NO_FILES.items():
            if key in dic or key == "explain_cells":
                dic[key] = value
        if tmp is not None:
            dic["dir_out"] = dic["dir_fig"] = str(tmp)
            dic["cache_dir"] = None
    return dic


# ======================================================================= hcwf/export.py

def load_cfg(path) -> dict:
    """The flat config dict of a run YAML as the run itself builds it (paths resolved, lead window shortened).

    Reads the file with the loader's own steps and the validator, but without its global cache.
    """

    dic = build_cfg(["-p", str(path)])
    HCWFValidator(dic).validate()
    return _complete(dic)


def _complete(dic: dict) -> dict:
    """Finish a flat config the way the loader does (paths, lead window); finishing a dict that is already done changes nothing."""
    from hcwf.lib.lead_window import apply_lead_window

    _resolve_paths(dic)
    if "lead_window" not in dic:
        apply_lead_window(dic)
    return dic


def _as_dict(cfg) -> dict:
    """A flat config dict from a dict, SimpleNamespace, hcwf.config.Config object or the path of a run YAML."""
    if isinstance(cfg, (str, Path)):
        return load_cfg(cfg)
    if hasattr(cfg, "to_flat"):
        cfg = cfg.to_flat()
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return _complete(dict(cfg))


# ======================================================================= the flows' case loops, written out
# (hcwf/app/spatial_far_mr_mae.py and hcwf/app/bin_skill_score.py: the lines that build the dict handed to
# compute_metrics_multiple_years / prepare_score_cache / create_score_results; the loop bodies are copied, the
# scoring calls are left out)

def deterministic_cases(dic: dict) -> list[tuple[tuple, dict]]:
    """[(combi, case_cfg)] of the forecast loop of spatial_far_mr_mae_map: the setting wins over the case."""
    setting = init_dataclass(Setting, dic)
    out = []
    for combi in list(product(*iter_list(dic))):
        case = make_case(Case, combi, dic)
        out.append((combi, {**asdict(case), **asdict(setting)}))
    return out


def deterministic_reference_cases(dic: dict) -> list[tuple[tuple, dict]]:
    """[(combi, case_cfg_ref)] of the reference loop of spatial_far_mr_mae_map: model_list is (ref_model,) while the other
    *_list keys stay the forecast models' (so the reference inherits models[0]'s accumulation and daily_boundary, and
    model_dims is None)."""
    setting = init_dataclass(Setting, dic)
    cfg_ref = dict(dic)
    cfg_ref["model_list"] = (dic["ref_model"],)
    out = []
    for combi in list(product(*iter_list(cfg_ref))):
        case = make_case(Case, combi, cfg_ref)
        case_ref = {'model_dir': setting.ref_model_dir,
                    'model_var': case.ref_model_var,
                    'file_pattern': setting.ref_model_file_pattern,
                    'unit_cvt': setting.ref_model_unit_cvt,
                    'model_dims': None
                    }
        case.update(case_ref)
        out.append((combi, {**asdict(case), **asdict(setting)}))
    return out


def ensemble_cases(dic: dict) -> list[tuple[tuple, dict, dict]]:
    """[(combi, prep_cfg, case_cfg)] of the loop of skill_score_in_bins: prep_cfg (full day_bins) goes to
    prepare_score_cache, case_cfg (day bins inside the window) to create_score_results; the case wins over the setting."""
    setting = init_dataclass(Setting, dic)
    out = []
    for combi in list(product(*iter_list(dic))):
        case = make_case(Case, combi, dic)
        full_day_bins = dic["day_bins"]
        day_bins_filtered = filter_bins_in_window(case.day_bins, case.verification_window)
        prep_case = copy.copy(case)
        prep_case.day_bins = full_day_bins
        prep_cfg = {**asdict(setting), **asdict(prep_case)}
        case.day_bins = day_bins_filtered
        out.append((combi, prep_cfg, {**asdict(setting), **asdict(case)}))
    return out


def ensemble_reference_cases(dic: dict) -> list[tuple[tuple, dict]]:
    """[(combi, ref_kwargs)]: what prepare_score_cache / create_score_results read for the reference model of a
    probabilistic run: the current model's case with reference_model_kwargs applied (the current model's accumulation and
    daily_boundary, ref_model_dims)."""
    return [(combi, reference_model_kwargs(prep_cfg)) for combi, prep_cfg, _ in ensemble_cases(dic)]


# ======================================================================= hcwf/driver.py:_run (the config part)

def driver_finish(cfg, cli_args=None) -> dict:
    """The flat config driver._run hands to execute(): from the CLI via load_run_config, or from a given dict or
    namespace (planned when it has no run_paths) and validated."""
    if cfg is None:
        dic = load_run_config(cli_args)
    else:
        dic = dict(vars(cfg)) if isinstance(cfg, SimpleNamespace) else dict(cfg)
        if "run_paths" not in dic:
            plan_run(dic, dic.get("config_file"))
        HCWFValidator(dic).validate()
    return dic


# ---------------------------------------------------------------------------------------------------------------
# Validation as it was before step S2 (hcwf/lib/assertion.py HCWFValidator, hcwf/lib/onset_config.py check_config, and
# the problems hcwf/lib/check.py `_units` and `_blend` reported), kept to compare hcwf.config.validate with.
# ---------------------------------------------------------------------------------------------------------------
from datetime import datetime
from hcwf.lib.onset_config import OnsetConfigError as _OnsetConfigError, onset_setup as _onset_setup


def check_config(dic: dict) -> None:
    """Validate a raw config dict: keys earlier versions read (the flat onset keys and the other removed ones) are
    refused with the message a YAML config gets, then the onset part is fully resolved."""
    from hcwf.lib import config_schema as schema

    removed = [key for key in schema.REMOVED_KEYS if dic.get(key) is not None]
    if removed:
        raise _OnsetConfigError("\n".join(schema.unknown_key_message(key) for key in removed))
    _onset_setup({"onset_definition": None, **dic})


class HCWFValidator:
    """Collects consistency errors for a configuration dict and raises them together."""

    def __init__(self, config_dict):
        """Wrap the config dict in a namespace and start with no errors."""
        # Convert the dictionary to an object so we can use self.cfg.param
        self.cfg = SimpleNamespace(**config_dict)
        self.errors = []

    def _add_error(self, msg):
        """Record one error message."""
        self.errors.append(msg)

    def validate(self):
        """Runs all consistency and logic checks."""
        self._check_temporal()
        self._check_list_lengths()
        self._check_model_names()
        self._check_tuple_ranges()
        self._check_forecast_limits()
        self._check_probabilistic_constraints()
        self._check_onset()
        self._check_explain_cells()

        if self.errors:
            report = "\n".join([f"  • {err}" for err in self.errors])
            raise HCWFConfigError(f"HCWF Configuration is invalid:\n{report}")
        return True

    def _check_temporal(self):
        """Check start_date/end_date ordering and the climatology year range."""
        try:
            sd = datetime(*self.cfg.start_date)
            ed = datetime(*self.cfg.end_date)
            if sd >= ed:
                self._add_error(f"start_date ({sd.date()}) is after end_date ({ed.date()})")

            sy = self.cfg.start_year_clim
            ey = self.cfg.end_year_clim
            if sd > ed:
                self._add_error(f"start_year_clim ({sy}) is after end_year_clim ({ey})")

            for key, name, first, last, bounds in (
                    ("years", "evaluation.years", sd.year, ed.year, "start_date and end_date"),
                    ("years_clim", "evaluation.climatology_years", sy, ey, "start_year_clim and end_year_clim")):
                outside = [y for y in getattr(self.cfg, key, None) or () if not first <= y <= last]
                if outside:
                    self._add_error(f"{name} {outside} lie outside {first}-{last}, the range of {bounds}. "
                                    f"Fix: list only years in that range, or widen the range")

        except Exception as e:
            self._add_error(f"Date formatting error: {e}")

    def _check_list_lengths(self):
        """Check that per-model _list entries match model_list (tolerance_days_list: verification windows)."""
        # We use the length of model_list as the 'Master' length
        master_len = len(self.cfg.model_list)

        # Access the underlying dictionary to find all keys ending in '_list'
        for key, value in vars(self.cfg).items():
            if key.endswith('_list') and isinstance(value, (list, tuple)):
                if key not in ('verification_window_list', 'tolerance_days_list'):
                    if len(value) != master_len:
                        self._add_error(f"Length Mismatch: '{key}' has {len(value)} items, \
                                        expected {master_len}")
                if key in ('tolerance_days_list',):
                    if len(value) != len(self.cfg.verification_window_list):
                        self._add_error(f"Length Mismatch: '{key}' has {len(value)} items, \
                                        expected {len(self.cfg.verification_window_list)}")

    def _check_model_names(self):
        """Check that model names contain no spaces."""
        # 1. Ensure no spaces in model names (critical for file paths/naming)
        for model in self.cfg.model_list:
            if " " in model:
                self._add_error(f"Invalid model name: '{model}' (Spaces are not allowed).")

    def _check_tuple_ranges(self):
        """Check that each (start, end) pair in verification windows and day_bins has start < end."""
        # 2. Ensure (start, end) pairs are logically ordered (start < end)
        for attr_name in ['verification_window_list', 'day_bins']:
            windows = getattr(self.cfg, attr_name)
            for window in windows:
                if window[0] >= window[1]:
                    self._add_error(f"Invalid range in {attr_name}: {window}. \
                            First number must be smaller than the second.")

    def _check_forecast_limits(self):
        """Check that no verification window extends beyond max_forecast_day."""
        # 3. Largest window value cannot exceed max_forecast_day
        max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
        if max_v_day > self.cfg.max_forecast_day:
            self._add_error(f"Verification window limit ({max_v_day}) \
                    exceeds max_forecast_day ({self.cfg.max_forecast_day}).")

    def _check_probabilistic_constraints(self):
        """In probabilistic mode, check the largest window end covers the largest day_bin end."""
        # Using .get() in case the key is missing to avoid AttributeError
        is_prob = getattr(self.cfg, 'probabilistic', False)

        if is_prob:
            # 5. Window max cannot be smaller than day_bins max
            max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
            max_bin_day = max([pair[1] for pair in self.cfg.day_bins])
            if max_v_day < max_bin_day:
                self._add_error(f"Window mismatch: Largest verification day \
                        ({max_v_day}) is smaller than largest day_bin ({max_bin_day}).")

    def _check_onset(self):
        """Check the onset settings resolve: every needed setting is known."""

        try:
            check_config(vars(self.cfg))
        except _OnsetConfigError as exc:
            self._add_error(str(exc))

    def _check_explain_cells(self):
        """Check explain_cells is None or a list of (lat, lon) pairs of numbers."""
        cells = getattr(self.cfg, "explain_cells", None)
        if cells is None:
            return
        ok = isinstance(cells, (list, tuple)) and all(
            isinstance(c, (list, tuple)) and len(c) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in c) for c in cells)
        if not ok:
            self._add_error("explain_cells must be None or a list of (lat, lon) pairs of numbers, "
                            "e.g. [(9.0, 38.75), (8.5, 39.0)]")

    def _check_graphics_dependencies(self):
        """Validates that plot settings match data availability and output requirements."""

        # 1. Panel Error Heatmap requires multiple items to compare (Matrix layout)
        if getattr(self.cfg, 'plot_panel_heatmap_error', False):
            if len(self.cfg.verification_window_list) <= 1 or len(self.cfg.model_list) <= 1:
                self._add_error("Graphics Conflict: 'plot_panel_heatmap_error' \
                        requires >1 model AND >1 verification window.")

        # 2. Panel Skill Heatmap logic
        if getattr(self.cfg, 'plot_panel_heatmap_skill', False):
            if len(self.cfg.verification_window_list) != 1:
                self._add_error("Graphics Conflict: 'plot_panel_heatmap_skill' \
                        only supports exactly ONE verification window.")

        # 3. Bar plots for Skill Scores (BSS/RPSS/AUC)
        if getattr(self.cfg, 'plot_bar_bss_rpss_auc', False):
            if len(self.cfg.verification_window_list) != 1:
                self._add_error("Graphics Conflict: 'plot_bar_bss_rpss_auc' \
                        only supports exactly ONE verification window.")


def check_units_problem(dic):
    """The problem `_units` of check.py reported: the text of the error line, or None."""
    from hcwf.spatial.units import UnitsSpec, UnitsSpecError

    try:
        {name: UnitsSpec.from_flat(dic, name) for name in ["observations", *dic["model_list"]]}
    except UnitsSpecError as exc:
        return f"units: {exc}"
    return None


def check_blend_problems(dic):
    """The error lines `_blend` of check.py reported."""
    from hcwf.blend.recipe import RecipeError, recipe_from_config

    try:
        recipe_from_config(dic, _onset_setup(dic))
    except RecipeError as exc:
        return [f"blend: {problem}" for problem in exc.problems]
    except Exception as exc:  # noqa: BLE001
        return [f"blend: {exc}"]
    return []


# ======================================================================= hcwf/stats/observed_cache.py (step S7)
# The observed-onset cache key as it was before the typed fingerprint_parts: the parts of one detection from the flat
# kwargs. (Calls the live `onset_setup`, `UnitsSpec.from_flat`, `units_files` and the typed options of the readers.)

def _fp_years(kwargs):
    years = set()
    for key in ("years", "years_clim"):
        value = kwargs.get(key)
        if value and not isinstance(value, str):
            years.update(int(y) for y in value)
    return sorted(years)


def _fp_obs_files(kwargs, years):
    from hcwf.cache import file_identity
    from hcwf.io.input import is_zarr, year_files

    if kwargs.get("obs_paths"):
        found = []
        for year in years:
            files = year_files(kwargs["obs_paths"], kwargs.get("obs_dir") or None, kwargs.get("obs_files_years"), year)
            found.append([file_identity(f) for f in files] or [f"missing {year}"])
        return found
    if is_zarr(kwargs["obs_dir"]):
        return [file_identity(kwargs["obs_dir"]), list(years)]
    found = []
    for year in years:
        for pattern in kwargs.get("obs_file_pattern") or ():
            path = Path(str(kwargs["obs_dir"])) / pattern.format(year)
            if path.exists():
                found.append(file_identity(path))
                break
        else:
            found.append([f"missing {year}"])
    return found


def _fp_mask_files(kwargs):
    from hcwf.cache import file_identity

    paths = []
    mask = kwargs.get("mask")
    if isinstance(mask, dict):
        paths += [mask.get("file"), (mask.get("condition") or {}).get("file")]
    paths += [kwargs.get("nc_mask"), kwargs.get("shpfile_dir")]
    return [file_identity(p) for p in paths if p]


def _fp_reference_part(reference):
    if reference is None:
        return None
    table = None if reference.table is None else reference.table.to_dict("list")
    return (reference.month_day, table, reference.computed, reference.statistic)


def _fp_options(kwargs):
    from typed_io import reader_inputs

    got = reader_inputs(kwargs)
    options = {**got.options.ingest(False, got.obs.units, check_units=got.select.check_units)}
    if kwargs.get("onset_horizon"):
        options["horizon"] = kwargs["onset_horizon"]
    if kwargs.get("reference_onset") is not None:
        options["reference_onset"] = _fp_reference_part(kwargs["reference_onset"])
    return options


def fingerprint_parts(rain, thresh, kwargs) -> dict:
    """The hash of each fingerprint part for one observed-onset detection, from the flat kwargs (the kwargs version)."""
    import json

    import numpy as np

    from hcwf.cache import part_hash
    from hcwf.io.input import units_files
    from hcwf.spatial.units import UnitsSpec
    from hcwf.lib.onset_config import onset_setup
    from hcwf.onset import definition as definition_mod
    from hcwf.stats.observed_cache import threshold_hash

    setup = onset_setup(kwargs)
    years = _fp_years(kwargs)
    spec = UnitsSpec.from_flat(kwargs, "observations")
    parts = {
        "obs_files": part_hash(_fp_obs_files(kwargs, years)),
        "obs_ingest": part_hash(kwargs.get("obs_var"), kwargs.get("obs_unit_cvt"), setup.obs_day_boundary,
                                kwargs.get("obs_file_pattern")),
        "definition": part_hash(definition_mod.definition_hash(setup.definition), setup.season, setup.dry_check),
        "thresholds": threshold_hash(thresh),
        "years": part_hash(years),
    }
    if spec is None:
        lat, lon = (np.asarray(rain.coords[name] if isinstance(rain.coords, dict) else rain[name].values)
                    for name in ("lat", "lon"))
        parts["cells"] = part_hash(kwargs.get("region"), json.dumps(kwargs.get("mask"), sort_keys=True, default=str),
                                   _fp_mask_files(kwargs), bool(kwargs.get("polygon")), bool(kwargs.get("grid_point")),
                                   kwargs.get("lat_select") if kwargs.get("grid_point") else None,
                                   kwargs.get("lon_select") if kwargs.get("grid_point") else None, lat, lon)
    else:
        parts["units"] = part_hash(repr(spec), units_files(spec), kwargs.get("obs_dims"),
                                   sorted(kwargs.get("keep_unit_ids") or ()), setup.lead_label)
    options = _fp_options(kwargs)
    if options:
        parts["options"] = part_hash(options)
    return parts
