"""Helpers for resolving file-system paths and restoring keyword-only arguments."""
import inspect
from pathlib import Path


def set_dir(path_str, work_dir=None):
    """
    Resolve a path string into an absolute, OS-agnostic Path.
    Relative paths are resolved against `work_dir` if given, otherwise the current directory.
    """
    # 1. Convert to an OS-agnostic Path object (handles \ and / automatically)
    p = Path(path_str).expanduser()

    if p.is_absolute():
        return p.resolve()

    elif not work_dir:
        resource_target = p.resolve()
    else:
        wd = Path(work_dir).expanduser().resolve()
        resource_target = (wd / p).resolve()

    return resource_target


def restore_args(func, kwargs, bound_args):
    """
    Restore keyword-only parameters of `func` back into kwargs.
    """
    sig = inspect.signature(func)
    new_kwargs = dict(kwargs)

    for name, param in sig.parameters.items():
        if (
            param.kind is param.KEYWORD_ONLY
            and name in bound_args
            and name not in new_kwargs
        ):
            new_kwargs[name] = bound_args[name]

    return new_kwargs



