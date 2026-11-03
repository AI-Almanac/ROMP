"""Helper for resolving file-system paths."""
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
