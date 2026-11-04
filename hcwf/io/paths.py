"""Helper for resolving file-system paths."""
from pathlib import Path

__all__ = ["set_dir"]


def set_dir(path_str, work_dir=None):
    """Resolve a path string into an absolute, OS-agnostic path.

    ``~`` is expanded. A relative path is resolved against ``work_dir`` if given, otherwise against the current
    directory.

    Parameters
    ----------
    path_str : str or pathlib.Path
        The path to resolve; backslash and slash separators are both understood.
    work_dir : str or pathlib.Path, optional
        The folder a relative path is taken from; None (or an empty value) means the current directory.

    Returns
    -------
    pathlib.Path
        The absolute, resolved path.
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
