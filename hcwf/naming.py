"""Format tuples as strings, for file names, labels and figure titles.

The helpers here are small and dependency-free. Tuples such as a verification window ``(1, 15)`` or a combination
``('a', (1, 15), 'X')`` are turned into tags like ``a_1-15_X`` that name output files and label plots.
"""

__all__ = ["combi_to_str", "tuple_to_str", "tuple_to_str_range"]


def combi_to_str(combi, sep="_", tuple_sep="-", suffix=""):
    """Convert a combination such as ``('a', (1, 15), 'X')`` into a string tag.

    Strings and numbers are used as they are; tuples are joined with ``tuple_sep``.

    Parameters
    ----------
    combi : tuple
        The items of the combination; each is a string, a number or a tuple of those.
    sep : str, default "_"
        Separator between the items.
    tuple_sep : str, default "-"
        Separator between the elements of an item that is a tuple.
    suffix : str, default ""
        Text appended to the result.

    Returns
    -------
    str
        The joined tag, for example ``"a_1-15_X"``.
    """
    parts = []
    for item in combi:
        if isinstance(item, tuple):
            # join tuple elements with tuple_sep
            parts.append(tuple_sep.join(map(str, item)))
        else:
            parts.append(str(item))
    return sep.join(parts) + suffix


def tuple_to_str(item):
    """Join the elements of a tuple with ``"-"``.

    Parameters
    ----------
    item : tuple
        The elements to join; each is converted with ``str``.

    Returns
    -------
    str
        The joined text, for example ``"1-15"`` for ``(1, 15)``.
    """
    return "-".join(map(str, item))


def tuple_to_str_range(item):
    """Join only the first and last element of a tuple, ignoring the middle ones.

    Parameters
    ----------
    item : tuple
        The elements; may be empty.

    Returns
    -------
    str
        ``""`` for an empty tuple, the single element for a one-element tuple, else ``"first-last"``.
    """
    if len(item) == 0:
        return ""
    elif len(item) == 1:
        return str(item[0])
    else:
        return f"{item[0]}-{item[-1]}"
