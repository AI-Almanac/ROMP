"""Helpers for formatting tuples as strings (file names, labels and figure titles)."""


def combi_to_str(combi, sep="_", tuple_sep="-", suffix=""):
    """
    Convert a tuple like ('a', (1,15), 'X') into a string tag:
    - Strings or numbers → as is
    - Tuples → joined with `tuple_sep`
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
    """Join the elements of a tuple with "-"."""
    return "-".join(map(str, item))


def tuple_to_str_range(item):
    """ ignore all middle elements and only join the first and last items of a tuple """
    if len(item) == 0:
        return ""
    elif len(item) == 1:
        return str(item[0])
    else:
        return f"{item[0]}-{item[-1]}"
