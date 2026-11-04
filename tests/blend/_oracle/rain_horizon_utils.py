"""Shared validation for fixed per-model rainfall forecast horizons."""


import numpy as np


def _positive_int(value, field_name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a positive integer, got {value!r}."
        ) from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    return int(numeric)


def resolve_rain_day_max(model_spec, probability_day_max):
    """
    Resolve a model's rainfall horizon and validation policy.

    An omitted rain_day_max preserves the legacy day_max + 10 minimum. An
    explicit value activates exact fixed-horizon validation unless the spec
    explicitly requests ``rain_horizon_policy: truncate``.
    """
    explicit = model_spec.get("rain_day_max") is not None
    value = (
        model_spec["rain_day_max"]
        if explicit
        else _positive_int(probability_day_max, "day_max") + 10
    )
    policy = model_spec.get("rain_horizon_policy")
    if policy is None:
        policy = "exact" if explicit else "legacy"
    policy = str(policy).strip().lower()
    if policy not in {"legacy", "exact", "truncate"}:
        raise ValueError(
            "rain_horizon_policy must be one of: legacy, exact, truncate."
        )
    if explicit and policy == "legacy":
        raise ValueError(
            "An explicit rain_day_max cannot use rain_horizon_policy: legacy; "
            "use truncate to allow later columns while validating days 1..N."
        )
    if not explicit and policy != "legacy":
        raise ValueError(
            f"rain_horizon_policy: {policy} requires an explicit rain_day_max."
        )
    return _positive_int(value, "rain_day_max"), policy


def _parse_day_value(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def validate_day_coordinate(
    day_values, rain_day_max, context, *, allow_extra=False
):
    """Validate a NetCDF lead coordinate against the integers 1..N."""
    rain_day_max = _positive_int(rain_day_max, "rain_day_max")
    parsed = [_parse_day_value(value) for value in day_values]
    invalid = [repr(value) for value, day in zip(day_values, parsed) if day is None]
    valid = [day for day in parsed if day is not None]
    duplicates = sorted({day for day in valid if valid.count(day) > 1})
    actual = set(valid)
    required = set(range(1, rain_day_max + 1))
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    nonpositive = [day for day in extra if day < 1]
    later = [day for day in extra if day > rain_day_max]

    problems = []
    if invalid:
        problems.append(f"non-integer day values={invalid[:10]}")
    if duplicates:
        problems.append(f"duplicate days={duplicates[:10]}")
    if missing:
        problems.append(f"missing days={missing[:10]}")
    if nonpositive:
        problems.append(f"nonpositive days={nonpositive[:10]}")
    if later and not allow_extra:
        problems.append(f"extra days={later[:10]}")
    if problems:
        requirement = (
            f"days 1..{rain_day_max}, with later days permitted"
            if allow_extra
            else f"exactly days 1..{rain_day_max}"
        )
        raise ValueError(
            f"{context}: strict rainfall horizon requires {requirement}; "
            + "; ".join(problems)
        )
    return valid

