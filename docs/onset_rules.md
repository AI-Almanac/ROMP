# Onset rules

ROMP detects rainy-season onset with a pluggable **onset rule**. One rule object
is applied to both observations and forecasts, so a benchmark can never compare
a forecast under one definition against observations under another.

## Choosing a rule

In `config.in`:

```python
onset_rule = "two_stage"
onset_rule_params = {"stage1_days": 10, "stage1_mm": 10, "stage2_days": 20, "stage2_mm": 20}
```

On the CLI: `--onset_rule two_stage --onset_rule_params '{"stage1_mm": 10}'`.
In containers: `ROMP_ONSET_RULE=two_stage ROMP_ONSET_RULE_PARAMS='{"stage1_mm": 10}'`.

Parameters not given take the rule's defaults. The default rule is `legacy`,
which reads the original `wet_init / wet_spell / dry_spell / dry_extent` keys,
so existing configs run unchanged.

Rules are validated at config load: unknown names, unknown parameters, and
out-of-bounds values are reported before any data is read.

## Built-in rules

| name | definition | parameters (defaults) |
|---|---|---|
| `legacy` | Every day of a `wet_spell`-day window ≥ `wet_init` mm and the window total > threshold. If `dry_extent > wet_spell`, rejected when a run of `dry_spell` days < `wet_init` occurs within `dry_extent` days of the candidate. | `wet_init` 1.0, `wet_spell` 3, `dry_spell` 0, `dry_extent` 0 |
| `two_stage` | ≥ `stage1_mm` over the first `stage1_days`, then ≥ `stage2_mm` over the following `stage2_days` (NiMet-style). Ignores the threshold file. | `stage1_days` 10, `stage1_mm` 10, `stage2_days` 20, `stage2_mm` 20, `inclusive` True |
| `moron_robertson` | First day > `wet_day_mm` and the `wet_days` total > threshold; not followed within `follow_days` by any `dry_window`-day period totalling < `dry_min_mm`. | `wet_day_mm` 1.0, `wet_days` 5, `threshold` None (= threshold file), `follow_days` 30, `dry_window` 10, `dry_min_mm` 5 |

"Threshold" is the per-cell value from `thresh_file`, or the scalar
`wet_threshold` when no file is given.

Every rule reports a `lookahead`: how many days after a candidate day it needs
to see. Forecasts must supply `max_forecast_day + lookahead` steps; `two_stage`
at its defaults needs 29 days of lookahead, so with a 30-day forecast only very
early onsets are verifiable. The validator warns when `lookahead >=
max_forecast_day`.

Programmatic access:

```python
from momp.stats.onset_rule import list_rules, resolve_rule
list_rules()                                   # {"legacy": LegacyRule, ...}
rule = resolve_rule("two_stage", {"stage1_mm": 12})
rule.lookahead, rule.label(), rule.fingerprint(), rule.describe()
```

`label()` is `name` at defaults and `name (k=v, ...)` otherwise; `fingerprint()`
is a stable hash of the resolved definition. Use them to tag results so
runs under different definitions are never compared as if they were the same.

## Writing a rule

A rule is a frozen dataclass subclassing `OnsetRule`. Parameters are declared
with `param(...)`, whose bounds and units drive validation and any UI:

```python
from dataclasses import dataclass
from typing import ClassVar

from momp.stats.onset_rule import OnsetRule, finalize, lead_all, lead_sum, param, register


@register
@dataclass(frozen=True)
class WetDaysThenTotal(OnsetRule):
    name: ClassVar[str] = "wet_days_then_total"
    doc: ClassVar[str] = "n_wet consecutive wet days, then total_mm over the next total_days."

    wet_mm: float = param(1.0, min=0, max=50, unit="mm", doc="Wet-day threshold")
    n_wet: int = param(3, min=1, max=15, unit="days")
    total_days: int = param(10, min=1, max=60, unit="days")
    total_mm: float = param(30.0, min=0, max=500, unit="mm")

    @property
    def lookahead(self) -> int:
        return self.n_wet + self.total_days - 1

    def __call__(self, rain, thresh, *, dim):
        wet = lead_all(rain >= self.wet_mm, dim, self.n_wet)
        total = lead_sum(rain, dim, self.total_days, start=self.n_wet)
        return finalize(wet & (total >= self.total_mm))
```

### The contract

`rule(rain, thresh, dim=...)` receives daily rainfall (mm) with an arbitrary
set of dimensions, one of which is `dim` (`"time"` for observations, `"step"`
for forecasts), and returns a **bool** array of the same shape where
`mask[d]` is True iff a season starting on day `d` satisfies the definition.

* **Causal, bounded lookahead** — `mask[d]` depends only on
  `rain[d : d+lookahead+1]`.
* **Fails closed** — NaN inside the inspected window, or fewer than
  `lookahead` days remaining, gives False.
* **Broadcasts** — only `dim` is touched; other dimensions pass through.
* `thresh` is the per-cell threshold; a rule may use it or ignore it.

### Window helpers

All are aligned to the candidate day `d` and return NaN where the window
overruns the series or contains NaN, so missing data fails closed by default.

| helper | meaning |
|---|---|
| `lead_sum(rain, dim, days, start=0)` | total over `[d+start, d+start+days)` |
| `lead_all(cond, dim, days, start=0)` | `cond` holds on every day of that window (False if it overruns) |
| `has_dry_run(rain, dim, days, run_days, dry_below, start=0)` | 1.0/0.0 whether a `run_days` run of days `< dry_below` lies inside the window; compare `== 0` to require none |
| `min_sub_sum(rain, dim, days, sub_days, start=0)` | minimum total over every `sub_days` sub-window |
| `finalize(mask)` | NaN → False, cast to bool (call this last) |

### Testing a rule

The conformance suite runs automatically against every registered rule and
checks the contract (dtype/shape, causality, exact lookahead, NaN behaviour,
broadcast invariance, parameter bounds, identity, and a non-degeneracy health
check at defaults). Add a hand-case test for the semantics of your parameters
in `tests/test_onset_rule_builtins.py`, then:

```bash
pytest tests/test_onset_rule_conformance.py tests/test_onset_rule_builtins.py
```

If the conformance suite fails on causality or lookahead, `lookahead` does not
match what `__call__` actually reads; on NaN, a condition is treating missing
data as informative.

## Notes on the refactor

`momp.stats.detect` no longer contains detection logic of its own. The
original scalar and xarray implementations are frozen in
`tests/_reference_detect.py`, and `tests/test_onset_rule_legacy_equivalence.py`
proves `legacy` reproduces them. Two pre-existing inconsistencies were removed
in the process:

* the original observed path used a dry window one day longer than the
  forecast path and skipped the veto when the window overran the series;
* a NaN inside the dry window was treated as "not dry" and let onset through.

Both paths now share `LegacyRule`, which follows the forecast-path semantics
and fails closed on NaN. With `dry_extent = 0` (the shipped default) neither
change has any effect.
