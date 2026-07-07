# Decision 004: Baseline Season And Window Boundaries

## Status

Accepted current default.

## Decision

Season selection for the Stage-2 baseline-day product applies only to each
row's `reference_time`. Fixed feature windows are evaluated against the
complete Stage-1 time axis and are not clipped to the selected season.

For example, with `--season-months 6 7 8`, a June 1 baseline day is retained
because its `reference_time` is in JJA. Its antecedent and accumulation windows
may use timestamps from May.

The same un-clipped windows are used to calculate `event_adjacent`. A selected
event occurring outside JJA still marks a JJA reference day as event-adjacent
when it falls within an active fixed window.

Baseline rows are dropped only when an active required window extends beyond
the complete Stage-1 dataset boundary. Crossing a selected-season boundary is
not a reason to drop or truncate a row.

## Rationale

The season defines the target population of reference days, while the fixed
windows describe the physical history leading into each reference day.
Clipping windows at a season boundary would shorten early-season histories,
make otherwise equivalent rows use different accumulation intervals, and
break direct comparisons with event-centered fixed-window features.

## Consequences

- JJA baseline products represent JJA reference days, not JJA-only source
  samples.
- Early-June features may include May data, and early-September exclusion does
  not alter August reference-day windows.
- Sample-count variables expose missing timestamps within complete dataset
  boundaries; they do not encode season clipping.
- Stage-1 products must include enough pre-season coverage to support the
  configured antecedent windows.
