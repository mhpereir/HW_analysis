# Decision 001: Event-feature Windows

## Status

Accepted current default.

## Decision

Event features use fixed timestamp windows relative to each event `peak_time`.
Timestamp slices are inclusive.

```text
heat_budget_pre      = (-96, 0) hours
lwa_pre_peak         = (-96, 0) hours
antecedent_state     = (-168, -24) hours
antecedent_change    = (-168, 0) hours
near_peak            = (-24, 24) hours
decay                = (0, 72) hours
```

## Rationale

Fixed windows make the Stage-2 feature table reproducible and easy to compare
across events. Centering on `peak_time` keeps the feature product aligned with
the event summaries in Stage 1.

The current `lwa_pre_peak=(-96,0)` contract matches
`scripts/event_features/event_feature_config.py` and keeps LWA exposure strictly
pre-peak through the peak timestamp.

## Consequences

- Inclusive hourly windows contain 97 samples for `(-96,0)` when all hourly
  timestamps are present.
- Sample-count variables must be retained so downstream PCA and clustering can
  identify boundary events or missing data.
- Adaptive growth-window features based on `dTdt > 0` remain out of scope for
  this product version.
