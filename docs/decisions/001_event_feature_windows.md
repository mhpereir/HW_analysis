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
`scripts/event_features/event_feature_config.py` and captures the four days
leading into and including the peak timestamp.

## Consequences

### Explicit sensitivity windows

Both Stage-2 builders accept a positive integer `--integration-hours H`.
Omitting it retains the 96-hour default. An override changes only
`heat_budget_pre` and `lwa_pre_peak` to `(-H, 0)`; all other windows and
Stage-1 event definitions remain unchanged. Configuration belongs to each
builder call and must not mutate process-global defaults. Feature metadata,
boundary eligibility, sample counts, and baseline event adjacency must all
use the resolved windows. Matching event and baseline tables must use the
same override.

The PNW sensitivity campaign uses 96, 168, 336, and 504 hours (4, 7, 14, and 21 days)
for TAS q90, surface-to-700-hPa, 1940-2024 JJA in `pnw_bartusek` and
`pnw_hotz`. Complete inclusive hourly windows contain 97, 169, 337, and 505
samples. Rebuild both Stage-2 products for all four windows, retaining accepted
48/96-hour products, and write each new region/window pair to a fresh namespace.
Select JJA events/references while
retaining the full Stage-1 time axis, including May for early-June anchors.

These are absolute-temperature budget integrals, including both warming and
cooling. They do not establish uninterrupted heating, anomaly onset, or an
event-specific development duration. The 2021 event remains in the reference
heatwave population. The original seven-day antecedent-state diagnostic does
not move with the integration start and must not be described as preceding a
longer budget. Longer baseline windows can contain earlier events; recompute
adjacency and report clean-baseline counts instead of assuming fixed membership.

Validate synthetic long windows, independent source reductions, no leakage
between successive overrides, actual metadata/counts, coverage, population
selection, baseline adjacency, and heat-budget closure. The dedicated
`schedule_stage2_window_sensitivity.sh` builds and independently validates
both tables per job, retaining hashes, configuration, source provenance and
2021 integrated-heating ranks beside the products. Render the established
event-versus-clean-baseline adiabatic/advection comparison in both full and
presentation layouts from each matching pair, using distinct filenames and
the shared plotting style. Plot interpretation must account for the changing
clean-baseline membership. Run the longest-window canary before the remaining
campaign. Scientific acceptance additionally
requires scheduler/log review; a validation file alone is insufficient.

- Inclusive hourly windows contain 97 samples for `(-96,0)` when all hourly
  timestamps are present.
- Sample-count variables must be retained so downstream consumers can identify
  boundary events or missing data.
- Adaptive growth-window features based on `dTdt > 0` remain out of scope for
  this product version.
