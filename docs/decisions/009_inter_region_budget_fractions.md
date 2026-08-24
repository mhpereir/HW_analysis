# Decision 009: Inter-Region Heat-Budget Fractions

## Status

Draft for the inter-region comparison branch.

## Decision

The inter-region heat-budget diagnostic compares existing Stage-2 event and
clean-baseline rows using signed fractions of gross pre-peak budget activity.
For each row, define:

```text
G = abs(I_adiabatic_pre) + abs(I_advection_pre) + abs(I_diabatic_pre)

f_adiabatic = I_adiabatic_pre / G
f_advection = I_advection_pre / G
f_dyn = I_dyn_pre / G
f_diabatic = I_diabatic_pre / G
```

Rows with non-finite required variables or `G <= 0` have undefined fractions
and are excluded from the corresponding distribution summaries. The
diagnostic reads the canonical stored `I_dyn_pre` variable and verifies its
documented identity with `I_adiabatic_pre + I_advection_pre`; it does not
reconstruct `I_dyn_pre` for plotting.

The diagnostic consumes only variables already present in the Stage-2 event
and baseline-day products:

- `I_adiabatic_pre`;
- `I_advection_pre`;
- `I_dyn_pre`;
- `I_diabatic_pre`; and
- `event_adjacent` for selecting clean baseline rows.

The existing Stage-2 tables remain authoritative and unchanged. The fractions
and regional distribution summaries are lightweight deterministic reductions
computed in memory through reusable code in `src/diagnostics.py`.

## Figure contract

One figure compares all requested regions on common axes. Regions occupy rows
and the four signed fractions occupy columns. A fifth column reports gross
budget activity `G` in kelvin so relative composition is not confused with
absolute magnitude.

For each region and quantity, the figure shows:

- the median;
- the interquartile range as a thick interval;
- the 10th to 90th percentile range as a thin interval;
- events as the foreground population; and
- clean baseline days, where `event_adjacent == 0`, as a lighter comparison
  population.

All fraction panels use a fixed `[-1, 1]` scale centered on zero. Positive
fractions indicate heating and negative fractions indicate cooling. Region
order is explicit and remains the same in every panel.

## Rationale

The existing per-region scatter matrices preserve absolute contribution
magnitudes and event-versus-baseline point clouds, but independent regional
axis scaling and large baseline populations make inter-region composition
differences difficult to compare. Direct ratios such as
`I_diabatic_pre / I_dyn_pre` are unsuitable because net dynamical heating can
cross or approach zero when adiabatic and advective contributions cancel.

Normalizing by the sum of absolute component magnitudes keeps every component
fraction bounded, preserves whether a component heats or cools, and exposes
internal dynamical cancellation. The separate gross-activity column retains
the absolute-magnitude context lost during normalization.

These fractions describe signed budget composition. They do not assign causal
attribution and do not imply that the component medians sum to one after
summarizing rows independently.

## Compatibility

Stage 1 and Stage 2 product contracts are unchanged. Existing regional plots,
schedulers, and output paths remain unchanged. The new diagnostic writes a
distinct inter-region figure and can consume either the accepted 96-hour
Stage-2 products or a separately identified compatible Stage-2 run, but all
regions in one figure must use matching window and integration semantics.

## Validation

- Synthetic reduction tests must cover signs, bounded component fractions,
  stored `I_dyn_pre` use, identity mismatch, non-finite rows, and zero gross
  activity.
- Plot tests must verify one row per region, common fraction limits, event and
  clean-baseline interval layers, readable region labels, and the gross
  activity panel.
- CLI tests must verify explicit region-to-event-to-baseline path mapping and
  non-overwriting output behavior.
- A real-data draft must validate compatible Stage-2 metadata, finite summary
  populations, all seven expected regions, and original-resolution figure
  readability before scientific interpretation.
