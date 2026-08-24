# Decision 007: In-Memory I_dyn Sign Matching

## Status

Accepted for command-center task A2.8.

## Decision

Positive and negative integrated dynamical-heating event populations are
matched in memory from the canonical `I_dyn_pre` variable in the Stage-2
event-feature table. The matching configuration is a tracked static settings
file consumed by plotting and diagnostic workflows. Matched membership is not
published as another durable data product.

The Stage-2 builders calculate the grouping metric as:

```text
I_dyn_pre = I_adiabatic_pre + I_advection_pre
```

The negative-`I_dyn_pre` population is the reference and the positive-
`I_dyn_pre` population is the candidate population. Events with exactly zero
`I_dyn_pre` are excluded from both populations.

Reusable matching logic belongs in `src/selectors.py`. Plotting scripts may
load the tracked settings and Stage-2 table, read `I_dyn_pre`, call the selector,
and use the returned event indices immediately. Plotting scripts do not
reconstruct the dynamical sum or implement their own assignment or caliper
logic.

## Matching method

The selector performs deterministic one-to-one matching without replacement.
It supports one or more event-level matching variables and a positive
pooled-standard-deviation caliper for each variable.

For each matching variable:

1. calculate the pooled within-group standard deviation from the complete
   positive and negative candidate populations;
2. divide every absolute cross-group pair difference by that scale; and
3. reject a pair if any standardized difference exceeds that variable's
   configured caliper.

Among the valid pairs, the assignment first maximizes the number of retained
pairs and then minimizes total root-mean-square standardized distance. Equal
weight is given to every matching variable. Events are sorted by `event_id`
before assignment so results do not depend on Stage-2 row order.

The selector returns matched negative and positive event indices, their event
IDs, pair distances, pooled scales, calipers, and method metadata. It does not
mutate or write the Stage-2 dataset.

## Settings contract

The tracked A2.8 settings define:

- the canonical `I_dyn_pre` group variable and reference sign;
- the supported matching and standardization methods;
- named matching-variable families;
- named specifications linking a family to an SD caliper;
- variables used for the common balance audit; and
- the specifications and caliper grid used by each exploratory figure.

Settings are validated before matching. Unknown methods, missing families,
non-positive or non-finite calipers, duplicate variables, and references to
undefined specifications must fail clearly. Figure-run summaries report the
settings path and SHA-256 checksum.

## Rationale

Matched membership depends on scientific choices such as the matching
variables, calipers, reference population, and desired balance-retention
tradeoff. Multiple useful specifications may coexist and define different
estimands. Embedding one selected population in the canonical Stage-2 table
would make those choices appear to be event features, while publishing every
configuration as a separate product would add avoidable provenance and input
management.

The selection is computationally light relative to plotting and is fully
determined by the Stage-2 table, settings, and selector implementation.
Recomputing it for each figure therefore keeps the canonical data boundary
simple without hiding reusable logic in plotting scripts.

Variables whose SMD improves even though they were not explicitly matched are
described as incidentally co-balanced. This does not, by itself, establish a
linear correlation or causal relationship with a matching variable.

## Compatibility

Stage 1 remains unchanged. Stage-2 event and baseline products gain the
canonical `I_dyn_pre` variable, so existing generated Stage-2 files must be
rebuilt before updated consumers run. Matching-aware figures consume the
rebuilt event product plus the tracked A2.8 settings. The stored equation is
unchanged, but changing the canonical integration window can change
its magnitude, sign, and resulting matched membership relative to preliminary
or historical Stage-2 diagnostics. Those diagnostics must therefore be
regenerated whenever they adopt a rebuilt Stage-2 table.

This decision does not establish one final scientific matching specification.
The settings retain the single-variable and multivariable candidates needed to
evaluate that choice explicitly.

The matched face-advection climatological-anomaly figure is a separate
consumer, not a replacement for the all-event face-advection figure. Its
production scheduler explicitly selects the tracked `peak_anomaly_0p20`
specification. The renderer uses component color and sign-population line style
as independent encodings: positive `I_dyn_pre` is solid and negative
`I_dyn_pre` is dashed.

The matched spatial workflow is also a separate consumer. Because the existing
spatial product stores group means rather than per-event spatial fields, the
matched population must be selected before spatial averaging. A dedicated
builder therefore recomputes `peak_anomaly_0p20` membership in memory from the
canonical Stage-2 table, then writes a separate durable matched composite with
the selected event IDs, pair IDs, pair distances, settings checksum, Stage-2
checksum, and matching-method metadata. This is a derived physical composite,
not a standalone matched-membership product. The unmatched spatial composite
and plot remain unchanged.

## Validation

- Synthetic selector tests must cover single- and multivariable calipers,
  maximum-cardinality matching, one-to-one assignment, deterministic row-order
  invariance, timedelta conversion, and invalid inputs.
- Settings-loader tests must cover the tracked file and representative invalid
  schemas.
- Plot tests must verify that all figures are generated through the selector
  rather than a private matching implementation and that consumers read
  `I_dyn_pre` without reconstructing it.
- Matched composite tests must verify exact Stage-2-to-Stage-1 event-ID and
  peak-time alignment, equal positive and negative event counts, separate
  output paths, and the solid-positive/dashed-negative line contract.
- Matched spatial tests must verify selector use, equal sign counts, retained
  event and pair audit variables, matching provenance, a separate product
  marker and paths, the requested lags, and a six-panel figure whose title and
  row labels identify the 0.20 pooled-SD matched `I_dyn_pre` populations.
- The production figure run must retain a machine-readable summary with exact
  Stage-2 input and settings checksums so documented pair counts and SMD values
  can be traced to their inputs.
- A representative 48-hour PNW Bartusek Stage-2 run must refresh the documented
  pair counts and SMD diagnostics before any matching result derived from a
  48-hour product is accepted.
