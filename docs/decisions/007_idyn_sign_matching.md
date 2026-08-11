# Decision 007: In-Memory I_dyn Sign Matching

## Status

Accepted for command-center task A2.8.

## Decision

Positive and negative integrated dynamical-heating event populations are
matched in memory from an unchanged Stage-2 event-feature table. The matching
configuration is a tracked static settings file consumed by plotting and
diagnostic workflows. Matched membership is not published as another durable
data product.

For A2.8, the grouping metric is derived at runtime as:

```text
I_dyn = I_adiabatic_pre + I_advection_pre
```

The negative-`I_dyn` population is the reference and the positive-`I_dyn`
population is the candidate population. Events with exactly zero `I_dyn` are
excluded from both populations.

Reusable matching logic belongs in `src/selectors.py`. Plotting scripts may
load the tracked settings and unchanged Stage-2 table, derive `I_dyn`, call the
selector, and use the returned event indices immediately. Plotting scripts do
not implement their own assignment or caliper logic.

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

- the `I_dyn` component variables and reference sign;
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

Stage 1 and Stage 2 products remain unchanged. Existing unmatched figures and
workflows continue to operate without a settings file. Matching-aware figures
consume the same Stage-2 product plus the tracked A2.8 settings.

This decision does not establish one final scientific matching specification.
The settings retain the single-variable and multivariable candidates needed to
evaluate that choice explicitly.

## Validation

- Synthetic selector tests must cover single- and multivariable calipers,
  maximum-cardinality matching, one-to-one assignment, deterministic row-order
  invariance, timedelta conversion, and invalid inputs.
- Settings-loader tests must cover the tracked file and representative invalid
  schemas.
- Plot tests must verify that all figures are generated through the selector
  rather than a private matching implementation.
- A representative PNW Bartusek Stage-2 run must reproduce the documented pair
  counts and SMD diagnostics before A2.8 is completed.
