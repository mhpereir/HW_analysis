# Decision 009: Use 48-Hour Stage-2 Integration Windows

## Status

Accepted.

## Decision

The active Stage-2 event and baseline products use these inclusive integration
windows relative to their row anchor:

```text
heat_budget_pre = (-48, 0) hours
lwa_pre_peak    = (-48, 0) hours
```

For event rows, the anchor is `peak_time`. For baseline rows, the anchor is
`reference_time`, and the LWA window is exposed as `lwa_pre_reference` in
baseline-facing metadata.

The established endpoint convention remains unchanged. A complete hourly
slice from lag -48 through lag 0 contains 49 samples. The 48-hour name denotes
the lag span and is not a claim that the inclusive sum contains only 48 hourly
values.

The `antecedent_state`, `antecedent_change`, `near_peak`, and `decay` windows
remain unchanged. In particular, this decision does not shorten the
antecedent-state mean or redefine event or baseline populations.

## Rationale

The analysis should emphasize dynamical and LWA contributions closer to the
event peak or baseline reference time. Applying the same 48-hour lag span to
event and baseline products preserves direct comparison and keeps the stored
`I_dyn_pre` identity unchanged:

```text
I_dyn_pre = I_adiabatic_pre + I_advection_pre
```

## Compatibility

This decision supersedes only the 96-hour `heat_budget_pre` and `lwa_pre_peak`
defaults in [decision 001](001_event_feature_windows.md). Existing 96-hour
datasets, figures, matching summaries, and spatial composites remain valid for
their recorded configuration and must not be overwritten or relabeled as
48-hour products.

New production datasets and figures must use isolated run namespaces that
identify the 48-hour configuration. Consumers must verify the stored window
metadata rather than infer it from a generic filename.

Changing the integration window can change `I_dyn_pre` magnitude and sign.
Matching and spatial workflows are therefore not interchangeable across the
96-hour and 48-hour datasets. They require separate regeneration if a future
analysis elects to consume the 48-hour products.

## Validation

- Synthetic event and baseline tests require 49 samples for both complete
  integration windows and verify their `-48,0` metadata.
- Event and baseline products must retain identical integration definitions
  and exact `I_dyn_pre` closure, including missing-value propagation.
- Production runs must preserve the event and baseline populations defined by
  Stage 1 while recording complete sample counts, finite required fields, and
  run-scoped provenance.
- Event-only and event-versus-clean-baseline figures must consume only the new
  48-hour Stage-2 tables and pass automated artifact checks plus
  original-resolution visual inspection.
