# Decision 009: Antecedent Temperature and Budget Initial State

## Status

Accepted. Implement alongside the existing Stage-2 fixed-window features.

## Definitions

Let the anchor be event `peak_time` or baseline `reference_time`, and let
`b = WINDOWS["heat_budget_pre"][0]` in hours. Configure only the positive
integer `ANTECEDENT_TEMPERATURE_DURATION_HOURS = 72`. Derive the antecedent
interval as `[b-72,b)` and the initial-state timestamp as `anchor+b` at run
time. Neither the initial lag nor the antecedent endpoint is independently
configurable. Require a pre-anchor budget ending at zero.

Store TAS and atmospheric `T_mean` anomalies averaged over that interval and
sampled exactly at the budget start. Also store TAS anomaly exactly at the
anchor. For the current budget these are 72 samples in `[-168,-96)` and a
point at `-96h`; a 48-hour budget gives `[-120,-48)` and `-48h`.

TAS uses Stage-1 `tas_anom` or `tas_region-tas_climatology`, preserving the
baseline of `tas_anom_peak`. Atmospheric temperature uses the matching Stage-1
regional calendar-hour climatology companion. These are different documented
climatology sources, not interchangeable anomaly baselines.

Means require the complete expected hourly timestamp sequence and all finite
values. Missing samples yield NaN, with timestamp and finite-value counts.
Exact points never interpolate or select the nearest timestamp. Include the
antecedent interval in boundary checks and the baseline event-adjacency union;
do not clip source windows to the selected season.

## Compatibility and interpretation

Keep `tas_anom_peak` as the maximum over the event, not the value at its
anchor (decision 005). Keep `T_anom_mean_ant` on inclusive `[-168,-24]`.
Existing budget integrals remain inclusive hourly sums (97 samples for 96h).
Do not anomalize their source fields when adding the new temperature metrics.

The new event-only 2x2 diagnostic uses anchor TAS anomaly on every y-axis,
antecedent means on the left, initial states on the right, TAS above and
atmospheric temperature below. Color by stored `I_dTdt_pre` using one scale.
Use one common finite population and highlight the largest stored
`tas_anom_peak`. Lines `y=x+c` on the TAS row denote surface anomaly
differences only, not atmospheric budget integrals. No such lines belong on
the atmospheric-temperature row. This is a diagnostic of pre-existing warmth,
not by itself evidence of its cause.

The simplified budget comparison retains the existing two-panel presentation
layout. Add `x+y=c` references on its diabatic-residual versus net-dynamical
panel, and one shared population annotation. Preserve the full layout.

## Migration and acceptance

Both builders require an explicit matching climatology companion; old Stage-2
products remain valid for old consumers but must be rebuilt for the new plot.
Record resolved lags, endpoint rules, source and climatology provenance in
NetCDF metadata. CSV carries the same values and coverage columns; NetCDF is
the metadata authority.

Test 96h and 48h configurations, exact/half-open boundaries, incomplete and
nonfinite data, event/baseline parity, season/adjacency behavior, incompatible
climatologies, round trips, source immutability, and unchanged legacy features.
Plot tests must cover common masks, normalization, reference-line semantics,
anchor/maximum distinction, and rejection of old products. Validate PNW Hotz
then PNW Bartusek with independent source reductions on Venus, reconcile
anchor-versus-maximum anomaly differences, and inspect original PNGs.
Single-year Eulerian heat-budget figures are outside this change.
