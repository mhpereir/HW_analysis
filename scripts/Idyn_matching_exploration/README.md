# Exploratory matching of heatwaves by I_dyn sign

## Status and question

This is an isolated exploration for command-center task A2.8. It is not yet an
active Stage-2 product or scientific default.

The question is whether positive and negative integrated dynamical-heating
heatwaves can be compared after matching their event severity. The initial
severity variable is `tas_anom_peak`, the maximum temperature anomaly within
each detected heatwave event.

For this exploration:

```text
I_dyn = I_adiabatic_pre + I_advection_pre
```

Both terms are Stage-2 sums over the inclusive 72-hour pre-peak window
`(-72, 0)`. Positive `I_dyn` events are the candidate population and negative
`I_dyn` events are the reference population.

## Data snapshot and provenance

The source is the canonical PNW Bartusek surface-to-700 hPa, tas-q90,
1940-2024 Stage-1 product on Venus:

```text
/home/mhpereir/HW_analysis/results/stage1/a2_7_climatology_20260806/
harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc
```

The existing compact Stage-2 table was compared with an in-memory Stage-2
rebuild from that exact Stage-1 file. Event IDs and every variable used here
were identical, including `I_advection_pre`, `I_adiabatic_pre`, peak severity,
duration, season timing, and antecedent anomaly. The compact input used for the
local figures was:

```text
/home/mhpereir/HW_analysis/results/stage2_event_features/
hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc
sha256: 5df97ddaaffb7be26fca0fdfd4979ee728faa27506b8003d101f6ffc59455252
```

The Stage-2 universe requires complete June-August events. It contains 258
events, all with finite and nonzero `I_dyn`:

| I_dyn sign | Events | Mean `tas_anom_peak` |
| --- | ---: | ---: |
| Negative | 90 | 3.179 K |
| Positive | 168 | 3.679 K |

Across all events, the Pearson correlation between `I_dyn` and
`tas_anom_peak` is 0.339. The unmatched standardized mean difference (SMD) in
peak anomaly is 0.598, where SMD is positive-group mean minus negative-group
mean divided by the pooled within-group standard deviation.

![Unmatched I_dyn populations and peak-anomaly distributions](../../results/Idyn_matching_exploration/idyn_population_overview.png)

## Primary exploratory match

The primary specification uses deterministic, one-to-one optimal matching
without replacement:

- reference population: negative `I_dyn` events;
- candidate population: positive `I_dyn` events;
- matching variable: `tas_anom_peak`;
- distance: absolute difference divided by the pooled candidate SD;
- caliper: 0.20 pooled SD; and
- objective: maximize pair count first, then minimize total distance.

This produces 90 pairs. Every negative event is retained, while 90 of 168
positive events are retained. The excluded 78 positive events lie outside the
selected comparison set.

| Pair diagnostic | Value |
| --- | ---: |
| Mean absolute anomaly difference | 0.023 K |
| Maximum absolute anomaly difference | 0.128 K |
| Peak-anomaly SMD before matching | 0.598 |
| Peak-anomaly SMD after matching | 0.011 |

![Before and after peak-anomaly matching](../../results/Idyn_matching_exploration/tas_anom_matching_diagnostics.png)

The matched comparison therefore describes positive events that resemble the
negative-event population in peak anomaly. It does not describe the full
positive-event population, especially its extreme warm tail.

## What else becomes balanced

The table below audits variables that were not used by the primary match.
Duration and days from June 21 are reported in days. Only `tas_anom_peak` was
constrained by the matching algorithm.

| Variable | Negative mean before | Positive mean before | SMD before | Negative mean after | Positive mean after | SMD after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peak temperature anomaly [K] | 3.179 | 3.679 | 0.598 | 3.179 | 3.185 | 0.011 |
| Peak temperature [K] | 291.190 | 291.848 | 0.378 | 291.190 | 291.662 | 0.281 |
| Peak threshold excess [K] | 0.708 | 1.218 | 0.632 | 0.708 | 0.746 | 0.068 |
| Integrated threshold excess [K day] | 1.496 | 3.197 | 0.504 | 1.496 | 1.443 | -0.032 |
| Duration [day] | 2.444 | 3.125 | 0.340 | 2.444 | 2.267 | -0.111 |
| Days from June 21 [day] | 21.278 | 25.411 | 0.162 | 21.278 | 28.622 | 0.301 |
| Antecedent mean anomaly [K] | 1.237 | 0.479 | -0.621 | 1.237 | 0.104 | -1.018 |

Peak threshold excess and integrated threshold excess become well balanced as
a consequence of matching peak anomaly. Absolute peak temperature retains a
moderate difference. Season timing becomes less balanced, and the large
antecedent-anomaly contrast becomes larger.

Those remaining contrasts are not automatically matching failures. They may
be part of the mechanism that distinguishes the dynamical-sign populations.
The scientific design must decide which variables define comparability and
which variables remain outcomes or explanatory diagnostics.

![Balance audit and multi-variable retention sensitivity](../../results/Idyn_matching_exploration/covariate_balance_and_sensitivity.png)

## Retention sensitivity

Using the same 0.20 pooled-SD caliper separately for every requested variable:

| Matching variables | Matched pairs |
| --- | ---: |
| Peak anomaly | 90 |
| Peak anomaly and days from June 21 | 73 |
| Peak anomaly and duration | 79 |
| Peak anomaly, days from June 21, and duration | 40 |

Adding season timing or duration improves balance on those variables but
changes the retained negative-event population and therefore changes the
comparison being made. The three-variable specification retains fewer than
half of the 90 negative events.

## Questions for the next design step

1. Should severity mean the existing event maximum `tas_anom_peak`, or the
   anomaly sampled specifically at the Stage-2 anchor `peak_time`?
2. Should season timing be a required matching variable, a stratification
   variable, or only a post-match diagnostic?
3. Is duration part of event comparability, or is it a possible consequence of
   the different dynamical evolution?
4. Should antecedent temperature anomaly remain available as a mechanism to
   explain different `I_dyn`, rather than being balanced away?
5. Which existing temporal and spatial composites should first consume the
   matched event IDs as a sensitivity analysis?

## Reproduce the exploration

The script consumes only the compact Stage-2 table. Generated figures remain
under the ignored `results/` tree.

```bash
mamba activate dev_env
python scripts/Idyn_matching_exploration/explore_idyn_matching.py \
  --input-path /path/to/hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc \
  --output-dir results/Idyn_matching_exploration
```

Use `--caliper` to explore another per-variable pooled-SD caliper. Use
`--overwrite` only when intentionally replacing prior exploratory figures.
