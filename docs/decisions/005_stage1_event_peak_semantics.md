# Decision 005: Stage-1 Event Peak Semantics

## Status

Accepted existing Stage-1 behavior, recorded retrospectively.

## Decision

Stage-1 event summaries distinguish the event anchor from named per-variable
peak metrics.

The selected `threshold_variable` determines both the event interval and the
variable used to define the event anchor:

| `threshold_variable` | Event-ID source | `peak_time` and `peak_value` source |
| --- | --- | --- |
| `tas` | `hw_event_id` | `tas_region` |
| `lwa` | `lwa_event_id` | `lwa_region` |
| `lwa_a` | `lwa_a_event_id` | `lwa_a_region` |
| `lwa_c` | `lwa_c_event_id` | `lwa_c_region` |

For each contiguous daily threshold-exceedance event, `peak_time` is the time
of the maximum value of the configured peak variable within the event
interval, and `peak_value` is that maximum. If the maximum occurs on multiple
event days, `peak_time` uses the first occurrence.

Named peak metrics are independent reductions over the same event interval:

```text
tas_peak      = max(tas_region over event days)
tas_anom_peak = max(tas_region - tas_climatology over event days)
tas_excess_peak = max(max(tas_region - hw_threshold, 0) over event days)
lwa_a_peak    = max(lwa_a_region over event days)
lwa_c_peak    = max(lwa_c_region over event days)
```

These metrics are calculated using one sample per calendar day. They are not
values sampled at `peak_time`, and their maxima are not required to occur on
the same day.

For `threshold_variable=tas`, `peak_value` and `tas_peak` are equivalent.
However, `tas_anom_peak` may still occur on a different day because the
climatology varies through the event. For an LWA-family threshold variable,
`peak_time` is anchored to the selected LWA-family maximum, while `tas_peak`
and `tas_anom_peak` remain independent temperature maxima within the
LWA-defined event interval.

## Rationale

The threshold variable defines the event universe and supplies a physically
meaningful anchor for event-centered windows and composites. Independent
per-variable maxima preserve useful event-severity summaries without requiring
all diagnostics to peak at the event anchor.

This behavior was implemented before decision records were introduced. This
record makes the existing Stage-1 contract explicit rather than changing the
product.

## Consequences

- Downstream fixed-window features and peak-aligned composites use `peak_time`,
  so their anchor follows the selected threshold variable.
- `tas_peak` and `tas_anom_peak` describe temperature severity during the
  event, but they must not be interpreted as temperature values at
  `peak_time`.
- For LWA-defined events, the LWA peak day, absolute-temperature peak day, and
  temperature-anomaly peak day may all differ.
- Analyses requiring temperature at the event anchor must sample
  `tas_region` or its anomaly at `peak_time`; they must not substitute
  `tas_peak` or `tas_anom_peak`.
- Tests and product documentation should preserve the distinction between the
  event anchor and independent named peak metrics.
