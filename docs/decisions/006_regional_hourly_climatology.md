# Decision 006: Regional Hourly Climatology Baseline

## Status

Accepted for the initial climatological-anomaly workflow.

## Decision

Regional temporal anomalies use a fixed 1940-2024 climatology matched by
calendar month, calendar day, and UTC hour. The climatology includes all finite
observations, including heatwave dates, and uses an unsmoothed arithmetic mean.

Climatology is subtracted from individual timestamps before event stacking,
event means, event-percentile calculation, and display smoothing. Absolute
Stage-1 values continue to define events and event selection.

## Rationale

Month-day-hour matching removes the seasonal and diurnal mean while avoiding
the leap-year shift produced by native day-of-year grouping. Including all
observations defines a standard climatology; excluding heatwaves would instead
define a conditional non-heatwave comparison. Applying anomalies before event
reduction preserves the meaning of event percentile envelopes.

## Compatibility

Existing absolute plotting entrypoints and output files remain unchanged.
Dedicated `_clim_anom.py` entrypoints consume the climatology companion and
write separate anomaly figures. Stage 1 remains a single standard product and
does not receive appended climatology or anomaly variables.

Top-event climatological-anomaly figures retain event membership, ranking,
event boundaries, and peak timestamps from the absolute Stage-1 event table.
Only the plotted timestamp-level variables and their all-event reference
distribution are anomalized. The figure title identifies the climatology
period and makes clear that the displayed rank and peak TAS are absolute.

## Deferred sensitivity analyses

- a centered evolving climate-normal baseline;
- leave-one-year-out means;
- conditional non-heatwave baselines; and
- optional calendar smoothing.

These alternatives must be named explicitly and may not silently replace the
fixed 1940-2024 baseline.
