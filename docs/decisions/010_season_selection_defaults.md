# Decision 010: Seasonal Analysis Populations

## Status

Accepted for the seasonal-default fix. Existing production artifacts retain
their original provenance and membership.

## Decision

Season selection changes rows, never the underlying hourly source record.
Months use the timestamps stored in the product; this decision does not change
the time basis of event detection.

| Analysis | Default population |
| --- | --- |
| All/split temporal composites, top-event plots and their reference curves, event summaries, face-advection composites | Events whose entire detected start-to-end interval falls within June, July and August (JJA/full events). |
| Stage-2 event integrals and feature plots | Events whose `peak_time`, the endpoint of the pre-peak integral, falls in JJA. |
| Stage-2 baseline integrals and comparisons | Non-event days whose `reference_time`, the endpoint of the pre-reference integral, falls in JJA. |
| Explicit window-sensitivity comparisons | Retain `require_full_event=True` for the conservative, complete-JJA event cohort across integration lengths. Baselines retain JJA endpoints and complete antecedent history. |

For Stage 2, neither the detected event start nor the integration start has to
be in JJA. An event beginning May 30 and peaking June 2 is eligible. An integral
ending June 5 includes its complete May history, including for a 21-day
window. August endpoints are treated identically. September endpoints are
excluded. Required-window checks still use the complete Stage-1 coverage.
Other feature windows, including those extending after the anchor, also stay
intact. Baseline adjacency continues to include out-of-season heatwaves in
the history, as specified in [decision 004](004_baseline_season_windows.md).

CLI defaults implement this policy without flags. `--all-seasons` explicitly
requests the full saved population. `--season-months` selects other months.
`--require-full-event` and `--no-require-full-event` choose complete-event or
peak-only selection where event membership applies. The full-event default
is enabled for Stage-1 composites and disabled for Stage 2. Explicitly asking
for full events together with all seasons is an error.

PBS wrappers forward optional whitespace-separated `HWA_SEASON_OPTIONS` as
CLI arguments and otherwise defer to these Python defaults. For example,
`HWA_SEASON_OPTIONS="--all-seasons"` requests a full-population run, while
`HWA_SEASON_OPTIONS="--require-full-event"` consumes historical JJA event
tables. Use matching options on producers and consumers, and choose fresh
output paths. These options contain flags and month numbers, not shell code.

Top-event ranking and the all-event mean/percentile reference must use the
same selected event table. Rank on absolute TAS even for anomaly figures.
Do not average only the top N to obtain the reference. Split groups must be
formed after season selection. Climatology construction and subtraction,
smoothing and peak alignment are unchanged.

## Consumer guard and compatibility

Stage-2 consumers check the declared season, membership rule and actual anchor
dates before selection, matching or plotting. A default JJA consumer rejects
all-season tables, missing or contradictory provenance, and rows outside JJA.
It does not silently trim a file or relabel its population. A full-population
analysis requires both an all-season producer and an explicit all-season
consumer. Custom seasons must likewise agree.

Historical JJA tables with `require_full_event=1` remain usable by explicitly
requesting `--require-full-event`. They are a stricter population than the new
default; regenerating from Stage 1 is required to recover omitted
boundary-crossing events. A consumer cannot restore those rows. Existing
tables without the new descriptive metadata remain usable when their old
season attributes and timestamps unambiguously establish the requested rule.

Matched face composites validate the Stage-2 source's JJA membership, then
restrict to complete JJA events before matching. They may therefore consume
either JJA endpoint tables or historical full-event JJA tables; the resulting
composite population always follows the requested composite rule.

Stage-1 source coverage and regional hourly climatologies retain supporting
months. Diurnal cycles select JJA on native Stage-1 timestamps before grouping
by local hour; their help and metadata must distinguish these two operations.
Threshold time-series plots retain annual detection context, and region maps
have no event-season population. These are intentional exceptions to
event-population selection.

## Scope and integration dependencies

This fix starts from master `3736891`. It covers its active temporal, Stage-2,
matching and spatial-builder consumers. Separate unmerged branches contain
component-anomaly/window-sensitivity producers, top-event face plots, top-event
maps and inter-region fractions. The window-sensitivity workflow intentionally
retains its explicit `require_full_event=True` selection; its matching
component-anomaly reference cohort must remain consistent. This is an approved
exception to the generic Stage-2 endpoint default, not a requirement to change
the sensitivity population. The integration history still spans all required
months.

Top-event face plots, top-event maps and inter-region fractions must adopt the
JJA/full-event default, including ranking and reference populations, with
explicit full-population modes preserved. Their consumers must guard inherited
membership. Apply these changes on their own branches after validating this
fix on Venus and regenerating the standard top-event figures. Existing outputs
retain their original provenance and must not be relabeled or overwritten.
This fix does not import those unrelated features or authorize their production
reruns.

## What the regression checks protect

These software regression checks guard against a fixed bug returning after a
future code change. They use small synthetic examples with different May, JJA
and September events, not a requirement that every plot have identical curves.
In particular, Stage-2 endpoint populations need not have the same event count
as Stage-1 full-event composites.

1. **Same reference population:** the all-event composite and top-event
   reference must contain the same event IDs, mean and percentile envelopes,
   before and after the same smoothing, for both absolute and anomalized data.
   This catches the original hidden use of all saved events in top-event plots.
2. **Rank after selection:** an extremely hot May event must not occupy a JJA
   top-N slot; a split threshold must not be shifted by out-of-season events.
3. **Full event versus endpoint:** a May-to-June event is excluded from the
   default composite but retained in default Stage 2 when its peak is in June.
   A late-August event ending in September follows the same rule.
4. **Keep the history:** an early-June 21-day integral must equal a direct sum
   of all inclusive hourly samples, including May. Sample counts and baseline
   adjacency must agree. This catches accidental truncation and the associated
   loss of early-June rows; it does not compare different integration lengths.
5. **Fail on incompatible input:** an all-season table or a misleading JJA
   label with May rows must fail before plotting. Historical strict subsets
   require the explicit matching mode. Matching producer/consumer all-season
   options must still succeed, preserving future full-population analysis.

Local synthetic tests establish code behavior. Production acceptance still
requires an authorized, exact-commit Venus smoke run in a fresh namespace,
membership counts/provenance checks and visual inspection of the figures.

The tracked seasonal validation job runs the complete regression suite and a
read-only audit of the campaign's saved Stage-1 and climatology inputs. The
audit checks their recorded hashes, independently checks full-JJA membership
and ranking, and compares the actual all/top entrypoints' complete reference
arrays before and after smoothing in both representations. It also checks
real-data Stage-2 endpoint membership, explicit full-event compatibility and
inclusive 21-day May history for early-June anchors. Audit outputs are separate
from production figures. Regeneration uses the existing top-event schedulers,
unchanged inputs and layout, default season options and new output paths.
The shared Venus environment's complete `pip check` output is retained as a
diagnostic; project imports and the full regression suite are mandatory gates.
An unrelated shared-environment dependency must not trigger an unreviewed
environment change. The first attempt exposed the existing `wrf-python`
Basemap dependency, neither of which is imported by this repository.
