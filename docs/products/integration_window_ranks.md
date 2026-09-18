# Integration-window heating ranks

## Purpose and compatibility

This exploratory derived product tests the June 2021 Pacific Northwest event's
heating rank at every integer integration length from 4 through 31 days. It
consumes accepted Stage-1, regional hourly climatology and 4-day Stage-2 event
products. It does not redefine heatwaves, change Stage-2 features, or require
new baseline-day tables. The previously accepted 4/7/14/21-day products remain
unchanged. Component-wise anomaly integrals are a separate future analysis.

The first production comparison uses `pnw_bartusek` and `pnw_hotz`, TAS q90,
surface to 700 hPa, full JJA events, and the fixed 1940-2024 climatology. The
target is identified by its exact June 29, 2021 peak and verified event ID,
not by selecting whichever 2021 event ranks highest at a particular window.

## Heating and correction

Let $S$ be Stage-1 `dTdt`, in K per hour, and $t_p$ the unchanged event peak.
For an integer window length $d$, retain the Stage-2 inclusive hourly sum:

$$
I_{\mathrm{raw}}(d) = \sum_{h=-24d}^{0} S(t_p+h\,\mathrm{hour})\,(1\,\mathrm{hour}).
$$

There are $24d+1$ samples. The temperature correction is an endpoint change
in the atmospheric `T_mean` climatology, matched by month, day and UTC hour:

$$
C(d) = \overline{\langle T\rangle}(t_p)
- \overline{\langle T\rangle}(t_p-d\,\mathrm{day}).
$$

$$
I_{\mathrm{corrected}}(d) = I_{\mathrm{raw}}(d) - C(d).
$$

All three quantities have units of K. Apply each event's own correction before
ranking the entire cohort. Positive and negative climatological changes retain
their signs. This correction is specifically the requested temperature endpoint
change; it is not silently replaced by the inclusive sum of climatological
`dTdt`. Those discrete operators need not agree exactly. Preserve the existing
raw integral for comparison with accepted Stage-2 results.

Use the existing all-observation climatology, including heatwave dates, without
smoothing. Match its region, pressure boundaries, Stage-1 contract, reference
years and source checksum. Require finite endpoints and complete source-year
counts. T2m climatology must not substitute for atmospheric `T_mean`.

## Population, coverage and rank

The accepted 4-day Stage-2 event table defines the candidate population. Verify
its IDs and exact event timestamps against the Stage-1 full-JJA event selection.
Require strictly increasing hourly timestamps, finite `dTdt`, and both exact
window endpoints for every eligible event/window; gaps must not be summed over
silently. Do not interpolate missing temperatures or extend the source period.

The primary curves use the intersection of eligible events across all requested
windows. The target must belong to that fixed cohort. Record per-window
eligibility and a reason for every exclusion. The May-October source starts at
May 1, 01 UTC, so a June 1, 00 UTC peak lacks the first sample and climatological
endpoint for a 31-day window. Report the resulting cohort sizes explicitly.

Also retain ranks within each window's available population as a diagnostic of
the fixed-cohort choice. Preserve finite values for candidates outside the
common cohort where shorter windows are complete. Ineligible ranks are missing.

Use descending competition ranks with exact numerical ties:

$$
r_i = 1 + \#\{j : I_j > I_i\}.
$$

Tied events receive the same rank, with gaps after ties. Store tie multiplicity.
Rank 1 is the largest heating total. Windows overlap and the sweep is
exploratory; it is not an independent significance test, an onset estimate, or
evidence for uninterrupted heating throughout the window.

## Saved handoff and figures

The builder under `scripts/integration_window_analysis/` writes a NetCDF table
with dimensions `event` and `integration_days`, a long-form CSV, and a target
event CSV. Store event IDs, peaks, window starts, sample and finite counts,
climatological endpoint values/counts, raw/corrected heating, eligibility,
exclusion reasons, fixed/available-cohort ranks, tie counts and cohort sizes.
Metadata records the correction and rank conventions, source paths/checksums,
reference period, generating commit, and target identity.

The plotter consumes these saved products. Use one panel per region, raw and
corrected curves on the same rank axis, daily markers, integer ticks, rank 1
at the top, and a visible cohort size. Identify the atmospheric-temperature
correction and reference period. Use the shared plotting style and export PNG
and PDF into fresh external artifact paths. Keep all source data immutable.

Minor ticks use fixed integer spacing: one day on the integration axis and
one fifth of the regular major-rank spacing (at least one rank) on the rank
axis. The extra major tick at rank 1 must not determine the minor spacing.
For example, major ranks 1, 25, 50, ... use minor ranks 5, 10, 15, 20, 30,
... . This is a display-only rule; saved ranks and major tick labels retain
their existing meanings. Validate tick positions after drawing and export,
including the irregular first major interval and the shared regional axes.

## Validation and execution

- Synthetic tests cover known warming and cooling corrections, re-ranking all
  events, exact ties, leap/non-leap calendar matching, missing endpoints,
  interior gaps/nonfinite tendencies, incompatible climatologies, fixed cohort
  preservation, target identity, no-overwrite behavior and plotted rank axes.
- Reproduce each raw integral independently from direct source slices and each
  endpoint difference from a calendar-key lookup. Independently count ranks and
  verify the saved product, population and exclusions.
- Compare the 4/7/14/21-day raw values against the accepted Stage-2 products and
  independently check the climatological endpoints against source-year means.
- Run local `dev_env` tests, then a commit-pinned PBS smoke and production run.
  Record PBS status/resources, clean logs, input/output hashes and image review.

See the [workflow README](../../scripts/integration_window_analysis/README.md),
[climatology contract](stage1_regional_hourly_climatology.md),
[Stage-2 contract](stage2_event_features.md) and
[artifact contract](../workflows/artifacts.md).
