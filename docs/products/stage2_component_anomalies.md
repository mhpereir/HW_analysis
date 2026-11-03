# Stage-2 component-anomaly heating budgets

## Purpose and compatibility

This derived product reproduces the accepted event-versus-clean-baseline
comparisons using anomalies of each normalized atmospheric heat-budget
component. The first campaign uses pnw_bartusek and pnw_hotz, TAS q90,
surface to 700 hPa, full JJA events, and 4/7/14/21-day windows. Its reference
population is the matching pair of accepted event and baseline tables from
`322b879_20260918_pnw_4_7_14_21d`.

Use only the anomaly budget. Do not rebuild raw totals, apply a temperature
endpoint correction, or add raw-total reference curves. Preserve existing
products. The accepted tables supply row identities, anchor timestamps,
non-budget features, severity values and baseline adjacency flags unchanged.
The clean-baseline population can differ between windows, exactly as in the
original campaign; it must not change between raw and anomaly representations
of the same window.

Seasonal selection follows [decision 010](../decisions/010_season_selection_defaults.md).
This matched window-sensitivity cohort retains complete JJA events, rather
than adopting the generic Stage-2 peak-in-JJA default. Validate both the
reference metadata and the actual event intervals/baseline reference dates;
reject incompatible populations without trimming or reselecting rows.
Historical references without the newer descriptive season attributes remain
valid when their original season metadata and timestamps establish this rule.
Keep the complete antecedent hourly record, including May for early-June
anchors. The climatology remains the fixed all-observation climatology;
matching the reference population does not redefine it as an event mean.

Ordinary regional campaigns may instead use accepted JJA-peak Stage-2
references, including TAS- and LWA-selected populations at 48 and 96 hours.
The builder must explicitly request that policy with
`--no-require-full-event`; its default remains complete JJA events for
compatibility with the window-sensitivity campaign. Never trim or reselect
reference rows to make a policy match. Saved anomaly products retain the
reference selection metadata, and readers validate the declared rule and
actual timestamps. Plotting consumers still require the requested policy to
match that declaration. Each selector uses a climatology with the exact
corresponding Stage-1 checksum, even when its physical tendencies agree with
another selector's source.

## Calculation

For each tendency X (dTdt, advection, adiabatic, diabatic), use the existing
fixed all-observation 1940-2024 arithmetic-mean regional hourly climatology,
matched by calendar month, day and UTC hour. Subtract it before integrating:

$$
I'_X(d) = \sum_{h=-24d}^{0}
[X(t_a+h\,\mathrm{hour})-\overline{X}(t_a+h\,\mathrm{hour})]
\,(1\,\mathrm{hour}).
$$

The anchor is the stored event peak or baseline reference time. The inclusive
windows contain 97, 169, 337 or 505 hourly samples. All integrals have units K.
The 48-hour regional products contain 49 samples under the same operator.
Missing hours, missing climatology keys, nonfinite samples, incomplete
climatological source-year counts and mismatched regions or pressure layers
are errors, not reasons to silently change the reference population.

$$
I'_{\mathrm{dyn}} = I'_{\mathrm{advection}}+I'_{\mathrm{adiabatic}},
\qquad
I'_{dT/dt} = I'_{\mathrm{dyn}}+I'_{\mathrm{diabatic}}.
$$

The total is integrated anomalous tendency. It is not the raw total and is
not required to equal the earlier rank study's temperature-endpoint-corrected
quantity: that study uses a different discrete operator. The diabatic term
remains the upstream budget residual. A positive component anomaly means
more heating or less cooling than usual; a negative anomaly means less
heating or more cooling. The clean non-event population is distinct from the
all-observation climatology and need not have zero mean.

## Saved products and provenance

Write `event_features_clim_anom.nc` and `baseline_features_clim_anom.nc`, with
matching CSV files. Preserve the reference row coordinates and every
non-budget field. Replace the five raw budget features with:

- `I_dTdt_anom_pre`
- `I_advection_anom_pre`
- `I_adiabatic_anom_pre`
- `I_dyn_anom_pre`
- `I_diabatic_anom_pre`

Use product markers `stage_2_event_component_anomalies` and
`stage_2_baseline_component_anomalies`, contract version 1, and
`heat_budget_representation="climatological_anomaly"`. Variables identify the
source tendency, units, inclusive window, climatology reference and integration
operator. Record region, boundaries, years, original product marker, source and
reference file paths/checksums, generating commit, interpreter and PBS identity.
Do not label copied non-budget features as new anomaly calculations.

Climatology must match the exact Stage-1 checksum. Verify accepted reference
hashes against their original manifest and that their stored Stage-1 input path
matches the supplied source. Write into a fresh output directory only.

## Figures and implementation

Keep the full four-panel and presentation two-panel layouts, muted clean
baseline points, event severity colors, shared colorbar and shared style.
Mark integral axes with primes, explicitly identify climatological anomalies
and the reference period, and interpret diagonal sum lines as anomalous total
heating. Reject mixed absolute/anomaly inputs or incompatible anomaly pairs.
The plotter consumes saved products, not hourly source data.

The comparison CLI accepts both raw and component-anomaly product markers and
validates seasonal membership for either representation. Pass
`--require-full-event` when plotting this campaign's saved tables. The raw
window-sensitivity runner passes that option explicitly. Other Stage-2
consumers retain their endpoint-based default and reject incompatible input.
The component-anomaly scheduler forwards `HWA_SEASON_OPTIONS` to its builder;
set it to `--no-require-full-event` for ordinary regional campaigns. The
builder writes both layouts in PNG and PDF from the saved anomaly tables.

Reusable calculation and validation belong in `src/`; product I/O belongs in
`src/analysis_io.py`. The thin CLI under `scripts/integration_window_analysis/`
and tracked OpenPBS scheduler handle explicit paths and provenance. This
feature develops on `feat/stage2-component-anomalies`; its dependency on the
accepted window-sensitivity branch is recorded in command-center task A2.40.

## Acceptance

Synthetic tests must cover nonzero seasonal heating, component cancellation,
zero anomalies for the climatological state, calendar matching across leap
years, both populations, all four windows, input immutability and rejection of
incompatible or incomplete inputs. Verify selection fields against the accepted
tables and independently integrate source-minus-climatology samples for every
row. Check dynamical and total anomaly closure, saved metadata and round trips.
Independently verify all four climatological tendency means and counts against
the Stage-1 calendar-hour groups before validating the anomaly integrals.
Exercise 48/96-hour JJA-peak references for both selected event-ID sources,
including events crossing May/June, while retaining full-event rejection and
historical-product compatibility tests. Preserve every non-budget feature,
including event severity and baseline adjacency, exactly.
Also exercise saved anomaly products through the comparison CLI, including
historical season metadata, rejection of mismatched season requests or falsely
labeled JJA rows, and complete May history for a 21-day June anchor. Retain the
shared full-JJA top-event/all-event reference regression checks in both
absolute and climatological-anomaly representations.

Require local lint/format/shell checks and the full test suite, then queued
Venus smoke and scientific/visual acceptance for the exact deployed commit.
Inspect both layouts, source/output hashes, scheduler accounting and logs.
Retain immutable manifests, validation reports, PNG/PDF figures and a README
in fresh external artifact paths.
