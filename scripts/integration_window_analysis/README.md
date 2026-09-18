# Integration-window analysis

## Status and purpose

This directory contains the dedicated workflow for exploring how heatwave
heating totals and event rankings depend on integration time and the expected
climatological temperature change. The first implementation is the daily
4-31-day rank sweep described by the
[heating-rank contract](../../docs/products/integration_window_ranks.md).

The accepted initial comparison covers 4, 7, 14 and 21 days in `pnw_bartusek` and
`pnw_hotz`, using the existing TAS q90, surface-to-700-hPa, 1940-2024 JJA
Stage-2 products. The June 2021 event remains part of the regional heatwave
population used for ranking.

The initial question is how the heating totals and rankings change after
subtracting the heating expected from the regional seasonal cycle. Anomalies
of individual budget components and event-specific heating-onset definitions
are possible later extensions, with their own documented scientific choices.

## Entrypoints and ownership

| Entrypoint | Responsibility |
| --- | --- |
| `build_heating_comparison.py` | Read Stage 1, the accepted 4-day Stage-2 event population and its hourly climatology; save the 4-31-day raw and endpoint-corrected heating and rank table. |
| `plot_heating_comparison.py` | Read saved regional tables and plot the target event's raw and corrected ranks through the shared plotting style. |

Keep these entrypoints thin. Reuse the existing
[climatology machinery](../../src/climatology.py) and
[fixed-window reductions](../event_features/fixed_window_features.py), with
new reusable analysis functions under `src/`, tests under `tests/`, and Venus
entrypoints under `schedulers/`. Plotting consumes the saved comparison table.
Generated datasets, figures, logs and manifests follow the external
[artifact-location contract](../../docs/workflows/artifacts.md).

## Current rank-sweep calculation

The current test subtracts the calendar-matched endpoint change in the
atmospheric `T_mean` climatology from each event's raw integrated heating:

$$
I_{\mathrm{corrected}}(d) = I_{\mathrm{raw}}(d)
- [\overline{\langle T\rangle}(t_p)
- \overline{\langle T\rangle}(t_p-d\,\mathrm{day})].
$$

The primary ranks use a fixed cohort with complete coverage at every window;
available-population ranks and all exclusions are also saved. See the linked
product contract for exact ties, target selection, metadata and validation.
This workflow needs event products only. New baseline-day products are not
required to test the event's rank.

Run the builder separately for each region, with explicit input paths:

```bash
python scripts/integration_window_analysis/build_heating_comparison.py \
  --input-path "$STAGE1_PATH" --climatology-path "$CLIMATOLOGY_PATH" \
  --reference-dir "$REGIONAL_REFERENCE_DIR" --output-dir "$REGIONAL_OUTPUT_DIR" \
  --min-days 4 --max-days 31 --target-peak 2021-06-29T00:00:00

python scripts/integration_window_analysis/plot_heating_comparison.py \
  --input-paths "$BARTUSEK_COMPARISON" "$HOTZ_COMPARISON" \
  --output-stem "$OUTPUT_DIR/rank_sensitivity"
```

The reference directory contains accepted `4d`, `7d`, `14d` and `21d`
subdirectories, each with `event_features.nc`. The builder publishes
`heating_comparison.nc`, `all_events.csv`, `target_event.csv`,
`excluded_events.csv`, `validation.json` and `manifest.json`. The plotter
publishes PNG, PDF and figure provenance JSON.

Production runs use `schedulers/schedule_integration_window_ranks.sh` and
require `PROJECT_ROOT`, `EXPECTED_COMMIT`, `STAGE1_DIR`, `CLIMATOLOGY_DIR`,
`REFERENCE_DIR` and a fresh `OUTPUT_DIR`. Pass artifact/log root overrides
explicitly through PBS. `REGIONS` defaults to both PNW regions; use
`REGIONS=pnw_bartusek` and `RUN_TESTS=1` for the initial queued smoke. Each job
requests one CPU, 4 GB and 15 minutes, with numerical threads limited to one.
The builder and plotter record the supplied commit and PBS identity.

For a plotting-only rerun, set `COMPARISON_DIR` to an accepted run directory
containing each region's `heating_comparison.nc`. The same scheduler then
reads those immutable tables and writes only the PNG, PDF and figure manifest
to a fresh `OUTPUT_DIR`. `STAGE1_DIR`, `CLIMATOLOGY_DIR` and `REFERENCE_DIR`
are not required in this mode. Both directories must be beneath the resolved
artifact root. Use `RUN_TESTS=1` for the queued plotting smoke; verify the
source-table hashes and inspect the regenerated figure before acceptance.

Local validation uses synthetic data only:

```bash
python -m pytest -q -W error tests/test_integration_window_analysis.py
```

## Later tendency-anomaly comparison

The following broader comparison remains planned. Its matched tendency
integral differs discretely from the current temperature endpoint correction.

Use the existing [event-feature](../../docs/products/stage2_event_features.md)
and [baseline-day](../../docs/products/stage2_baseline_features.md) products
for each requested window, together with the compatible
[regional hourly climatology](../../docs/products/stage1_regional_hourly_climatology.md).
Retain event IDs, peak/reference timestamps, baseline adjacency flags and
population selection from the source products.

Let $S(t)$ denote the atmospheric temperature tendency stored as `dTdt`, and
let $W$ be the source product's inclusive pre-peak or pre-reference window.
The proposed comparison is

$$
I_{\mathrm{raw}} = \sum_{t\in W} S(t)\,\Delta t,
\qquad I_{\mathrm{clim}} = \sum_{t\in W} \overline{S}(t)\,\Delta t.
$$

$$
I_{\mathrm{adjusted}} = I_{\mathrm{raw}} - I_{\mathrm{clim}}.
$$

`I_dTdt_pre` supplies the raw integral. Match the climatological `dTdt` by
calendar month, day and UTC hour for each row, then use the same window and
hourly integration convention. The current 4/7/14/21-day windows contain
97/169/337/505 hourly samples, with $\Delta t = 1$ hour. All three heating
quantities have units of K.

Use the fixed 1940-2024, all-observation climatology already defined by its
product contract, including heatwave dates. The clean non-event comparison
population is distinct from that climatological reference. Its adjusted
mean is not required to be zero.

The atmospheric tendency climatology must describe the same region and
pressure boundaries as the budget. A T2m climatology, a single summer-average
correction, or a climatological temperature endpoint difference must not be
silently substituted for the matched tendency integral. Retain May coverage
for windows preceding early-June anchors.

## Comparison output and interpretation

The first derived comparison table should retain, for each region, population
and integration window:

- event ID or baseline reference identity and its anchor timestamp;
- raw, expected climatological and adjusted heating;
- raw and adjusted event ranks and the event-population size;
- window bounds, hourly sample count and baseline adjacency where applicable;
- source paths/checksums, climatology reference and generating commit.

Apply the correction to every event before calculating adjusted ranks within
each region/window population. Rank 1 denotes the largest heating total; tie
handling and any differences in eligible populations must be explicit.
Treat this as a derived comparison product while the established Stage-2
products retain their existing definitions.

Adjusted heating measures the departure of the integrated tendency from its
expected calendar-window value. It does not determine uninterrupted heating
duration, an onset date, or a statistical outlier classification. Integrating
temperature anomalies themselves would produce K hours or K days and answer
a different question.

## Implementation and validation sequence

1. Document the comparison schema, ranking convention and compatibility
   checks before implementing the builder.
2. Validate calendar-hour matching, inclusive sample counts, complete finite
   coverage and region/pressure/reference compatibility. Check that adjusted
   integrals equal raw minus climatological integrals and independently
   reproduce direct integration of tendency anomalies.
3. Validate population preservation and raw/adjusted rankings, including ties
   and the June 2021 event, using focused synthetic tests in local `dev_env`.
4. Add the plotting consumer with explicit window, region and representation
   labels, following the [shared plotting workflow](../../docs/workflows/plotting.md).
5. Follow the repository's local validation and Venus production gates, using
   fresh output namespaces and retaining detailed provenance beside outputs.

The [pipeline overview](../../docs/pipeline_overview.md), product contracts and
[documentation-first process](../../docs/README.md) remain authoritative.
