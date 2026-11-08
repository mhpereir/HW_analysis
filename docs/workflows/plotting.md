# Workflow: Plotting and Shared Style

## Purpose

All active figures share one visual contract through `src/plot_style.py`.
Plotting scripts consume prepared Stage 1, Stage 2, composite, or spatial
products. They do not own ingestion, event detection, or reusable scientific
reductions.

## Shared style authority

`src/plot_style.py` owns:

- publication widths and aspect defaults;
- paper and legend font sizes;
- line, reference-line, and scatter styles;
- the 300 DPI export default;
- variable display names and colors;
- diagnostic and boundary-face colors;
- time-axis and numeric-axis formatting;
- standard axis, legend, zero-line, and layout helpers; and
- `save_figure()` for numeric formatting and consistent export.

Add a reusable visual choice there rather than defining a new palette, label
mapping, font configuration, or export convention in one script.

Publication canvases remain 6 inches wide for a single column and 12 inches
for two columns; the standard single-panel height/width ratio is 0.6. Shared
labels, ticks and titles use 18 pt, and legends and statistical annotations
use 14 pt. These replace the former 14/9 pt settings so text stays readable
when figures are reduced to manuscript column width (at 3.5 inches, roughly
10.5/8.2 pt). The change applies to newly rendered shared-style figures;
existing artifacts are retained. Review representative single- and multi-panel
layouts for clipping and overlap when changing these settings.

`save_figure()` normally crops unused margins. Use `crop=False` with a layout
that fits all artists inside the canvas when an exact publication aspect is
required; PNG and PDF then preserve the requested canvas dimensions.
`format_integer_axis()` preserves whole-number labels through export; combine
it with `limit_major_ticks(integer=True)` for adaptive integer tick spacing.
Other numeric axes retain the two-decimal default.

`src/plotting.py` is the shared renderer for Stage-1 temporal composites and
top-event figures. Its callers therefore use the shared style indirectly even
when they do not import `plot_style` themselves.

`src/plot_paths.py` owns structured default output directories for Stage-1
plots beneath the [external artifact root](artifacts.md):

```text
$HWA_ARTIFACT_ROOT/plots_<plot_name>/
  region_<region>/
  boundary_<bottom>_<top>/
  time_range_<start>_<end>/
```

## Active figure families

| Input | Figure family | Entrypoints |
| --- | --- | --- |
| Stage 1 | all-event and split temporal composites | `scripts/plot_composite_timeseries_all.py`, `scripts/plot_composite_timeseries_split.py` |
| Stage 1 plus regional hourly climatology | all-event and split climatological-anomaly composites | `scripts/plot_composite_timeseries_all_clim_anom.py`, `scripts/plot_composite_timeseries_split_clim_anom.py` |
| Stage 1 plus regional hourly climatology | face-resolved advection climatological anomalies | `scripts/plot_advection_direction_exploration_clim_anom.py` |
| Stage 1 | top-event face-resolved advection values against an all-event reference | `scripts/plot_advection_direction_exploration_top_events.py` |
| Stage 1 plus regional hourly climatology | top-event face-resolved advection climatological anomalies against an all-event reference | `scripts/plot_advection_direction_exploration_top_events_clim_anom.py` |
| Stage 1 plus regional hourly climatology, Stage 2 event features, and matching settings | matched face-resolved advection climatological anomalies | `scripts/plot_advection_direction_exploration_matched_clim_anom.py` |
| Stage 1 | top-event traces | `scripts/plot_top_events.py` |
| Stage 1 plus regional hourly climatology | top-event climatological-anomaly traces | `scripts/plot_top_events_clim_anom.py` |
| Stage 1 | diurnal, threshold, and event-summary diagnostics | `scripts/plot_diurnal_cycle.py`, `scripts/plot_threshold_timeseries.py`, `scripts/plot_event_summary.py` |
| Stage 1 plus prepared regional hourly T2m (or explicit daily-only mode) | temperature scatter plots and variability statistics | `scripts/plot_temperature_variability.py` |
| Stage 1 run inventory | Northern Hemisphere regional-domain overview | `scripts/region_vis/plot_stage1_regions.py` |
| Stage 2 event features | feature grids, splits, and combined comparisons | `event_feature_grid_plot.py`, `plot_event_feature.py`, `plot_event_feature_split.py`, `plot_event_feature_split_combined.py` under `scripts/event_features/` |
| Stage 2 event features | adiabatic, advection, and diabatic event diagnostics | `plot_adiabatic_advection_comparison.py`, `plot_adiabatic_diabatic_advection.py` under `scripts/event_features/` |
| Stage 2 event and baseline features | event-versus-baseline comparisons | `plot_adiabatic_advection_comparison_baseline.py`, `plot_adiabatic_diabatic_advection_baseline.py` under `scripts/event_features/` |
| Stage 2 event features plus tracked settings | matched positive/negative `I_dyn` diagnostics | `scripts/idyn_matching_exploration/explore_idyn_matching.py` |
| Saved integration-window heating-rank tables | target-event raw and climatology-corrected rank curves | `scripts/integration_window_analysis/plot_heating_comparison.py` |
| Spatial composite product | sign-by-lag T2m/Z500 maps | `scripts/spatial_composites/plot_dyn_net_spatial_composites.py` |
| Matched spatial composite product | matched positive/negative `I_dyn_pre` sign-by-lag T2m/Z500 maps | `scripts/spatial_composites/plot_matched_dyn_pre_spatial_composites.py` |

Scripts under `scripts/event_features/old/` are legacy and are not active
figure entrypoints.

For GMT hour grouping, pressure-layer inputs, sample counts and numerical
companions, follow the [diurnal-cycle workflow](diurnal_cycle.md).

## Surface and atmospheric temperature variability

`scripts/plot_temperature_variability.py` is a single-file diagnostic of the
relationship between regional T2m and the saved EHB `T_mean`. It consumes a
Stage-1 product and, for the hourly comparison, a separately prepared regional
hourly T2m NetCDF. It does not load spatial ERA5, reconstruct EHB temperature,
detect events, or change existing products. The small statistical reductions
are kept in this script for this explicitly self-contained diagnostic.

Stage-1 `tas_region` is **daily-native**, projected onto hourly timestamps. It
is the authoritative daily event-selection temperature, not hourly T2m. The
hourly companion must contain `tas_region(time)` by default, temperature
units (`K` or Celsius), `native_time_resolution="hourly"`
on that variable, and a dataset `region` matching Stage 1. Its preparation must
use the same regional definition and cosine-latitude weighting as event T2m.
The [hourly preparation script](../products/regional_hourly_t2m.md) supplies
this product on the native 0.25-degree grid; daily event T2m uses the existing
2-degree grid, so daily source differences are reported rather than assumed
to vanish. Alternatively, `--stage1-hourly-t2m` uses the optional Stage-1
`tas_region_hourly` field after the prepared cache has been ingested. The
script checks matching stored bounds when present and records companion
metadata; a region label alone cannot certify the upstream spatial reduction.
Daily-projected inputs, duplicate, unordered or non-hour-aligned timestamps,
and incompatible metadata are errors. Missing hours are never interpolated.
Use `--daily-only` explicitly if the hourly companion is not yet available.

The default population is every complete UTC day in JJA over all available
years, including heatwave and non-heatwave days. `--all-seasons` uses every
available month, not necessarily a complete calendar year; `--season-months`,
`--start-year` and `--end-year` provide explicit selections. Both comparisons
use the same days with all 24 finite hourly samples in each required field.
Daily atmospheric temperature is the arithmetic mean of those 24 hours.
The daily scatter uses Stage-1 `tas_region`, checked for constancy within a
UTC day. With an hourly companion, an additional daily-from-hourly statistical
comparison and differences from daily event T2m expose source disagreement.

The default `--figure-layout diagnostic` shows hourly and daily scatter plots
and yearly standard-deviation ratios. `--figure-layout paper` instead exports
only the daily scatter, its OLS fit, and a compact annotation with sample size,
Pearson r and the atmospheric/surface SD ratio. It uses the shared single-column
6-by-3.6-inch canvas, preserves that aspect in PNG/PDF, and leaves region,
period and population descriptions to the manuscript caption and saved
metadata. There is no hourly panel, yearly ratio panel or figure title.
Paper temperature axes display whole kelvin tick labels. This changes only
tick formatting, not the temperature values or statistical precision.
This is a rendering option: it uses exactly the same daily pairs and retains
all numerical tables, including hourly statistics when an hourly input is
supplied. It also works with explicit `--daily-only` input mode. No existing
dataset needs rebuilding. Statistics use every retained pair, independently of plotting:
Pearson r, Spearman rho, OLS of `T_mean` on T2m (slope, intercept and R-squared),
means, sample standard deviations and variances (`ddof=1`), interquartile
ranges, and atmospheric-to-surface SD and variance ratios, pooled and per year.
Exported columns and statistical keys use `tas_region`, consistent with
Stage 1 and the prepared hourly product; time scale is recorded separately.
The daily-pair table also includes `tas_region_hourly`, the daily mean of
the native hourly companion, when available. The raw ERA5 source field is
named `t2m`; it is normalized to `tas_region` during regional preparation.
A ratio greater than one means greater atmospheric temperature variability
on the stated sample. Undefined statistics (including constant series) are
null, not zero. No independent-sample p-values or causal interpretations are
reported. These are raw-temperature statistics; shared seasonal cycles and
trends remain, so they do not establish an anomaly relationship.

Use `--plot-kind hexbin` to replace the point layer with hexagonal sample
counts. The default remains `scatter`, and both figure layouts support either
kind. `--hexbin-gridsize` sets the number of hexagons across the x range
(default 35, minimum 2). Every retained pair contributes once, empty cells are
hidden (`mincnt=1`), and a shared sequential colormap with logarithmic count
normalization exposes both sparse and crowded cells. The colorbar identifies
days per hexagon for daily data and hours per hexagon for hourly data, using
the shared legend font size for its label and ticks. Fit lines
and annotations always use the original pairs, without bin weighting or
smoothing. Validate count conservation, known repeated-point bin counts,
unchanged numerical exports, integer paper ticks after export, and readability
of the colorbar within the fixed single-column canvas.

For example, inside an authorized Venus PBS allocation:

```bash
python scripts/plot_temperature_variability.py \
  --input-path /absolute/path/to/stage1.nc \
  --hourly-t2m-path /absolute/path/to/prepared_regional_hourly_t2m.nc \
  --output-dir /absolute/path/to/new_temperature_variability_attempt
```

The output directory must be new. It contains PNG/PDF figures,
`statistics.csv`, `daily_pairs.csv`, and `statistics.json` with coverage,
selection, definitions, input paths/hashes/metadata, execution Git/script
provenance, numerical results and output hashes. Figure filenames are
`temperature_variability` for the diagnostic layout and
`temperature_daily_scatter` for the paper layout. JSON also records the layout,
canvas dimensions, shared font sizes, crop policy and style-source hash.
Hexbin filenames are `temperature_variability_hexbin` and
`temperature_daily_hexbin`, respectively. JSON records the plot kind, tick
precision and binning settings, including count meaning and normalization.
Daily-only diagnostic output explicitly marks the hourly panels unavailable.
Local validation checks that the paper panel contains every selected daily
pair and its fitted line, preserves the standard canvas and fonts, and exports
the same statistical results as the diagnostic. It also uses synthetic known
correlations and variance ratios, unit conversion, missing-day/season/year
boundaries, source disagreement, invalid inputs, immutability and exclusive
output creation, followed by the full `dev_env` suite. Production acceptance
requires an authorized clean-commit Venus PBS run, independent numerical
checks and original-resolution figure review. The separate hourly preparation
has its own producer and scheduler; neither workflow submits jobs itself.

Use `schedulers/schedule_temperature_variability.sh` for an existing hourly
cache. Supply `INPUT_PATH`, `HOURLY_T2M_PATH`, `CACHE_MANIFEST_PATH`, a fresh
`OUTPUT_DIR`, and the usual commit and storage settings. `START_YEAR` and
`END_YEAR` default to 1940 and 2024; `SEASON_MONTHS` defaults to `6 7 8`.
Set `FIGURE_LAYOUT=paper` for the single-panel paper figure; the default is
`diagnostic`. `PLOT_KIND` defaults to `scatter`; set it to `hexbin` for the
count view, with optional `HEXBIN_GRIDSIZE` (default 35). For the PNW Bartusek
paper renders, reuse the accepted Stage-1 and
hourly-cache inputs and the 1940-2024 JJA selection in a fresh output directory.
This job does not rebuild or modify the cache. It writes the diagnostic and
`independent_validation.json`. The validator checks the entire cache's hourly
coverage and hash, source-file identities, direct native-grid means at four
hours in the first/middle/last source years, and independently recomputes all
saved daily pairs and pooled/yearly statistics. Raw-grid checks are sampled,
not a second full spatial reduction. Input hashes are checked again at the
end. Scheduler completion still requires log, provenance, and visual review
before scientific acceptance.

## Required behavior

The event-versus-clean-baseline comparison also consumes the saved
[component-anomaly products](../products/stage2_component_anomalies.md).
Both inputs must use the same representation, region, pressure layer, window
and climatological reference. Anomaly figures label all heating integrals with
primes and identify the reference period; lower-panel diagonals represent the
anomalous total. Peak-TAS coloring and accepted population membership are
unchanged. Absolute inputs retain their existing rendering.

The integration-window rank figure follows the
[heating-rank product contract](../products/integration_window_ranks.md).
It reads saved comparisons, uses `HEATING_RANK_STYLES`, displays one regional
panel per input, and puts rank 1 at the top of an integer axis. Both curves
use the same fixed cohort across all integration windows. Identify the
climatological atmospheric-temperature endpoint correction, reference years
and population size. PNG and PDF outputs must not overwrite earlier figures.

- Use the non-interactive Matplotlib `Agg` backend for batch rendering.
- The regional-domain overview discovers unique regions from the Stage 1
  products in a supplied run directory, verifies their product marker and
  stored bounds against `src/config.py`, and draws unfilled, distinctly
  colored domain boundaries on one Northern Hemisphere map. It is a read-only
  inventory diagnostic: it creates no analysis product and does not change the
  Stage 1 contract or existing consumers. The default run is
  `bf232281_20260819`, but both input and output paths remain configurable.
  Because this diagnostic is intentionally self-contained, its Venus OpenPBS
  entrypoint is co-located under `scripts/region_vis/`.
- Open Stage 1 through `src.analysis_io.open_harmonized_timeseries()` when its
  product contract is required.
- Keep analysis and selection logic in reusable modules, not in visual styling
  code.
- Matching-aware figures must load a tracked static settings file and call the
  reusable matching implementation in `src/selectors.py`. They may recompute
  lightweight event indices in memory, but must not carry a private assignment
  implementation or mutate the Stage-2 input.
- The matched face-advection figure must leave the existing all-event figure
  unchanged. It applies a named settings specification to the Stage-2 event
  table, maps the returned event IDs to the authoritative Stage-1 event table,
  and validates matching Stage-1 and Stage-2 peak timestamps before building
  composites. Component identity remains color encoded. Positive `I_dyn_pre`
  uses solid lines and negative `I_dyn_pre` uses dashed lines.
- The absolute, climatological-anomaly, and matched-population face-advection
  entrypoints each preserve the existing unsmoothed two-panel figure and write
  a second `_smoothed.png` figure. The second figure uses the shared centered
  display-only running mean, defaults to 24 hourly samples, requires complete
  windows at the lag boundaries, and identifies the smoothing window in its
  title. Smooth signed face tendencies before deriving grouped curves. For the
  matched figure, smooth each sign population independently. Do not alter event
  membership, climatology subtraction, matching, or source composites.
- The raw and climatological-anomaly top-event face-advection entrypoints rank
  the filtered absolute Stage-1 event table by descending `tas_peak`, with a
  default of 10 events. For every selected event they overlay the corresponding
  all-event mean as solid lines and the individual top-event trace as dashed
  lines in the existing two-panel face/grouped layout. The raw entrypoint plots
  Stage-1 tendencies directly in `K hr-1`; the anomaly entrypoint subtracts the
  timestamp-matched hourly climatology and uses anomaly labels. Component color
  remains the identity channel. Hourly and centered 24-hour-smoothed figures
  use separate filenames; smooth the reference and event face tendencies
  independently before deriving grouped curves. Titles and a line-style legend
  must distinguish the reference from the ranked event. The existing three
  face-advection figure families remain unchanged.
- The matched spatial figure must consume a separate matched spatial-composite
  product built from the canonical Stage-2 table and tracked settings. It must
  not infer matched membership from an already averaged all-event spatial
  product. Its title and row labels identify `I_dyn_pre`, the named matching
  specification, its pooled-SD caliper, and the equal pair count. The existing
  all-event spatial product and figure remain unchanged.
- The production matching exploration runs through
  `schedulers/schedule_explore_idyn_matching.sh`. It publishes the four
  README-linked PNGs and a `matching_summary.json` containing the input and
  settings checksums plus the numerical diagnostics used to refresh the
  exploratory Markdown.
- Event, baseline, matching, and spatial-composite consumers must read the
  canonical Stage-2 `I_dyn_pre` variable. They must not reconstruct it from
  `I_adiabatic_pre` and `I_advection_pre`.
- Use `plot_style.VARIABLE_NAME_MAPPING` and shared color dictionaries for
  existing variables.
- Use `plot_style.publication_figsize()` and shared line-width constants.
- Use `plot_style.style_axis()` or `style_axes()` where appropriate.
- Use `plot_style.legend_kwargs()` for standard legends.
- Event-versus-clean-baseline figures place the shared "Clean baseline days"
  and "Events" legend in the upper-right corner of the first panel. Later
  panels do not repeat that legend.
- Event-versus-clean-baseline figures retain clean baseline days as the muted
  background population and color foreground event points by the requested
  event-feature variable, defaulting to `tas_anom_peak`. The event layers use
  the same `gist_heat_r` mapping and peak-TAS-anomaly label as the corresponding
  event-only diagnostic, with one shared event-derived color normalization and
  colorbar across all panels. Event-only severity variables remain absent from
  and are never inferred for baseline-day rows.
- The four-panel event-only and event-versus-clean-baseline
  adiabatic/advection comparison entrypoints support `full` and `presentation`
  layouts. `full` remains the default and preserves the existing 2-by-2
  figure. `presentation` retains the first panel, advection versus adiabatic
  heating, and the fourth panel, diabatic heating versus `I_dyn,net`, in a
  2-row by 1-column figure. Both retained panels show their own x-axis label
  because their x variables differ. Stored values, event-severity colour
  mapping, the shared colorbar, and baseline-versus-event layering retain
  their full-layout meanings. The presentation-specific population and
  annotations are defined below. When no output path is supplied, figures use a
  distinct `_presentation.png` filename so they cannot overwrite the default
  four-panel product. The presentation canvas is 7.5 inches wide by 9 inches
  high, widening the original single-column canvas by 25% without changing its
  height so the scatter panels and shared colorbar have more horizontal space.
  Synthetic plot tests must confirm both layouts, presentation dimensions, and
  their CLI routing.
- In both adiabatic/advection comparison layouts, the event-only x-axes pad
  the finite plotted range by 5% on either side, including zero in the range,
  so extreme event markers are not cut in half. The presentation layout uses
  the shared major-tick limiter with at most four intervals on each x-axis;
  retain the shared two-decimal formatting and existing font and canvas sizes.
- The adiabatic/advection baseline comparison uses a stable, visible Events
  legend proxy sampled from the midpoint of the shared severity colormap,
  with a contrasting outline. The proxy identifies the event population,
  not an individual event or severity value; the unchanged colorbar remains
  the quantitative severity key. Do not derive the legend marker from the
  first event, whose color may be white. Keep the legend in the first panel.
- Both comparison presentation layouts selectively adopt the budget guides
  from `agent/stage2-antecedent-temperature` (`955f12d`, refined in `cd10710`).
  The lower panel is titled `Diabatic Residual vs I_dyn,net`: `I_diabatic_pre`
  is a closure residual, not a separately measured direct heating term. Add
  labelled dotted `x+y=c` guides at 5 K increments and a shared footer explaining
  `I_dyn,net + I_diabatic = I_dT/dt`. These are constant integrated atmospheric
  warming guides, not fitted relationships or surface-temperature anomalies.
  The shared `plot_style` helper clips guides to existing linear-axis limits
  without changing the plotted range. For unusually broad ranges, increase
  spacing to an integer multiple of 5 K to cap the guide count at 17. Omit
  corner-only segments shorter than 10% of the panel in both dimensions.
  Place labels at segment midpoints so the larger shared fonts stay clear of
  the corners and adjacent labels; preserve the guide values and axis limits.
- Each presentation uses one common finite population for all five budget
  fields (`I_adiabatic_pre`, `I_advection_pre`, stored `I_dyn_pre`,
  `I_dTdt_pre`, and `I_diabatic_pre`) and the requested event colour variable,
  if any. This intentionally includes `I_dTdt_pre` even though it is represented
  by reference guides rather than a scatter axis. Apply the shared finite-row
  selector in `src/selectors.py`; baseline rows must also have
  `event_adjacent == 0`. Use the retained events to normalize colours, and fail
  clearly before allocating a figure if any required population is empty.
  Replace per-panel count boxes with one figure-level count header. Full
  layouts retain their existing per-panel filtering and count annotations.
- Presentation event markers default to 40 points squared and opacity 0.9 in
  both renderers and their PBS launchers. Explicit size/opacity overrides take
  precedence. Baseline markers remain size 24 and opacity 0.2; full-layout
  defaults remain unchanged. Keep the visible Events legend proxy, padded
  limits, four-interval x ticks, two-decimal formatting, and 7.5-by-9-inch
  canvas from the earlier presentation-legibility repair.
- No Stage-2 rebuild, climatology input, new temperature variables, or budget
  window change is required for these presentation additions. They do not
  modify source datasets or the separate antecedent-temperature diagnostic.
  Test common masks (including missing values in different fields), retained
  colour normalization, empty populations, source immutability, guide values
  and clipping, shared counts, size/opacity defaults and overrides, and full
  layout compatibility. Check marker clearance, guide labels, footer and tick
  legibility after shared 300 DPI export, including negative ranges and constant
  data. Validate fresh, non-overwriting Venus renders at original resolution
  before production acceptance.
- Event-versus-clean-baseline figures pad each finite plotted data range by 5%
  on both sides while keeping zero reference lines inside the padded range.
  Shared x-axes use the combined plotted x-data, while each panel derives its
  y-range from its own clean-baseline and event points.
- Use shared numeric-axis helpers for reusable tick-spacing and formatting
  rules, including integer-only axes.
- Use `plot_style.save_figure()` for ordinary Matplotlib figures.
- Map exports may use a dedicated writer when Cartopy layout requires it, but
  must still use shared theme, size, and DPI settings.
- Add or update synthetic plot tests when changing visual semantics or the
  shared style API.
- Climatological-anomaly figures must identify the representation and baseline
  period in their title or metadata, include a zero reference where
  appropriate, and never overwrite their absolute-value counterparts.

### Extended temporal-composite layout

Extended all-event, split-event, climatological-anomaly, and top-event figures
use a 5x2 panel grid with exactly ten axes, one y-axis per panel:

| Row | Left column | Right column |
| --- | --- | --- |
| 1 | temperature (`T_mean`) | anticyclonic and cyclonic LWA |
| 2 | `dTdt` | cloud cover |
| 3 | advection | soil moisture |
| 4 | adiabatic heating | longwave and shortwave radiative heating |
| 5 | diabatic heating | sensible surface heating |

The temperature panel contains no volume trace. Cloud cover and soil moisture
have separate panels, and latent surface heating is omitted. Absolute cloud
cover is bounded to the physical fraction range from zero to one, while a
climatological anomaly is not. Anomalous temperature, cloud cover, and soil
moisture include a zero reference. Keep the IQR, split-bin style, and top-event
all-event-reference keys in the first panel as applicable, alongside the
temperature key. Preserve the existing variable colors and sensible-heating
sign convention.

This changes the shared extended renderer and the variables selected and
smoothed by its six absolute/anomaly entrypoints. Volume and latent heating
are no longer required for extended rendering; existing Stage-1 and climatology
products remain compatible and retain those stored variables. Compact paper
and presentation layouts, event selection and ranking, peak alignment,
percentile reductions, smoothing of retained variables, CLI flags, and output
naming remain unchanged. Historical figures remain valid artifacts; new
production validation and subsequent regeneration use fresh output namespaces.

Synthetic tests must check the panel order, exactly ten axes, separated cloud
and soil traces, absence of volume and latent heating, reference and percentile
displays, legend keys, bounds, signs, and source immutability for all-event,
split-event, and top-event figures in both representations. Extended entrypoints
must select only the displayed variables and retain unsmoothed daily LWA.
Before production acceptance, run the full local suite and representative
Venus PBS renders, inspecting the saved figures at original resolution.

PBL diagnostics and panels are inactive under
[decision 008](../decisions/008_retire_pbl_diagnostics.md).

### Presentation temporal layout

The four active all-event, split-event, absolute, and
climatological-anomaly composite entrypoints also accept
`--layout presentation`. This alternate view is a 16:9, 3x2 figure intended
for slides. It reuses the same event selection, peak alignment, composite
statistics, percentile displays, split-bin styles, anomaly convention, and
raw plus 24-hour-smoothed output behavior as the corresponding paper figure.

The presentation grid is arranged by row as follows:

| Row | Left column | Right column |
| --- | --- | --- |
| 1 | anticyclonic and cyclonic LWA | advection |
| 2 | temperature | adiabatic heating |
| 3 | `dTdt` | diabatic heating |

The temperature panel contains `T_mean` only. Volume and the remaining
extended surface diagnostics are intentionally omitted so that the figure has
exactly six axes and remains legible in a presentation. Presentation outputs
use a distinct `*_presentation` plot directory and filename by default, and
must not overwrite the paper-layout products. `--layout presentation` and
`--plot-extended-variables` are mutually exclusive.

The absolute and climatological-anomaly top-event entrypoints accept the same
presentation layout and panel order. They preserve the selected event trace,
the all-event mean and IQR reference, peak-relative day x-axis, event start/end
markers, event-peak marker, ranking, and raw plus 24-hour-smoothed output
behavior. The temperature panel contains only `T_mean`, with the event trace
and reference keys both retained. Climatological anomalies are constructed
before the event trace, all-event reference, IQR, or smoothing is calculated,
while selection and ranking remain absolute. Presentation and anomaly outputs
use distinct plot directories and filename tokens so they cannot overwrite
the extended absolute paper figures.

### Top-event time axis

All top-event layouts, including the compact paper, extended 5x2 paper, and
3x2 presentation views, label the x-axis `Lag from event peak (days)`. Plot
each source timestamp at its elapsed time from the exact Stage-1 `peak_time`,
divided by 24 hours. Negative values precede the peak, zero marks the peak,
and positive values follow it. Preserve fractional days and sub-hour offsets;
do not round timestamps to calendar dates or resample the hourly traces.

Convert the all-event reference's `lag_hour` coordinate to days for both its
mean and IQR. Place the event start and end markers on the same elapsed-day
axis and the peak marker at zero. Use shared numeric tick formatting that
prefers whole-day major ticks when the range permits and retains fractional
major ticks for short windows. Minor ticks fall at integer days, spaced one
day apart, with ticks coincident with major ticks omitted. Do not subdivide
days into fractional minor ticks, even for short windows. Preserve this
formatting on export, including twin axes.

This is a display-only change for both absolute and climatological-anomaly
top-event figures, including raw and smoothed outputs. Stage-1 timestamps,
event selection and ranking, window extraction, reference reductions,
smoothing, y-values, and output naming remain unchanged. Composite plots
continue to use their existing lag-hour axes. Existing figures remain
historical artifacts; production validation uses a fresh output namespace.

Synthetic regressions must check elapsed-day coordinates across every panel,
reference means and IQRs, exact peak and boundary markers, non-midnight peaks,
source immutability, and whole-day minor ticks after export in each layout.
Check that major and minor ticks together mark every integer day in the
standard event window, while short windows retain distinct major labels.
Before production acceptance, render representative presentation and extended
paper figures through Venus PBS and inspect the saved axes at original resolution.

### Composite labels and surface-flux signs

The all-event and split-event climatological-anomaly composite figures use the
symbol `Δ` in panel-axis labels instead of spelling out `anomaly`. Place `Δ`
immediately before the plotted variable name when the axis identifies a
variable, for example `ΔT_mean [K]` or `ΔLWA [m hPa]`. For a shared or
units-only tendency axis, place the symbol immediately before the units, for
example `Δ [K hr-1]`. Figure titles continue to identify the
climatological-anomaly representation explicitly.

ERA5 sensible and latent surface heat fluxes use the source convention that
positive values are directed toward the surface. In every extended temporal
figure, render sensible heating (`sshf_heating_rate_approx`) with the opposite
sign, so positive plotted values denote heat transfer into the atmosphere and
can be compared directly with the atmospheric diabatic-heating sign. Apply the
same sign reversal to composite means, event-percentile bounds, top-event
traces, and their reference composites. Latent heating retains its stored
source convention but is no longer displayed in this layout.

This is a display transform only. It must not mutate Stage 1, the climatology
companion, or any assembled composite or top-event dataset. Synthetic plotting
tests must verify the `Δ` labels, the reversed absolute and anomaly traces and
bounds, top-event and reference signs, and source-data immutability.

## Validation

Run the plotting and spatial plot tests locally:

```bash
mamba activate dev_env
python -m pytest -q \
  tests/test_plotting.py \
  tests/test_plot_*.py \
  tests/test_region_vis.py \
  tests/test_spatial_composite_plot.py
```

For a production figure change, also render a representative product through
PBS on Venus and inspect the saved artifact, labels, units, panel ordering,
legibility, and output path.

Synthetic tests for both top-event face-advection variants must verify ranking,
season-filter propagation, peak-relative event extraction, solid-reference and
dashed-event line semantics, independent display smoothing,
grouped-component identities, stable filenames, source-data immutability,
no-overwrite behavior, and complete scheduler publication. The anomaly variant
must additionally verify timestamp-level climatology subtraction.

For the presentation feature comparisons, first run the commit-pinned
`schedulers/schedule_presentation_feature_smoke.sh` on Venus. It exercises the
synthetic plotting/export regressions in `dev_env` with warnings treated as
errors. Then render the unchanged accepted Stage-2 inputs through the existing
event-only and baseline-comparison schedulers into fresh commit-scoped paths.
Record input/output checksums, counts, clean logs and original-resolution
inspection; a successful PBS job alone does not establish visual acceptance.
