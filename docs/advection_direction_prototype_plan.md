# Advection Direction Prototype Plan

## Status

Planned on branch `advection-direction-prototype`.

This document defines the goal, scientific interpretation, implementation
sequence, compatibility requirements, and intended outputs for a standalone
advection-direction time-series prototype. The prototype uses an enhanced
Stage-1 copy in an exploration subfolder. It does not yet change the canonical
Stage-1 contract or any production figure.

## Goal

Recover directional information that is hidden by the current scalar
`advection(time)` tendency and test how best to show it alongside
peak-aligned heatwave time series.

The first prototype will answer three questions:

1. How much of the net advective temperature tendency comes through the zonal,
   meridional, horizontal, and top boundaries?
2. Are component ratios or bounded component shares stable and interpretable
   across the seven days before and after the heatwave peak?
3. Can a sequence of 24-hour face-contribution glyphs show which boundaries
   contribute warming or cooling without being mistaken for physical wind
   vectors?

The initial scientific case is:

```text
region = pnw_bartusek
heat-budget boundaries = surface to 700 hPa
event definition = tas q90
analysis years = 1940-2024
season = June-August
event window = peak_time +/- 7 days
```

## Motivation

The current plots show that `advection` and `adiabatic` frequently cancel.
The scalar net-advection term does not reveal whether that cancellation occurs
during predominantly zonal, meridional, or vertical exchange, nor whether the
transport direction changes through the event life cycle.

Directional information should be recovered without changing the meaning or
sign of the existing `advection` term. Existing EHB files are immutable
upstream inputs.

## Confirmed Raw Inputs

A read-only inspection of the Venus EHB product
`pnw_bartusek_surface_700hPa_1940_2025/annual/heat_budget_2024.nc` confirmed
hourly, time-indexed variables for:

```text
flux_contribution_west
flux_contribution_east
flux_contribution_south
flux_contribution_north
flux_contribution_top

mass_flux_contribution_west
mass_flux_contribution_east
mass_flux_contribution_south
mass_flux_contribution_north
mass_flux_contribution_top
```

The upstream EHB implementation defines the heat-advection term as
`-integral(T U dot dA)` and applies inward-positive face signs:

```text
west  = +1
east  = -1
south = +1
north = -1
top   = +1
```

The same face signs are applied to the mass-flux contributions.

The intended Stage-1 extension uses the heat-flux contributions only. The
mass-flux variables confirm that a boundary-throughflow diagnostic would be
possible, but that diagnostic is not needed for the current scientific goal.

The raw NetCDF attributes observed in the inspected file contain inherited
metadata that do not reliably describe the variables or units. The prototype
must therefore validate the upstream formulas and write fresh, explicit
metadata. It must not propagate raw attributes without review.

A read-only 2024 comparison on Venus found zero maximum absolute error for all
three initial reconstruction checks:

```text
sum(face heat contributions) - raw advection_term
existing Stage-1 volume - raw domain_volume
normalized face sum - existing Stage-1 advection
```

## Scientific Interpretation

### Advective tendency components

Let each `F_face` be the signed raw heat-flux contribution. After applying the
same normalization as the existing Stage-1 `advection` variable,

```text
A_face = F_face / domain_volume * 3600
```

the candidate hourly heating-rate components are:

```text
A_zonal      = A_west + A_east
A_meridional = A_south + A_north
A_horizontal = A_zonal + A_meridional
A_vertical   = A_top
A_total      = A_horizontal + A_vertical
```

For a surface lower boundary, there is no bottom-face contribution. A later
fixed-pressure-bottom case must include the bottom face and document its sign.

The primary reconstruction invariant is:

```text
A_total == existing Stage-1 advection
```

within an explicitly tested numerical tolerance.

These component tendencies describe contributions to domain-mean temperature
change. They do not directly describe the direction of the mean wind.

### Ratios and shares

Direct ratios such as `A_zonal / A_meridional` and
`A_vertical / A_horizontal` become unstable when the denominator approaches
zero. The prototype will compare:

- direct signed ratios with a documented near-zero denominator mask;
- bounded absolute shares, such as
  `abs(A_zonal) / (abs(A_zonal) + abs(A_meridional))`;
- component-space angles calculated with `atan2`; and
- the unnormalized component tendencies, which remain the reference
  diagnostic.

No ratio will replace the underlying components.

The initial denominator mask is `0.005 K hr-1`. In the inspected 2024 data,
this is near the lower tail of the absolute meridional and horizontal
contributions. The value remains explicit in the CLI, plot title, and output
metadata so it can be assessed against the full composite.

### Face-contribution visualization

The prototype will preserve all signed heat contributions rather than reduce
them to one horizontal vector. This is important because multiple faces can
simultaneously contribute warming or cooling, and a single vector would discard
that information.

For each daily glyph, average each face's heating-rate contribution over a
complete 24-hour window. In an event composite, average each face contribution
within event and then across events. The glyph should use:

- face position to identify west, east, south, north, and top;
- color to distinguish warming from cooling;
- bar, spoke, or marker size to show contribution magnitude; and
- an explicit legend stating that the marks represent heating-rate
  contributions, not airflow direction.

Arrowheads should be avoided unless testing shows that they add information
without implying physical wind direction. A small domain-box schematic with
signed face bars is the preferred initial design.

The top-face contribution should be displayed separately or above the domain
box because it has no faithful direction in a two-dimensional map view.

## Prototype Architecture

The prototype remains isolated from Stage 1 until its scientific representation
is selected and validated.

```text
raw EHB face contributions + existing Stage-1 event definitions
  -> enhanced Stage-1 copy in an exploration subfolder
  -> peak-aligned in-memory experimental composite
  -> standalone comparison figures
```

The plotting entrypoint will consume prepared experimental data. It will not
load raw EHB files, rebuild event IDs, or hide the directional reductions in
plotting code.

Generated datasets, figures, and logs will remain outside Git under ignored
result paths.

## Planned Modifications

### Phase 1: Source audit and reusable calculations

- Audit a small selection of years and both supported bottom-boundary modes for
  variable presence, time alignment, finite values, and metadata drift.
- Confirm from the upstream EHB implementation whether `advection_term` is
  exactly the sum of the signed face heat contributions used here.
- Add reusable, source-independent directional calculations in
  `src/advection_direction.py`.
- Define explicit variable names, units, signs, formulas, and provenance.
- Add synthetic tests for each face, simultaneous opposing-face contributions,
  complete cancellation, top exchange, and near-zero-denominator cases.

### Phase 2: Experimental Stage-1 builder

- Add a thin builder such as
  `scripts/build_stage1_advection_exploration.py`.
- Load raw EHB inputs through `src/data_io.py`.
- Open Stage 1 through `src.analysis_io.open_harmonized_timeseries()`.
- Require exact hourly alignment between raw EHB data and Stage 1.
- Save a complete enhanced Stage-1 copy containing the original variables plus
  normalized face contributions under
  `results/stage1/advection_direction_exploration/`.
- Derive grouped components and ratios from the face contributions rather than
  persisting redundant variables.
- Build the peak-aligned composite in memory using the existing Stage-1
  `peak_time` and event-selection semantics.
- Refuse to overwrite an existing prototype product unless the CLI explicitly
  authorizes it.

### Phase 3: Standalone plotting prototype

- Add a thin plotter such as
  `scripts/plot_advection_direction_exploration.py`.
- Use `src/plot_style.py` and the non-interactive Matplotlib backend.
- Render unnormalized component tendencies as the scientific reference.
- Compare the most promising ratio, bounded-share, and angle representations.
- Render one 24-hour face-contribution glyph for each daily lag from day `-7`
  through day `+7`.
- Put the glyphs in a dedicated ribbon or inset cells so each day's west, east,
  south, north, and top contributions remain distinguishable.
- Encode warming or cooling by color and contribution magnitude by marker size
  or length. Always include a scale and a clear `advective heating-rate
  contribution` label.
- Add a tracked OpenPBS smoke scheduler for the `pnw_bartusek` case.

### Phase 4: Review and select a representation

Review the prototype for:

- directional interpretability;
- behavior during component cancellation;
- sensitivity to 24-hour window placement;
- event-to-event spread and directional cancellation;
- sensitivity to low contribution magnitude;
- legibility in a seven-day event window; and
- whether the daily face glyph adds information beyond the component traces.

Record the selected scientific default in a decision document before changing
the Stage-1 contract or current plots.

### Phase 5: Stage-1 integration

Only after prototype approval:

- update `docs/products/stage1_harmonized_timeseries.md` first;
- add the normalized signed face contributions as the minimal auditable
  Stage-1 extension;
- use provisional Stage-1 names `advection_west`, `advection_east`,
  `advection_south`, `advection_north`, and `advection_top`, each in
  `K hr-1` with an explicit contribution-to-domain-tendency description;
- derive zonal, meridional, horizontal, vertical, ratios, and shares downstream
  unless profiling demonstrates a reason to persist them;
- update `src/harmonize.py`, `src.analysis_io.py`, the Stage-1 builder, metadata,
  and tests;
- write new Stage-1 products beside existing products during validation rather
  than overwriting them;
- preserve compatibility for existing Stage-1 files and consumers during a
  documented migration period;
- rebuild affected Stage-2 products only if directional features are added to
  their contracts; and
- integrate the selected representation into active composite plots through
  `src/plotting.py`.

## Intended Prototype Outputs

Generated paths:

```text
results/stage1/advection_direction_exploration/
  harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc

results/plots_advection_direction_exploration/
  region_pnw_bartusek/
    boundary_surface_700hPa/
      time_range_1940_2024/
        advection_face_contributions.png
```

`advection_face_contributions.png` will show:

- individual signed face contributions;
- total, zonal, meridional, horizontal, and vertical contributions;
- signed ratios with invalid regions masked;
- the daily 24-hour face-contribution glyph sequence.

Each product and figure will record the region, pressure boundaries, event
definition, years, season filter, lag window, 24-hour aggregation convention,
input paths, and source commit.

## Compatibility Requirements

- Existing raw EHB files remain immutable.
- Existing Stage-1 and Stage-2 files remain valid and are not overwritten.
- Existing plots must render unchanged during Phases 1-4.
- Experimental variables must not be added to the current required Stage-1
  variable set until the contract and migration are approved.
- Extra variables in a future Stage-1 product must not silently change current
  Stage-2 reductions or plot layouts.
- Directional component signs must reconstruct the existing `advection`
  tendency before any visual interpretation is accepted.

## Validation Plan

Local validation in `dev_env`:

- unit tests with small synthetic datasets;
- exact tests of face-sign and grouping formulas;
- reconstruction of total advection from components;
- 24-hour window completeness and boundary behavior;
- face-component averaging before grouped reductions;
- masked behavior for near-zero denominators;
- metadata and unit validation;
- plotting tests for glyph count, lag placement, scale, labels, and non-empty
  output; and
- the full repository test suite.

Venus validation through OpenPBS:

- one short-year data-product smoke test before the full period;
- exact source and Stage-1 time alignment;
- expected dimensions, coordinates, variables, units, and signs;
- reconstruction-error statistics;
- event count, season-selection count, and complete-window count;
- non-empty NetCDF and PNG outputs;
- visual inspection of face placement, sign colors, magnitude scaling, and
  legibility; and
- recorded commit, job ID, exit status, elapsed time, CPU time, and peak memory.

## Decisions Required After the Prototype

1. Should the primary representation be component tendencies, bounded shares,
   angles, face glyphs, or a combination?
2. Should the top-face contribution be shown as its own scalar trace, in the
   face glyph, or both?
3. Should 24-hour windows be trailing, centered, or fixed daily bins relative
   to `peak_time`?
4. Should face-contribution magnitude be absolute, normalized within each
   event, or normalized across the composite?
5. Should all face contributions become required Stage-1 variables, including
   the bottom face for fixed-pressure-bottom products?

Implementation should not begin until the source audit resolves the sign,
normalization, and reconstruction questions in this plan.
