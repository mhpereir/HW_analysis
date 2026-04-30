# HW Analysis Pipeline Architecture

## Purpose

This document serves as the grounding architecture outline for the new heatwave analysis pipeline. The goal of the project is to investigate Eulerian heat flux diagnostics in conjunction with other variables relevant to Pacific Northwest heatwave development, while keeping the codebase modular, extensible, and easy to revise.

This project brings together data products from several prior workflows, including:

- heatwave thresholds
- local wave activity (LWA)
- Eulerian heat budget diagnostics
- planetary boundary layer (PBL) height
- ARCO-only variables such as cloud fraction and surface radiative fluxes

The immediate purpose of this document is to define a stable architectural direction before implementation proceeds too far.

---

## High-level design principles

The new pipeline should follow the same broad modular philosophy as the earlier LWA analysis project, but with stronger separation between data harmonization, event selection, diagnostics, and plotting.

The main design principles are:

1. **Dataset-first rather than plot-first**
   - The code should first construct harmonized, analysis-ready datasets.
   - Plotting should consume prepared outputs, not perform core analysis steps internally.

2. **Regional time series as the primary analysis product**
   - The primary unit of analysis should be a regional time series dataset.
   - Gridded products should remain available as an optional branch for later diagnostics and map-based analyses.

3. **Explicit handling of timestep mismatches**
   - The architecture must support variables with different native temporal resolutions.
   - Native resolution and analysis resolution should be treated as separate concepts.

4. **Reusable selector and event logic**
   - Event masks, event IDs, duration filters, and top-event selection should be built once in reusable modules.
   - These should not be buried inside plotting scripts.

5. **Separation of concerns**
   - Preprocessing, event definition, diagnostics, compositing, and plotting should live in distinct modules.

---

## Scientific scope

The current working scope includes the following variables.

### Locally available

- surface temperature
- HW threshold
- LWA (anticyclonic, cyclonic)
- LWA threshold
- PBL height in pressure units
- Eulerian heat budget diagnostics, including:
  - volume mean temperature
  - domain volume
  - time tendency of regional mean temperature
  - net advection (this term is coming in with the wrong sign convention; needs to be fixed upon harmonization)
  - adiabatic term
  - diabatic term

### ARCO-sourced

- cloud cover fraction
- net solar radiation at the surface
- net longwave radiation at the surface

---

## Main architectural decision

The recommended architecture is a **two-stage top-level pipeline**:

1. **Build an analysis-ready regional dataset** which contains all of the necessary variables for analysis, in a uniform timeseries.
2. **Run selectors, event detection, composites, and plotting against that dataset**

To avoid ambiguity:

- **Stage** should refer only to these two top-level pipeline stages.
- **Workflow layer** should refer to the finer-grained internal modules and processing steps inside those stages.

In other words, the `src/` package is not defining more than two top-level stages. It is defining the reusable workflow layers that implement those two stages.

Practical module mapping:

- **Top-level Stage 1: build the analysis-ready regional dataset**
  - primary modules: `data_io.py`, `preprocess.py`, `harmonize.py`
  - handoff/storage module: `analysis_io.py`
  - supporting modules: `config.py`, selected utilities from `diagnostics.py` when they contribute derived analysis variables
- **Top-level Stage 2: analyze that dataset**
  - primary modules: `analysis_io.py`, `selectors.py`, `events.py`, `composites.py`, `plotting.py`
  - supporting modules: `config.py`, selected utilities from `diagnostics.py`

---

## Recommended package structure

The project should use a `src/` package, similar to the LWA analysis project, but with more explicit module boundaries.

```text
HW_analysis/
├── src/
│   ├── config.py
│   ├── data_io.py
│   ├── analysis_io.py
│   ├── preprocess.py
│   ├── harmonize.py
│   ├── selectors.py
│   ├── events.py
│   ├── composites.py
│   ├── diagnostics.py
│   └── plotting.py
├── scripts/
│   ├── build_regional_timeseries.py
│   ├── make_event_composites.py
│   ├── plot_composite_timeseries.py
│   └── plot_top_events.py
├── notebooks/
├── tests/
│   ├── test_preprocess.py
│   ├── test_selectors.py
│   ├── test_composites.py
│   └── test_time_alignment.py
└── README.md
```

---

## Module responsibilities

### `config.py`

This module should define:

- file paths
- region definitions
- seasons
- source-specific constants
- output paths
- default analysis settings
- optional run presets

It should remain declarative and should not perform data loading or computations.

### `data_io.py`

This module should be responsible for:

- opening raw source datasets
- handling source-specific filename conventions
- standardizing variable names where possible
- preserving xarray-native lazy loading

This layer should not perform major transformations beyond what is necessary to return valid, source-consistent data objects.

### `analysis_io.py`

This module should handle IO for internal pipeline products, especially the handoff between the two top-level stages.

Responsibilities should include:

- saving the harmonized analysis-ready regional dataset produced by Stage 1
- reopening the saved analysis-ready regional dataset for Stage 2 workflows
- validating required metadata and dataset conventions on read
- managing filenames or paths for stable internal products

This module should not contain source-specific raw data loading logic. That boundary should remain in `data_io.py`.

### `preprocess.py`

This module should contain only low-level preprocessing tasks, such as:

- flooring time coordinates
- dropping leap days
- unit conversions
- coordinate standardization
- area-weighted regional means
- day-of-year climatologies and anomalies
- resampling or interpolation utilities

This module should **not** contain event logic, composite logic, or plotting code.

### `harmonize.py`

This module should transform raw source variables into a common analysis framework.

Responsibilities should include:

- mapping variables from different sources to a shared naming convention
- aligning time coordinates
- aligning source-specific spatial domains
- applying region averaging when needed
- harmonizing selected variables to a target analysis timestep
- assembling a common analysis-ready dataset

This is the module that turns heterogeneous source fields into a stable internal representation.

### `selectors.py`

This module should define reusable selection logic such as:

- heatwave threshold exceedance masks
- LWA threshold exceedance masks
- compound selection masks
- seasonal filtering (ensure selected events occur entirely within a given season ie JJA)
- quantile-based selection modes
- top-N event selection modes

Selectors should produce boolean masks or filtered subsets, but should not themselves compute composites.

### `events.py`

This module should convert masks into explicit event objects or event IDs.

Responsibilities should include:

- converting boolean masks into contiguous event IDs
- filtering by event duration
- identifying event peaks
- assigning event ranks
- extracting event-level summary metadata

This module should define the concept of an event as a first-class object in the pipeline.

### `composites.py`

This module should handle:

- peak-aligned composites
- event-centered extraction windows
- ensemble member composites
- event-mean and ensemble-mean reductions
- percentile envelopes across events or members
- top-event extraction

It should accept harmonized datasets plus event definitions, then return derived composite products.

### `diagnostics.py`

This module should compute or manage domain-specific derived diagnostics, such as:

- combined radiative metrics
- residual checks
- transformed or normalized diagnostics
- optional derived event metrics for ranking or labeling

This module is for scientific diagnostic logic, not generic preprocessing.

### `plotting.py`

This module should contain only plotting functions.

Plot functions should:

- accept prepared datasets or tables
- not load raw data
- not compute masks internally
- not redefine event logic internally

This is a strict design goal. Plotting should be a consumer, not a hidden analysis layer.

---

## Detailed workflow layers

These layers are internal to the two-stage pipeline above.

### Workflow layer 1: Raw data access

Raw datasets are opened from:

- locally stored files
- threshold products from previous workflows
- ARCO-backed variables

At this layer, data remain close to source form.

### Workflow layer 2: Harmonization

All variables are standardized into a common internal representation.

This should include:

- common time axis handling
- optional daily or hourly analysis resolution
- spatial averaging where required
- shared naming conventions
- source metadata retention

The output of this layer is the main analysis-ready dataset, which completes top-level Stage 1.

### Workflow layer 3: Event selection

Event masks are built from threshold-based or ranked criteria.

Examples:

- heatwave-selected periods
- anticyclonic LWA-selected periods
- compound HW and LWA periods
- top-N hottest events
- top-N LWA events

These are then converted into event IDs and event-level summary information.

### Workflow layer 4: Composite generation

Given an analysis dataset and an event definition, the code should produce:

- mean composites centered on event peaks
- ensemble mean and spread
- percentile envelopes across events
- individual-event extracts for selected top events

This layer is part of top-level Stage 2.

### Workflow layer 5: Plotting and export

Prepared composite products are rendered into:

- multi-row time series panels
- top-event individual traces
- optional future map composites
- optional summary tables or serialized outputs

This layer is also part of top-level Stage 2.

---

## Primary analysis dataset design

The most important intermediate product should be a regional time series dataset.

### Recommended dimensions

- `time` #hourly
- optional `member`
- optional `event`
- optional `lag_tas`   #day relative to peak tas within an event ID; set to 0 when event ID = 0
- optional `lag_lwa_a` #day relative to peak lwa_a during within an event ID; set to 0 when event ID = 0
- optional `lag_lwa_c` #day relative to peak lwa_c during within an event ID; set to 0 when event ID = 0

The initial ERA5 workflow should transform all input variables into a standardized hourly_time axis, which will keep the simple name `time`.

### Candidate variables

Daily event-definition variables:

- `tas_region(daily_time -> hourly_time)`
- `hw_threshold(daily_time -> hourly_time)`
- `lwa_a_region(daily_time -> hourly_time)`
- `lwa_c_region(daily_time -> hourly_time)`
- `lwa_a_threshold(dayofyear -> hourly_time)`
- `lwa_c_threshold(dayofyear -> hourly_time)`
- `hw_flag(daily_time -> hourly_time)`
- `lwa_flag(daily_time -> hourly_time)`
- `hw_event_id(daily_time -> hourly_time)`
- `lwa_event_id(daily_time -> hourly_time)`

Hourly diagnostic variables:

- `T_mean(hourly_time)`
- `volume(hourly_time)`
- `dTdt(hourly_time)`
- `advection(hourly_time)`
- `adiabatic(hourly_time)`
- `diabatic(hourly_time)`
- `cloud_frac(hourly_time)` #pending implementation
- `rad_sw_net_sfc(hourly_time)`
- `rad_lw_net_sfc(hourly_time)`
- `pbl_p(hourly_time)`

Projected hourly event labels:

- `hw_event_id_hourly(hourly_time)`
- `lwa_a_event_id_hourly(hourly_time)`
- `lwa_c_event_id_hourly(hourly_time)`

The projected hourly event labels are produced by flooring each `hourly_time` timestamp to its calendar day and looking up the corresponding daily event ID. Every hour on an event day receives the same event ID; hours on non-event days receive `0`.

This dataset should also carry metadata describing:

- variable source
- native timestep
- analysis timestep
- preprocessing choices
- region name
- threshold settings used

---

## Temporal resolution strategy

This is one of the most important architectural questions.

Some variables currently exist at daily resolution, while others are hourly. The architecture should explicitly distinguish:

- **native timestep**: the resolution provided by the source
- **analysis timestep**: the resolution chosen for a specific analysis run

### Recommended initial approach

For version 1 of the pipeline:

- define event masks and event IDs on a **daily** axis
- preserve hourly Eulerian heat-budget diagnostics on an **hourly** axis
- project daily event IDs onto hourly diagnostics using calendar-day lookup
- avoid aggregating hourly heat-budget variables to daily unless a specific analysis requires it

### Why daily event definitions with hourly targets

- the threshold logic for HW and LWA is daily-oriented
- the Eulerian heat-budget diagnostics are naturally hourly and should remain available at native cadence
- a daily-to-hourly event-ID projection is easier to validate than a mixed-frequency merge on one `time` coordinate
- this keeps future daily composites and hourly composites both possible

### Future support

Hourly event definitions can be added later by:

- introducing hourly-compatible selectors
- revisiting LWA temporal resolution
- defining explicit policies for upsampling or holding daily quantities
- adding separate hourly event-ID products when thresholds are truly hourly

---

## Selection and event logic

Selection logic should be configurable and reusable.

### Selector types to

- `hw`
- `lwa_a`
- `lwa_c`
- `compound_hw_lwa`
- `top_n_hw`
- `top_n_lwa`

### Event features to support

- contiguous event IDs
- minimum duration filters
- seasonal filtering
- peak alignment
- ranking by event maximum
- ranking by event-integrated quantity

This design is strongly preferred over treating threshold exceedance days as isolated, because the scientific targets are event evolutions rather than independent timesteps.

---

## Composite design

Composite functions should work on harmonized analysis datasets and event objects.

### Required

- event-centered mean trajectories
- ensemble mean and spread
- percentile bounds
- optional top-event traces

### Default

The default alignment should be:

- centered on the peak of the selector variable used to define the event

This should be configurable if needed later.

### Primary comparison mode

The default comparison pattern should be:

- ERA5 as a single reference time series
- CanESM as member-wise composites summarized into ensemble mean and spread

---

## Plotting priorities

The target composite figure currently envisioned includes multiple vertically stacked panels on a shared x-axis.

### Planned composite figure panels

1. regional mean temperature on the left axis and volume on the right axis
2. time tendency of regional mean temperature
3. net advection, adiabatic, and diabatic terms
4. net solar and longwave surface radiation on one axis, cloud fraction on the other
5. LWA anticyclonic and cyclonic on one axis, PBL height on the other

In addition, the workflow should support:

- plotting a few individual high-impact events
- selecting top events by temperature or by LWA
- changing threshold quantiles without hardcoding them

### Plotting rule

Plotting functions should never handle raw loading or event generation internally.

---

## Recommended implementation phases

To reduce risk, implementation should proceed in stages.

### Phase 1

- build harmonized regional time series dataset
- support HW-based and LWA-based selectors
- implement one composite engine
- implement a reduced three-row plot:
  - `t_mean` and `volume`
  - `dTdt`
  - `adv_net`, `adiabatic`, `diabatic`

### Phase 2

- add ARCO cloud and radiation variables
- add PBL height
- add top-event extraction and individual-event plots

### Phase 3

- add hourly analysis support
- add optional gridded composite support
- add richer event ranking and comparison diagnostics

This order keeps the most difficult and most failure-prone tasks early:

- time harmonization
- threshold alignment
- event logic
- composite validation

---

## Important design constraints

### Keep preprocessing narrow

`preprocess.py` should not become a catch-all module.

It should remain limited to:

- time handling
- coordinate handling
- averaging
- anomaly calculations
- unit normalization
- basic resampling

If event logic, composite logic, or scientific diagnostics accumulate there, the codebase will become harder to maintain.

### Avoid script-level monoliths

The new project should avoid scripts that:

- load data
- preprocess
- build masks
- compute composites
- plot
- export

all in one place.

Scripts should orchestrate module calls, not define the analysis.

### Prefer reusable intermediate products

Where practical, save intermediate regional time series datasets or composite outputs so that the same expensive harmonization steps do not need to be repeated for every figure.

---

## Initial implementation decisions

The following decisions should be treated as the current default architecture:

- **Primary analysis unit:** regional time series
- **Primary event definition:** contiguous events, not isolated selected days
- **Primary cadence for v1:** daily event definitions projected onto hourly diagnostics
- **Primary alignment for composites:** event peak
- **Primary comparison mode:** ERA5 reference versus CanESM ensemble summary
- **Primary design pattern:** build once, analyze many times

These defaults can be revised later, but they provide a stable starting point.

---

## Open questions to revisit later

The following questions remain open and should be revisited during implementation:

1. Should LWA ultimately be recomputed at hourly resolution?
2. Which analyses, if any, should aggregate hourly diagnostics to daily summaries?
3. Should the first analysis dataset be saved as NetCDF, Zarr, or both?
4. How should top events be ranked:
   - selector maximum
   - temperature maximum
   - event-integrated diagnostic
5. Should compound selectors be handled as first-class selector types or composed from simpler selectors?
6. How much gridded support should be retained in the first implementation?

---

## Summary

The recommended architecture is a modular, dataset-first pipeline centered on a harmonized regional analysis dataset. Event selection and event definition should be reusable and explicit, while composites and plotting should operate downstream of that harmonized data product.

The most important architectural shift relative to the earlier LWA project is this:

**do not organize the new project around individual plotting scripts; organize it around reusable analysis stages and stable intermediate datasets.**

This document should serve as the baseline grounding reference for the first implementation phase and can be revised as design decisions become more concrete.
