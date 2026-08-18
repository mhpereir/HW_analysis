# Decision 008: Retire PBL Diagnostics From Active Products

## Status

Accepted.

## Decision

Planetary-boundary-layer top-pressure diagnostics are no longer part of the
active `HW_analysis` product or plotting contracts. New Stage-1 harmonized
regional time series, regional hourly climatologies, Stage-2 extended feature
products, and temporal-composite figures omit `pbl_p_mean`, `pbl_p_p05`,
`pbl_p_p95`, and `pbl_p_mean_ant`.

The standalone raw-data path configuration and `src.data_io.open_era5_pbl_p()`
remain available for explicitly scoped ad hoc work. They are not production
Stage-1 inputs and their availability must not gate an active pipeline run.

Extended 5x2 temporal figures retain their existing dimensions. Cloud cover
moves into the former PBL panel, and soil moisture occupies the final panel on
its own axis. Absolute cloud cover remains bounded to the physical fraction
range from zero to one. Climatological cloud-cover anomalies are not bounded to
that range.

## Rationale

The PBL diagnostic has not added useful interpretive value to the current
analysis, while its separate regional retrieval has delayed otherwise-ready
Stage-1 production. Removing it simplifies the active data dependency graph and
allows every region to proceed once its heat-budget and remaining ERA5-family
inputs are available.

Separate soil-moisture and cloud-cover panels also remove the final figure's
twin-axis comparison between variables with unrelated units and ranges.

## Compatibility

Stage-1 contract version 2 remains current because that version is defined by
the normalized signed boundary-face tendencies. PBL diagnostics were optional
extended variables, so their removal does not require a contract-version bump.

Existing Stage-1, climatology, and Stage-2 products that contain PBL variables
remain readable. Updated consumers ignore those extra variables. Existing
artifacts and figures are not overwritten; new products and plots are written
to run-specific paths and omit PBL fields and panels.

The raw PBL loader is retained for ad hoc use, but no active product builder,
feature extractor, plotter, or PBS workflow may depend on it.

## Validation

- New full-diagnostic Stage-1 products contain every documented non-PBL
  diagnostic and no `pbl_p*` variable.
- New regional hourly climatologies contain no PBL mean, standard deviation, or
  count variable.
- New extended Stage-2 event and baseline products omit `pbl_p_mean_ant`.
- Extended all-event, split-event, climatological-anomaly, and top-event plots
  preserve the 5x2 layout with separate cloud-cover and soil-moisture panels.
- A representative Venus run verifies that PBL input availability is not
  inspected or required.
