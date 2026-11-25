# 011: Explicit boundary-flux budget partition

## Decision and scope

Support `original` (the default) and `boundary_flux` partitions in the same
regional products. The first campaign is PNW Bartusek, TAS q90, surface-700 hPa,
1940-2024. This is a regrouping of signed control-volume boundary fluxes, not
a local vertical-gradient diagnostic. Upstream EHB inputs remain immutable.

Let V be `advection_top` plus `advection_bottom` when present. Store:

```
adv_horizontal = advection - V
vertical_contribution = adiabatic + V
```

Verify horizontal advection against the sum of the four lateral faces, and
verify that the new pair equals `advection + adiabatic`. Preserve the original
variables, residual `diabatic`, event definitions, coordinates and diagnostics.
The residual remains an upstream closure residual, not directly measured
diabatic heating. Reject missing required faces, inconsistent units or face
closure, and partial or inconsistent partition extensions. Fixed-pressure
layers require the bottom face; surface layers may omit it.

## Product extension and compatibility

This is an additive version-1 `budget_partition_contract_version` extension
to Stage-1 contract v2. `budget_partitions` identifies `original,boundary_flux`.
New variables carry units, signed formula and boundary-flux provenance.
Legacy products remain valid for original-partition consumers. An explicitly
requested boundary-flux plot must fail when its fields are absent.

The normal Stage-1 builder adds the extension. A separate migration entrypoint
may derive it from a validated immutable Stage-1 v2 source and save a new
product with source path and SHA256. It must not reload upstream inputs or
change existing arrays. Publication uses a fresh output namespace.

Rebuild the hourly climatology from the extended Stage 1, including mean,
sample standard deviation and count for both new variables. Compute these
from hourly samples; never add component standard deviations or quantiles.
Carry partition metadata and require compatible companions for the selected
partition. Subtract climatology before stacking events and computing spreads.

Stage-2 event and baseline builders include `I_adv_horizontal_pre` and
`I_vertical_contribution_pre` when this extension is present, using the same
inclusive windows and finite-sample rules as the original terms. The derived
anomaly tables include `I_adv_horizontal_anom_pre` and
`I_vertical_contribution_anom_pre`. Original integrals retain their meanings;
`I_dyn_pre` and its anomalous counterpart remain partition-independent.

## Figures and pilot

All/split temporal CLI consumers and Stage-2 event-versus-baseline comparisons
accept `--budget-partition original|boundary_flux`. Use the shared style for
explicit horizontal-advection and total-vertical labels. Select actual named
fields before any sample reductions; do not relabel persisted original data.
Use separate output paths for the two partitions. Preserve original defaults,
the ten-panel layout, JJA legend summaries and hourly/24-hour display variants.

The 96-hour Stage-2 pilot uses 97 inclusive hourly samples and reproduces the
accepted peak-in-JJA event cohort and clean-baseline flags. Generate absolute
and climatological-anomaly comparisons in both full four-panel and existing
two-panel layouts. Temporal composites retain their full-JJA event selection.
Do not introduce local-day or exact-duration window changes in this experiment.

## Validation and acceptance

Test signed top/bottom regrouping, closure, missing and inconsistent fields,
units and metadata, legacy compatibility, independent local-window integrals,
sample-based climatological spreads and unchanged input arrays. Exercise both
partitions through temporal all/split and both Stage-2 layouts, with correct
anomaly labels and unchanged cohorts. Run the full local `dev_env` suite and
lint, then exact-commit PBS validation and the PNW pilot. Independently compare
every retained source variable and reference row, check all hourly/integrated
identities and climatology counts, retain hashes/provenance, and inspect
representative figures at original resolution. Scheduler success alone is
insufficient. Master integration, scientific acceptance and retirement remain
separate decisions.

## Execution

Use `scripts/build_stage1_budget_partitions.py --input-path OLD --output-path NEW`
for the additive Stage-1 migration, then the standard climatology and Stage-2
builders. The anomaly builder defaults to its historical full-JJA cohort;
`--no-require-full-event` explicitly selects the declared peak-in-JJA reference
contract without changing any reference rows.

`schedulers/schedule_budget_partition_pilot.sh` accepts `RUN_MODE=smoke` for
targeted queued tests, followed by `RUN_MODE=pilot` for validated production.
Both require `PROJECT_ROOT`, `EXPECTED_COMMIT`, `INPUT_PATH`, `REFERENCE_DIR`,
`OUTPUT_DIR` and external `LOG_DIR`. The reference directory contains the
accepted event/baseline NetCDF files and their original checksum manifest.
The pilot regenerates both schemas and refuses any existing output directory.
It verifies all source arrays, calendar-hour statistics, every 96-hour budget
integral, reference fields and anomaly reductions independently. It writes
64 PNGs (56 temporal, eight integration), eight PDFs and six NetCDF products.
The four temporal PBS wrappers and baseline-comparison wrapper also accept
`BUDGET_PARTITION=original|boundary_flux`, with explicit fresh output paths.

The complete pilot requests one CPU, 40 GB and 30 minutes. The initial PNW
attempt exceeded its 8 GB allocation while rendering temporal figures, reaching
about 27 GiB; retain that cancelled attempt and use a fresh output namespace
for the larger allocation. Smoke-only submissions can override this with one
CPU, 4 GB and five minutes. Compare measured peak memory with the allocation
before expanding the campaign.
