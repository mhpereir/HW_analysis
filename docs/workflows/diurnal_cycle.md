# GMT diurnal-cycle diagnostics

`scripts/plot_diurnal_cycle.py` consumes an existing Stage-1 product and
compares all retained June-August hourly samples in two classes:
`hw_event_id > 0` and `hw_event_id == 0`. It preserves the stored daily TAS
heatwave classification. It does not select only complete JJA events, align
events at their peaks, rebuild thresholds, smooth the series, or subtract a
climatology. Means and sample quartiles are calculated independently at each
hour. The IQR describes sample spread, not uncertainty in the mean.

## Time convention and compatibility

The default offset is zero. Calendar-month selection, event membership and
hour grouping therefore all use native GMT/UTC timestamps. Figures identify
the axis as `GMT hour (UTC+0)`. This is the convention for the PNW layer
comparison and the investigation of the 09:00 GMT cooling feature.

The explicit `--local-utc-offset-hours` option remains available for historical
comparisons. It shifts hour labels only; month filtering and heatwave membership
remain tied to native timestamps. Such a shift can place the GMT day boundary
inside the displayed cycle. It does not define local-calendar-day heatwaves.
The `local_hour` coordinate name is retained for API compatibility, including
when its values are GMT hours. Existing Stage-1 products are unchanged.

The default figure filename identifies the selected months and time offset.
Titles identify the actual input's pressure boundaries, region and year range.
Counts report distinct contributing GMT dates rather than rounding hourly
sample counts into days. The numerical composite also retains total hourly
counts, counts by hour and nonmissing counts for each variable, so incomplete
coverage is visible without silently changing the sample population.
Hour ticks use integers. The volume axis uses adaptive scientific precision
so the small volume variations in a fixed pressure layer retain distinct
tick labels in the original m2 Pa units.

## Pressure-layer comparison

Use the accepted TAS q90 products for `pnw_bartusek`, 1940-2024, with identical
time axes and `hw_event_id` values in all three layers: surface-700 hPa,
700-500 hPa and 500-300 hPa. The elevated-layer products preserve the surface
reference's heatwave definitions. The pressure-layer producer does not need
to be rerun to make these figures.

Supply each accepted input explicitly. The boundary CLI options describe that
input; they do not extract another layer from a supplied NetCDF file. Labels
come from stored metadata. Numerical values are retained in their Stage-1
units, including K hr-1 for temperature tendencies.

The reusable reducer lives in `src/diurnal.py`. The existing script keeps
its public builder imports for compatibility. `--composite-output-path` saves
the plotted means, quartiles and counts through `src/analysis_io.py`, with
the source path and SHA-256. Production requires this companion file to make
the comparison independently inspectable. Both output files must be new paths.

## Venus execution and acceptance

Use the tracked `schedulers/schedule_diurnal_cycle.sh` after authorized
publication and clean commit-pinned deployment. Pass `PROJECT_ROOT`,
`EXPECTED_COMMIT`, `HWA_ARTIFACT_ROOT`, `HWA_LOG_ROOT`, `LOG_DIR`, `INPUT_PATH`,
`OUTPUT_PATH`, `COMPOSITE_OUTPUT_PATH`, `REGION`, `BOTTOM_BOUNDARY`,
`TOP_BOUNDARY`, `THRESHOLD_VARIABLE`, `QUANTILE`, `TIME_START` and `TIME_END`
explicitly through PBS. The scheduler fixes JJA and offset zero and requests
one CPU, 4 GB and ten minutes per figure.

Also pass `EXPECTED_INPUT_SHA256` and a fresh `VALIDATION_OUTPUT_PATH`.
After plotting, the scheduler invokes `scripts/validate_diurnal_cycle.py`.
Its independent NumPy reduction checks every stored mean, quartile and sample
count against the source, verifies metadata and checksums, and checks thermal
closure. It records time-axis and heatwave-ID hashes for comparison across
layers and JJA sample counts by year. A failed check fails the job. This does
not replace the subsequent original-resolution visual review.

Local synthetic regressions must cover GMT month boundaries and class
membership, shifted-hour compatibility, hourly sample counts, direct
mean/quantile agreement, layer metadata, NetCDF round trips and no-overwrite
behavior. Run the full local suite before handoff.

For production, first render a representative upper-layer product. Validate
the exact source hash, GMT/JJA selection, all 24 hourly bins, class and
variable counts, means and quartiles against an independent direct reduction,
and stored pressure metadata. Confirm matching populations across layers and
check the composite heat-budget identity. Inspect all final PNGs at original
resolution and preserve checksums, commit, PBS identities and logs beside the
run evidence. A completed PBS job alone is not scientific acceptance.

The vertical comparison can establish whether the feature appears in multiple
layers. It cannot by itself identify the origin of a residual diabatic term.
