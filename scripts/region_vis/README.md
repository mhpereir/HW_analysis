# Stage 1 Region Visualization

This self-contained plotting entrypoint inventories a Stage 1 run directory and
draws each distinct regional averaging domain as a colored wireframe on one
Northern Hemisphere map. It reads only NetCDF metadata, verifies the Stage 1
product marker, and checks the stored latitude and longitude bounds against the
canonical definitions in `src/config.py`.

The default run, `bf232281_20260819`, contains 14 products representing seven
unique regions. The two threshold variants for a region produce one map
boundary.

Run the small metadata and plotting workflow locally with:

```bash
mamba activate dev_env
python scripts/region_vis/plot_stage1_regions.py \
  --run-dir results/stage1/runs/bf232281_20260819 \
  --output-path results/region_vis/stage1_regional_domains_bf232281_20260819.png \
  --expected-region-count 7
```

Production rendering on Venus uses
`scripts/region_vis/schedule_plot_stage1_regions.sh`. Its submission workflow
must provide the clean deployed `PROJECT_ROOT` and exact `EXPECTED_COMMIT`.
`RUN_DIR`, `OUTPUT_PATH`, `EXPECTED_REGION_COUNT`, `LOG_DIR`, and
`VENUS_MAMBA_ENV` remain configurable environment variables.
