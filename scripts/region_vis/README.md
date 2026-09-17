# Stage 1 Region Visualization

This self-contained plotting entrypoint inventories a Stage 1 run directory and
draws each distinct regional averaging domain as a colored wireframe on one
Northern Hemisphere map. It reads only NetCDF metadata, verifies the Stage 1
product marker, and checks the stored latitude and longitude bounds against the
canonical definitions in `src/config.py`.

The default run, `bf232281_20260819`, contains 14 products representing seven
unique regions. The two threshold variants for a region produce one map
boundary.

Within an authorized Venus PBS job, the plotting CLI uses external artifacts:

```bash
mamba activate dev_env
export HWA_ARTIFACT_ROOT="${HWA_ARTIFACT_ROOT:-$HOME/HW-analysis/artifacts}"
python scripts/region_vis/plot_stage1_regions.py \
  --run-dir "$HWA_ARTIFACT_ROOT/stage1/runs/bf232281_20260819" \
  --output-path "$HWA_ARTIFACT_ROOT/region_vis/FRESH_ATTEMPT/stage1_regional_domains.png" \
  --expected-region-count 7
```

Production rendering on Venus uses
`scripts/region_vis/schedule_plot_stage1_regions.sh`. Its submission workflow
must provide the clean deployed `PROJECT_ROOT` and exact `EXPECTED_COMMIT`.
`RUN_DIR`, `OUTPUT_PATH`, `EXPECTED_REGION_COUNT`, `LOG_DIR`, and
`VENUS_MAMBA_ENV` remain configurable environment variables.
Follow [artifact locations](../../docs/workflows/artifacts.md) when submitting;
prepared paths must be under `HWA_ARTIFACT_ROOT`, and logs under `HWA_LOG_ROOT`.
