# HW_analysis Agent Instructions

## Project scope

This repository is for analysis and visualization of heatwaves using:

- pre-calculated Eulerian Heat Budget (EHB) data;
- raw or locally prepared ERA5 data and related diagnostics; and
- reusable analysis datasets assembled by this repository.

EHB calculations themselves are out of scope. Treat EHB files as immutable
upstream inputs. The active pipeline is dataset-first: assemble shared products
once, then make all figures and analyses consume those products.

## Venus-only execution

The scripts and production paths in this repository are exclusively for the
UVic Climate Lab Venus HPC environment. Required scientific data is expected to
already exist on Venus. Do not make local execution of production workflows a
goal, download replacement data, or add fallback data sources merely because
the inputs are unavailable in a local checkout.

- Develop and test code in the local Git checkout.
- Never edit source directly on Venus.
- Use Git commits and clean fast-forward deployment to move code to
  `/home/mhpereir/HW_analysis` on Venus.
- Use the Mamba-managed `dev_env` environment on Venus.
- Run data-intensive scripts through the tracked OpenPBS schedulers, not on the
  Venus login node.
- Treat raw data, assembled datasets, figures, logs, caches, and environments as
  external or generated artifacts. Do not commit them.
- Do not transfer a mixed code-and-data repository root with `rsync`.
- Require explicit authorization before pushing, deploying, transferring data,
  or submitting or modifying PBS jobs.

Read-only inspection of the Venus checkout, data paths, job state, and logs is
appropriate when needed to understand or diagnose a workflow.

## Local Python environment

Use the local Mamba environment named `dev_env` for compilation checks, unit
tests, and small prototypes:

```bash
mamba activate dev_env
python -c "import sys; print(sys.executable)"
```

This environment is intended to mirror the `dev_env` environments used on
Venus and Alliance HPC and the Python environment in the Eulerian Heat Budget
Google container image. Use it as the first compatibility gate for shared code
and dependencies.

Environment similarity does not replace validation on the target platform.
Venus production workflows must still run through OpenPBS, Alliance workflows
through Slurm, and Google workflows through their container build and service
gates. Before testing, confirm that the resolved interpreter and required
packages are present. Do not silently fall back to the base environment or
install ad hoc packages without updating the relevant dependency contract.

## Active data products

There are two active reusable product stages:

1. **Stage 1 - harmonized regional time series**
   - Produced by `scripts/build_stage1_harmonized_timeseries.py`.
   - Ingests pre-calculated EHB output, ERA5 inputs, thresholds, LWA, and
     optional surface, soil-moisture, cloud, and PBL diagnostics.
   - Is the canonical handoff from heterogeneous source data to downstream
     analysis.
   - Must be opened through `src.analysis_io` when Stage-1 contract validation
     is required.

2. **Stage 2 - analysis feature datasets**
   - Produced by the builders in `scripts/event_features/`.
   - Includes event-feature and baseline-day feature datasets derived from
     Stage 1.
   - Must consume Stage 1 rather than reload raw data or rebuild thresholds,
     event IDs, or harmonization logic.

All plotting scripts should consume Stage-1 or Stage-2 products. Plotting code
must not silently become another ingestion or dataset-construction path.

Stages 3 and 4 are not part of the active workflow. Their PCA and clustering
implementations under `scripts/event_features/old/`, along with related
schedulers, documentation, and tests, are legacy material. Do not extend, run,
or restore these stages unless the user explicitly reactivates them.

## Plotting contract

`src/plot_style.py` is the single shared plotting-style module. All new and
modified plotting scripts must use it so figures remain consistent across the
project.

- Import `plot_style` from `src`.
- Reuse its theme, sizes, colors, labels, line widths, axis formatting, legend
  helpers, and `save_figure()` function.
- Add a generally useful style choice to `src/plot_style.py` rather than
  duplicating constants or local `matplotlib` configuration across scripts.
- Keep scientific data preparation separate from visual styling.
- Add or update plot tests when changing shared style behavior or figure
  semantics.

## Code responsibilities

- `src/config.py`: Venus data paths, regions, source constants, and defaults.
- `src/data_io.py`: source-specific file discovery and raw data loading.
- `src/preprocess.py`: coordinate, unit, time, averaging, anomaly, and
  resampling utilities.
- `src/harmonize.py`: source alignment and Stage-1 construction.
- `src/analysis_io.py`: stable validation and I/O for assembled products.
- `src/selectors.py` and `src/events.py`: reusable selection and event logic.
- `src/composites.py` and `src/diagnostics.py`: analysis-ready computations.
- `src/plotting.py` and `src/plot_style.py`: plotting prepared products.
- `scripts/`: thin command-line entrypoints.
- `schedulers/`: Venus OpenPBS execution entrypoints.

Prefer reusable, tested functions in `src/` over analysis logic embedded in a
plotting script or scheduler. Keep paths configurable through command-line
arguments or shared configuration instead of adding new hardcoded paths.

## Validation

Run feasible unit and plotting tests locally from the repository root:

```bash
mamba activate dev_env
python -m pytest
```

Use targeted tests while developing, then run the full suite before handing off
a change. Tests must use small synthetic fixtures and must not require access to
the Venus production datasets.

For changes that affect data ingestion, product construction, or production
plotting, local tests are necessary but not sufficient. After the exact commit
has been pushed and cleanly deployed with authorization, run a short PBS smoke
test on Venus and validate:

- expected dimensions, coordinates, variables, metadata, units, and signs;
- Stage-1 or Stage-2 product contract markers;
- time coverage and event or baseline selection counts;
- output paths and non-empty artifacts; and
- figure creation through the shared plotting style.

Do not weaken validation or substitute fabricated production data when a Venus
input is missing. Report the missing path or upstream product clearly.
