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

- Invoke the `venus-hpc` skill before deploying, submitting, monitoring, or
  diagnosing Venus work.
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

## Documentation authority

Before changing the pipeline, read [docs/README.md](docs/README.md) and follow
its reading order.

- [docs/pipeline_overview.md](docs/pipeline_overview.md) is the canonical
  current architecture and ownership map.
- `docs/products/` contains active Stage 1 and Stage 2 data contracts.
- `docs/workflows/` contains temporal composites, spatial composites, and
  shared plotting procedures.
- `docs/decisions/` contains active scientific defaults.
- `docs/legacy/` contains inactive Stage 3, Stage 4, and historical material.

This file defines execution and contribution constraints. It is not a second
pipeline specification. The documentation is the stable source of truth for
intended behavior, architecture, scientific contracts, and compatibility.
Tests are executable conformance checks, and the code is the implementation.

Plan and document new development before implementing it. Follow the
documentation-first change process in `docs/README.md`: identify affected
contracts and consumers, record the intended change, define compatibility and
validation requirements, and only then change code and tests. Preserve
documented behavior by default when an inconsistency is found. Do not revise a
contract after the fact merely to match an accidental implementation change.

## Engineering boundaries

- Build reusable datasets before figures. Plotting code must consume prepared
  Stage 1, Stage 2, composite, or spatial products.
- Keep source-specific loading in `src/data_io.py`, regional harmonization in
  `src/harmonize.py`, product I/O in `src/analysis_io.py`, and reusable
  selection, event, composite, and diagnostic logic in `src/`.
- Keep `scripts/` as thin CLI entrypoints and `schedulers/` as Venus OpenPBS
  entrypoints.
- Keep full spatial ERA5 fields outside Stage 1 and Stage 2. Follow
  [docs/workflows/spatial_composites.md](docs/workflows/spatial_composites.md)
  for the separate daily-field, climatology, and spatial-composite path.
- Use `src/plot_style.py` directly or through `src/plotting.py` for every
  active figure. Follow
  [docs/workflows/plotting.md](docs/workflows/plotting.md) before changing
  shared visual behavior.
- Keep paths configurable through command-line arguments or shared
  configuration instead of adding new hardcoded paths.
- Do not extend, run, or restore legacy Stage 3 or Stage 4 workflows unless the
  user explicitly reactivates them.

## Track long-running work

- Invoke the `command-center` skill for qualifying asynchronous runs.
- In addition to the shared schema, record the intended product, authoritative
  branch and commit, Venus checkout, scheduler script, scientific
  configuration, requested resources, input paths, log path, and expected
  outputs.
- Mark the task complete only after the expected Stage 1, Stage 2, composite,
  spatial, or figure artifacts pass their documented contracts and scientific
  checks.
- Preserve detailed immutable provenance beside project outputs.

## Validation

Run feasible unit and plotting tests locally from the repository root:

```bash
mamba activate dev_env
python -m pytest
```

Use targeted tests while developing, then run the full suite before handing off
a change. Treat passing tests as evidence that the implementation conforms to
the documented contracts, not as authority to override those contracts. Tests
must use small synthetic fixtures and must not require access to the Venus
production datasets.

For changes that affect data ingestion, product construction, or production
plotting, local tests are necessary but not sufficient. After the exact commit
has been pushed and cleanly deployed with authorization, run a short PBS smoke
test on Venus and validate:

- expected dimensions, coordinates, variables, metadata, units, and signs;
- Stage 1, Stage 2, or spatial product contract markers;
- time coverage and event or baseline selection counts;
- output paths and non-empty artifacts; and
- figure creation through the shared plotting style.

Record the exact PBS job ID, deployed commit, scheduler script, scientific
configuration, environment, input and output paths, requested resources,
terminal state, log path, and completed validation with every production
claim.

Do not weaken validation or substitute fabricated production data when a Venus
input is missing. Report the missing path or upstream product clearly.
