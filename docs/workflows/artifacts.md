# Stable artifact and scheduler-log locations

Code checkouts and durable research products have independent lifetimes. All
active HW_analysis entrypoints use the same external storage defaults on Venus,
regardless of the selected Git checkout or submission directory:

| Setting | Default | Contents |
| --- | --- | --- |
| `PROJECT_ROOT` | Required for PBS jobs | Clean, commit-pinned source checkout |
| `HWA_ARTIFACT_ROOT` | `$HOME/HW-analysis/artifacts` | Prepared inputs, datasets, figures, run manifests and checksums |
| `HWA_LOG_ROOT` | `$HOME/HW-analysis/logs` | Scheduler/application logs |
| `LOG_DIR` | `$HWA_LOG_ROOT` | Optional run-specific subdirectory of the log root |

These roots are absolute paths, not paths relative to `PROJECT_ROOT` or
`PBS_O_WORKDIR`. Overrides must also be absolute. Configure roots before
starting Python or submitting a job. Python defaults live in
`src/artifact_paths.py`; shell entrypoints source `config/artifact_paths.sh`.
Conformance tests keep their defaults and validation aligned.

## Inputs, outputs and compatibility

The artifact root replaces the old checkout-local `results/` prefix. Existing
relative names are unchanged: `stage1/`, `stage1_climatology/`,
`stage2_event_features/`, `stage2_baseline_features/`, `spatial_composites/`,
`plots_*/` and existing run-scoped namespaces remain directly under this root.
Do not add an extra `results/` level. Product contents, scientific defaults and
figure styles are unchanged.

Both prepared inputs and outputs of active PBS jobs must be absolute paths
beneath `HWA_ARTIFACT_ROOT`. Jobs reject relative paths, the root directory
itself, traversal or symlinks escaping it, and checkout-local legacy paths.
`LOG_DIR` must be the log root or a subdirectory. Storage roots and the selected
source checkout must be disjoint: neither may contain the other. Direct CLI
path overrides remain available for synthetic tests and explicit inspection;
production uses the PBS guards.

Raw ERA5, upstream EHB, thresholds, LWA, environments and tracked configuration
files retain their existing locations. They are not migrated by this policy.
Stage 3 and Stage 4 remain inactive and their legacy entrypoints are excluded.

Use fresh, explicitly named run/attempt output paths for production. Stable
storage does not authorize overwriting an accepted product. Stage-2 and
matching schedulers now refuse replacement of existing final products, including
at staged publication. Older diurnal, event-summary and threshold wrappers now
require explicit `INPUT_PATH` and `OUTPUT_PATH` (or threshold `OUTPUT_DIR`),
and the combined spatial job requires `COMPOSITE_OUTPUT_PATH` and
`FIGURE_OUTPUT_PATH`. The older q75 split wrapper also requires `INPUT_PATH`.
Their former implicit overwrite behavior is not retained.
The Stage-1 and region-inventory schedulers also reject existing outputs.
Existing builder publication checks remain in force; do not infer a product's
generating commit from the current code checkout. Record exact input/output
paths, commit, PBS ID and validation evidence in the existing task/run records and immutable
manifests. Historical provenance is evidence, not a configuration file: do not
rewrite old paths in it.

## Submission and deployment

Every active scheduler requires `PROJECT_ROOT` and `EXPECTED_COMMIT`, verifies
the deployed SHA and cleanliness, and loads path configuration from that
checkout. Submit only an authorized, clean deployment containing this policy.
Older commit-pinned checkouts do not inherit it; a moved checkout or repaired
Git link does not update its scheduler defaults.

For example, after selecting the deployed checkout and a fresh output path:

```bash
export PROJECT_ROOT=/absolute/path/to/HW_analysis-COMMIT
export EXPECTED_COMMIT=FULL_VALIDATED_COMMIT
export HWA_ARTIFACT_ROOT="$HOME/HW-analysis/artifacts"
export HWA_LOG_ROOT="$HOME/HW-analysis/logs"
export INPUT_PATH="$HWA_ARTIFACT_ROOT/stage1/<prepared-input>.nc"
export OUTPUT_DIR="$HWA_ARTIFACT_ROOT/plots_top_events/<fresh-attempt>"
export LOG_DIR="$HWA_LOG_ROOT/<fresh-attempt>"
cd "$PROJECT_ROOT"
qsub -v PROJECT_ROOT,EXPECTED_COMMIT,HWA_ARTIFACT_ROOT,HWA_LOG_ROOT,INPUT_PATH,OUTPUT_DIR,LOG_DIR \
  schedulers/schedule_top_events.sh
```

Root overrides in an interactive shell are not automatically copied by PBS:
pass them explicitly with `qsub -v`. Avoid exporting the entire local
environment with `-V`. The tracked daily-spatial array submitter forwards its
resolved roots and commit pin automatically, including in its dry-run output.

Application logs use PBS job IDs under `LOG_DIR` and refuse to replace an
existing log. Preflight errors after logging starts are retained there too.
PBS failures before the script can load its configuration are scheduler-level
failures; inspect `qstat -fx`/PBS diagnostics rather than assuming an application
log exists.

## Migration and acceptance

The user copied the old results contents into `artifacts/` and retained the
source. This policy neither deletes that source nor installs compatibility
symlinks. Preserve old logs and repaired-checkout metadata backups. A file-size
inventory establishes copy coverage, not checksum or scientific acceptance.

Before integration, run the full local `dev_env` suite, shell syntax checks,
and path/default/guard conformance tests. The tracked presentation-feature
smoke scheduler also runs artifact-path, Stage-2 scheduler and spatial-shell
regressions in Venus `dev_env`. After authorized commit, push and clean
deployment, run that smoke job and a short figure job using an existing copied input
and a fresh output namespace. Verify the input, nonempty output, expected
product/figure contract, recorded commit, and log location. Only then use that
revision for subsequent scheduled tasks. Other unfinished branches must absorb
this change and validate their branch-specific schedulers before production.

At branch/worktree retirement, retain artifacts and logs independently. The
Git lifecycle and shared cleanup skill still govern removal of source paths.
