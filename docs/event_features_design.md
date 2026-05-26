# Event-level Fixed-window Feature Extraction Design

## Purpose

Add a new Stage-2 workflow for converting each heatwave event into a compact event-level feature vector suitable for pathway comparison, exploratory clustering, dimensionality reduction, and later statistical analysis.

The script should consume the saved Stage-1 harmonized regional time-series dataset produced by `build_regional_timeseries.py`. It should not rebuild thresholds, event IDs, or the harmonized time series. Its job is to summarize already-detected events into fixed-window metrics centered on each event's `peak_time`.

This workflow is intentionally separate from composite plotting. Composites summarize the event population as trajectories; event features summarize each event as one row in a table.

Adaptive growth-window features based on `dTdt > 0` should be deferred to a later module/version. The first implementation should focus only on fixed windows relative to each event peak.

---

## Pipeline role

This is a **Stage-2 analysis product**.

It should consume the Stage-1 harmonized dataset and produce an event-level feature table. It should not modify the Stage-1 product.

Recommended role in the existing pipeline:

```text
Stage 1:
  build_regional_timeseries.py
      -> harmonized_regional_timeseries.nc

Stage 2:
  composites/
      -> event-centered composite plots
  top_events/
      -> individual-event diagnostic plots
  event_features/
      -> event-level fixed-window feature table
```

The event-feature table can later be consumed by separate clustering/PCA/EOF-style scripts.

---

## Recommended file and folder structure

Going forward, Stage-2 scripts that need multiple files should live in their own folders.

Recommended structure:

```text
HW_analysis/
├── src/
│   ├── analysis_io.py
│   ├── selectors.py
│   ├── events.py
│   ├── composites.py
│   ├── event_features.py        # reusable feature-building logic, optional first-version target
│   └── ...
├── scripts/
│   ├── build_regional_timeseries.py
│   ├── composites/
│   │   ├── plot_composite_timeseries_all.py
│   │   ├── plot_composite_timeseries_split.py
│   │   └── plot_top_events.py
│   ├── event_features/
│   │   ├── build_event_features.py
│   │   ├── build_event_feature_pca.py
│   │   ├── event_feature_config.py
│   │   └── README.md
│   └── diagnostics/
│       ├── plot_event_summary.py
│       ├── plot_threshold_timeseries.py
│       └── plot_diurnal_cycle.py
└── results/
    ├── stage1/
    └── event_features/
```

For a first implementation, it is acceptable to keep most logic inside:

```text
scripts/event_features/build_event_features.py
```

The script should remain a thin CLI wrapper around reusable functions.

---

## Inputs

### Required input

A saved Stage-1 harmonized regional time-series dataset, normally opened with:

```python
from src import analysis_io

ds = analysis_io.open_harmonized_timeseries(input_path)
```

The dataset should contain the standard time-indexed variables and the event-summary table appended by `build_regional_timeseries.py`.

Core required time-indexed variables:

```text
T_mean(time)
volume(time)
dTdt(time)
advection(time)
adiabatic(time)
diabetic/time residual variable: diabatic(time)
tas_region(time)
tas_climatology(time)
hw_threshold(time)
hw_event_id(time)
lwa_a_region(time)
lwa_c_region(time)
```

Core required event-level variables:

```text
event_id(event)
start_time(event)
end_time(event)
duration(event)
peak_time(event)
tas_peak(event)
tas_anom_peak(event)
tas_excess_peak(event)
tas_excess_integral(event)
lwa_a_peak(event)
lwa_c_peak(event)
```

### Optional extended input variables

When the Stage-1 dataset was built with full diagnostics, the feature workflow should also use:

```text
soil_moisture(time)
cloud_cover(time)
pbl_p_mean(time)
nslr_heating_rate_approx(time)
nssr_heating_rate_approx(time)
sshf_heating_rate_approx(time)
slhf_heating_rate_approx(time)
surface_energy_heating_rate_approx(time)
```

These should be used only when running in extended mode.

---

## Default versus extended modes

The script should support two operating modes.

### Default mode

Default mode should work with the core harmonized dataset only.

It should not require `soil_moisture`, `cloud_cover`, PBL, or surface flux/radiation diagnostics.

Default-mode feature output should include:

```text
I_dTdt_pre
I_advection_pre
I_adiabatic_pre
I_diabatic_pre
I_lwa_a_pre_peak
I_lwa_c_pre_peak
T_anom_mean_ant
days_from_solstice
duration
tas_peak
tas_anom_peak
tas_excess_peak
tas_excess_integral
lwa_a_peak
lwa_c_peak
```

Recommended default clustering matrix:

```text
I_adiabatic_pre
I_advection_pre
I_diabatic_pre
I_lwa_a_pre_peak
T_anom_mean_ant
days_from_solstice
duration
tas_anom_peak
tas_excess_integral
```

### Extended mode

Extended mode should include all default features plus land-surface, cloud, PBL, and surface energy diagnostics when present.

Extended-mode feature output should additionally include:

```text
soil_moisture_mean_ant
soil_moisture_change
cloud_cover_mean_ant
pbl_p_mean_ant
I_nslr_pre
I_nssr_pre
I_sshf_pre
I_slhf_pre
I_surface_energy_pre
```

Recommended extended clustering matrix:

```text
I_adiabatic_pre
I_advection_pre
I_diabatic_pre
I_lwa_a_pre_peak
soil_moisture_mean_ant
soil_moisture_change
cloud_cover_mean_ant
T_anom_mean_ant
days_from_solstice
duration
tas_anom_peak
tas_excess_integral
```

Note that the clustering matrix should be selected downstream from the feature table. The first version of this script should produce features, not run clustering.

---

## Surface flux sign convention

The approximate surface-energy heating-rate variables retain the ERA5/source sign convention unless explicitly transformed elsewhere.

For `sshf_heating_rate_approx` and `slhf_heating_rate_approx`, negative values indicate the surface is losing energy. For sensible heat flux, this corresponds to upward energy transfer from the surface into the atmosphere.

The event-feature builder should preserve native signs by default and record this clearly in metadata.

Optional future addition:

```text
--atmosphere-oriented-surface-fluxes
```

which would add transformed variables such as:

```text
I_sshf_atm_pre = -1 * I_sshf_pre
I_slhf_atm_pre = -1 * I_slhf_pre
```

Do not make this transformation silently.

---

## Config file

Create:

```text
scripts/event_features/event_feature_config.py
```

This file should be declarative only. It should define paths, variable groups, feature windows, and feature naming conventions. It should not open data or compute features.

Suggested initial content:

```python
"""Configuration for fixed-window event feature extraction."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage1"
    / "harmonized_regional_timeseries_pnw_bartusek_tas_q90_1940_2024.nc"
)

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "event_features"
    / "hw_event_features_fixed_windows.nc"
)

DEFAULT_CSV_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "event_features"
    / "hw_event_features_fixed_windows.csv"
)

TIME_DIM = "time"
EVENT_DIM = "event"
PEAK_TIME_NAME = "peak_time"
EVENT_ID_NAME = "event_id"

# Fixed windows in hours relative to peak_time.
WINDOWS = {
    "heat_budget_pre": (-96, 0),
    "lwa_pre_peak": (-96, 24),
    "antecedent_state": (-168, -24),
    "antecedent_change": (-168, 0),
    "near_peak": (-24, 24),
    "decay": (0, 72),
}

# Variables integrated over time.
DEFAULT_INTEGRAL_FEATURES = {
    "dTdt": "heat_budget_pre",
    "advection": "heat_budget_pre",
    "adiabatic": "heat_budget_pre",
    "diabatic": "heat_budget_pre",
    "lwa_a_region": "lwa_pre_peak",
    "lwa_c_region": "lwa_pre_peak",
}

EXTENDED_INTEGRAL_FEATURES = {
    "nslr_heating_rate_approx": "heat_budget_pre",
    "nssr_heating_rate_approx": "heat_budget_pre",
    "sshf_heating_rate_approx": "heat_budget_pre",
    "slhf_heating_rate_approx": "heat_budget_pre",
    "surface_energy_heating_rate_approx": "heat_budget_pre",
}

# Variables averaged over fixed windows.
DEFAULT_MEAN_FEATURES = {
    "tas_anom": "antecedent_state",
}

EXTENDED_MEAN_FEATURES = {
    "soil_moisture": "antecedent_state",
    "cloud_cover": "antecedent_state",
    "pbl_p_mean": "antecedent_state",
}

# Variables where start/end differences are useful.
EXTENDED_CHANGE_FEATURES = {
    "soil_moisture": "antecedent_change",
}

# Event-summary variables copied directly from the Stage-1 event table.
EVENT_SUMMARY_FEATURES = (
    "event_id",
    "start_time",
    "end_time",
    "peak_time",
    "duration",
    "tas_peak",
    "tas_anom_peak",
    "tas_excess_peak",
    "tas_excess_integral",
    "lwa_a_peak",
    "lwa_c_peak",
)

# Seasonal coordinate feature.
SOLSTICE_MONTH = 6
SOLSTICE_DAY = 21

# Optional event filtering defaults.
DEFAULT_SEASON_MONTHS = (6, 7, 8)
REQUIRE_FULL_EVENT = False
```

The important design point is that the windows are configurable without editing the computational logic.

---

## Script entrypoint

Create:

```text
scripts/event_features/build_event_features.py
```

Recommended CLI:

```text
python scripts/event_features/build_event_features.py \
    --input-path results/stage1/harmonized_regional_timeseries_pnw_bartusek_tas_q90_1940_2024.nc \
    --output-path results/event_features/hw_event_features_fixed_windows.nc \
    --csv-output-path results/event_features/hw_event_features_fixed_windows.csv \
    --season-months 6 7 8
```

Recommended command-line options:

```text
--input-path
--output-path
--csv-output-path
--use-extended-variables
--allow-missing-extended
--season-months 6 7 8
--require-full-event
--overwrite
```

Recommended behavior:

- `--use-extended-variables` requests extended features.
- If `--use-extended-variables` is passed, missing extended variables should raise an error by default.
- If `--allow-missing-extended` is also passed, compute all available extended features and skip missing ones with a clear warning.
- If neither option is passed, run in default mode and ignore extended variables even if they exist.

---

## Core computation design

### 1. Open the harmonized dataset

Use the existing Stage-1/Stage-2 handoff function:

```python
from src import analysis_io

ds = analysis_io.open_harmonized_timeseries(args.input_path)
```

### 2. Extract the event table

The Stage-1 harmonized dataset contains both time-indexed variables and event-level summary variables. Extract the event-level subset by selecting variables with the `event` dimension only.

Pseudo-logic:

```python
def event_summary_table(ds, event_dim="event"):
    names = [
        name
        for name, da in ds.data_vars.items()
        if event_dim in da.dims and set(da.dims).issubset({event_dim})
    ]
    if not names:
        raise ValueError("Input dataset contains no event-summary variables.")
    return ds[names]
```

### 3. Optional seasonal filtering

If `--season-months` is provided, apply event-level seasonal filtering before feature extraction.

Use existing selector logic:

```python
from src import selectors

event_table = selectors.select_events_by_season(
    event_table,
    args.season_months,
    require_full_event=args.require_full_event,
)
```

If no events remain, fail clearly.

### 4. Compute `tas_anom` if absent

The feature table needs `T_anom_mean_ant`, based on near-surface regional temperature anomaly rather than `T_mean`.

If a direct `tas_anom` variable is absent, compute:

```python
ds["tas_anom"] = ds["tas_region"] - ds["tas_climatology"]
```

This should not be written back to the Stage-1 file; it is only a temporary derived variable for feature extraction.

### 5. Event-centered fixed-window slicing

For each event, use `peak_time` and fixed hour offsets to extract windows from the `time` dimension.

For a window `(start_lag, end_lag)` in hours:

```python
start = peak_time + np.timedelta64(start_lag, "h")
end = peak_time + np.timedelta64(end_lag, "h")
window = ds.sel(time=slice(start, end))
```

Use actual timestamps, not integer positions. This protects against missing hours and makes missing-sample diagnostics easier.

### 6. Integral features

For variables in `K hr-1`, summing hourly values over an hourly axis gives `K`.

Initial implementation can assume the Stage-1 dataset is hourly and compute:

```python
integral = window[var].sum("time", skipna=True)
```

Metadata should record:

```text
integral_method = hourly_sum_assuming_1h_spacing
```

A later version can replace this with timestep-weighted integration if non-hourly products become possible.

Recommended feature names:

```text
I_dTdt_pre
I_advection_pre
I_adiabatic_pre
I_diabatic_pre
I_lwa_a_pre_peak
I_lwa_c_pre_peak
I_nslr_pre
I_nssr_pre
I_sshf_pre
I_slhf_pre
I_surface_energy_pre
```

Note: LWA integrals are exposure metrics, not energy or heating contributions. Metadata should describe them as:

```text
LWA exposure over fixed window
```

### 7. Mean antecedent-state features

For antecedent-state variables:

```python
mean_value = window[var].mean("time", skipna=True)
```

Recommended feature names:

```text
T_anom_mean_ant
soil_moisture_mean_ant
cloud_cover_mean_ant
pbl_p_mean_ant
```

### 8. Change features

For `soil_moisture_change`, use a robust window-end minus window-start estimate rather than a single timestamp.

Recommended method:

```text
mean over final 24 h of antecedent_change window
minus
mean over first 24 h of antecedent_change window
```

For the default `antecedent_change = (-168, 0)` window:

```text
soil_moisture_change = mean(soil_moisture[-24h, 0h]) - mean(soil_moisture[-168h, -144h])
```

This reduces sensitivity to individual hourly samples.

### 9. Solstice timing feature

For each event, compute:

```text
days_from_solstice
```

where solstice is approximated as June 21 of the event year.

Suggested implementation:

```python
solstice = np.datetime64(f"{year}-06-21")
days_from_solstice = (peak_day - solstice) / np.timedelta64(1, "D")
```

This is more interpretable than raw day-of-year for the current heatwave-pathway question.

Optional later addition:

```text
sin_doy
cos_doy
```

but these should not be required in the first version.

---

## Output

The script should write an event-feature table as NetCDF and optionally CSV.

Recommended NetCDF dimensions:

```text
event
```

Recommended coordinate:

```text
event
```

Recommended default-mode variables:

```text
event_id(event)
start_time(event)
end_time(event)
peak_time(event)
duration(event)

tas_peak(event)
tas_anom_peak(event)
tas_excess_peak(event)
tas_excess_integral(event)
lwa_a_peak(event)
lwa_c_peak(event)

I_dTdt_pre(event)
I_advection_pre(event)
I_adiabatic_pre(event)
I_diabatic_pre(event)
I_lwa_a_pre_peak(event)
I_lwa_c_pre_peak(event)

T_anom_mean_ant(event)
days_from_solstice(event)

n_samples_heat_budget_pre(event)
n_samples_lwa_pre_peak(event)
n_samples_antecedent_state(event)
```

Recommended extended-mode additions:

```text
soil_moisture_mean_ant(event)
soil_moisture_change(event)
cloud_cover_mean_ant(event)
pbl_p_mean_ant(event)

I_nslr_pre(event)
I_nssr_pre(event)
I_sshf_pre(event)
I_slhf_pre(event)
I_surface_energy_pre(event)

n_samples_antecedent_change(event)
```

Recommended global attributes:

```text
pipeline_stage = "stage_2_event_features"
feature_method = "fixed_windows_relative_to_peak_time"
input_path = ...
extended_variables_used = 0 or 1
adaptive_windows_used = 0
heat_budget_window_hours = "-96,0"
lwa_window_hours = "-96,0"
antecedent_state_window_hours = "-168,-24"
antecedent_change_window_hours = "-168,0"
```

Each feature variable should also have attributes such as:

```text
source_variable
window_name
window_lag_hours
operation = "sum" | "mean" | "change" | "copy"
units
```

---

## Validation checks

The script should fail clearly when required core variables are absent.

Required default variables:

```text
T_mean
volume
dTdt
advection
adiabatic
diabatic
tas_region
tas_climatology
lwa_a_region
lwa_c_region
peak_time
duration
tas_peak
tas_anom_peak
tas_excess_integral
```

Required extended variables when `--use-extended-variables` is passed without `--allow-missing-extended`:

```text
soil_moisture
cloud_cover
pbl_p_mean
nslr_heating_rate_approx
nssr_heating_rate_approx
sshf_heating_rate_approx
slhf_heating_rate_approx
surface_energy_heating_rate_approx
```

Additional checks:

```text
- all selected events must have finite peak_time
- requested windows must overlap the dataset time range
- missing edge-window samples should be counted per event
- n_valid_samples should be stored for each configured window
- output path should not be overwritten unless --overwrite is passed
```

Useful diagnostic variables:

```text
n_samples_heat_budget_pre(event)
n_samples_lwa_pre_peak(event)
n_samples_antecedent_state(event)
n_samples_antecedent_change(event)
```

These will identify events near dataset boundaries or events with missing hourly data.

---

## Clustering matrix design

The first version should not perform clustering. It should produce a clean feature table from which clustering inputs can be selected.

### Default-mode clustering matrix

```text
I_adiabatic_pre
I_advection_pre
I_diabatic_pre
I_lwa_a_pre_peak
T_anom_mean_ant
days_from_solstice
duration
tas_anom_peak
tas_excess_integral
```

### Extended-mode clustering matrix

```text
I_adiabatic_pre
I_advection_pre
I_diabatic_pre
I_lwa_a_pre_peak
soil_moisture_mean_ant
soil_moisture_change
cloud_cover_mean_ant
T_anom_mean_ant
days_from_solstice
duration
tas_anom_peak
tas_excess_integral
```

Before clustering, variables should be standardized. Do not cluster raw variables with different units and magnitudes.

Recommended later consumers:

```text
scripts/event_features/build_event_feature_pca.py
scripts/event_features/cluster_event_feature_pca.py
scripts/event_features/plot_event_feature_space.py
scripts/event_features/plot_event_feature_pca.py
scripts/event_features/plot_cluster_composites.py
```

---

## PCA dataset output design

After the fixed-window feature table is built, a second Stage-2 script should construct a PCA-ready dataset and write the PCA transform itself. This script should be based on the exploratory `event_feature_clustering.py` workflow, but renamed and narrowed to a reproducible PCA builder:

```text
scripts/event_features/build_event_feature_pca.py
```

The current exploratory script is useful as the foundation because it already:

- opens the event-feature NetCDF table;
- defines derived feature names such as heat-budget fractions, square-root LWA exposure, cosine season phase, and log TAS excess integral;
- maps derived variables back to their required source variables;
- validates that required variables are present;
- converts event-level variables into numeric arrays;
- optionally standardizes values for diagnostic plotting.

The PCA builder should keep those feature-preparation utilities, but should replace the plotting-only endpoint with a dataset-writing endpoint. It should not perform clustering. Clustering should remain a later consumer of the PCA score dataset.

### PCA script entrypoint

Recommended command:

```text
python scripts/event_features/build_event_feature_pca.py \
    --input-path results/event_features/hw_event_features_fixed_windows.nc \
    --output-path results/event_features/hw_event_feature_pca.nc \
    --features I_dTdt_pre f_adiabatic_pre f_diabatic_pre f_advection_pre \
               sqrt_I_lwa_a_pre_peak T_anom_mean_ant cos_days_from_solstice duration \
    --overwrite
```

Recommended command-line options:

```text
--input-path
--output-path
--features
--n-components
--scaler standard | robust
--drop-missing-events
--overwrite
```

Default behavior should be:

```text
--scaler standard
--drop-missing-events
--n-components unset, meaning keep min(n_events, n_features) components
```

### Recommended default PCA feature matrix

The first default matrix should emphasize pathway structure rather than event impact:

```text
I_dTdt_pre
f_adiabatic_pre
f_diabatic_pre
f_advection_pre
sqrt_I_lwa_a_pre_peak
T_anom_mean_ant
cos_days_from_solstice
duration
```

The following variables should usually be kept as diagnostic/outcome variables rather than PCA inputs in the first pass:

```text
tas_anom_peak
log10_tas_excess_integral
tas_excess_integral
```

They may be copied into the PCA output dataset as event metadata so that later plots can test whether PCA axes or clusters correspond to event severity.

### Derived PCA input variables

The PCA builder should support the same derived variables as the current exploratory feature-space script.

Heat-budget fractions:

```text
f_adiabatic_pre = I_adiabatic_pre / budget_activity_pre
f_diabatic_pre  = I_diabatic_pre  / budget_activity_pre
f_advection_pre = I_advection_pre / budget_activity_pre

budget_activity_pre = abs(I_adiabatic_pre) + abs(I_diabatic_pre) + abs(I_advection_pre)
```

Other derived variables:

```text
sqrt_I_lwa_a_pre_peak = sqrt(I_lwa_a_pre_peak), valid only when I_lwa_a_pre_peak >= 0
cos_days_from_solstice = cos(days_from_solstice * 2*pi / 365)
log10_tas_excess_integral = log10(tas_excess_integral), valid only when tas_excess_integral > 0
```

The PCA builder should preserve the current script's source-variable mapping pattern so that derived variables can be requested through `--features` while validation still checks the actual source variables in the feature table.

### Standardization before PCA

PCA should be run on standardized event-level features, not on raw variables. The script should compute all physical features and derived variables first, then standardize each selected PCA column across events.

For the default `standard` scaler:

```text
X_scaled[:, j] = (X[:, j] - mean_j) / std_j
```

The script should save the fitted centering and scaling values to the output dataset. This is required for reproducibility and for projecting later events into the same PCA space.

Optional robust scaling can be added for sensitivity tests:

```text
X_scaled[:, j] = (X[:, j] - median_j) / IQR_j
```

This should not replace the default standard scaler unless outliers are found to dominate the PCA.

### PCA output dataset

The PCA builder should write a NetCDF dataset. Recommended dimensions:

```text
event
pc
feature
event_original
```

Recommended coordinates:

```text
event            # retained event IDs, preferably from event_id rather than row number
pc               # PC1, PC2, PC3, ...
feature          # selected PCA input feature names
event_original   # original event coordinate from the input feature table
```

Recommended variables:

```text
pc_score(event, pc)
pc_loading(pc, feature)
explained_variance(pc)
explained_variance_ratio(pc)
cumulative_explained_variance_ratio(pc)
feature_center(feature)
feature_scale(feature)
feature_matrix(event, feature)
feature_matrix_scaled(event, feature)
valid_event_mask_original(event_original)
```

Recommended copied event metadata:

```text
event_id(event)
start_time(event)
end_time(event)
peak_time(event)
duration(event)
tas_peak(event)
tas_anom_peak(event)
tas_excess_peak(event)
tas_excess_integral(event)
lwa_a_peak(event)
lwa_c_peak(event)
```

Optional copied diagnostic variables:

```text
I_dTdt_pre(event)
I_adiabatic_pre(event)
I_diabatic_pre(event)
I_advection_pre(event)
I_lwa_a_pre_peak(event)
T_anom_mean_ant(event)
days_from_solstice(event)
log10_tas_excess_integral(event)
```

The key downstream variable is:

```text
pc_score(event, pc)
```

This is the matrix that later clustering scripts should consume.

The key interpretation variable is:

```text
pc_loading(pc, feature)
```

This records the feature weights for each principal component. Large positive or negative loadings indicate which standardized input features define a PC axis.

### PCA output metadata

Recommended global attributes:

```text
pipeline_stage = "stage_2_event_feature_pca"
input_path = ...
source_feature_table = ...
pca_features = comma-separated feature list
scaler = "standard" or "robust"
pca_implementation = "sklearn.decomposition.PCA"
n_input_events = ...
n_valid_events = ...
n_features = ...
n_components = ...
missing_event_policy = "drop_missing_events"
clustering_performed = 0
```

Each output variable should include enough metadata to avoid ambiguity. For example:

```text
pc_score.description = "Principal-component scores for each retained event."
pc_loading.description = "Feature loading vectors for standardized PCA input variables."
explained_variance_ratio.description = "Fraction of standardized feature variance explained by each PC."
feature_center.description = "Centering value subtracted from each PCA input feature."
feature_scale.description = "Scaling value used for each PCA input feature."
```

### PCA validation checks

The script should fail clearly when:

```text
- the input feature table has no event dimension;
- a requested PCA feature cannot be resolved to source variables;
- a required source variable is absent;
- fewer than two events remain after finite-value filtering;
- fewer than two features are selected;
- any selected feature has zero variance after filtering;
- n_components exceeds min(n_valid_events, n_features);
- output path exists and --overwrite is not passed.
```

The script should also report how many events were dropped because of missing or invalid PCA inputs.

### Relationship to clustering

`build_event_feature_pca.py` should stop after writing the PCA dataset. It should not assign cluster labels.

Recommended later consumers:

```text
scripts/event_features/cluster_event_feature_pca.py
scripts/event_features/plot_event_feature_pca.py
scripts/event_features/plot_cluster_composites.py
```

The clustering script should consume:

```text
pc_score(event, pc)
```

with a user-selected subset such as:

```text
PC1 PC2 PC3
```

Cluster labels, when implemented, should be written to a separate clustering dataset or appended to a copied PCA dataset as:

```text
cluster_label(event)
cluster_probability(event, cluster)   # only for probabilistic methods such as GMM
```

### PCA design cautions

PCA identifies orthogonal axes of variance, not physical mechanisms by itself. A PC should be interpreted through its loadings, scatter plots, and event composites.

Because the heat-budget fractions are compositional and partly redundant, the PCA should be tested with both:

```text
I_dTdt_pre + all three heat-budget fractions
```

and a reduced matrix such as:

```text
I_dTdt_pre + f_adiabatic_pre + f_advection_pre
```

If the leading PCs change substantially when one fraction is removed, the interpretation should be treated as sensitive to the compositional encoding.

---

## Non-goals for first version

Do not include adaptive `dTdt > 0` growth-window integrals yet.

Do not perform clustering.

Do not compute composites.

Do not add new event definitions.

Do not modify the Stage-1 harmonized dataset.

Do not move event detection or ranking logic into this script.

---

## Implementation order

1. Create `scripts/event_features/event_feature_config.py`.
2. Create a minimal `scripts/event_features/build_event_features.py` that opens the harmonized dataset and extracts the event table.
3. Implement fixed-window slicing around `peak_time`.
4. Add heat-budget integrals for `dTdt`, `advection`, `adiabatic`, and `diabatic`.
5. Add LWA exposure features.
6. Add `T_anom_mean_ant` and `days_from_solstice`.
7. Add event-summary passthrough variables.
8. Add optional extended variables.
9. Add missing-sample diagnostics.
10. Write NetCDF output.
11. Add optional CSV output.
12. Add validation and overwrite checks.
13. Rename/adapt `event_feature_clustering.py` into `build_event_feature_pca.py`.
14. Add derived PCA input variables and source-variable validation.
15. Add event-level standardization and PCA fitting.
16. Write the PCA NetCDF dataset.
17. Keep clustering as a separate downstream script.

---

## Design cautions

Fixed-window integrals should be interpreted as pathway descriptors, not as complete causal attribution.

For example, `I_adiabatic_pre` and `I_advection_pre` summarize tendencies over the same pre-peak window, but clustering will still be sensitive to:

- window choice,
- event alignment,
- seasonal timing,
- heatwave threshold definition,
- event duration,
- missing samples near dataset boundaries,
- and whether variables are standardized before clustering.

The config file should therefore record the exact windows used, and the output dataset should preserve those choices in metadata.
