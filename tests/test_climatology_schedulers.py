import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULERS = (
    REPO_ROOT / "schedulers" / "schedule_build_stage1_hourly_climatology.sh",
    REPO_ROOT / "schedulers" / "schedule_plot_composite_timeseries_all_clim_anom.sh",
    REPO_ROOT / "schedulers" / "schedule_plot_composite_timeseries_split_clim_anom.sh",
    REPO_ROOT
    / "schedulers"
    / "schedule_plot_advection_direction_exploration_clim_anom.sh",
    REPO_ROOT
    / "schedulers"
    / "schedule_plot_advection_direction_exploration_matched_clim_anom.sh",
)
ABSOLUTE_PLOT_SCHEDULERS = (
    REPO_ROOT / "schedulers" / "schedule_composite_timeseries_all.sh",
    REPO_ROOT / "schedulers" / "schedule_composite_timeseries_split.sh",
    REPO_ROOT / "schedulers" / "schedule_top_events.sh",
)


def test_climatology_schedulers_are_commit_verified_and_syntax_valid():
    for scheduler in SCHEDULERS:
        text = scheduler.read_text()
        assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
        assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
        assert "status --porcelain --untracked-files=normal" in text
        assert "mamba activate \"${VENUS_MAMBA_ENV:-dev_env}\"" in text
        subprocess.run(["bash", "-n", str(scheduler)], check=True)


def test_absolute_plot_schedulers_are_commit_verified_and_syntax_valid():
    for scheduler in ABSOLUTE_PLOT_SCHEDULERS:
        text = scheduler.read_text()
        assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
        assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
        assert 'INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"' in text
        assert "status --porcelain --untracked-files=normal" in text
        assert "mamba activate \"${VENUS_MAMBA_ENV:-dev_env}\"" in text
        assert "/home/mhpereir/HW_analysis" not in text
        subprocess.run(["bash", "-n", str(scheduler)], check=True)


def test_absolute_plot_schedulers_require_explicit_non_overwriting_outputs():
    all_text = ABSOLUTE_PLOT_SCHEDULERS[0].read_text()
    split_text = ABSOLUTE_PLOT_SCHEDULERS[1].read_text()
    top_text = ABSOLUTE_PLOT_SCHEDULERS[2].read_text()

    assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in all_text
    assert 'test ! -e "${OUTPUT_PATH}"' in all_text
    assert 'test -s "${SMOOTHED_OUTPUT_PATH}"' in all_text

    assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in split_text
    assert 'for split_variable in "${split_variable_list[@]}" peak_time; do' in split_text
    assert 'test ! -e "${derived_output_path}"' in split_text
    assert split_text.count('test -s "${derived_output_path}"') == 2

    assert 'OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"' in top_text
    assert 'test ! -e "${OUTPUT_DIR}"' in top_text
    assert 'test -d "${OUTPUT_DIR}"' in top_text


def test_climatology_builder_requires_explicit_products():
    text = SCHEDULERS[0].read_text()

    assert 'INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"' in text
    assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in text
    assert "build_stage1_hourly_climatology.py" in text
    assert "refusing to overwrite" in text


def test_anomaly_plot_schedulers_require_stage1_climatology_and_output():
    for scheduler in SCHEDULERS[1:]:
        text = scheduler.read_text()
        assert 'INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"' in text
        assert (
            'CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:?CLIMATOLOGY_PATH is required}"'
            in text
        )
        assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in text
        assert "--input-path \"${INPUT_PATH}\"" in text
        assert "--climatology-path \"${CLIMATOLOGY_PATH}\"" in text


def test_matched_advection_scheduler_requires_stage2_settings_and_atomic_output():
    text = SCHEDULERS[4].read_text()

    assert 'EVENT_FEATURES_PATH="${EVENT_FEATURES_PATH:?EVENT_FEATURES_PATH is required}"' in text
    required_settings = (
        'MATCHING_SETTINGS_PATH="${MATCHING_SETTINGS_PATH:'
        '?MATCHING_SETTINGS_PATH is required}"'
    )
    assert required_settings in text
    assert 'MATCHING_SPECIFICATION="${MATCHING_SPECIFICATION:-peak_anomaly_0p20}"' in text
    assert "plot_advection_direction_exploration_matched_clim_anom.py" in text
    assert '--event-features-path "${EVENT_FEATURES_PATH}"' in text
    assert '--matching-settings-path "${MATCHING_SETTINGS_PATH}"' in text
    assert '--matching-specification "${MATCHING_SPECIFICATION}"' in text
    assert 'export PYTHONWARNINGS=error' in text
    assert 'mv "${STAGED_OUTPUT}" "${OUTPUT_PATH}"' in text
    assert text.count('test ! -e "${OUTPUT_PATH}"') == 2


def test_split_anomaly_scheduler_matches_absolute_production_split_matrix():
    text = SCHEDULERS[2].read_text()
    absolute_text = (
        REPO_ROOT / "schedulers" / "schedule_composite_timeseries_split.sh"
    ).read_text()

    expected_variables = _shell_array_values(
        absolute_text,
        variable_name="split_variable_list",
    )
    assert _shell_array_values(text, variable_name="split_variable_list") == (
        expected_variables
    )
    assert 'SPLIT_QUANTILE="${SPLIT_QUANTILE:-0.75}"' in text
    assert 'SPLIT_YEAR="${SPLIT_YEAR:-1982}"' in text
    assert '--split-variable peak_time' in text
    assert '--split-years "${SPLIT_YEAR}"' in text


def test_split_anomaly_scheduler_accepts_regional_plot_configuration():
    text = SCHEDULERS[2].read_text()

    expected_defaults = {
        "REGION": "pnw_bartusek",
        "BOTTOM_BOUNDARY": "surface",
        "TOP_BOUNDARY": "700",
        "THRESHOLD_VARIABLE": "tas",
        "QUANTILE": "90",
        "TIME_START": "1940",
        "TIME_END": "2024",
        "WINDOW_DAYS": "7",
        "SMOOTHING_WINDOW": "24",
    }
    for variable, default in expected_defaults.items():
        assert f'{variable}="${{{variable}:-{default}}}"' in text

    expected_arguments = {
        "region": "REGION",
        "bottom-boundary": "BOTTOM_BOUNDARY",
        "top-boundary": "TOP_BOUNDARY",
        "threshold-variable": "THRESHOLD_VARIABLE",
        "quantile": "QUANTILE",
        "start-year": "TIME_START",
        "end-year": "TIME_END",
        "window-days": "WINDOW_DAYS",
        "smoothing-window": "SMOOTHING_WINDOW",
    }
    for argument, variable in expected_arguments.items():
        assert text.count(f'--{argument} "${{{variable}}}"') == 2


def test_all_anomaly_scheduler_accepts_regional_plot_configuration():
    text = SCHEDULERS[1].read_text()

    expected_defaults = {
        "REGION": "pnw_bartusek",
        "BOTTOM_BOUNDARY": "surface",
        "TOP_BOUNDARY": "700",
        "THRESHOLD_VARIABLE": "tas",
        "QUANTILE": "90",
        "TIME_START": "1940",
        "TIME_END": "2024",
        "WINDOW_DAYS": "7",
        "SMOOTHING_WINDOW": "24",
    }
    for variable, default in expected_defaults.items():
        assert f'{variable}="${{{variable}:-{default}}}"' in text

    expected_arguments = {
        "region": "REGION",
        "bottom-boundary": "BOTTOM_BOUNDARY",
        "top-boundary": "TOP_BOUNDARY",
        "threshold-variable": "THRESHOLD_VARIABLE",
        "quantile": "QUANTILE",
        "start-year": "TIME_START",
        "end-year": "TIME_END",
        "window-days": "WINDOW_DAYS",
        "smoothing-window": "SMOOTHING_WINDOW",
    }
    for argument, variable in expected_arguments.items():
        assert f'--{argument} "${{{variable}}}"' in text

    assert 'test ! -e "${OUTPUT_PATH}"' in text
    assert 'test ! -e "${SMOOTHED_OUTPUT_PATH}"' in text
    assert 'test -s "${OUTPUT_PATH}"' in text
    assert 'test -s "${SMOOTHED_OUTPUT_PATH}"' in text


def test_split_anomaly_scheduler_preflights_and_validates_all_outputs():
    text = SCHEDULERS[2].read_text()

    assert 'for split_variable in "${split_variable_list[@]}" peak_time; do' in text
    assert 'for split_variable in "${split_variable_list[@]}"; do' in text
    assert text.count('test ! -e "${derived_output_path}"') == 1
    missing_smoothed = (
        'test ! -e "${derived_output_path%.*}_smoothed.${OUTPUT_SUFFIX}"'
    )
    assert text.count(missing_smoothed) == 1
    assert text.count('test -s "${derived_output_path}"') == 2
    written_smoothed = (
        'test -s "${derived_output_path%.*}_smoothed.${OUTPUT_SUFFIX}"'
    )
    assert text.count(written_smoothed) == 2


def _shell_array_values(text: str, *, variable_name: str) -> tuple[str, ...]:
    match = re.search(
        rf"{variable_name}=\(\n(?P<body>.*?)\n\)",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return tuple(re.findall(r'"([^"]+)"', match.group("body")))
