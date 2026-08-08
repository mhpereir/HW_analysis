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
)


def test_climatology_schedulers_are_commit_verified_and_syntax_valid():
    for scheduler in SCHEDULERS:
        text = scheduler.read_text()
        assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
        assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
        assert "status --porcelain --untracked-files=normal" in text
        assert "mamba activate \"${VENUS_MAMBA_ENV:-dev_env}\"" in text
        subprocess.run(["bash", "-n", str(scheduler)], check=True)


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
