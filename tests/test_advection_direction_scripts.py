from argparse import Namespace
from pathlib import Path

from HW_analysis.scripts import (
    build_stage1_advection_exploration as build_script,
    plot_advection_direction_exploration as plot_script,
)


def test_builder_defaults_to_isolated_stage1_subfolder():
    args = build_script.finalize_args(_base_args())

    assert args.input_path.parent.name == "stage1"
    assert args.output_path.parent.name == "advection_direction_exploration"
    assert args.output_path.name == args.input_path.name
    assert args.output_path != args.input_path


def test_builder_rejects_output_equal_to_input():
    args = _base_args()
    shared = Path("/tmp/shared-stage1.nc")
    args.input_path = shared
    args.output_path = shared

    try:
        build_script.finalize_args(args)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("Expected equal input and output paths to fail.")


def test_plot_defaults_to_enhanced_stage1_and_separate_plot_tree():
    args = _base_args()
    args.window_days = 7
    args.ratio_epsilon = 1e-4
    args.season_months = [6, 7, 8]
    args.require_full_event = True

    out = plot_script.finalize_args(args)

    assert out.input_path.parent.name == "advection_direction_exploration"
    assert "plots_advection_direction_exploration" in out.output_path.parts
    assert out.output_path.name == "advection_face_contributions.png"


def _base_args() -> Namespace:
    return Namespace(
        region="pnw_bartusek",
        bottom_boundary="surface",
        top_boundary="700",
        threshold_variable="tas",
        quantile="90",
        start_year=1940,
        end_year=2024,
        input_path=None,
        output_path=None,
        start_year_ehb=1940,
        end_year_ehb=2025,
        heat_budget_root=None,
        overwrite=False,
    )
