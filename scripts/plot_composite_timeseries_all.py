"""Plot peak-aligned composites from the saved Stage-1 harmonized dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io, composites, plot_paths, plotting, selectors

PLOT_NAME = "composite_timeseries_all"
DEFAULT_OUTPUT_FILENAME = "hw_all_events_composite.png"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / f"plots_{PLOT_NAME}" / DEFAULT_OUTPUT_FILENAME
)
PRESENTATION_PLOT_NAME = f"{PLOT_NAME}_presentation"
PRESENTATION_DEFAULT_OUTPUT_FILENAME = "hw_all_events_composite_presentation.png"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_SMOOTHING_WINDOW = 24
COMPOSITE_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
)
EXTENDED_COMPOSITE_VARIABLES: tuple[str, ...] = COMPOSITE_VARIABLES + (
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
)
PRESENTATION_COMPOSITE_VARIABLES = plotting.PRESENTATION_PLOT_VARIABLES
SMOOTHED_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
)
EXTENDED_SMOOTHED_VARIABLES: tuple[str, ...] = SMOOTHED_VARIABLES + (
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
)
PRESENTATION_SMOOTHED_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for composite time-series plotting."""
    parser = argparse.ArgumentParser(
        description="Plot peak-aligned composite time series for all HW events."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path where the composite PNG will be written.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Number of days to include on each side of event peak time.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Hourly running-mean window for the smoothed composite figure.",
    )
    parser.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=None,
        metavar="MONTH",
        help="Optional calendar months to retain before compositing, e.g. 6 7 8.",
    )
    parser.add_argument(
        "--require-full-event",
        action="store_true",
        help="Require the full event interval to fall within --season-months.",
    )
    parser.add_argument(
        "--plot-extended-variables",
        action="store_true",
        help="Plot optional extended diagnostics when present in the input dataset.",
    )
    parser.add_argument(
        "--layout",
        choices=plotting.COMPOSITE_LAYOUTS,
        default=plotting.PAPER_COMPOSITE_LAYOUT,
        help="Figure layout. Presentation uses a widescreen six-panel grid.",
    )
    args = parser.parse_args()
    plot_name, default_output_filename = _default_plot_destination(args.layout)
    return plot_paths.finalize_stage1_plot_paths(
        args,
        parser,
        plot_name=plot_name,
        default_output_filename=default_output_filename,
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate composite plotting CLI arguments."""
    if args.window_days < 0:
        raise ValueError("--window-days must be >= 0.")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be >= 1.")
    if args.require_full_event and args.season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if getattr(
        args, "layout", plotting.PAPER_COMPOSITE_LAYOUT
    ) == plotting.PRESENTATION_COMPOSITE_LAYOUT and getattr(
        args, "plot_extended_variables", False
    ):
        raise ValueError(
            "--layout presentation cannot be combined with --plot-extended-variables."
        )
    if args.season_months is not None:
        _validate_season_months(args.season_months)


def main() -> int:
    """Open the harmonized dataset and write the all-event composite figure."""
    args = parse_args()
    validate_args(args)

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        variables = _composite_variables(args.plot_extended_variables, args.layout)
        smoothed_variables = _smoothed_variables(
            args.plot_extended_variables,
            args.layout,
        )
        composite_kwargs = {
            "variables": variables,
            "pre_days": args.window_days,
            "post_days": args.window_days,
            "event_percentiles": (0.25, 0.5, 0.75),
        }
        if args.season_months is not None:
            event_table = selectors.select_events_by_season(
                ds,
                args.season_months,
                require_full_event=args.require_full_event,
            )
            if event_table.sizes.get("event", 0) == 0:
                months = " ".join(str(month) for month in args.season_months)
                raise ValueError(
                    f"No events remain after filtering to season months: {months}."
                )
            composite_kwargs["event_table"] = event_table

        composite = composites.all_event_peak_aligned_composite(
            ds,
            **composite_kwargs,
        )
        written = plotting.write_composite_timeseries_outputs(
            composite,
            args.output_path,
            smoothed_output_path=_smoothed_output_path(args.output_path),
            smoothing_window=args.smoothing_window,
            smoothed_variables=smoothed_variables,
            plot_extended_variables=args.plot_extended_variables,
            layout=args.layout,
        )
        print("Wrote HW all-event composite figures:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        ds.close()
    return 0


def _validate_season_months(months: list[int]) -> None:
    """Validate CLI season-month values."""
    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        values = ", ".join(str(month) for month in invalid)
        raise ValueError(
            f"--season-months values must be between 1 and 12; got {values}."
        )


def _composite_variables(
    plot_extended_variables: bool,
    layout: str = plotting.PAPER_COMPOSITE_LAYOUT,
) -> tuple[str, ...]:
    """Return variables required for the selected composite plot layout."""
    if layout == plotting.PRESENTATION_COMPOSITE_LAYOUT:
        return PRESENTATION_COMPOSITE_VARIABLES
    if plot_extended_variables:
        return EXTENDED_COMPOSITE_VARIABLES
    return COMPOSITE_VARIABLES


def _smoothed_variables(
    plot_extended_variables: bool,
    layout: str = plotting.PAPER_COMPOSITE_LAYOUT,
) -> tuple[str, ...]:
    """Return display-smoothed variables for the selected plot layout."""
    if layout == plotting.PRESENTATION_COMPOSITE_LAYOUT:
        return PRESENTATION_SMOOTHED_VARIABLES
    if plot_extended_variables:
        return EXTENDED_SMOOTHED_VARIABLES
    return SMOOTHED_VARIABLES


def _default_plot_destination(layout: str) -> tuple[str, str]:
    """Return the output namespace and filename for one figure layout."""
    if layout == plotting.PRESENTATION_COMPOSITE_LAYOUT:
        return PRESENTATION_PLOT_NAME, PRESENTATION_DEFAULT_OUTPUT_FILENAME
    return PLOT_NAME, DEFAULT_OUTPUT_FILENAME


def _smoothed_output_path(output_path: Path) -> Path:
    """Return sibling path for the smoothed composite figure."""
    if output_path.name == DEFAULT_OUTPUT_PATH.name:
        return output_path.with_name("hw_all_events_composite_smoothed.png")
    return output_path.with_name(f"{output_path.stem}_smoothed{output_path.suffix}")


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
