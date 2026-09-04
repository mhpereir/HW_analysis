"""Plot matched-sign face-advection climatological-anomaly composites."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.idyn_matching_exploration import matching_settings
from src import (
    advection_direction,
    advection_direction_plotting,
    analysis_io,
    climatology,
    composites,
    plot_paths,
    selectors,
)

PLOT_NAME = "advection_direction_exploration_matched_clim_anom"
DEFAULT_OUTPUT_FILENAME = "advection_face_contributions_matched_clim_anom.png"
EXPECTED_EVENT_FEATURE_STAGE = "stage_2_event_features"


@dataclass(frozen=True)
class MatchedAdvectionComposites:
    """Matched membership and the two prepared sign composites."""

    negative: xr.Dataset
    positive: xr.Dataset
    match: selectors.SignMatchResult
    specification: matching_settings.MatchingSpecification


def parse_args() -> argparse.Namespace:
    """Parse product, matching, and output arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot matched positive/negative I_dyn_pre face-advection "
            "climatological-anomaly composites."
        )
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument("--climatology-path", type=Path, default=None)
    parser.add_argument(
        "--event-features-path",
        type=Path,
        required=True,
        help="Canonical Stage-2 event-feature NetCDF product.",
    )
    parser.add_argument(
        "--matching-settings-path",
        type=Path,
        default=matching_settings.DEFAULT_SETTINGS_PATH,
        help="Tracked matching settings JSON.",
    )
    parser.add_argument(
        "--matching-specification",
        default=None,
        help="Named settings specification; defaults to plots.primary_specification.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=advection_direction_plotting.DEFAULT_SMOOTHING_WINDOW,
    )
    args = plot_paths.finalize_stage1_plot_paths(
        parser.parse_args(),
        parser,
        plot_name=PLOT_NAME,
        default_output_filename=DEFAULT_OUTPUT_FILENAME,
    )
    if args.climatology_path is None:
        args.climatology_path = analysis_io.default_regional_hourly_climatology_path(
            region=args.region,
            bottom_boundary=args.bottom_boundary,
            top_boundary=args.top_boundary,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    args.climatology_path = args.climatology_path.expanduser().resolve()
    args.event_features_path = args.event_features_path.expanduser().resolve()
    args.matching_settings_path = args.matching_settings_path.expanduser().resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid windows, missing inputs, and output replacement."""
    if args.window_days < 1:
        raise ValueError("--window-days must be >= 1.")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be >= 1.")
    for label, path in (
        ("Stage-1 input", args.input_path),
        ("climatology", args.climatology_path),
        ("Stage-2 event features", args.event_features_path),
        ("matching settings", args.matching_settings_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}.")
    if args.output_path.suffix.lower() != ".png":
        raise ValueError("--output-path must use the .png suffix.")
    smoothed_path = advection_direction_plotting.smoothed_output_path(args.output_path)
    existing = [path for path in (args.output_path, smoothed_path) if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Matched anomaly plot output already exists: {paths}.")


def open_event_features(path: str | Path) -> xr.Dataset:
    """Open and validate the canonical Stage-2 event-feature table."""
    input_path = Path(path).expanduser().resolve()
    ds = xr.open_dataset(
        input_path,
        engine="h5netcdf",
        decode_timedelta=True,
    )
    if ds.attrs.get("pipeline_stage") != EXPECTED_EVENT_FEATURE_STAGE:
        actual = ds.attrs.get("pipeline_stage")
        ds.close()
        raise ValueError(
            "Expected Stage-2 event features with "
            f"pipeline_stage={EXPECTED_EVENT_FEATURE_STAGE!r}; got {actual!r}."
        )
    return ds


def build_matched_composites(
    stage1: xr.Dataset,
    anomaly_source: xr.Dataset,
    event_features: xr.Dataset,
    *,
    settings: matching_settings.MatchingSettings,
    specification_id: str | None,
    window_days: int,
) -> MatchedAdvectionComposites:
    """Match Stage-2 rows, map IDs to Stage 1, and build sign composites."""
    selected_specification = specification_id or settings.primary_specification
    specification = settings.specification(selected_specification)
    family = settings.family(specification.family)
    match = selectors.match_events_by_metric_sign(
        event_features,
        settings.group_variable,
        match_variables=family.variables,
        caliper_sd=specification.caliper_sd,
        reference_sign=settings.reference_sign,
    )
    if match.pair_count == 0:
        raise ValueError(
            f"Matching specification {selected_specification!r} retained no pairs."
        )

    negative_features = event_features.isel(event=match.negative_indices)
    positive_features = event_features.isel(event=match.positive_indices)
    negative_events = selectors.select_events_by_id(
        stage1,
        match.negative_event_ids,
    )
    positive_events = selectors.select_events_by_id(
        stage1,
        match.positive_event_ids,
    )
    _validate_event_alignment(
        negative_events,
        negative_features,
        sign="negative",
    )
    _validate_event_alignment(
        positive_events,
        positive_features,
        sign="positive",
    )

    variables = (
        "advection",
        *(
            advection_direction.stage1_face_variable(face)
            for face in advection_direction.available_stage1_faces(anomaly_source)
        ),
    )
    common = {
        "variables": variables,
        "pre_days": window_days,
        "post_days": window_days,
        "event_percentiles": None,
    }
    negative = composites.all_event_peak_aligned_composite(
        anomaly_source,
        event_table=negative_events,
        **common,
    )
    positive = composites.all_event_peak_aligned_composite(
        anomaly_source,
        event_table=positive_events,
        **common,
    )
    matching_attrs = {
        "matching_group_variable": settings.group_variable,
        "matching_specification": specification.identifier,
        "matching_family": family.identifier,
        "matching_label": family.label,
        "matching_variables": ",".join(family.variables),
        "matching_caliper_sd": float(specification.caliper_sd),
        "matching_reference_sign": settings.reference_sign,
        "matching_pair_count": match.pair_count,
        "matching_settings_path": str(settings.source_path),
        "matching_settings_sha256": settings.sha256,
    }
    for sign, composite in (("negative", negative), ("positive", positive)):
        composite.attrs.update(anomaly_source.attrs)
        composite.attrs.update(matching_attrs)
        composite.attrs["matched_sign"] = sign
        composite.attrs["n_events"] = match.pair_count

    return MatchedAdvectionComposites(
        negative=negative,
        positive=positive,
        match=match,
        specification=specification,
    )


def _validate_event_alignment(
    stage1_events: xr.Dataset,
    stage2_events: xr.Dataset,
    *,
    sign: str,
) -> None:
    """Require exact IDs and peak times across selected Stage-1 and Stage-2 rows."""
    required = ("event_id", "peak_time")
    for label, table in (("Stage 1", stage1_events), ("Stage 2", stage2_events)):
        missing = [name for name in required if name not in table]
        if missing:
            raise ValueError(
                f"{label} {sign} event table is missing: {', '.join(missing)}."
            )
    stage1_ids = np.asarray(stage1_events["event_id"].values, dtype=np.int64)
    stage2_ids = np.asarray(stage2_events["event_id"].values, dtype=np.int64)
    if not np.array_equal(stage1_ids, stage2_ids):
        raise ValueError(f"Stage-1 and Stage-2 {sign} event IDs do not align.")
    stage1_peaks = np.asarray(stage1_events["peak_time"].values, dtype="datetime64[ns]")
    stage2_peaks = np.asarray(stage2_events["peak_time"].values, dtype="datetime64[ns]")
    if np.isnat(stage1_peaks).any() or np.isnat(stage2_peaks).any():
        raise ValueError(f"Matched {sign} events contain missing peak times.")
    if not np.array_equal(stage1_peaks, stage2_peaks):
        raise ValueError(f"Stage-1 and Stage-2 {sign} peak times do not align.")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 checksum for one input file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Build and write raw and smoothed matched-sign advection figures."""
    args = parse_args()
    validate_args(args)
    settings = matching_settings.load_matching_settings(args.matching_settings_path)
    stage1 = analysis_io.open_harmonized_timeseries(args.input_path)
    climate = analysis_io.open_regional_hourly_climatology(args.climatology_path)
    event_features = open_event_features(args.event_features_path)
    try:
        event_features_sha256 = sha256_file(args.event_features_path)
        variables = (
            "advection",
            *(
                advection_direction.stage1_face_variable(face)
                for face in advection_direction.available_stage1_faces(stage1)
            ),
        )
        anomaly_source = climatology.apply_regional_hourly_climatology(
            stage1,
            climate,
            variables=variables,
        )
        prepared = build_matched_composites(
            stage1,
            anomaly_source,
            event_features,
            settings=settings,
            specification_id=args.matching_specification,
            window_days=args.window_days,
        )
        for composite in (prepared.negative, prepared.positive):
            composite.attrs["climatology_path"] = str(args.climatology_path)
            composite.attrs["event_features_path"] = str(args.event_features_path)
            composite.attrs["event_features_sha256"] = event_features_sha256
        written = advection_direction_plotting.write_matched_advection_direction_exploration_outputs(
            prepared.negative,
            prepared.positive,
            args.output_path,
            smoothed_output_path=advection_direction_plotting.smoothed_output_path(
                args.output_path
            ),
            smoothing_window=args.smoothing_window,
        )
    finally:
        event_features.close()
        climate.close()
        stage1.close()

    print("Wrote matched advection-direction anomaly plots:")
    for path in written:
        print(f"  {path}")
    print(f"Matching specification: {prepared.specification.identifier}")
    print(f"Matched pairs: {prepared.match.pair_count}")
    print(f"Stage-2 SHA-256: {event_features_sha256}")
    print(f"Settings SHA-256: {settings.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
