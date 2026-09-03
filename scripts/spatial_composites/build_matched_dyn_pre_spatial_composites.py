"""Build lagged daily spatial composites for matched I_dyn_pre populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.idyn_matching_exploration import matching_settings
from scripts.spatial_composites import (
    build_dyn_net_spatial_composites as spatial_builder,
)
from src import selectors

DEFAULT_EVENT_FEATURES_PATH = spatial_builder.DEFAULT_EVENT_FEATURES_PATH
DEFAULT_DAILY_DIR = spatial_builder.DEFAULT_DAILY_DIR
DEFAULT_CLIMATOLOGY_PATH = spatial_builder.DEFAULT_CLIMATOLOGY_PATH
DEFAULT_SETTINGS_PATH = matching_settings.DEFAULT_SETTINGS_PATH
DEFAULT_MATCHING_SPECIFICATION = "peak_anomaly_0p20"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results/spatial_composites"
    / (
        "matched_dyn_pre_daily_spatial_composites_pnw_bartusek_"
        "tas_q90_1940_2024_peak_anomaly_0p20.nc"
    )
)
EXPECTED_EVENT_FEATURE_STAGE = "stage_2_event_features"
MATCHED_COMPOSITE_STAGE = "daily_matched_idyn_spatial_composites"


@dataclass(frozen=True)
class MatchedSpatialSelection:
    """Matched Stage-2 rows and their selection provenance."""

    events: xr.Dataset
    match: selectors.SignMatchResult
    specification: matching_settings.MatchingSpecification
    family: matching_settings.MatchingFamily


def parse_args() -> argparse.Namespace:
    """Parse matched-selection, spatial-input, and output arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build daily T2m/Z500 composites for matched positive and negative "
            "I_dyn_pre populations."
        )
    )
    parser.add_argument(
        "--event-features-path",
        type=Path,
        default=DEFAULT_EVENT_FEATURES_PATH,
    )
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument(
        "--climatology-path",
        type=Path,
        default=DEFAULT_CLIMATOLOGY_PATH,
    )
    parser.add_argument(
        "--matching-settings-path",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
    )
    parser.add_argument(
        "--matching-specification",
        default=DEFAULT_MATCHING_SPECIFICATION,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--lat-min",
        type=float,
        default=spatial_builder.LATITUDE_BOUNDS[0],
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=spatial_builder.LATITUDE_BOUNDS[1],
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=spatial_builder.LONGITUDE_BOUNDS[0],
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=spatial_builder.LONGITUDE_BOUNDS[1],
    )
    parser.add_argument(
        "--lag-start",
        type=int,
        default=spatial_builder.DAILY_LAGS[0],
    )
    parser.add_argument(
        "--lag-end",
        type=int,
        default=spatial_builder.DAILY_LAGS[-1],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing matched composite product.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject missing inputs, invalid spatial bounds, and unsafe replacement."""
    if args.lat_min >= args.lat_max:
        raise ValueError("--lat-min must be less than --lat-max.")
    if args.lon_min >= args.lon_max:
        raise ValueError("--lon-min must be less than --lon-max.")
    if args.lon_min < -180 or args.lon_max > 180:
        raise ValueError("Longitude bounds must lie within [-180, 180].")
    if args.lag_start > args.lag_end:
        raise ValueError("--lag-start must be less than or equal to --lag-end.")

    for label, path in (
        ("Stage-2 event features", args.event_features_path),
        ("daily spatial data directory", args.daily_dir),
        ("daily climatology", args.climatology_path),
        ("matching settings", args.matching_settings_path),
    ):
        resolved = path.expanduser().resolve()
        if label.endswith("directory"):
            exists = resolved.is_dir()
        else:
            exists = resolved.is_file()
        if not exists:
            raise FileNotFoundError(f"{label} does not exist: {resolved}.")

    output_path = args.output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".nc":
        raise ValueError("--output-path must use the .nc suffix.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Matched composite exists: {output_path}. Pass --overwrite to replace it."
        )


def open_event_features(path: str | Path) -> xr.Dataset:
    """Open and validate the canonical Stage-2 event-feature product."""
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


def prepare_matched_events(
    features: xr.Dataset,
    *,
    settings: matching_settings.MatchingSettings,
    specification_id: str,
) -> MatchedSpatialSelection:
    """Select paired sign populations and retain per-event pair audit fields."""
    specification = settings.specification(specification_id)
    family = settings.family(specification.family)
    match = selectors.match_events_by_metric_sign(
        features,
        settings.group_variable,
        match_variables=family.variables,
        caliper_sd=specification.caliper_sd,
        reference_sign=settings.reference_sign,
    )
    if match.pair_count == 0:
        raise ValueError(
            f"Matching specification {specification_id!r} retained no pairs."
        )

    selected_indices = np.concatenate(
        [match.positive_indices, match.negative_indices]
    )
    pair_ids = np.tile(np.arange(match.pair_count, dtype=np.int64), 2)
    pair_distances = np.tile(match.distances, 2)
    events = spatial_builder.prepare_events(
        features.isel({spatial_builder.EVENT_DIM: selected_indices})
    )
    events = events.assign_coords(
        {spatial_builder.EVENT_DIM: np.arange(selected_indices.size)}
    )
    events["matched_pair_id"] = (spatial_builder.EVENT_DIM, pair_ids)
    events["matched_pair_id"].attrs.update(
        {
            "long_name": "zero-based matched-pair identifier",
            "description": "Each identifier occurs once in each I_dyn_pre sign group.",
        }
    )
    events["matched_pair_distance"] = (
        spatial_builder.EVENT_DIM,
        pair_distances,
    )
    events["matched_pair_distance"].attrs.update(
        {
            "long_name": "root-mean-square standardized pair distance",
            "units": "1",
        }
    )

    counts = events["event_dyn_sign"].to_series().value_counts()
    if int(counts.get("positive", 0)) != match.pair_count:
        raise AssertionError("Positive matched-event count is inconsistent.")
    if int(counts.get("negative", 0)) != match.pair_count:
        raise AssertionError("Negative matched-event count is inconsistent.")

    events.attrs.update(_matching_attrs(settings, specification, family, match))
    events.attrs.update(
        _population_attrs(
            features,
            group_variable=settings.group_variable,
            pair_count=match.pair_count,
        )
    )
    return MatchedSpatialSelection(
        events=events,
        match=match,
        specification=specification,
        family=family,
    )


def build_matched_spatial_composites(
    features: xr.Dataset,
    *,
    settings: matching_settings.MatchingSettings,
    specification_id: str,
    daily_dir: str | Path,
    climatology_path: str | Path,
    lat_bounds: tuple[float, float] = spatial_builder.LATITUDE_BOUNDS,
    lon_bounds: tuple[float, float] = spatial_builder.LONGITUDE_BOUNDS,
    daily_lags: tuple[int, ...] = spatial_builder.DAILY_LAGS,
    event_features_path: str | Path | None = None,
    event_features_sha256: str = "",
) -> tuple[xr.Dataset, MatchedSpatialSelection]:
    """Return a provenance-rich spatial product for one matching specification."""
    selected = prepare_matched_events(
        features,
        settings=settings,
        specification_id=specification_id,
    )
    composite = spatial_builder.build_spatial_composites(
        selected.events,
        daily_dir=daily_dir,
        climatology_path=climatology_path,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        daily_lags=daily_lags,
        event_features_path=event_features_path,
    )
    composite.attrs.update(
        _matching_attrs(
            settings,
            selected.specification,
            selected.family,
            selected.match,
        )
    )
    composite.attrs.update(
        _population_attrs(
            features,
            group_variable=settings.group_variable,
            pair_count=selected.match.pair_count,
        )
    )
    composite.attrs.update(
        {
            "pipeline_stage": MATCHED_COMPOSITE_STAGE,
            "event_features_sha256": event_features_sha256,
        }
    )
    return composite, selected


def _matching_attrs(
    settings: matching_settings.MatchingSettings,
    specification: matching_settings.MatchingSpecification,
    family: matching_settings.MatchingFamily,
    match: selectors.SignMatchResult,
) -> dict[str, str | int | float]:
    """Return NetCDF-safe matching provenance attributes."""
    return {
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
        "matching_settings_schema_version": settings.schema_version,
        "matching_method": match.method,
        "matching_standardization": match.standardization,
        "matching_distance": match.distance_method,
        "matching_caliper_rule": match.caliper_rule,
        "matching_replacement": int(match.replacement),
        "matching_pooled_scales": json.dumps(
            dict(match.pooled_scales),
            sort_keys=True,
        ),
        "matching_calipers_sd": json.dumps(
            dict(match.calipers_sd),
            sort_keys=True,
        ),
    }


def _population_attrs(
    features: xr.Dataset,
    *,
    group_variable: str,
    pair_count: int,
) -> dict[str, int]:
    """Return source-population and unmatched-event audit counts."""
    values = np.asarray(features[group_variable].values, dtype=float)
    negative = int(np.count_nonzero(values < 0))
    positive = int(np.count_nonzero(values > 0))
    return {
        "matching_source_negative_count": negative,
        "matching_source_positive_count": positive,
        "matching_source_zero_count": int(np.count_nonzero(values == 0)),
        "matching_unmatched_negative_count": negative - pair_count,
        "matching_unmatched_positive_count": positive - pair_count,
    }


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 checksum of one input file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Build and write the separate matched spatial-composite product."""
    args = parse_args()
    validate_args(args)
    event_path = args.event_features_path.expanduser().resolve()
    settings = matching_settings.load_matching_settings(
        args.matching_settings_path
    )
    event_features_sha256 = sha256_file(event_path)
    features = open_event_features(event_path)
    try:
        composite, selected = build_matched_spatial_composites(
            features.load(),
            settings=settings,
            specification_id=args.matching_specification,
            daily_dir=args.daily_dir,
            climatology_path=args.climatology_path,
            lat_bounds=(args.lat_min, args.lat_max),
            lon_bounds=(args.lon_min, args.lon_max),
            daily_lags=tuple(range(args.lag_start, args.lag_end + 1)),
            event_features_path=event_path,
            event_features_sha256=event_features_sha256,
        )
    finally:
        features.close()

    written = spatial_builder.write_composite_product(
        composite,
        args.output_path,
    )
    print(f"Wrote matched I_dyn_pre spatial composites: {written}")
    print(f"Matching specification: {selected.specification.identifier}")
    print(f"Matched pairs: {selected.match.pair_count}")
    print(f"Stage-2 SHA-256: {event_features_sha256}")
    print(f"Settings SHA-256: {settings.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
