"""Build lagged daily T2m/Z500 composites split by event net dynamical sign."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402


DEFAULT_EVENT_FEATURES_PATH = (
    REPO_ROOT
    / "results/stage2_event_features"
    / "hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_DAILY_DIR = REPO_ROOT / "results/spatial_composites/daily"
DEFAULT_CLIMATOLOGY_PATH = (
    REPO_ROOT
    / "results/spatial_composites/climatology"
    / "era5_daily_doy_climatology_t2m_z500_global_1940_2024.nc"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results/spatial_composites"
    / "dyn_net_daily_spatial_composites_pnw_bartusek_tas_q90_1940_2024.nc"
)

GROUPS = ("positive", "negative")
GROUP_DIM = "dyn_sign"
LAG_DIM = "lag"
EVENT_DIM = "event"
DAILY_LAGS = tuple(range(-3, 4))
LATITUDE_BOUNDS = (10.0, 80.0)
LONGITUDE_BOUNDS = (-170.0, -40.0)
GEOPOTENTIAL_TO_HEIGHT_M_S2 = config.G_M_S2
REQUIRED_EVENT_VARIABLES = (
    "event_id",
    "peak_time",
    "I_dyn_pre",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build event-relative daily T2m/Z500 composites for positive and "
            "negative event I_dyn_net populations."
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
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--lat-min", type=float, default=LATITUDE_BOUNDS[0])
    parser.add_argument("--lat-max", type=float, default=LATITUDE_BOUNDS[1])
    parser.add_argument("--lon-min", type=float, default=LONGITUDE_BOUNDS[0])
    parser.add_argument("--lon-max", type=float, default=LONGITUDE_BOUNDS[1])
    parser.add_argument(
        "--lag-start",
        type=int,
        default=DAILY_LAGS[0],
        help="First daily lag relative to the event peak date (default: -3).",
    )
    parser.add_argument(
        "--lag-end",
        type=int,
        default=DAILY_LAGS[-1],
        help="Last daily lag relative to the event peak date (default: +3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing composite product.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.lat_min >= args.lat_max:
        raise ValueError("--lat-min must be less than --lat-max.")
    if args.lon_min >= args.lon_max:
        raise ValueError("--lon-min must be less than --lon-max.")
    if args.lon_min < -180 or args.lon_max > 180:
        raise ValueError("Longitude bounds must lie within [-180, 180].")
    if args.lag_start > args.lag_end:
        raise ValueError("--lag-start must be less than or equal to --lag-end.")
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Pass --overwrite to replace it."
        )


def main() -> int:
    args = parse_args()
    validate_args(args)

    event_path = args.event_features_path.expanduser().resolve()
    with xr.open_dataset(
        event_path,
        engine="h5netcdf",
        decode_timedelta=True,
    ) as features:
        events = prepare_events(features.load())

    composite = build_spatial_composites(
        events,
        daily_dir=args.daily_dir,
        climatology_path=args.climatology_path,
        lat_bounds=(args.lat_min, args.lat_max),
        lon_bounds=(args.lon_min, args.lon_max),
        daily_lags=tuple(range(args.lag_start, args.lag_end + 1)),
        event_features_path=event_path,
    )
    written = write_composite_product(composite, args.output_path)
    print(f"Wrote daily dynamical-sign spatial composites: {written}")
    return 0


def prepare_events(features: xr.Dataset) -> xr.Dataset:
    """Validate the Stage-2 table and return finite nonzero sign-labelled events."""
    missing = sorted(name for name in REQUIRED_EVENT_VARIABLES if name not in features)
    if missing:
        raise ValueError(
            "Event-feature table is missing required variables: " + ", ".join(missing)
        )
    if EVENT_DIM not in features.dims:
        raise ValueError(f"Event-feature table is missing dimension {EVENT_DIM!r}.")

    peak_values = np.asarray(features["peak_time"].values, dtype="datetime64[ns]")
    dyn_net = np.asarray(features["I_dyn_pre"].values, dtype=float)
    finite = (~np.isnat(peak_values)) & np.isfinite(dyn_net)
    if not finite.all():
        bad = int((~finite).sum())
        raise ValueError(f"Event-feature table contains {bad} non-finite required rows.")

    nonzero = dyn_net != 0
    keep = np.flatnonzero(nonzero)
    if keep.size == 0:
        raise ValueError("No nonzero I_dyn_net events are available.")
    selected = features.isel({EVENT_DIM: keep}).copy()
    selected["I_dyn_net"] = (EVENT_DIM, dyn_net[keep])
    selected["I_dyn_net"].attrs.update(
        {
            "description": (
                "Stage-2 I_dyn_pre retained under the spatial-product audit name."
            ),
            "source_variable": "I_dyn_pre",
        }
    )
    signs = np.where(dyn_net[keep] > 0, "positive", "negative")
    selected["event_dyn_sign"] = (EVENT_DIM, signs)
    selected.attrs = dict(features.attrs)
    selected.attrs["zero_dyn_net_events_excluded"] = int((~nonzero).sum())

    counts = {group: int(np.count_nonzero(signs == group)) for group in GROUPS}
    empty = [group for group, count in counts.items() if count == 0]
    if empty:
        raise ValueError("No events are available for group(s): " + ", ".join(empty))
    return selected


def build_spatial_composites(
    events: xr.Dataset,
    *,
    daily_dir: str | Path,
    climatology_path: str | Path,
    lat_bounds: tuple[float, float] = LATITUDE_BOUNDS,
    lon_bounds: tuple[float, float] = LONGITUDE_BOUNDS,
    daily_lags: tuple[int, ...] = DAILY_LAGS,
    event_features_path: str | Path | None = None,
) -> xr.Dataset:
    """Return equal-event-weight composites for each sign group and daily lag."""
    if not daily_lags:
        raise ValueError("At least one daily lag is required.")
    if len(set(daily_lags)) != len(daily_lags):
        raise ValueError("Daily lags must be unique.")
    if tuple(sorted(daily_lags)) != daily_lags:
        raise ValueError("Daily lags must be strictly increasing.")

    event_dates, event_signs = event_window_dates(events, daily_lags=daily_lags)
    group_counts = np.array(
        [np.count_nonzero(event_signs == group) for group in GROUPS],
        dtype=np.int64,
    )
    raw_weights = timestamp_weights(event_dates, event_signs, group_counts)
    climatology_weights = calendar_key_weights(
        event_dates,
        event_signs,
        group_counts,
    )

    raw_fields, raw_lat, raw_lon = reduce_annual_daily_fields(
        raw_weights,
        daily_dir=daily_dir,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )
    climate_fields, climate_lat, climate_lon = reduce_climatology_fields(
        climatology_weights,
        climatology_path=climatology_path,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )
    require_same_grid(raw_lat, raw_lon, climate_lat, climate_lon)

    coords = {
        GROUP_DIM: np.asarray(GROUPS, dtype=str),
        LAG_DIM: np.asarray(daily_lags, dtype=np.int64),
        "latitude": raw_lat,
        "longitude": raw_lon,
    }
    out = xr.Dataset(coords=coords)
    out[LAG_DIM].attrs.update(
        {
            "long_name": "day relative to heatwave peak date",
            "description": "integer day offset; 0 is the event peak date",
        }
    )
    for variable, units, long_name in (
        ("t2m", "K", "2 metre temperature"),
        ("z500", "m", "500 hPa geopotential height"),
    ):
        event_name = f"{variable}_event_mean"
        climatology_name = f"{variable}_climatology_mean"
        anomaly_name = f"{variable}_anomaly"
        out[event_name] = (
            (GROUP_DIM, LAG_DIM, "latitude", "longitude"),
            raw_fields[variable],
        )
        out[climatology_name] = (
            (GROUP_DIM, LAG_DIM, "latitude", "longitude"),
            climate_fields[variable],
        )
        out[anomaly_name] = out[event_name] - out[climatology_name]
        for name in (event_name, climatology_name, anomaly_name):
            out[name].attrs["units"] = units
        out[event_name].attrs["long_name"] = f"event composite {long_name}"
        out[climatology_name].attrs["long_name"] = (
            f"event-date- and lag-matched climatological {long_name}"
        )
        out[anomaly_name].attrs["long_name"] = (
            f"lagged event composite {long_name} anomaly"
        )

    out["event_count"] = (GROUP_DIM, group_counts)
    out["I_dyn_pre_mean"] = (
        GROUP_DIM,
        np.array(
            [
                float(
                    events["I_dyn_pre"]
                    .where(events["event_dyn_sign"] == group)
                    .mean()
                )
                for group in GROUPS
            ]
        ),
    )
    out["I_dyn_pre_mean"].attrs.update(
        {
            "units": "K",
            "long_name": "group mean integrated pre-peak dynamical contribution",
        }
    )
    out["I_dyn_net_mean"] = out["I_dyn_pre_mean"].copy(deep=False)
    out["I_dyn_net_mean"].attrs.update(
        {
            "units": "K",
            "long_name": (
                "compatibility alias for group mean integrated pre-peak "
                "dynamical contribution"
            ),
            "source_variable": "I_dyn_pre_mean",
        }
    )
    copy_event_audit_variables(out, events)
    out.attrs.update(
        {
            "pipeline_stage": "daily_dyn_net_spatial_composites",
            "event_features_path": (
                str(Path(event_features_path).expanduser().resolve())
                if event_features_path is not None
                else ""
            ),
            "daily_data_dir": str(Path(daily_dir).expanduser().resolve()),
            "climatology_path": str(Path(climatology_path).expanduser().resolve()),
            "daily_lags": ",".join(str(lag) for lag in daily_lags),
            "daily_window_description": (
                "individual complete UTC days relative to peak date"
            ),
            "event_weighting": (
                "equal event weight within each dynamical-sign and daily-lag composite"
            ),
            "climatology_matching": "calendar month and day",
            "latitude_bounds": f"{lat_bounds[0]},{lat_bounds[1]}",
            "longitude_bounds": f"{lon_bounds[0]},{lon_bounds[1]}",
            "geopotential_height_conversion": f"z / {GEOPOTENTIAL_TO_HEIGHT_M_S2} m s-2",
            "zero_dyn_net_events_excluded": int(
                events.attrs.get("zero_dyn_net_events_excluded", 0)
            ),
        }
    )
    return out


def event_window_dates(
    events: xr.Dataset,
    *,
    daily_lags: tuple[int, ...] = DAILY_LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    peaks = pd.DatetimeIndex(events["peak_time"].values).normalize()
    signs = np.asarray(events["event_dyn_sign"].values, dtype=str)
    dates = np.empty((len(peaks), len(daily_lags)), dtype="datetime64[ns]")
    for index, lag in enumerate(daily_lags):
        dates[:, index] = (peaks + pd.to_timedelta(lag, unit="D")).values
    return dates, signs


def timestamp_weights(
    event_dates: np.ndarray,
    event_signs: np.ndarray,
    group_counts: np.ndarray,
) -> dict[pd.Timestamp, np.ndarray]:
    """Return normalized group/lag weights keyed by actual calendar timestamp."""
    lag_count = event_dates.shape[1]
    weights: defaultdict[pd.Timestamp, np.ndarray] = defaultdict(
        lambda: np.zeros((len(GROUPS), lag_count), dtype=float)
    )
    for row, sign in zip(event_dates, event_signs, strict=True):
        group_index = GROUPS.index(str(sign))
        sample_weight = 1.0 / group_counts[group_index]
        for lag_index, value in enumerate(row):
            weights[pd.Timestamp(value).normalize()][
                group_index, lag_index
            ] += sample_weight
    assert_normalized_weights(weights)
    return dict(weights)


def calendar_key_weights(
    event_dates: np.ndarray,
    event_signs: np.ndarray,
    group_counts: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Return normalized group/lag weights keyed by climatological month/day."""
    lag_count = event_dates.shape[1]
    weights: defaultdict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros((len(GROUPS), lag_count), dtype=float)
    )
    for row, sign in zip(event_dates, event_signs, strict=True):
        group_index = GROUPS.index(str(sign))
        sample_weight = 1.0 / group_counts[group_index]
        for lag_index, value in enumerate(row):
            timestamp = pd.Timestamp(value)
            weights[(timestamp.month, timestamp.day)][
                group_index, lag_index
            ] += sample_weight
    assert_normalized_weights(weights)
    return dict(weights)


def assert_normalized_weights(weights: Mapping[object, np.ndarray]) -> None:
    total = np.sum(np.stack(list(weights.values())), axis=0)
    np.testing.assert_allclose(total, np.ones_like(total), atol=1e-12)


def reduce_annual_daily_fields(
    weights: Mapping[pd.Timestamp, np.ndarray],
    *,
    daily_dir: str | Path,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Stream required dates from annual daily files and apply group weights."""
    by_year: defaultdict[int, dict[pd.Timestamp, np.ndarray]] = defaultdict(dict)
    for date, value in weights.items():
        by_year[date.year][date] = value

    accumulators: dict[str, np.ndarray] | None = None
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    root = Path(daily_dir).expanduser().resolve()
    for year in sorted(by_year):
        path = root / f"ERA5_daily_t2m_z500_{year}.nc"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annual daily input: {path}")
        with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as source:
            ds = standardize_spatial_dataset(
                source,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
            )
            fields = load_weighted_fields(ds, by_year[year], context=str(path))
            lat = np.asarray(ds["latitude"].values, dtype=float)
            lon = np.asarray(ds["longitude"].values, dtype=float)
            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
                accumulators = {
                    name: np.zeros_like(values, dtype=np.float64)
                    for name, values in fields.items()
                }
            else:
                require_same_grid(reference_lat, reference_lon, lat, lon)
            assert accumulators is not None
            for name, values in fields.items():
                accumulators[name] += values

    if accumulators is None or reference_lat is None or reference_lon is None:
        raise ValueError("No annual daily fields were selected.")
    return accumulators, reference_lat, reference_lon


def reduce_climatology_fields(
    weights: Mapping[tuple[int, int], np.ndarray],
    *,
    climatology_path: str | Path,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load and weight required calendar days from the daily climatology."""
    path = Path(climatology_path).expanduser().resolve()
    with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as source:
        ds = standardize_spatial_dataset(
            source,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
        )
        time = pd.DatetimeIndex(ds["time"].values)
        keys = [(value.month, value.day) for value in time]
        if len(keys) != 366 or len(set(keys)) != 366:
            raise ValueError("Daily climatology must contain 366 unique calendar days.")
        key_to_index = {key: index for index, key in enumerate(keys)}
        missing = sorted(set(weights).difference(key_to_index))
        if missing:
            raise ValueError(f"Daily climatology is missing calendar days: {missing}")
        timestamp_mapping = {
            pd.Timestamp(time[key_to_index[key]]).normalize(): value
            for key, value in weights.items()
        }
        fields = load_weighted_fields(ds, timestamp_mapping, context=str(path))
        return (
            fields,
            np.asarray(ds["latitude"].values, dtype=float),
            np.asarray(ds["longitude"].values, dtype=float),
        )


def standardize_spatial_dataset(
    ds: xr.Dataset,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.Dataset:
    """Normalize coordinate names, validate variables, and crop before loading."""
    rename: dict[str, str] = {}
    for canonical, candidates in (
        ("time", ("time", "valid_time")),
        ("latitude", ("latitude", "lat")),
        ("longitude", ("longitude", "lon")),
    ):
        found = next((name for name in candidates if name in ds.coords), None)
        if found is None:
            raise ValueError(f"Spatial dataset is missing a {canonical} coordinate.")
        if found != canonical:
            rename[found] = canonical
    out = ds.rename(rename)
    missing = sorted(name for name in ("t2m", "z") if name not in out)
    if missing:
        raise ValueError("Spatial dataset is missing variables: " + ", ".join(missing))

    z = out["z"]
    pressure_dims = [dim for dim in z.dims if "pressure" in dim.lower()]
    if pressure_dims:
        if len(pressure_dims) != 1:
            raise ValueError("Z has ambiguous pressure dimensions.")
        pressure_dim = pressure_dims[0]
        levels = np.asarray(out[pressure_dim].values, dtype=float)
        if levels.size != 1 or not np.isclose(levels[0], 500.0):
            raise ValueError(f"Expected only Z500; found pressure levels {levels.tolist()}.")
        out = out.isel({pressure_dim: 0}, drop=True)
    elif "pressure_level" in out.coords:
        levels = np.asarray(out["pressure_level"].values, dtype=float)
        if levels.size != 1 or not np.isclose(levels[0], 500.0):
            raise ValueError(f"Expected only Z500; found pressure levels {levels.tolist()}.")

    latitude = np.asarray(out["latitude"].values, dtype=float)
    longitude = np.asarray(out["longitude"].values, dtype=float)
    normalized_lon = ((longitude + 180.0) % 360.0) - 180.0
    lat_indices = np.flatnonzero(
        (latitude >= lat_bounds[0]) & (latitude <= lat_bounds[1])
    )
    lon_indices = np.flatnonzero(
        (normalized_lon >= lon_bounds[0]) & (normalized_lon <= lon_bounds[1])
    )
    if lat_indices.size == 0 or lon_indices.size == 0:
        raise ValueError("Requested latitude/longitude bounds select no grid cells.")
    out = out.isel(latitude=lat_indices, longitude=lon_indices)
    out = out.assign_coords(
        longitude=("longitude", normalized_lon[lon_indices]),
    ).sortby(["latitude", "longitude"])
    return out[["t2m", "z"]]


def load_weighted_fields(
    ds: xr.Dataset,
    weights: Mapping[pd.Timestamp, np.ndarray],
    *,
    context: str,
) -> dict[str, np.ndarray]:
    """Load selected fields and reduce them with group-by-lag weights."""
    time = pd.DatetimeIndex(ds["time"].values).normalize()
    if time.has_duplicates:
        raise ValueError(f"{context} has duplicate daily timestamps.")
    time_to_index = {value: index for index, value in enumerate(time)}
    requested = sorted(weights)
    missing = [value for value in requested if value not in time_to_index]
    if missing:
        raise ValueError(f"{context} is missing required daily timestamps: {missing}")
    indices = [time_to_index[value] for value in requested]
    weight_matrix = np.stack([weights[value] for value in requested], axis=-1)

    t2m = np.asarray(ds["t2m"].isel(time=indices).load().values, dtype=np.float64)
    z = np.asarray(ds["z"].isel(time=indices).load().values, dtype=np.float64)
    if t2m.ndim != 3 or z.ndim != 3:
        raise ValueError(f"{context} fields must have time/latitude/longitude dimensions.")
    if not np.isfinite(t2m).all() or not np.isfinite(z).all():
        raise ValueError(f"{context} contains non-finite selected spatial fields.")
    return {
        "t2m": np.tensordot(weight_matrix, t2m, axes=(-1, 0)),
        "z500": np.tensordot(weight_matrix, z, axes=(-1, 0))
        / GEOPOTENTIAL_TO_HEIGHT_M_S2,
    }


def require_same_grid(
    reference_lat: np.ndarray,
    reference_lon: np.ndarray,
    candidate_lat: np.ndarray,
    candidate_lon: np.ndarray,
) -> None:
    if not np.array_equal(reference_lat, candidate_lat) or not np.array_equal(
        reference_lon,
        candidate_lon,
    ):
        raise ValueError("Daily data and climatology grids do not match exactly.")


def copy_event_audit_variables(out: xr.Dataset, events: xr.Dataset) -> None:
    out.coords[EVENT_DIM] = events[EVENT_DIM]
    required = (
        "event_id",
        "peak_time",
        "I_dyn_pre",
        "I_dyn_net",
        "event_dyn_sign",
    )
    optional = ("matched_pair_id", "matched_pair_distance")
    audit_names = required + tuple(name for name in optional if name in events)
    for name in audit_names:
        out[name] = events[name]
        out[name].attrs = dict(events[name].attrs)


def write_composite_product(ds: xr.Dataset, path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding: dict[str, dict[str, object]] = {}
    for name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.floating):
            encoding[name] = {"dtype": "float32", "zlib": True, "complevel": 4}
    ds.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
