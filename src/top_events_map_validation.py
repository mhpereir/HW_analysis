"""Independent checks against daily source fields for top-event map acceptance."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config

ABSOLUTE_TOLERANCE = 1e-10


def validate_against_sources(product: xr.Dataset) -> dict:
    """Independently rank events and directly sum the three daily source fields.

    Deliberately do not use the production selector, field loader, or reduction.
    This complements, rather than replaces, the saved-product contract check.
    """
    records = json.loads(product.attrs["source_files"])
    _check_source_identities(records)
    event_path = Path(product.attrs["event_features_path"])
    with event_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != product.attrs["event_features_sha256"]:
        raise ValueError("Source event-table checksum no longer matches the product.")
    with xr.open_dataset(
        event_path, engine="h5netcdf", decode_timedelta=True
    ) as events:
        metric = events[str(product.attrs["rank_metric"])].values
        peaks = pd.DatetimeIndex(events.peak_time.values)
        year_filter = str(product.attrs["peak_year_filter"])
        eligible = [
            index
            for index, value in enumerate(metric)
            if np.isfinite(value)
            and (year_filter == "all" or peaks[index].year == int(year_filter))
        ]
        # The established selector puts later source rows first when metrics tie.
        ranked = sorted(eligible, key=lambda index: (-metric[index], -index))
        ranked = ranked[: int(product.attrs["requested_top_n"])]
        np.testing.assert_array_equal(
            product.event_id.values, events.event_id.values[ranked]
        )
        np.testing.assert_array_equal(
            product.peak_time.values, events.peak_time.values[ranked]
        )
        np.testing.assert_array_equal(product.rank_value.values, metric[ranked])
    daily_paths = {
        int(Path(record["path"]).stem.rsplit("_", 1)[-1]): Path(record["path"])
        for record in records
        if Path(record["path"]).name.startswith("ERA5_daily_t2m_z500_")
    }
    maxima = {
        f"{name}_{suffix}": 0.0
        for name in ("t2m", "z500")
        for suffix in ("event_mean", "climatology_mean", "anomaly")
    }
    summaries = []
    with ExitStack() as stack:
        climate = stack.enter_context(
            xr.open_dataset(
                product.attrs["climatology_path"],
                engine="h5netcdf",
                decode_timedelta=True,
            )
        )
        annual = {}
        for event in range(product.sizes["event"]):
            peak = pd.Timestamp(product.peak_time.values[event]).normalize()
            days = [peak - pd.Timedelta(days=1), peak, peak + pd.Timedelta(days=1)]
            raw_days, climate_days = [], []
            for date in days:
                if date.year not in annual:
                    annual[date.year] = stack.enter_context(
                        xr.open_dataset(
                            daily_paths[date.year],
                            engine="h5netcdf",
                            decode_timedelta=True,
                        )
                    )
                raw_days.append(
                    _source_day(annual[date.year], product, date, climatology=False)
                )
                climate_days.append(
                    _source_day(climate, product, date, climatology=True)
                )
            for name, divisor in (("t2m", 1.0), ("z500", config.G_M_S2)):
                actual = (
                    (raw_days[0][name] + raw_days[1][name] + raw_days[2][name])
                    / 3
                    / divisor
                )
                baseline = (
                    (
                        climate_days[0][name]
                        + climate_days[1][name]
                        + climate_days[2][name]
                    )
                    / 3
                    / divisor
                )
                for suffix, expected in (
                    ("event_mean", actual),
                    ("climatology_mean", baseline),
                    ("anomaly", actual - baseline),
                ):
                    variable = f"{name}_{suffix}"
                    saved = product[variable].isel(event=event).values
                    np.testing.assert_allclose(
                        saved,
                        expected,
                        rtol=0,
                        atol=ABSOLUTE_TOLERANCE,
                        err_msg=f"Independent source check: {variable}, event index {event}",
                    )
                    maxima[variable] = max(
                        maxima[variable], float(np.max(np.abs(saved - expected)))
                    )
            summaries.append(
                {
                    "event_id": int(product.event_id.values[event]),
                    "peak_date": str(peak.date()),
                    "sample_dates": [str(day.date()) for day in days],
                }
            )
    _check_source_identities(records)
    return {
        "status": "passed",
        "method": "independent_event_ranking_and_three_daily_source_sums",
        "relative_tolerance": 0,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "max_absolute_errors": maxima,
        "events": summaries,
        "source_files": records,
    }


def _check_source_identities(records: list[dict]) -> None:
    for record in records:
        stat = Path(record["path"]).stat()
        if (
            stat.st_size != record["size_bytes"]
            or stat.st_mtime_ns != record["mtime_ns"]
        ):
            raise ValueError(f"Source file identity changed: {record['path']}")


def _source_day(
    source: xr.Dataset, product: xr.Dataset, date: pd.Timestamp, *, climatology: bool
) -> dict:
    """Select one day and the saved grid directly from the source coordinates."""
    names = {}
    for canonical, candidates in (
        ("time", ("valid_time", "time")),
        ("latitude", ("latitude", "lat")),
        ("longitude", ("longitude", "lon")),
    ):
        found = [name for name in candidates if name in source.coords]
        if len(found) != 1:
            raise ValueError(f"Source needs an unambiguous {canonical} coordinate.")
        names[canonical] = found[0]
    times = pd.DatetimeIndex(source[names["time"]].values)
    matches = np.flatnonzero(
        (times.month == date.month) & (times.day == date.day)
        if climatology
        else times == date
    )
    if matches.size != 1:
        raise ValueError(f"Expected exactly one source day for {date.date()}.")
    indexer = {names["time"]: int(matches[0])}
    for canonical in ("latitude", "longitude"):
        values = np.asarray(source[names[canonical]].values, dtype=float)
        if canonical == "longitude":
            values = (values + 180) % 360 - 180
        if np.unique(values).size != values.size:
            raise ValueError(f"Duplicate source {canonical} coordinates.")
        indices = pd.Index(values).get_indexer(product[canonical].values)
        if np.any(indices < 0):
            raise ValueError(f"Saved {canonical} grid is absent from the source.")
        indexer[names[canonical]] = indices
    row = source[["t2m", "z"]].isel(indexer)
    fields = {}
    for original, name in (("t2m", "t2m"), ("z", "z500")):
        values = (
            row[original]
            .squeeze(drop=True)
            .transpose(names["latitude"], names["longitude"])
        )
        fields[name] = np.asarray(values.load().values, dtype=np.float64)
        if not np.isfinite(fields[name]).all():
            raise ValueError(f"Non-finite {original} source on {date.date()}.")
    return fields
