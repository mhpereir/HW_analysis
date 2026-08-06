"""Face-resolved advective heating diagnostics for Stage 1.

Pipeline role:
- Stage-1 harmonization and downstream diagnostic layer.

Responsibilities:
- Normalize signed EHB face heat contributions to domain-mean heating rates.
- Validate that face contributions reconstruct the existing advection term.
- Derive grouped components and denominator-masked ratios.
- Aggregate composite face contributions into complete 24-hour lag bins.

Out of scope:
- Raw file discovery.
- Event detection or selection.
- Plot rendering.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


SECONDS_PER_HOUR = 3600.0
TIME_DIM = "time"
LAG_DIM = "lag_hour"
DAILY_WINDOW_DIM = "daily_window"
REQUIRED_FACES: tuple[str, ...] = ("west", "east", "south", "north", "top")
OPTIONAL_FACES: tuple[str, ...] = ("bottom",)
DEFAULT_RECONSTRUCTION_RTOL = 1e-10
DEFAULT_RECONSTRUCTION_ATOL = 1e-12
DEFAULT_RATIO_EPSILON = 5e-3


def add_face_advection_tendencies(
    stage1: xr.Dataset,
    heat_budget: xr.Dataset,
    *,
    rtol: float = DEFAULT_RECONSTRUCTION_RTOL,
    atol: float = DEFAULT_RECONSTRUCTION_ATOL,
) -> xr.Dataset:
    """Return Stage 1 with normalized, signed advection face contributions.

    Raw ``flux_contribution_*`` variables already include the EHB inward-face
    signs. Each is divided by domain volume and converted from K s-1 to K hr-1,
    matching the existing Stage-1 ``advection`` normalization.
    """
    _validate_tolerance(rtol=rtol, atol=atol)
    _require_dataset(stage1, "stage1")
    _require_dataset(heat_budget, "heat_budget")
    _require_variables(stage1, ("volume", "advection"), dataset_name="stage1")
    _require_variables(
        heat_budget,
        (
            "domain_volume",
            "advection_term",
            *(source_face_variable(face) for face in REQUIRED_FACES),
        ),
        dataset_name="heat_budget",
    )
    _validate_exact_time_alignment(stage1, heat_budget)

    faces = available_faces(heat_budget)
    _validate_one_dimensional_time_variables(
        stage1,
        ("volume", "advection"),
        dataset_name="stage1",
    )
    _validate_one_dimensional_time_variables(
        heat_budget,
        (
            "domain_volume",
            "advection_term",
            *(source_face_variable(face) for face in faces),
        ),
        dataset_name="heat_budget",
    )
    _assert_finite_positive(stage1["volume"], name="stage1 volume")
    _assert_allclose(
        stage1["volume"],
        heat_budget["domain_volume"],
        description="Stage-1 volume and raw EHB domain_volume",
        rtol=rtol,
        atol=atol,
    )

    raw_total = sum(
        (
            heat_budget[source_face_variable(face)]
            for face in faces
        ),
        start=xr.zeros_like(heat_budget["advection_term"]),
    )
    _assert_allclose(
        raw_total,
        heat_budget["advection_term"],
        description="summed raw face heat contributions and raw advection_term",
        rtol=rtol,
        atol=atol,
    )

    normalized: dict[str, xr.DataArray] = {}
    for face in faces:
        source_name = source_face_variable(face)
        output_name = stage1_face_variable(face)
        tendency = (
            heat_budget[source_name] / heat_budget["domain_volume"] * SECONDS_PER_HOUR
        ).rename(output_name)
        tendency = tendency.assign_coords({TIME_DIM: stage1[TIME_DIM]})
        tendency.attrs.update(
            {
                "long_name": f"Advective contribution through the {face} face",
                "units": "K hr-1",
                "source_variable": source_name,
                "face": face,
                "normalized_by": "domain_volume",
                "time_conversion_factor": SECONDS_PER_HOUR,
                "sign_convention": (
                    "positive contributes warming of the domain; "
                    "negative contributes cooling of the domain"
                ),
                "formula": f"{source_name} / domain_volume * 3600",
                "native_time_resolution": "hourly",
                "analysis_time_resolution": "hourly",
            }
        )
        normalized[output_name] = tendency

    normalized_total = sum(
        normalized.values(),
        start=xr.zeros_like(stage1["advection"]),
    )
    _assert_allclose(
        normalized_total,
        stage1["advection"],
        description="normalized face contributions and Stage-1 advection",
        rtol=rtol,
        atol=atol,
    )

    out = stage1.assign(normalized)
    out.attrs.update(
        {
            "advection_face_contributions": 1,
            "advection_face_contribution_faces": ",".join(faces),
            "advection_face_contribution_sign": (
                "positive warming contribution; negative cooling contribution"
            ),
            "advection_face_reconstruction_rtol": float(rtol),
            "advection_face_reconstruction_atol": float(atol),
        }
    )
    return out


def grouped_advection_components(ds: xr.Dataset) -> xr.Dataset:
    """Return zonal, meridional, horizontal, vertical, and total tendencies."""
    faces = available_stage1_faces(ds)
    required = tuple(stage1_face_variable(face) for face in REQUIRED_FACES)
    _require_variables(ds, required, dataset_name="face-contribution dataset")

    zonal = (ds["advection_west"] + ds["advection_east"]).rename(
        "advection_zonal"
    )
    meridional = (ds["advection_south"] + ds["advection_north"]).rename(
        "advection_meridional"
    )
    horizontal = (zonal + meridional).rename("advection_horizontal")
    vertical = ds["advection_top"]
    if "bottom" in faces:
        vertical = vertical + ds["advection_bottom"]
    vertical = vertical.rename("advection_vertical")
    total = (horizontal + vertical).rename("advection_face_total")

    out = xr.Dataset(
        {
            "advection_zonal": zonal,
            "advection_meridional": meridional,
            "advection_horizontal": horizontal,
            "advection_vertical": vertical,
            "advection_face_total": total,
        }
    )
    descriptions = {
        "advection_zonal": "Sum of west and east face advective contributions",
        "advection_meridional": "Sum of south and north face advective contributions",
        "advection_horizontal": "Sum of zonal and meridional advective contributions",
        "advection_vertical": "Sum of top and available bottom face contributions",
        "advection_face_total": "Sum of all available face advective contributions",
    }
    for name, description in descriptions.items():
        out[name].attrs.update(
            {
                "long_name": description,
                "units": "K hr-1",
                "sign_convention": (
                    "positive contributes warming of the domain; "
                    "negative contributes cooling of the domain"
                ),
            }
        )
    return out


def add_grouped_components_and_ratios(
    ds: xr.Dataset,
    *,
    ratio_epsilon: float = DEFAULT_RATIO_EPSILON,
) -> xr.Dataset:
    """Return a dataset with grouped components and masked signed ratios."""
    if ratio_epsilon < 0:
        raise ValueError("ratio_epsilon must be >= 0.")

    grouped = grouped_advection_components(ds)
    out = xr.merge([ds, grouped], compat="override")
    out["advection_meridional_zonal_ratio"] = masked_ratio(
        out["advection_meridional"],
        out["advection_zonal"],
        epsilon=ratio_epsilon,
        name="advection_meridional_zonal_ratio",
    )
    out["advection_horizontal_vertical_ratio"] = masked_ratio(
        out["advection_horizontal"],
        out["advection_vertical"],
        epsilon=ratio_epsilon,
        name="advection_horizontal_vertical_ratio",
    )
    out["advection_meridional_zonal_ratio"].attrs.update(
        {
            "long_name": "Meridional to zonal advective-tendency ratio",
            "numerator": "advection_meridional",
            "denominator": "advection_zonal",
        }
    )
    out["advection_horizontal_vertical_ratio"].attrs.update(
        {
            "long_name": "Horizontal to vertical advective-tendency ratio",
            "numerator": "advection_horizontal",
            "denominator": "advection_vertical",
        }
    )
    for name in (
        "advection_meridional_zonal_ratio",
        "advection_horizontal_vertical_ratio",
    ):
        out[name].attrs.update(
            {
                "units": "1",
                "denominator_mask_threshold": float(ratio_epsilon),
                "denominator_mask_threshold_units": "K hr-1",
            }
        )
    out.attrs.update(ds.attrs)
    out.attrs["advection_ratio_epsilon"] = float(ratio_epsilon)
    out.attrs["advection_ratio_epsilon_units"] = "K hr-1"
    return out


def masked_ratio(
    numerator: xr.DataArray,
    denominator: xr.DataArray,
    *,
    epsilon: float,
    name: str | None = None,
) -> xr.DataArray:
    """Return a signed ratio masked where the denominator is near zero."""
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0.")
    numerator, denominator = xr.align(numerator, denominator, join="exact")
    ratio = xr.where(np.abs(denominator) > epsilon, numerator / denominator, np.nan)
    if name is not None:
        ratio = ratio.rename(name)
    return ratio


def complete_daily_face_means(
    composite: xr.Dataset,
    *,
    window_hours: int = 24,
    lag_dim: str = LAG_DIM,
    output_dim: str = DAILY_WINDOW_DIM,
) -> xr.Dataset:
    """Average face contributions over sequential complete lag windows.

    The final inclusive lag endpoint is not duplicated into an incomplete
    one-sample window. For a -168 through +168 hourly composite this yields
    fourteen complete 24-hour bins spanning [-168, 168).
    """
    if window_hours < 1:
        raise ValueError("window_hours must be >= 1.")
    if lag_dim not in composite.coords:
        raise ValueError(f"composite is missing lag coordinate {lag_dim!r}.")

    face_names = tuple(
        stage1_face_variable(face)
        for face in available_stage1_faces(composite)
    )
    _require_variables(
        composite,
        tuple(stage1_face_variable(face) for face in REQUIRED_FACES),
        dataset_name="composite",
    )
    lag = np.asarray(composite[lag_dim].values)
    if lag.ndim != 1 or lag.size < 2:
        raise ValueError(f"{lag_dim!r} must be a one-dimensional hourly coordinate.")
    if not np.issubdtype(lag.dtype, np.number):
        raise TypeError(f"{lag_dim!r} must be numeric.")
    lag = lag.astype(float)
    if not np.all(np.isfinite(lag)) or not np.allclose(np.diff(lag), 1.0):
        raise ValueError(f"{lag_dim!r} must be finite, strictly hourly, and contiguous.")

    first = int(lag[0])
    stop = int(lag[-1])
    if not np.isclose(lag[0], first) or not np.isclose(lag[-1], stop):
        raise ValueError(f"{lag_dim!r} endpoints must be integer hours.")
    span = stop - first
    if span < window_hours or span % window_hours != 0:
        raise ValueError(
            f"Lag span {span} hours must be a positive multiple of window_hours="
            f"{window_hours}."
        )

    windows: list[xr.Dataset] = []
    centers: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for start in range(first, stop, window_hours):
        end = start + window_hours
        selected = composite[list(face_names)].where(
            (composite[lag_dim] >= start) & (composite[lag_dim] < end),
            drop=True,
        )
        if selected.sizes.get(lag_dim, 0) != window_hours:
            raise ValueError(
                f"Lag window [{start}, {end}) does not contain "
                f"{window_hours} hourly samples."
            )
        windows.append(selected.mean(lag_dim).expand_dims({output_dim: [len(windows)]}))
        centers.append((start + end - 1) / 2.0)
        starts.append(start)
        ends.append(end)

    out = xr.concat(windows, dim=output_dim)
    out = out.assign_coords(
        lag_hour_center=(output_dim, np.asarray(centers, dtype=float)),
        lag_day_center=(output_dim, np.asarray(centers, dtype=float) / 24.0),
        lag_hour_start=(output_dim, np.asarray(starts, dtype=np.int64)),
        lag_hour_end_exclusive=(output_dim, np.asarray(ends, dtype=np.int64)),
    )
    out.attrs.update(
        {
            "daily_window_hours": int(window_hours),
            "daily_window_convention": "sequential [start, end) lag-hour bins",
            "daily_window_reduction": "mean of composite face contributions",
        }
    )
    return out


def source_face_variable(face: str) -> str:
    """Return the raw EHB heat-flux variable for a face."""
    return f"flux_contribution_{face}"


def stage1_face_variable(face: str) -> str:
    """Return the Stage-1 heating-rate variable for a face."""
    return f"advection_{face}"


def available_faces(ds: xr.Dataset) -> tuple[str, ...]:
    """Return required raw faces plus an optional bottom face when present."""
    faces = list(REQUIRED_FACES)
    for face in OPTIONAL_FACES:
        if source_face_variable(face) in ds:
            faces.append(face)
    return tuple(faces)


def available_stage1_faces(ds: xr.Dataset) -> tuple[str, ...]:
    """Return required Stage-1 faces plus an optional bottom face when present."""
    faces = list(REQUIRED_FACES)
    for face in OPTIONAL_FACES:
        if stage1_face_variable(face) in ds:
            faces.append(face)
    return tuple(faces)


def _require_dataset(value: object, name: str) -> None:
    if not isinstance(value, xr.Dataset):
        raise TypeError(f"{name} must be an xarray.Dataset.")


def _require_variables(
    ds: xr.Dataset,
    names: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    missing = sorted(name for name in names if name not in ds)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required variables: {', '.join(missing)}"
        )


def _validate_one_dimensional_time_variables(
    ds: xr.Dataset,
    names: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    for name in names:
        if ds[name].dims != (TIME_DIM,):
            raise ValueError(
                f"{dataset_name} variable {name!r} must have dims ({TIME_DIM!r},); "
                f"got {ds[name].dims!r}."
            )


def _validate_exact_time_alignment(stage1: xr.Dataset, heat_budget: xr.Dataset) -> None:
    if TIME_DIM not in stage1.coords:
        raise ValueError(f"stage1 is missing coordinate {TIME_DIM!r}.")
    if TIME_DIM not in heat_budget.coords:
        raise ValueError(f"heat_budget is missing coordinate {TIME_DIM!r}.")
    stage1_time = np.asarray(stage1[TIME_DIM].values)
    heat_budget_time = np.asarray(heat_budget[TIME_DIM].values)
    if not np.array_equal(stage1_time, heat_budget_time):
        raise ValueError(
            "Stage-1 and raw EHB time coordinates must match exactly; "
            f"got {stage1_time.size} and {heat_budget_time.size} timestamps."
        )


def _assert_finite_positive(da: xr.DataArray, *, name: str) -> None:
    values = np.asarray(da.values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive.")


def _assert_allclose(
    actual: xr.DataArray,
    expected: xr.DataArray,
    *,
    description: str,
    rtol: float,
    atol: float,
) -> None:
    actual_values = np.asarray(actual.values, dtype=float)
    expected_values = np.asarray(expected.values, dtype=float)
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        max_abs_error = float(np.nanmax(np.abs(actual_values - expected_values)))
        raise ValueError(
            f"{description} do not match within rtol={rtol:g}, atol={atol:g}; "
            f"maximum absolute error is {max_abs_error:g}."
        )


def _validate_tolerance(*, rtol: float, atol: float) -> None:
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be >= 0.")
