"""Named boundary-flux regrouping and explicit plotting-role selection.

Persist both partitions. Selection maps roles to field names, never replaces
the values of original variables, and must precede nonlinear sample reductions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

ORIGINAL = "original"
BOUNDARY_FLUX = "boundary_flux"
CHOICES = (ORIGINAL, BOUNDARY_FLUX)
VARIABLES = ("adv_horizontal", "vertical_contribution")
METADATA = {
    "budget_partition_contract_version": 1,
    "budget_partitions": "original,boundary_flux",
    "budget_partition_method": "signed_boundary_flux_regrouping",
}
ROLE_NAMES = dict(zip(("advection", "adiabatic"), VARIABLES, strict=True))


def add_argument(parser) -> None:
    parser.add_argument(
        "--budget-partition",
        choices=CHOICES,
        default=ORIGINAL,
        help="Original terms or horizontal advection plus total vertical contribution.",
    )


def variable_name(name: str, partition: str = ORIGINAL) -> str:
    """Resolve a canonical plotting role to an actual persisted field name."""
    if partition not in CHOICES:
        raise ValueError(f"Unknown budget partition: {partition!r}.")
    if partition == ORIGINAL:
        return name
    for original, alternate in ROLE_NAMES.items():
        for prefix, suffix in (("", ""), ("I_", "_pre"), ("I_", "_anom_pre")):
            if name == f"{prefix}{original}{suffix}":
                return f"{prefix}{alternate}{suffix}"
    return name


def variable_names(names: Sequence[str], partition: str = ORIGINAL) -> tuple[str, ...]:
    return tuple(variable_name(name, partition) for name in names)


def resolve(ds: xr.Dataset, name: str) -> str:
    return variable_name(name, ds.attrs.get("budget_partition", ORIGINAL))


def extension_variables(
    ds: xr.Dataset, *, prefix: str = "", suffix: str = ""
) -> tuple[str, ...]:
    """Recognize a complete, explicitly declared additive product extension."""
    names = tuple(f"{prefix}{name}{suffix}" for name in VARIABLES)
    present = [name in ds for name in names]
    declared = any(key in ds.attrs for key in METADATA)
    if not any(present) and not declared:
        return ()
    if not all(present) or any(ds.attrs.get(k) != v for k, v in METADATA.items()):
        raise ValueError(
            "Incomplete or incompatible boundary-flux partition extension."
        )
    return names


def select(
    ds: xr.Dataset, partition: str = ORIGINAL, *, integrated=False
) -> xr.Dataset:
    """Return an explicitly selected shallow view without renaming any fields."""
    variable_name("advection", partition)
    if partition == BOUNDARY_FLUX:
        suffix = "_pre" if integrated else ""
        prefix = "I_" if integrated else ""
        if not extension_variables(ds, prefix=prefix, suffix=suffix):
            raise ValueError(
                "Boundary-flux partition requires regenerated named fields."
            )
    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs, budget_partition=partition)
    return out


def _assert_equal(actual: xr.DataArray, expected: xr.DataArray, label: str) -> None:
    if actual.dims != expected.dims or actual.sizes != expected.sizes:
        raise ValueError(f"Incompatible dimensions for {label}.")
    a, b = np.asarray(actual.values), np.asarray(expected.values)
    if (
        np.isinf(a).any()
        or np.isinf(b).any()
        or not np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
    ):
        raise ValueError(f"Boundary-flux identity failed: {label}.")


def _regrouped_values(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    faces = tuple(
        f"advection_{face}" for face in ("west", "east", "south", "north", "top")
    )
    required = (*faces, "advection", "adiabatic")
    bottom = ds.attrs.get("heat_budget_bottom_boundary", "surface")
    if bottom not in ("surface", "", None) or "advection_bottom" in ds:
        required += ("advection_bottom",)
    for name in required:
        if name not in ds:
            raise ValueError(f"Boundary-flux regrouping requires {name}.")
        if ds[name].dims != ("time",) or ds[name].attrs.get("units") != "K hr-1":
            raise ValueError(f"{name} must have dims (time,) and units K hr-1.")
    vertical = ds.advection_top
    if "advection_bottom" in ds:
        vertical = vertical + ds.advection_bottom
    horizontal = ds.advection - vertical
    lateral = (
        ds.advection_west + ds.advection_east + ds.advection_south + ds.advection_north
    )
    _assert_equal(ds.advection, lateral + vertical, "face sum = advection")
    _assert_equal(horizontal, lateral, "adv_horizontal = lateral face sum")
    return horizontal, ds.adiabatic + vertical


def add_boundary_flux_fields(ds: xr.Dataset) -> xr.Dataset:
    """Derive the additive extension, preserving all existing fields exactly."""
    if extension_variables(ds):
        validate_stage1(ds)
        return ds.copy(deep=False)
    horizontal, vertical = _regrouped_values(ds)
    vertical_faces = "advection_top"
    if "advection_bottom" in ds:
        vertical_faces += " + advection_bottom"
    out = ds.assign(adv_horizontal=horizontal, vertical_contribution=vertical)
    out.attrs = {**ds.attrs, **METADATA}
    for name, formula, label in (
        (
            "adv_horizontal",
            f"advection - ({vertical_faces})",
            "Horizontal boundary-flux advection",
        ),
        (
            "vertical_contribution",
            f"adiabatic + ({vertical_faces})",
            "Total vertical contribution",
        ),
    ):
        out[name].attrs = {
            "long_name": label,
            "units": "K hr-1",
            "formula": formula,
            "budget_partition": BOUNDARY_FLUX,
            "partition_method": METADATA["budget_partition_method"],
            "vertical_boundary_terms": vertical_faces,
            "sign_convention": "positive warming contribution; negative cooling contribution",
        }
    validate_stage1(out)
    return out


def validate_stage1(ds: xr.Dataset) -> None:
    if not extension_variables(ds):
        return
    horizontal, vertical = _regrouped_values(ds)
    for name, expected in zip(VARIABLES, (horizontal, vertical), strict=True):
        if ds[name].attrs.get("units") != "K hr-1":
            raise ValueError(f"Invalid boundary-flux units for {name}.")
        _assert_equal(ds[name], expected, name)
    _assert_equal(
        ds.adv_horizontal + ds.vertical_contribution,
        ds.advection + ds.adiabatic,
        "unchanged dynamical tendency",
    )


def validate_integrals(ds: xr.Dataset, *, anomaly=False) -> None:
    suffix = "_anom_pre" if anomaly else "_pre"
    names = extension_variables(ds, prefix="I_", suffix=suffix)
    if not names:
        return
    reference = ds[f"I_advection{suffix}"]
    for name in names:
        if ds[name].attrs.get("units") != "K" or ds[name].dims != reference.dims:
            raise ValueError(f"Invalid boundary-flux integral {name}.")
        for key in ("window_lag_hours", "window_endpoint_inclusion", "integral_method"):
            if ds[name].attrs.get(key) != reference.attrs.get(key):
                raise ValueError(f"Incompatible {key} for {name}.")
    _assert_equal(
        ds[names[0]] + ds[names[1]],
        ds[f"I_dyn{suffix}"],
        "unchanged integrated dynamical contribution",
    )
