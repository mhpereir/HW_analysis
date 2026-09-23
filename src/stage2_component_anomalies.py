"""Component-anomaly budgets on immutable accepted Stage-2 populations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from scripts.event_features.fixed_window_features import WindowReducer
from src import climatology

TENDENCIES = ("dTdt", "advection", "adiabatic", "diabatic")
RAW_TO_ANOMALY = {
    f"I_{name}_pre": f"I_{name}_anom_pre" for name in (*TENDENCIES, "dyn")
}
REPRESENTATION = "climatological_anomaly"
INTEGRAL_METHOD = "hourly_sum_assuming_1h_spacing"
PAIR_METADATA = (
    "region",
    "heat_budget_bottom_boundary",
    "heat_budget_top_boundary",
    "heat_budget_pre_window_hours",
    "climatology_start_year",
    "climatology_end_year",
    "climatology_matching",
    "climatology_event_exclusion",
)


def _window_hours(table: xr.Dataset) -> int:
    try:
        start, end = map(int, table.attrs["heat_budget_pre_window_hours"].split(","))
    except (KeyError, ValueError, AttributeError) as error:
        raise ValueError("Missing or invalid heat-budget window.") from error
    if start >= 0 or end != 0:
        raise ValueError(
            "Require a positive pre-anchor heat-budget span ending at zero."
        )
    return -start


def _validate_reference(table: xr.Dataset, kind: str, hours: int) -> None:
    required = {
        "pipeline_stage": f"stage_2_{kind}_features",
        "heat_budget_pre_window_hours": f"{-hours},0",
        "window_endpoint_inclusion": "inclusive",
        "integral_method": INTEGRAL_METHOD,
        "season_months": "6,7,8",
    }
    for key, expected in required.items():
        if table.attrs.get(key) != expected:
            raise ValueError(f"Reference {kind} requires {key}={expected!r}.")
    if table.attrs.get("heat_budget_representation", "absolute") != "absolute":
        raise ValueError("The reference must be an accepted absolute Stage-2 table.")
    if kind == "event" and int(table.attrs.get("require_full_event", 0)) != 1:
        raise ValueError("The reference requires full JJA events.")
    dim, anchor = (
        ("event", "peak_time")
        if kind == "event"
        else ("baseline_day", "reference_time")
    )
    if not table.sizes.get(dim) or table[anchor].dims != (dim,):
        raise ValueError(f"Invalid {kind} reference population.")
    if not pd.Index(table[dim].values).is_unique:
        raise ValueError("Reference row identities must be unique.")
    for name in RAW_TO_ANOMALY:
        if name not in table or table[name].dims != (dim,):
            raise ValueError(f"Missing reference budget feature {name}.")
        for key, expected in {
            "units": "K",
            "window_lag_hours": f"{-hours},0",
            "window_endpoint_inclusion": "inclusive",
            "integral_method": INTEGRAL_METHOD,
        }.items():
            if table[name].attrs.get(key) != expected:
                raise ValueError(f"Reference {name} has incompatible {key}.")
    if not np.all(table.n_samples_heat_budget_pre.values == hours + 1):
        raise ValueError("Reference heat-budget sample counts are incomplete.")
    if kind == "baseline" and not np.isin(table.event_adjacent.values, [0, 1]).all():
        raise ValueError("Invalid reference baseline adjacency flags.")


def _validate_climate(source: xr.Dataset, climate: xr.Dataset) -> None:
    climatology.validate_regional_hourly_climatology(
        climate, required_variables=TENDENCIES
    )
    climatology.validate_climatology_source_compatibility(source, climate)
    for key in (
        "region",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
        "stage1_contract_version",
    ):
        if key not in source.attrs:
            raise ValueError(f"Stage-1 metadata missing {key}.")
    for key in ("start_year", "end_year"):
        if int(source.attrs[key]) != int(climate.attrs[f"climatology_{key}"]):
            raise ValueError("Source and climatology reference years must agree.")
    if climate.attrs.get("climatology_event_exclusion") != "none":
        raise ValueError("An all-observation climatology is required.")
    if climate.attrs.get("climatology_method") != "arithmetic mean over source years":
        raise ValueError("The fixed arithmetic-mean climatology is required.")
    years = (
        int(climate.attrs["climatology_end_year"])
        - int(climate.attrs["climatology_start_year"])
        + 1
    )
    for name in TENDENCIES:
        for ds, dim in ((source, "time"), (climate, "climatology_time")):
            if ds[name].dims != (dim,) or ds[name].attrs.get("units") not in {
                "K hr-1",
                "K h-1",
                "K hour-1",
            }:
                raise ValueError(f"{name} must be one-dimensional in K per hour.")
        if not np.isfinite(climate[name].values).all() or not np.all(
            climate[f"{name}_count"].values == years
        ):
            raise ValueError(
                f"Incomplete climatological source-year coverage for {name}."
            )
    residual = climate.dTdt - climate.advection - climate.adiabatic - climate.diabatic
    if not np.allclose(residual, 0, atol=1e-10, rtol=0):
        raise ValueError("Climatological heat budget does not close.")


def _complete_windows(anomalies: xr.Dataset, anchors: np.ndarray, hours: int) -> None:
    times = anomalies.time.values.astype("datetime64[ns]")
    if np.isnat(anchors).any() or not np.array_equal(
        anchors, anchors.astype("datetime64[h]")
    ):
        raise ValueError("Anchors must be finite whole-hour timestamps.")
    for anchor in anchors:
        expected = anchor + np.arange(-hours, 1).astype("timedelta64[h]")
        a, b = np.searchsorted(times, [expected[0], expected[-1]])
        if not np.array_equal(times[a : b + 1], expected):
            raise ValueError(f"Incomplete hourly window ending at {anchor}.")
        for name in TENDENCIES:
            if not np.isfinite(anomalies[name].values[a : b + 1]).all():
                raise ValueError(
                    f"Nonfinite {name} samples in window ending at {anchor}."
                )


def build_component_anomaly_pair(
    source: xr.Dataset, climate: xr.Dataset, events: xr.Dataset, baseline: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """Integrate tendency anomalies without recalculating raw totals or selections."""
    hours = _window_hours(events)
    _validate_climate(source, climate)
    for kind, reference in (("event", events), ("baseline", baseline)):
        _validate_reference(reference, kind, hours)
    if not pd.Index(events.event_id.values).is_unique:
        raise ValueError("Reference event IDs must be unique.")
    for name in ("event_id", "peak_time", "start_time", "end_time"):
        if not np.array_equal(events[name], source[name].sel(event=events.event)):
            raise ValueError(f"Reference event {name} does not match Stage 1.")
    anomalies = climatology.apply_regional_hourly_climatology(
        source, climate, variables=TENDENCIES
    )
    windows = {"heat_budget_pre": (-hours, 0)}
    reducer = WindowReducer(anomalies, windows=windows)
    outputs = []
    for kind, reference, anchor_name, dim in (
        ("event", events, "peak_time", "event"),
        ("baseline", baseline, "reference_time", "baseline_day"),
    ):
        anchors = reference[anchor_name].values.astype("datetime64[ns]")
        _complete_windows(anomalies, anchors, hours)
        out = reference.drop_vars(list(RAW_TO_ANOMALY)).copy(deep=True)
        for name in TENDENCIES:
            feature = f"I_{name}_anom_pre"
            out[feature] = (dim, reducer.sums(name, anchors, "heat_budget_pre"))
            out[feature].attrs = dict(reference[f"I_{name}_pre"].attrs)
            out[feature].attrs.update(
                long_name=f"Integrated climatological anomaly of {name}",
                data_representation=REPRESENTATION,
                anomaly_method="source tendency minus calendar-hour climatological tendency",
            )
        out["I_dyn_anom_pre"] = out.I_advection_anom_pre + out.I_adiabatic_anom_pre
        out.I_dyn_anom_pre.attrs = dict(out.I_advection_anom_pre.attrs)
        out.I_dyn_anom_pre.attrs.pop("source_variable", None)
        out.I_dyn_anom_pre.attrs.update(
            long_name="Integrated dynamical heating anomaly",
            source_variables="I_advection_anom_pre,I_adiabatic_anom_pre",
            formula="I_advection_anom_pre + I_adiabatic_anom_pre",
        )
        out.attrs.update(
            pipeline_stage=f"stage_2_{kind}_component_anomalies",
            component_anomaly_contract_version=1,
            reference_pipeline_stage=reference.attrs["pipeline_stage"],
            heat_budget_representation=REPRESENTATION,
            anomaly_method="subtract calendar-hour tendency climatology before inclusive hourly sum",
            active_integral_sources=",".join(TENDENCIES),
            derived_budget_variables=",".join(RAW_TO_ANOMALY.values()),
            copied_reference_variables=",".join(
                reference.drop_vars(list(RAW_TO_ANOMALY)).data_vars
            ),
        )
        for key in (
            "region",
            "heat_budget_bottom_boundary",
            "heat_budget_top_boundary",
        ):
            out.attrs[key] = source.attrs[key]
        for key in (
            "climatology_start_year",
            "climatology_end_year",
            "climatology_matching",
            "climatology_event_exclusion",
        ):
            out.attrs[key] = climate.attrs[key]
        for name in RAW_TO_ANOMALY.values():
            for key in (
                "climatology_start_year",
                "climatology_end_year",
                "climatology_matching",
            ):
                out[name].attrs[key] = climate.attrs[key]
        validate_component_anomalies(out)
        outputs.append(out)
    return tuple(outputs)


def validate_component_anomalies(table: xr.Dataset) -> None:
    """Validate the saved anomaly-only budget contract and its closure."""
    marker = table.attrs.get("pipeline_stage")
    if marker not in {
        "stage_2_event_component_anomalies",
        "stage_2_baseline_component_anomalies",
    }:
        raise ValueError("Not a Stage-2 component-anomaly product.")
    if (
        int(table.attrs.get("component_anomaly_contract_version", 0)) != 1
        or table.attrs.get("heat_budget_representation") != REPRESENTATION
    ):
        raise ValueError(
            "Missing component-anomaly representation or contract version."
        )
    if any(name in table for name in RAW_TO_ANOMALY):
        raise ValueError("Anomaly products must not contain raw budget features.")
    if any(key not in table.attrs for key in PAIR_METADATA):
        raise ValueError("Incomplete component-anomaly provenance.")
    dim, anchor = (
        ("event", "peak_time")
        if marker == "stage_2_event_component_anomalies"
        else ("baseline_day", "reference_time")
    )
    if not table.sizes.get(dim) or table[anchor].dims != (dim,):
        raise ValueError("Invalid anomaly population.")
    hours = _window_hours(table)
    if not np.all(table.n_samples_heat_budget_pre.values == hours + 1):
        raise ValueError("Invalid anomaly sample counts.")
    for name in RAW_TO_ANOMALY.values():
        if (
            name not in table
            or table[name].dims != (dim,)
            or not np.isfinite(table[name]).all()
        ):
            raise ValueError(f"Missing, nonfinite or invalid anomaly feature {name}.")
        for key, expected in {
            "units": "K",
            "data_representation": REPRESENTATION,
            "window_lag_hours": f"{-hours},0",
            "window_endpoint_inclusion": "inclusive",
            "integral_method": INTEGRAL_METHOD,
        }.items():
            if table[name].attrs.get(key) != expected:
                raise ValueError(f"Invalid {key} for {name}.")
    for residual in (
        table.I_dyn_anom_pre - table.I_advection_anom_pre - table.I_adiabatic_anom_pre,
        table.I_dTdt_anom_pre - table.I_dyn_anom_pre - table.I_diabatic_anom_pre,
    ):
        if not np.allclose(residual, 0, atol=1e-8, rtol=0):
            raise ValueError("Anomalous heat budget does not close.")


def comparison_plot_views(
    baseline: xr.Dataset, events: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset, bool]:
    """Return ephemeral legacy-name views; never disguise an anomaly as raw input."""
    reps = [
        table.attrs.get("heat_budget_representation", "absolute")
        for table in (baseline, events)
    ]
    if reps[0] != reps[1] or reps[0] not in {"absolute", REPRESENTATION}:
        raise ValueError(
            "Both comparison tables must use the same budget representation."
        )
    if reps[0] == "absolute":
        if any(
            name in table
            for table in (baseline, events)
            for name in RAW_TO_ANOMALY.values()
        ):
            raise ValueError(
                "Anomaly features require explicit representation metadata."
            )
        return baseline, events, False
    for table, kind in ((baseline, "baseline"), (events, "event")):
        validate_component_anomalies(table)
        if table.attrs["pipeline_stage"] != f"stage_2_{kind}_component_anomalies":
            raise ValueError("Wrong anomaly population type.")
    for key in PAIR_METADATA:
        if baseline.attrs[key] != events.attrs[key]:
            raise ValueError(f"Anomaly comparison inputs disagree on {key}.")
    names = {value: key for key, value in RAW_TO_ANOMALY.items()}
    return baseline.rename(names), events.rename(names), True
