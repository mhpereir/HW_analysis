import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from HW_analysis.src import analysis_io, events, harmonize, pressure_layer


@pytest.fixture
def layer_inputs():
    time = pd.date_range("2001-05-01T01:00", "2001-10-31T22:00", freq="h")
    n = len(time)
    volume = 8e16
    raw = xr.Dataset(
        {
            "T_domain_avg": ("time", np.full(n, 260.0)),
            "domain_volume": ("time", np.full(n, volume)),
            "advection_term": ("time", np.full(n, 0.21 * volume / 3600)),
            "adiabatic_term": ("time", np.full(n, 0.7 * volume / 3600)),
            "diabatic_term": ("time", np.full(n, -0.41 * volume / 3600)),
            "dT_dt": ("time", np.full(n, 0.5 * volume / 3600)),
            "net_mass_advection": ("time", np.full(n, 21e10)),
        },
        coords={"time": time},
    )
    raw.domain_volume.attrs.update(
        zg_bottom_mode="pressure_level",
        zg_bottom_pressure_pa=70000,
        zg_top_pressure_pa=50000,
    )
    for i, face in enumerate(pressure_layer.FACES, 1):
        raw[f"flux_contribution_{face}"] = (
            "time",
            np.full(n, i * 0.01 * volume / 3600),
        )
        raw[f"mass_flux_contribution_{face}"] = ("time", np.full(n, i * 1e10))
    for face, p in [("bottom", 70000), ("top", 50000)]:
        for prefix in ["flux_contribution", "mass_flux_contribution"]:
            raw[f"{prefix}_{face}"].attrs["face_pressure_pa"] = p
    days = pd.date_range("2001-05-01", "2001-10-31", freq="D")

    def daily(values):
        return xr.DataArray(
            np.broadcast_to(values, (len(days),)).copy(),
            dims="time",
            coords={"time": days},
        )

    ids = np.zeros(len(days), dtype=int)
    ids[1:4] = 1
    ids[70:72] = 2
    hw = {
        "tas_region": daily(np.where(ids, 285.0, 275.0)),
        "tas_climatology": daily(280.0),
        "hw_threshold": daily(282.0),
        "hw_exceedance_mask": daily(ids > 0),
        "hw_event_id": daily(ids),
    }
    lwa = [
        {
            f"{prefix}_region": daily(np.where(ids, 10.0, 1.0)),
            f"{prefix}_threshold": daily(5.0),
            f"{prefix}_exceedance_mask": daily(ids > 0),
            f"{prefix}_event_id": daily(ids),
        }
        for prefix in ("lwa", "lwa_a", "lwa_c")
    ]
    diagnostics = {}
    for key, source_name in harmonize.FULL_DIAGNOSTIC_SOURCE_VARIABLES.items():
        diagnostics[key] = xr.Dataset(
            {source_name: (("time", "lat", "lon"), np.full((n, 2, 2), 10.0))},
            coords={"time": time, "lat": [45.0, 55.0], "lon": [-125.0, -115.0]},
        )
    attrs = {
        "region": "pnw_bartusek",
        "quantile": "90",
        "threshold_variable": "tas",
        "add_full_diagnostics": 1,
        "cloud_cover_source_layout": "global-hourly-grid",
        "start_year": 2001,
        "end_year": 2001,
    }

    def build(budget):
        ds = harmonize.build_regional_analysis_dataset(
            heat_budget=budget,
            hw_event_products=hw,
            lwa_event_products=lwa,
            full_diagnostics=diagnostics,
            region="pnw_bartusek",
            attrs=attrs,
        )
        return xr.merge(
            [
                ds,
                events.build_event_summary_table(
                    ds, "hw_event_id", peak_variable="tas_region"
                ),
            ]
        )

    surface = raw.drop_vars(
        ["flux_contribution_bottom", "mass_flux_contribution_bottom"]
    ).copy(deep=True)
    surface["domain_volume"] = surface.domain_volume * 2
    surface["advection_term"] = sum(
        surface[f"flux_contribution_{f}"] for f in pressure_layer.FACES[:-1]
    )
    reference = build(surface)
    return raw, reference, build(raw)


def test_reused_regional_fields_match_full_raw_source_build(layer_inputs):
    raw, reference, direct = layer_inputs
    result = harmonize.replace_heat_budget(reference, raw)
    assert "advection_bottom" not in reference
    assert "advection_bottom" in result
    for name in direct.data_vars:
        np.testing.assert_array_equal(
            result[name].values, direct[name].values, err_msg=name
        )
    np.testing.assert_allclose(
        result.sshf_heating_rate_approx, reference.sshf_heating_rate_approx * 2
    )
    assert (
        "not a measured heating"
        in result.sshf_heating_rate_approx.attrs["physical_interpretation"]
    )
    report = pressure_layer.validate_product(result, reference, raw)
    assert report["event_table_exact"]
    assert report["events"] == 2


@pytest.mark.parametrize(
    "fault",
    [
        "missing_hour",
        "missing_bottom",
        "wrong_pressure",
        "closure",
        "mass",
        "nan",
        "volume",
    ],
)
def test_annual_source_validation_rejects_invalid_science(layer_inputs, fault):
    raw = layer_inputs[0].copy(deep=True)
    if fault == "missing_hour":
        raw = raw.isel(time=slice(1, None))
    elif fault == "missing_bottom":
        raw = raw.drop_vars("flux_contribution_bottom")
    elif fault == "wrong_pressure":
        raw.flux_contribution_bottom.attrs["face_pressure_pa"] = 80000
    elif fault == "closure":
        raw.diabatic_term.values[0] *= 2
    elif fault == "mass":
        raw.net_mass_advection.values[0] *= 2
    elif fault == "nan":
        raw.T_domain_avg.values[0] = np.nan
    elif fault == "volume":
        raw.domain_volume.values[0] = 0
    with pytest.raises(ValueError):
        pressure_layer.validate_annual_budget(raw, year=2001, bottom=700, top=500)


def test_source_and_reference_contracts(layer_inputs):
    raw, reference, _ = layer_inputs
    report = pressure_layer.validate_annual_budget(raw, year=2001, bottom=700, top=500)
    assert report["hours"] == 4414
    selected = pressure_layer.select_reference(
        reference,
        years=[2001],
        region="pnw_bartusek",
        threshold_variable="tas",
        quantile="90",
    )
    xr.testing.assert_equal(reference, selected)
    with pytest.raises(ValueError, match="quantile"):
        pressure_layer.select_reference(
            reference,
            years=[2001],
            region="pnw_bartusek",
            threshold_variable="tas",
            quantile="95",
        )
    with pytest.raises(ValueError, match="complete-year coverage"):
        pressure_layer.select_reference(
            reference,
            years=[2000, 2001],
            region="pnw_bartusek",
            threshold_variable="tas",
            quantile="90",
        )
    with pytest.raises(ValueError, match="matching times"):
        harmonize.replace_heat_budget(reference, raw.isel(time=slice(1, None)))


@pytest.mark.parametrize("threshold_variable", ["tas", "lwa_a"])
def test_cli_round_trip_hashes_manifest_and_no_overwrite(
    layer_inputs, tmp_path, threshold_variable
):
    raw, reference, _ = layer_inputs
    if threshold_variable == "lwa_a":
        hourly = reference.drop_dims("event")
        reference = xr.merge(
            [
                hourly,
                events.build_event_summary_table(
                    hourly, "lwa_a_event_id", peak_variable="lwa_a_region"
                ),
            ]
        )
        reference.attrs["threshold_variable"] = threshold_variable
    raw.to_netcdf(tmp_path / "heat_budget_2001.nc", engine="h5netcdf")
    ref_path = analysis_io.save_harmonized_timeseries(
        reference, tmp_path / "reference.nc"
    )
    manifest = {
        "production_start_year": 2001,
        "production_end_year": 2001,
        "annual_dir": str(tmp_path),
        "request": {
            "bbox": [40, 60, -130, -110],
            "zg_bottom": "pressure_level",
            "zg_bottom_pressure": 70000,
            "zg_top_pressure": 50000,
        },
        "git": {"dirty": False, "commit": "test-fixture"},
        "surface_behaviour": {"allow_bottom_overflow": False},
    }
    manifest_path = tmp_path / "production_run.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "output"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/build_stage1_pressure_layer.py"
    )
    command = [
        sys.executable,
        "-W",
        "error",
        str(script),
        "--reference",
        str(ref_path),
        "--reference-sha256",
        pressure_layer.sha256(ref_path),
        "--heat-budget-manifest",
        str(manifest_path),
        "--region",
        "pnw_bartusek",
        "--threshold-variable",
        threshold_variable,
        "--bottom-hpa",
        "700",
        "--top-hpa",
        "500",
        "--start-year",
        "2001",
        "--end-year",
        "2001",
        "--output-dir",
        str(output),
    ]
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    provenance = json.loads((output / "manifest.json").read_text())
    assert provenance["status"] == "validated"
    assert provenance["validation"]["events"] == 2
    saved_path = Path(provenance["output"]["path"])
    assert pressure_layer.sha256(saved_path) == provenance["output"]["sha256"]
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert "Refusing existing output directory" in refused.stderr
    assert pressure_layer.sha256(saved_path) == provenance["output"]["sha256"]


def test_manifest_requires_matching_region_and_bounds():
    manifest = {
        "request": {
            "bbox": [40, 60, -130, -110],
            "zg_bottom": "pressure_level",
            "zg_bottom_pressure": 70000,
            "zg_top_pressure": 50000,
        },
        "git": {"dirty": False},
        "surface_behaviour": {"allow_bottom_overflow": False},
    }
    pressure_layer.validate_manifest(
        manifest, region="pnw_bartusek", bottom=700, top=500
    )
    with pytest.raises(ValueError, match="pressure bounds"):
        pressure_layer.validate_manifest(
            manifest, region="pnw_bartusek", bottom=800, top=500
        )
    with pytest.raises(ValueError, match="region"):
        pressure_layer.validate_manifest(
            manifest, region="pnw_hotz", bottom=700, top=500
        )
