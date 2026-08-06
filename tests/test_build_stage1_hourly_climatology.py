import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts import build_stage1_hourly_climatology as builder
from HW_analysis.src import climatology, data_io


def test_finalize_args_builds_standard_input_and_output_paths():
    args = argparse.Namespace(
        region="pnw_bartusek",
        bottom_boundary="surface",
        top_boundary="700",
        threshold_variable="tas",
        quantile="90",
        start_year=1940,
        end_year=2024,
        input_path=None,
        output_path=None,
        overwrite=False,
    )

    out = builder.finalize_args(args)

    assert out.input_path.name == (
        "harmonized_regional_timeseries_"
        "pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc"
    )
    assert out.output_path.name == (
        "regional_hourly_climatology_"
        "pnw_bartusek_surface_700hPa_1940_2024.nc"
    )


def test_require_canonical_source_rejects_contract_version_1():
    ds = _canonical_source().assign_attrs(stage1_contract_version=1)

    with pytest.raises(ValueError, match="contract version 2"):
        builder.require_canonical_climatology_source(ds)


def test_require_canonical_source_rejects_legacy_cloud_cover():
    ds = _canonical_source().assign_attrs(cloud_cover_source_layout="legacy-regional")

    with pytest.raises(ValueError, match="global-hourly-grid"):
        builder.require_canonical_climatology_source(ds)


def test_require_canonical_source_accepts_all_standard_variables():
    builder.require_canonical_climatology_source(_canonical_source())


def test_climatology_variables_include_optional_bottom_face():
    ds = _canonical_source()
    ds["advection_bottom"] = ("time", [1.0])

    variables = builder.climatology_variables_for_source(ds)

    assert variables[-1] == "advection_bottom"


def test_file_sha256_is_stable(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"stage-1 climatology")

    assert builder.file_sha256(path) == (
        "66de80e09c49037aa007d6e5662de951266d5d6f654e18f02d9b4e0361703165"
    )


def _canonical_source() -> xr.Dataset:
    time = np.array(["2000-05-01T00:00"], dtype="datetime64[h]")
    ds = xr.Dataset(
        {
            name: ("time", np.array([float(index)]))
            for index, name in enumerate(climatology.DEFAULT_VARIABLES)
        },
        coords={"time": time},
        attrs={
            "stage1_contract_version": 2,
            "cloud_cover_source_layout": data_io.CLOUD_COVER_LAYOUT_GLOBAL,
        },
    )
    return ds
