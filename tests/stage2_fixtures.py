"""Small explicit climatology companions for Stage-2 tests."""

from src.climatology import build_regional_hourly_climatology


def stage1_attrs():
    return {
        "region": "synthetic",
        "stage1_contract_version": 2,
        "heat_budget_bottom_boundary": "surface",
        "heat_budget_top_boundary": "700hPa",
    }


def climatology_for(ds):
    return build_regional_hourly_climatology(ds, variables=["T_mean"])
