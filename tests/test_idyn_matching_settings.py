import json

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.Idyn_matching_exploration import matching_settings
from HW_analysis.src import selectors


def test_default_matching_settings_are_valid_and_complete():
    settings = matching_settings.load_matching_settings()

    assert settings.schema_version == 1
    assert settings.group_variables == ("I_adiabatic_pre", "I_advection_pre")
    assert settings.reference_sign == "negative"
    assert settings.primary_specification == "peak_anomaly_0p20"
    assert settings.specification("peak_anomaly_0p20").caliper_sd == 0.2
    assert settings.family("peak_anomaly").variables == ("tas_anom_peak",)
    assert settings.method == selectors.SIGN_MATCH_METHOD
    assert settings.standardization == selectors.SIGN_MATCH_STANDARDIZATION
    assert settings.distance == selectors.SIGN_MATCH_DISTANCE
    assert settings.caliper_rule == selectors.SIGN_MATCH_CALIPER_RULE
    assert len(settings.sha256) == 64


def test_derive_group_metric_does_not_mutate_stage2_table():
    settings = matching_settings.load_matching_settings()
    event_table = xr.Dataset(
        data_vars={
            "I_adiabatic_pre": ("event", [1.0, -2.0]),
            "I_advection_pre": ("event", [0.5, 0.25]),
        },
        coords={"event": [0, 1]},
    )

    result = matching_settings.derive_group_metric(event_table, settings)

    assert "I_dyn" not in event_table
    assert result.name == "I_dyn"
    np.testing.assert_allclose(result.values, [1.5, -1.75])
    assert result.attrs["source_variables"] == "I_adiabatic_pre,I_advection_pre"


def test_matching_settings_reject_unknown_plot_specification(tmp_path):
    raw = json.loads(matching_settings.DEFAULT_SETTINGS_PATH.read_text())
    raw["plots"]["primary_specification"] = "missing_specification"
    path = tmp_path / "invalid_settings.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="undefined identifier"):
        matching_settings.load_matching_settings(path)


def test_matching_settings_reject_nonpositive_caliper(tmp_path):
    raw = json.loads(matching_settings.DEFAULT_SETTINGS_PATH.read_text())
    raw["specifications"]["peak_anomaly_0p20"]["caliper_sd"] = 0
    path = tmp_path / "invalid_caliper.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="finite positive"):
        matching_settings.load_matching_settings(path)
