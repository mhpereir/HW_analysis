import json

import pytest
from HW_analysis.scripts.idyn_matching_exploration import matching_settings
from HW_analysis.src import selectors


def test_default_matching_settings_are_valid_and_complete():
    settings = matching_settings.load_matching_settings()

    assert settings.schema_version == 2
    assert settings.group_variable == "I_dyn_pre"
    assert settings.reference_sign == "negative"
    assert settings.primary_specification == "peak_anomaly_0p20"
    assert settings.specification("peak_anomaly_0p20").caliper_sd == 0.2
    assert settings.family("peak_anomaly").variables == ("tas_anom_peak",)
    assert settings.method == selectors.SIGN_MATCH_METHOD
    assert settings.standardization == selectors.SIGN_MATCH_STANDARDIZATION
    assert settings.distance == selectors.SIGN_MATCH_DISTANCE
    assert settings.caliper_rule == selectors.SIGN_MATCH_CALIPER_RULE
    assert len(settings.sha256) == 64


def test_matching_settings_reject_legacy_runtime_sum_group(tmp_path):
    raw = json.loads(matching_settings.DEFAULT_SETTINGS_PATH.read_text())
    raw["group"] = {
        "name": "I_dyn",
        "operation": "sum",
        "variables": ["I_adiabatic_pre", "I_advection_pre"],
        "reference_sign": "negative",
    }
    path = tmp_path / "legacy_group.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="Invalid keys for group"):
        matching_settings.load_matching_settings(path)


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
