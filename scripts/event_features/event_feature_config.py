"""Configuration for fixed-window event feature extraction."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage1"
    / "harmonized_regional_timeseries_pnw_bartusek_tas_q90_1940_2024.nc"
)

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "event_features"
    / "hw_event_features_fixed_windows.nc"
)

DEFAULT_CSV_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "event_features"
    / "hw_event_features_fixed_windows.csv"
)

TIME_DIM = "time"
EVENT_DIM = "event"
PEAK_TIME_NAME = "peak_time"
EVENT_ID_NAME = "event_id"

# Fixed windows in hours relative to peak_time. Timestamp slices are inclusive.
WINDOWS = {
    "heat_budget_pre": (-96, 0),
    "lwa_pre_peak": (-96, 24),
    "antecedent_state": (-168, -24),
    "antecedent_change": (-168, 0),
    "near_peak": (-24, 24),
    "decay": (0, 72),
}

DEFAULT_INTEGRAL_FEATURES = {
    "dTdt": "heat_budget_pre",
    "advection": "heat_budget_pre",
    "adiabatic": "heat_budget_pre",
    "diabatic": "heat_budget_pre",
    "lwa_a_region": "lwa_pre_peak",
    "lwa_c_region": "lwa_pre_peak",
}

EXTENDED_INTEGRAL_FEATURES = {
    "nslr_heating_rate_approx": "heat_budget_pre",
    "nssr_heating_rate_approx": "heat_budget_pre",
    "sshf_heating_rate_approx": "heat_budget_pre",
    "slhf_heating_rate_approx": "heat_budget_pre",
    "surface_energy_heating_rate_approx": "heat_budget_pre",
}

DEFAULT_MEAN_FEATURES = {
    "tas_anom": "antecedent_state",
}

EXTENDED_MEAN_FEATURES = {
    "soil_moisture": "antecedent_state",
    "cloud_cover": "antecedent_state",
    "pbl_p_mean": "antecedent_state",
}

EXTENDED_CHANGE_FEATURES = {
    "soil_moisture": "antecedent_change",
}

EVENT_SUMMARY_FEATURES = (
    "event_id",
    "start_time",
    "end_time",
    "peak_time",
    "duration",
    "tas_peak",
    "tas_anom_peak",
    "tas_excess_peak",
    "tas_excess_integral",
    "lwa_a_peak",
    "lwa_c_peak",
)

DEFAULT_FEATURE_NAMES = {
    "dTdt": "I_dTdt_pre",
    "advection": "I_advection_pre",
    "adiabatic": "I_adiabatic_pre",
    "diabatic": "I_diabatic_pre",
    "lwa_a_region": "I_lwa_a_pre_peak",
    "lwa_c_region": "I_lwa_c_pre_peak",
    "tas_anom": "T_anom_mean_ant",
}

EXTENDED_FEATURE_NAMES = {
    "soil_moisture": "soil_moisture_mean_ant",
    "cloud_cover": "cloud_cover_mean_ant",
    "pbl_p_mean": "pbl_p_mean_ant",
    "nslr_heating_rate_approx": "I_nslr_pre",
    "nssr_heating_rate_approx": "I_nssr_pre",
    "sshf_heating_rate_approx": "I_sshf_pre",
    "slhf_heating_rate_approx": "I_slhf_pre",
    "surface_energy_heating_rate_approx": "I_surface_energy_pre",
    "soil_moisture_change": "soil_moisture_change",
}

SOLSTICE_MONTH = 6
SOLSTICE_DAY = 21

INTEGRAL_METHOD = "hourly_sum_assuming_1h_spacing"
FEATURE_METHOD = "fixed_windows_relative_to_peak_time"
PIPELINE_STAGE = "stage_2_event_features"
