import numpy as np
import xarray as xr

from src import composites


def test_all_event_peak_aligned_composite_uses_dataset_event_table_by_default():
    ds = _make_composite_dataset()

    composite = composites.all_event_peak_aligned_composite(
        ds,
        variables=("T_mean", "volume"),
        pre_days=0,
        post_days=0,
    )

    assert composite.sizes["lag_hour"] == 1
    np.testing.assert_allclose(composite["T_mean"].values, [282.0])
    np.testing.assert_allclose(composite["volume"].values, [12.0])
    assert composite["event_percentile_T_mean"].dims == ("quantile", "lag_hour")
    np.testing.assert_allclose(
        composite["event_percentile_T_mean"].values[:, 0],
        [281.1, 282.0, 282.9],
    )
    assert composite.attrs["composite_reduction"] == "mean over all HW events"


def test_all_event_peak_aligned_composite_can_disable_event_percentile_bands():
    ds = _make_composite_dataset()

    composite = composites.all_event_peak_aligned_composite(
        ds,
        variables=("T_mean", "volume"),
        pre_days=0,
        post_days=0,
        event_percentiles=None,
    )

    assert "event_percentile_T_mean" not in composite
    assert composite.sizes == {"lag_hour": 1}
    np.testing.assert_allclose(composite["T_mean"].values, [282.0])


def test_event_percentile_envelope_ignores_event_only_metadata():
    stacked = composites.stack_events_centered_on_peak(
        _make_composite_dataset(),
        _make_composite_dataset(),
        variables=("T_mean", "volume"),
        pre_days=0,
        post_days=0,
    )

    envelope = composites.event_percentile_envelope(stacked)

    assert "T_mean" in envelope
    assert "event_id" not in envelope
    assert "peak_time" not in envelope
    assert envelope["T_mean"].dims == ("quantile", "lag_hour")


def _make_composite_dataset() -> xr.Dataset:
    time = np.array(
        [
            "2000-05-01T00:00",
            "2000-05-01T01:00",
            "2000-05-01T02:00",
            "2000-05-01T03:00",
        ],
        dtype="datetime64[h]",
    )
    event = np.arange(2)
    return xr.Dataset(
        data_vars={
            "T_mean": ("time", np.array([280.0, 281.0, 282.0, 283.0])),
            "volume": ("time", np.array([10.0, 11.0, 12.0, 13.0])),
            "event_id": ("event", np.array([1, 2], dtype=np.int64)),
            "peak_time": (
                "event",
                np.array(
                    ["2000-05-01T01:00", "2000-05-01T03:00"],
                    dtype="datetime64[h]",
                ),
            ),
        },
        coords={"time": time, "event": event},
    )
