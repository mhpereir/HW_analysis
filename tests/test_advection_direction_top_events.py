import numpy as np
import pytest
import xarray as xr
from HW_analysis.src import advection_direction_top_events


def test_build_inputs_uses_filtered_absolute_events_for_rank_and_reference(
    monkeypatch,
):
    stage1 = _make_stage1()
    original = stage1.copy(deep=True)
    filtered = stage1.isel(event=[0, 2])
    plot_source = stage1.copy(deep=True)
    plot_source.attrs["data_representation"] = "source marker"
    reference = xr.Dataset(attrs={"n_events": 2})
    stacked = xr.Dataset(
        {
            "event_id": ("event", [3, 1]),
            "selection_rank": ("event", [1, 2]),
        },
        coords={"event": [3, 1]},
    )
    captured = {}

    def fake_season(source, months, *, require_full_event):
        captured["season"] = (source, months, require_full_event)
        return filtered

    def fake_reference(source, **kwargs):
        captured["reference"] = (source, kwargs)
        return reference

    def fake_stack(source, event_table, **kwargs):
        captured["stack"] = (source, event_table, kwargs)
        return stacked

    monkeypatch.setattr(
        advection_direction_top_events.selectors,
        "select_events_by_season",
        fake_season,
    )
    monkeypatch.setattr(
        advection_direction_top_events.composites,
        "all_event_peak_aligned_composite",
        fake_reference,
    )
    monkeypatch.setattr(
        advection_direction_top_events.composites,
        "stack_events_centered_on_peak",
        fake_stack,
    )

    out_reference, out_events = advection_direction_top_events.build_top_event_inputs(
        plot_source,
        stage1,
        data_representation="absolute",
        top_n=2,
        window_days=5,
        season_months=[6, 7, 8],
        require_full_event=True,
    )

    season_source, season_months, require_full_event = captured["season"]
    assert season_source is not stage1
    xr.testing.assert_identical(season_source, stage1)
    assert season_months == [6, 7, 8]
    assert require_full_event is True
    reference_source, reference_kwargs = captured["reference"]
    assert reference_source is plot_source
    assert reference_kwargs["event_table"] is filtered
    assert reference_kwargs["event_percentiles"] is None
    assert reference_kwargs["pre_days"] == 5
    assert reference_kwargs["post_days"] == 5
    stack_source, ranked_events, stack_kwargs = captured["stack"]
    assert stack_source is plot_source
    np.testing.assert_array_equal(ranked_events["event_id"].values, [3, 1])
    np.testing.assert_array_equal(ranked_events["selection_rank"].values, [1, 2])
    assert stack_kwargs["pre_days"] == 5
    assert stack_kwargs["post_days"] == 5
    assert out_reference is reference
    assert out_events is stacked
    assert out_reference.attrs["data_representation"] == "absolute"
    assert out_events.attrs["top_event_rank_metric"] == "tas_peak"
    assert out_events.attrs["top_event_reference_event_count"] == 2
    assert out_events.attrs["top_event_selected_count"] == 2
    xr.testing.assert_identical(stage1, original)


def test_build_inputs_rejects_unsupported_representation():
    stage1 = _make_stage1()

    with pytest.raises(ValueError, match="Unsupported.*representation"):
        advection_direction_top_events.build_top_event_inputs(
            stage1,
            stage1,
            data_representation="raw-ish",
        )


def test_build_inputs_rejects_different_face_sets():
    stage1 = _make_stage1()
    plot_source = stage1.drop_vars("advection_top")

    with pytest.raises(ValueError, match="missing required variables"):
        advection_direction_top_events.build_top_event_inputs(
            plot_source,
            stage1,
            data_representation="absolute",
        )


def test_select_reference_events_rejects_empty_season(monkeypatch):
    stage1 = _make_stage1()
    monkeypatch.setattr(
        advection_direction_top_events.selectors,
        "select_events_by_season",
        lambda *args, **kwargs: stage1.isel(event=[]),
    )

    with pytest.raises(ValueError, match="No events remain"):
        advection_direction_top_events.select_reference_events(
            stage1,
            season_months=[6, 7, 8],
            require_full_event=True,
        )


def test_build_inputs_extracts_ranked_peak_centered_raw_windows():
    stage1 = _make_realistic_stage1()
    original = stage1.copy(deep=True)

    reference, events = advection_direction_top_events.build_top_event_inputs(
        stage1,
        stage1,
        data_representation="absolute",
        top_n=2,
        window_days=7,
        season_months=[6, 7, 8],
        require_full_event=True,
    )

    assert reference.sizes["lag_hour"] == 337
    assert events.sizes == {"event": 2, "lag_hour": 337}
    assert reference.attrs["n_events"] == 3
    np.testing.assert_array_equal(events["event_id"].values, [20, 30])
    np.testing.assert_array_equal(events["selection_rank"].values, [1, 2])
    np.testing.assert_array_equal(events["lag_hour"].values, np.arange(-168, 169))
    assert events.attrs["data_representation"] == "absolute"
    xr.testing.assert_identical(stage1, original)


def _make_stage1() -> xr.Dataset:
    time = np.arange(
        np.datetime64("2000-06-01T00"),
        np.datetime64("2000-06-01T03"),
        np.timedelta64(1, "h"),
    )
    faces = {
        "advection_west": np.array([0.1, 0.2, 0.3]),
        "advection_east": np.array([-0.1, -0.1, -0.1]),
        "advection_south": np.array([0.05, 0.05, 0.05]),
        "advection_north": np.array([-0.02, -0.02, -0.02]),
        "advection_top": np.array([0.01, 0.01, 0.01]),
    }
    return xr.Dataset(
        {
            **{name: ("time", values) for name, values in faces.items()},
            "advection": ("time", sum(faces.values())),
            "event_id": ("event", [1, 2, 3]),
            "tas_peak": ("event", [300.0, 301.0, 305.0]),
            "peak_time": (
                "event",
                np.array(
                    [
                        "2000-06-01T00",
                        "2000-06-01T01",
                        "2000-06-01T02",
                    ],
                    dtype="datetime64[h]",
                ),
            ),
        },
        coords={"time": time, "event": [0, 1, 2]},
    )


def _make_realistic_stage1() -> xr.Dataset:
    time = np.arange(
        np.datetime64("2000-06-01T00"),
        np.datetime64("2000-09-01T00"),
        np.timedelta64(1, "h"),
    )
    phase = np.linspace(0, 8 * np.pi, time.size)
    faces = {
        "advection_west": 0.04 + 0.01 * np.sin(phase),
        "advection_east": -0.02 + 0.005 * np.cos(phase),
        "advection_south": 0.015 + 0.004 * np.sin(phase / 2),
        "advection_north": -0.01 + 0.003 * np.cos(phase / 2),
        "advection_top": -0.005 + 0.002 * np.sin(phase / 3),
    }
    peak_times = np.array(
        ["2000-06-15T12", "2000-07-15T12", "2000-08-15T12"],
        dtype="datetime64[h]",
    )
    return xr.Dataset(
        {
            **{name: ("time", values) for name, values in faces.items()},
            "advection": ("time", sum(faces.values())),
            "event_id": ("event", [10, 20, 30]),
            "tas_peak": ("event", [302.0, 307.0, 305.0]),
            "start_time": ("event", peak_times - np.timedelta64(1, "D")),
            "end_time": ("event", peak_times + np.timedelta64(1, "D")),
            "peak_time": ("event", peak_times),
        },
        coords={"time": time, "event": np.arange(3)},
        attrs={"region": "synthetic"},
    )
