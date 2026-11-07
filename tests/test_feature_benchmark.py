import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_features import compare_features, select_events


def test_feature_comparison_rejects_differences_and_nonfinite_values():
    reference = [np.array([0., 1.], dtype=np.float32)]
    assert compare_features(reference, reference)["passed"]
    assert not compare_features(reference, [reference[0] + .01])["passed"]
    assert not compare_features(reference, [np.array([np.nan, 1.], np.float32)])["passed"]
    with pytest.raises(ValueError, match="shape"):
        compare_features(reference, [np.zeros(3, np.float32)])


def test_video_selection_is_one_interval_per_movie():
    table = pd.DataFrame([
        dict(type="Video", run=str(i), filepath=f"movie-{i}.mp4",
             offset=0., duration=900., frequency=24.)
        for i in range(4)
    ])
    events, selection = select_events(table, "video")
    assert len(events) == len(selection) == 4
    assert [e.offset for e in events] == [120., 300., 480., 660.]
    assert all(e.duration == 4. and e.frequency == 24. for e in events)


def test_video_selection_validates_the_full_requested_duration():
    table = pd.DataFrame([
        dict(type="Video", run=str(i), filepath=f"movie-{i}.mp4",
             offset=[120., 300., 480., 660.][i], duration=60., frequency=24.)
        for i in range(4)
    ])
    events, selection = select_events(table, "video", video_duration=60)
    assert all(e.duration == 60 for e in events)
    assert all(row["duration_s"] == 60 for row in selection)
    table.loc[0, "duration"] = 4.
    with pytest.raises(ValueError, match="not covered"):
        select_events(table, "video", video_duration=60)


def test_video_selection_rejects_a_gap_in_event_coverage():
    table = pd.DataFrame([
        dict(type="Video", run=str(i), filepath=f"movie-{i}.mp4",
             offset=180. if i == 0 else 0., duration=900., frequency=24.)
        for i in range(4)
    ])
    with pytest.raises(ValueError, match="not covered"):
        select_events(table, "video")
