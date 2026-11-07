from types import SimpleNamespace

from benchmarks.evaluate_video_precision import overlapping, score_passes


def test_alignment_selects_events_crossing_window_boundaries():
    events = [SimpleNamespace(start=0, stop=60), SimpleNamespace(start=60, stop=120),
              SimpleNamespace(start=120, stop=180), SimpleNamespace(start=180, stop=240)]
    assert overlapping(events, 100, 100) == events[1:]
    assert overlapping(events, 60, 60) == events[1:2]


def test_all_accuracy_margins_are_required():
    assert score_passes(1e-4, 1e-3, -1e-3)
    assert not score_passes(-1.01e-4, 0, 0)
    assert not score_passes(0, 1.01e-3, 0)
    assert not score_passes(0, 0, 1.01e-3)
    assert not score_passes(float("nan"), 0, 0)
