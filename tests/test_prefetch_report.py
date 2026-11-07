import pytest

from benchmarks.render_features_report import verify_prefetch


@pytest.fixture
def result():
    runs = []
    order = [(0, "baseline"), (0, "candidate"), (1, "candidate"),
             (1, "baseline"), (2, "baseline"), (2, "candidate")]
    for repeat, condition in order:
        runs.append({"repeat": repeat, "condition": condition,
                     "wall_s": 10. if condition == "baseline" else 9.,
                     "peak_allocated_bytes": 1024,
                     "shapes": [[20, 1408, 120]], "output_dtype": "float32",
                     "output_timing": [{"frequency": 2., "duration": 60.}]})
    return {"status": "measured", "accepted": True, "num_frames": 64,
            "model": "facebook/vjepa2-vitg-fpc64-256", "model_dtype": "torch.float16",
            "baseline_mode": "serial_deferred_cpu_reads",
            "candidate_mode": "production_single_window_prefetch",
            "cache_bypassed": True, "model_load_in_timing": True,
            "selection": [{"movie_ordinal": 0, "offset_s": 120., "duration_s": 60.}],
            "runs": runs, "median_times_s": {"baseline": 10., "candidate": 9.},
            "ratio_of_median_times": .9,
            "comparisons": [{"repeat": i, "bitwise_equal": True, "passed": True,
                             "finite": True, "compared_elements": 3379200,
                             "failed_elements": 0, "max_abs": 0.} for i in range(3)]}


def test_prefetch_report_accepts_complete_alternating_pairs(result):
    groups = verify_prefetch(result)
    assert len(groups["baseline"]) == len(groups["candidate"]) == 3


def test_prefetch_report_rejects_partial_pairs(result):
    result["runs"].pop()
    with pytest.raises(AssertionError):
        verify_prefetch(result)


def test_prefetch_report_rejects_nonidentical_features(result):
    result["comparisons"][1]["bitwise_equal"] = False
    with pytest.raises(AssertionError, match="bitwise"):
        verify_prefetch(result)


def test_prefetch_report_rejects_false_acceptance(result):
    for row in result["runs"]:
        if row["condition"] == "candidate":
            row["wall_s"] = 9.8
    result["median_times_s"]["candidate"] = 9.8
    result["ratio_of_median_times"] = .98
    with pytest.raises(AssertionError, match="ratio exceeds"):
        verify_prefetch(result)


def test_prefetch_report_checks_actual_order(result):
    result["runs"][2], result["runs"][3] = result["runs"][3], result["runs"][2]
    with pytest.raises(AssertionError):
        verify_prefetch(result)
