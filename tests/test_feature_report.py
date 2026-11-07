from copy import deepcopy

import pytest

from benchmarks.render_features_report import GATES, build_summary, verify_cohort, verify_video_pair


def test_half_precision_summary_uses_original_fp32_denominator():
    baseline = [{"wall_s": 20, "peak_allocated_bytes": 8 * 2**30},
                {"wall_s": 22, "peak_allocated_bytes": 8 * 2**30}]
    optimized = [{"wall_s": 10, "peak_allocated_bytes": 4 * 2**30},
                 {"wall_s": 12, "peak_allocated_bytes": 4 * 2**30}]
    half = [{"wall_s": 2, "peak_allocated_bytes": 2 * 2**30},
            {"wall_s": 2.2, "peak_allocated_bytes": 2 * 2**30}]
    paired = {"baseline": baseline, "optimized": optimized}
    _, original_s, candidate_s, ratio, original_gib, candidate_gib = build_summary(
        paired, half, paired, paired)[1]
    assert original_s == 21 and candidate_s == pytest.approx(2.1)
    assert ratio == pytest.approx(.1)
    assert original_gib == 8 and candidate_gib == 2


def test_video_ratio_rejects_different_sampling_or_output_grid():
    original = {"model": "facebook/vjepa2-vitg-fpc64-256", "selection": [120, 300, 480, 660],
                "torch": "test", "hip": "test", "transformers": "test",
                "neuralset": "test", "device": "test", "runs": [{
                    "shapes": [[20, 1408, 8]] * 4, "output_dtype": "float32",
                    "output_timing": [{"frequency": 2.0, "duration": 4.0}] * 4}]}
    half = deepcopy(original)
    verify_video_pair(original, half)
    half["selection"][0] = 121
    with pytest.raises(AssertionError, match="selection"):
        verify_video_pair(original, half)
    half = deepcopy(original)
    half["runs"][0]["output_timing"][0]["frequency"] = 1.0
    with pytest.raises(AssertionError):
        verify_video_pair(original, half)


def test_cohort_report_rejects_partial_counts_and_hidden_failed_subjects():
    runs = ["MOVIE1_CC1", "MOVIE2_HO1", "MOVIE3_CC2", "MOVIE4_HO2"]
    data = {"status": "passed", "gates": GATES, "eval_lag": -5, "frames_per_clip": 64,
            "n_subjects": 168, "n_movies": 4, "n_comparisons": 672, "expected_comparisons": 672,
            "quantile_probabilities": [0,.05,.25,.5,.75,.95,1], "n_failed_comparisons": 0,
            "runs": [{"run": run, "n_subjects": 168, "n_failed_subjects": 0,
                      "subject_mean_r_delta_quantiles": [0] * 7, "mean_parcel_r_delta": 0,
                      "baseline_mean_parcel_r": .2, "candidate_mean_parcel_r": .2,
                      "max_abs_subject_mean_r_delta": 0, "max_abs_parcel_r_delta": 0,
                      "max_abs_relative_mse_change": 0} for run in runs]}
    verify_cohort(data)
    partial = deepcopy(data)
    partial["n_comparisons"] = 671
    with pytest.raises(AssertionError):
        verify_cohort(partial)
    data["runs"][0]["max_abs_parcel_r_delta"] = .002
    with pytest.raises(AssertionError):
        verify_cohort(data)
