from copy import deepcopy

import pytest

from benchmarks.render_features_report import build_summary, verify_video_pair


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
