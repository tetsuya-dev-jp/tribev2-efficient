import math

import pytest
import torch

from benchmarks.benchmark_inference import compare_predictions


def test_identical_predictions_pass():
    reference = [torch.tensor([[1.0, -2.0], [3.0, 4.0]])]
    result = compare_predictions(reference, [reference[0].clone()], "same")
    assert result["allclose"]
    assert result["finite_elements"] == 4
    assert result["tolerance_failure_elements"] == 0
    assert result["max_abs_error"] == 0.0
    assert result["relative_l2"] == pytest.approx(0.0)
    assert result["centered_correlation"] == pytest.approx(1.0)


def test_empty_predictions_are_not_evidence_of_agreement():
    assert not compare_predictions([], [], "empty")["allclose"]


def test_failed_tolerance_returns_statistics():
    reference = [torch.ones(4, dtype=torch.float64)]
    candidate = [reference[0] + 0.1]
    result = compare_predictions(reference, candidate, "offset")
    assert not result["allclose"]
    assert result["tolerance_failure_elements"] == 4
    assert result["nonfinite_candidate_elements"] == 0
    assert result["max_abs_error"] == pytest.approx(0.1)
    assert result["relative_l2"] == pytest.approx(0.1)


def test_nonfinite_values_are_counted_separately():
    reference = [torch.tensor([1.0, 2.0, 3.0, 4.0])]
    candidate = [torch.tensor([1.0, 2.0, math.nan, math.inf])]
    result = compare_predictions(reference, candidate, "nonfinite")
    assert not result["allclose"]
    assert result["nonfinite_candidate_elements"] == 2
    assert result["nonfinite_reference_elements"] == 0
    assert result["tolerance_failure_elements"] == 0
    assert result["finite_elements"] == 2
    assert result["max_abs_error"] == 0.0


def test_global_centered_correlation_uses_all_batches():
    reference = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    candidate = [value * 2.0 + 1.0 for value in reference]
    result = compare_predictions(reference, candidate, "affine")
    assert result["centered_correlation"] == pytest.approx(1.0)


def test_shape_mismatch_reports_error():
    reference = [torch.zeros(2, 2)]
    candidate = [torch.zeros(2, 3)]
    result = compare_predictions(reference, candidate, "shape")
    assert not result["allclose"]
    assert "error" in result
