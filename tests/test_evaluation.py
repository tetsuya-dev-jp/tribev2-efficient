import numpy as np
import pytest
import torch

from benchmarks.evaluate_cached import brain_scores, errors


def test_fixed_lag_and_parcel_aggregation():
    labels = np.repeat([f"region-{index}" for index in range(360)], 2)
    rng = np.random.default_rng(7)
    bold = rng.normal(size=(20, len(labels)))
    predictions = np.zeros_like(bold)
    predictions[:-5] = bold[5:]
    correlation, mse, n_vertices = brain_scores(predictions, bold, labels)
    np.testing.assert_allclose(correlation, np.ones(360), atol=1e-14)
    assert mse == 0 and n_vertices == 720


def test_error_statistics_and_failed_gate():
    reference = torch.tensor([1.0, 2.0], dtype=torch.float64)
    candidate = reference + 1e-6
    maximum, squared_error, squared_norm = errors(reference, candidate)
    assert maximum == pytest.approx(1e-6)
    assert squared_error == pytest.approx(2e-12)
    assert squared_norm == 5
    with pytest.raises(AssertionError):
        errors(reference, reference + 1)
