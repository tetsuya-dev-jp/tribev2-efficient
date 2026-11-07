import pytest
import torch
from neuraltrain.models.common import SubjectLayers
from neuralset.dataloader import SegmentData

from tribev2.model import FmriEncoder


def make_model(
    *,
    bias: bool = True,
    low_rank_head: int | None = 6,
    average_subjects: bool = True,
    n_subjects: int = 0,
    subject_dropout: float | None = 0.1,
    hidden: int = 8,
    n_outputs: int = 5,
    n_output_timesteps: int = 4,
):
    subject_layers = SubjectLayers(
        n_subjects=n_subjects,
        bias=bias,
        subject_dropout=subject_dropout if average_subjects else None,
        average_subjects=average_subjects,
    )
    config = FmriEncoder(
        encoder=None,
        linear_baseline=True,
        hidden=hidden,
        subject_layers=subject_layers,
        low_rank_head=low_rank_head,
    )
    return config.build(
        feature_dims={"text": (2, 3)},
        n_outputs=n_outputs,
        n_output_timesteps=n_output_timesteps,
    )


def make_batch(batch_size: int = 2, n_times: int = 11) -> SegmentData:
    return SegmentData(
        data={"text": torch.randn(batch_size, 2, 3, n_times)},
        segments=[object()] * batch_size,
    )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("pool_outputs", [False, True])
@pytest.mark.parametrize(
    ("n_times", "n_output_timesteps"),
    [(11, 4), (4, 4), (3, 5)],
)
def test_optimized_output_matches_legacy(
    bias: bool, pool_outputs: bool, n_times: int, n_output_timesteps: int
) -> None:
    torch.manual_seed(0)
    model = make_model(
        bias=bias, n_output_timesteps=n_output_timesteps
    ).eval()
    batch = make_batch(n_times=n_times)

    with torch.no_grad():
        legacy = model(batch, pool_outputs=pool_outputs).clone()
        model.optimize_for_inference()
        optimized = model(batch, pool_outputs=pool_outputs)

    expected_timesteps = n_output_timesteps if pool_outputs else n_times
    assert optimized.shape == (2, 5, expected_timesteps)
    torch.testing.assert_close(optimized, legacy, rtol=1e-5, atol=1e-6)


def test_no_low_rank_head_selects_and_clones_shared_row() -> None:
    model = make_model(low_rank_head=None, n_subjects=2).eval()
    batch = make_batch()
    assert model.predictor.weights.shape == (3, 8, 5)

    expected_weights = model.predictor.weights[2].clone()
    expected_bias = model.predictor.bias[2].clone()
    with torch.no_grad():
        legacy = model(batch).clone()
        model.optimize_for_inference()
        optimized = model(batch)

    weights = model.predictor.weights
    assert weights.shape == (1, 8, 5)
    assert weights.untyped_storage().nbytes() == (
        weights.numel() * weights.element_size()
    )
    assert model.predictor.n_subjects == 0
    torch.testing.assert_close(weights[0], expected_weights)
    torch.testing.assert_close(model.predictor.bias[0], expected_bias)
    torch.testing.assert_close(optimized, legacy, rtol=1e-5, atol=1e-6)


def test_original_state_dict_strict_roundtrip() -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    state_dict = {key: value.clone() for key, value in model.state_dict().items()}

    restored = make_model().eval()
    restored.load_state_dict(state_dict, strict=True)

    batch = make_batch()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch), model(batch))


def test_original_mode_still_supports_backprop() -> None:
    model = make_model()
    model.train()

    out = model(make_batch())
    out.square().mean().backward()

    assert model.low_rank_head.weight.grad is not None
    assert model.predictor.weights.grad is not None
    assert model.projectors["text"].weight.grad is not None


def test_optimize_fuses_low_rank_head() -> None:
    model = make_model().eval()
    assert model.low_rank_head.weight.shape == (6, 8)
    assert model.predictor.weights.shape == (1, 6, 5)

    model.optimize_for_inference()

    assert isinstance(model.low_rank_head, torch.nn.Identity)
    assert model.predictor.weights.shape == (1, 8, 5)
    assert model.predictor.n_subjects == 0
    assert not model.predictor.weights.requires_grad
    assert "low_rank_head.weight" not in model.state_dict()


def test_state_dict_is_unchanged_until_optimize_is_called() -> None:
    model = make_model().eval()
    keys_before = tuple(model.state_dict())

    with torch.no_grad():
        model(make_batch())

    assert tuple(model.state_dict()) == keys_before
    assert "low_rank_head.weight" in keys_before
    assert model.predictor.weights.shape == (1, 6, 5)


def test_repeated_optimization_is_a_noop() -> None:
    model = make_model().eval()
    batch = make_batch()

    model.optimize_for_inference()
    with torch.no_grad():
        first = model(batch).clone()

    assert model.optimize_for_inference() is model
    with torch.no_grad():
        second = model(batch)

    assert model.predictor.weights.shape == (1, 8, 5)
    torch.testing.assert_close(second, first)


def test_optimize_requires_eval_mode() -> None:
    model = make_model()
    assert model.training

    with pytest.raises(RuntimeError, match="eval mode"):
        model.optimize_for_inference()


def test_optimize_rejects_non_average_subject_head() -> None:
    model = make_model(average_subjects=False, n_subjects=2).eval()

    with pytest.raises(RuntimeError, match="average-subject"):
        model.optimize_for_inference()


def test_optimize_rejects_average_head_without_subject_dropout() -> None:
    model = make_model(subject_dropout=None).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}

    with pytest.raises(RuntimeError, match="subject_dropout"):
        model.optimize_for_inference()

    after = model.state_dict()
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert after[key].shape == value.shape
        torch.testing.assert_close(after[key], value)


def test_optimized_model_rejects_training() -> None:
    model = make_model().eval()
    model.optimize_for_inference()

    with pytest.raises(RuntimeError, match="Training is not supported"):
        model.train()
