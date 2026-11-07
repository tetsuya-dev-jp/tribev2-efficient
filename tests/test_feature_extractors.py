# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model, VJEPA2VideoProcessor

from neuralset.extractors.base import BaseExtractor
from neuralset.extractors.image import HuggingFaceImage
from neuralset.extractors.video import HuggingFaceVideo, _VideoImage

from tribev2.feature_extractors import (
    EfficientVJEPA2Video,
    _read_processed_frame_window,
    _token_mean,
    _vjepa2_token_means,
)

VJEPA2_MODEL = "facebook/vjepa2-vitg-fpc64-256"
MODEL_IMAGE = {"model_name": VJEPA2_MODEL, "infra": {"keep_in_ram": False}}


class _RepoFreeImage(HuggingFaceImage):
    _skip_repo_check = True


def _tiny_model() -> VJEPA2Model:
    torch.manual_seed(0)
    config = VJEPA2Config(
        crop_size=32,
        frames_per_clip=4,
        tubelet_size=2,
        patch_size=16,
        hidden_size=24,
        num_attention_heads=4,
        num_hidden_layers=3,
        pred_hidden_size=12,
        pred_num_attention_heads=3,
        pred_num_hidden_layers=1,
        pred_num_mask_tokens=1,
        num_pooler_layers=1,
    )
    return VJEPA2Model(config).eval()


@pytest.mark.parametrize("cache_n_layers", [None, 2, 3, 5])
def test_token_means_match_captured_hidden_states(
    cache_n_layers: int | None,
) -> None:
    model = _tiny_model()
    pixel_values = torch.randn(2, 4, 3, 32, 32)

    with torch.inference_mode():
        outputs = model(
            pixel_values_videos=pixel_values,
            output_hidden_states=True,
            skip_predictor=True,
        )
        actual = _vjepa2_token_means(model.encoder, pixel_values, cache_n_layers)

    states = outputs.hidden_states
    assert states is not None
    assert len(states) == model.config.num_hidden_layers + 1

    image = _RepoFreeImage(
        model_name=VJEPA2_MODEL,
        device="cpu",
        cache_n_layers=cache_n_layers,
        token_aggregation="mean",
    )
    expected = torch.stack(
        [
            image._aggregate_tokens(torch.stack([state[idx] for state in states]))
            for idx in range(pixel_values.shape[0])
        ]
    )

    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_uses_last_hidden_state_before_final_layer_norm() -> None:
    model = _tiny_model()
    pixel_values = torch.randn(1, 4, 3, 32, 32)

    with torch.inference_mode():
        outputs = model(
            pixel_values_videos=pixel_values,
            output_hidden_states=True,
            skip_predictor=True,
        )
        actual = _vjepa2_token_means(model.encoder, pixel_values, 4)

    assert outputs.hidden_states is not None
    before_norm = outputs.hidden_states[-1].mean(dim=1)
    after_norm = outputs.last_hidden_state.mean(dim=1)
    torch.testing.assert_close(actual[:, -1], before_norm, rtol=1e-6, atol=1e-7)
    assert not torch.equal(before_norm, after_norm)


def test_rejects_non_vjepa2_model() -> None:
    with pytest.raises(ValueError, match="VJEPA2 checkpoint"):
        EfficientVJEPA2Video(
            frequency=2,
            image={"model_name": "facebook/dinov2-base", "infra": {"keep_in_ram": False}},
        )


def test_rejects_non_mean_token_aggregation() -> None:
    with pytest.raises(ValueError, match="token_aggregation='mean'"):
        EfficientVJEPA2Video(
            frequency=2,
            image={**MODEL_IMAGE, "token_aggregation": "max"},
        )


def test_rejects_layer_type() -> None:
    with pytest.raises(ValueError, match="layer_type"):
        EfficientVJEPA2Video(frequency=2, layer_type="mit", image=MODEL_IMAGE)


def test_has_own_registration_and_cache_namespace() -> None:
    extractor = EfficientVJEPA2Video(
        frequency=2,
        image={**MODEL_IMAGE, "cache_n_layers": 20},
    )

    assert extractor.image.cache_n_layers == 20
    assert extractor.compute_dtype == "float32"
    assert BaseExtractor._CLASSES["EfficientVJEPA2Video"] is EfficientVJEPA2Video

    parent_infra = HuggingFaceVideo.model_fields["infra"].default
    own_infra = EfficientVJEPA2Video.model_fields["infra"].default
    assert own_infra is not parent_infra
    assert own_infra.timeout_min == parent_infra.timeout_min
    assert own_infra.version == parent_infra.version
    assert own_infra._infra_method.method.__qualname__ == (
        "EfficientVJEPA2Video._get_data"
    )


class _FakeVideo:
    filename = "fake.mp4"

    def __init__(self, height: int = 2, width: int = 2) -> None:
        self.height = height
        self.width = width
        self.read_times: list[float] = []

    def get_frame(self, time: float) -> np.ndarray:
        self.read_times.append(time)
        rows, columns = np.mgrid[0 : self.height, 0 : self.width]
        base = int(round(float(time) * 1000)) % 256
        frame = np.stack(
            [
                (columns + base) % 256,
                (rows + base) % 256,
                (columns + rows + base) % 256,
            ],
            axis=-1,
        )
        return frame.astype(np.uint8)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, *, videos: list[np.ndarray], return_tensors: str) -> dict:
        assert return_tensors == "pt"
        self.calls.append(len(videos))
        self.shapes.append(np.asarray(videos[0]).shape)
        frames = torch.stack(
            [
                torch.as_tensor(np.asarray(video), dtype=torch.float32)
                for video in videos
            ]
        )
        return {"pixel_values_videos": frames[None]}


def test_processed_frame_window_clamps_and_preserves_order() -> None:
    video = _FakeVideo()
    processor = _FakeProcessor()

    pixels, cache = _read_processed_frame_window(
        video, 0.25, [1.0, 0.5, 0.0], {}, processor, None
    )

    assert video.read_times == [0.0, 0.25]
    assert processor.calls == [2]
    assert torch.equal(pixels[0, 0], pixels[0, 1])
    assert not torch.equal(pixels[0, 0], pixels[0, 2])
    assert set(cache) == {0.0, 0.25}
    assert torch.equal(cache[0.0], pixels[0, 0])
    assert torch.equal(cache[0.25], pixels[0, 2])


def test_processed_frame_window_reuses_overlap_and_evicts_old() -> None:
    video = _FakeVideo()
    processor = _FakeProcessor()
    _, first = _read_processed_frame_window(
        video, 0.25, [1.0, 0.5, 0.0], {}, processor, None
    )
    first_zero = first[0.0]
    first_quarter = first[0.25]

    pixels, second = _read_processed_frame_window(
        video, 0.75, [1.0, 0.5, 0.0], first, processor, None
    )

    assert video.read_times == [0.0, 0.25, 0.75]
    assert processor.calls == [2, 1]
    assert torch.equal(pixels[0, 0], first_zero)
    assert torch.equal(pixels[0, 1], first_quarter)
    assert set(second) == {0.0, 0.25, 0.75}

    pixels, third = _read_processed_frame_window(
        video, 1.75, [1.0, 0.5, 0.0], second, processor, None
    )

    assert video.read_times == [0.0, 0.25, 0.75, 1.25, 1.75]
    assert processor.calls == [2, 1, 2]
    assert set(third) == {0.75, 1.25, 1.75}
    assert 0.0 not in third
    assert 0.25 not in third
    assert torch.equal(pixels[0, 0], second[0.75])
    assert len(third) <= 64


def test_processed_frame_window_keeps_distinct_timestamps_with_same_rounded_key() -> None:
    video = _FakeVideo()
    processor = _FakeProcessor()

    _, cache = _read_processed_frame_window(
        video, 1.0002, [0.0001, 0.0], {}, processor, None
    )

    assert len(video.read_times) == 2
    assert processor.calls == [2]
    assert len(cache) == 2
    assert len({f"{timestamp:.3f}" for timestamp in cache}) == 1


def test_processed_frame_window_skips_processor_when_fully_cached() -> None:
    video = _FakeVideo()
    processor = _FakeProcessor()
    first_pixels, first = _read_processed_frame_window(
        video, 0.25, [1.0, 0.5, 0.0], {}, processor, None
    )

    pixels, second = _read_processed_frame_window(
        video, 0.25, [1.0, 0.5, 0.0], first, processor, None
    )

    assert processor.calls == [2]
    assert video.read_times == [0.0, 0.25]
    assert torch.equal(pixels, first_pixels)
    assert set(second) == set(first)


def test_processed_frame_window_resizes_before_processor() -> None:
    video = _FakeVideo(height=8, width=4)
    processor = _FakeProcessor()

    pixels, _ = _read_processed_frame_window(
        video, 0.25, [1.0, 0.5, 0.0], {}, processor, max_imsize=4
    )

    assert processor.shapes == [(4, 2, 3)]
    assert pixels.shape == (1, 3, 4, 2, 3)


def _full_processed(
    video: _FakeVideo,
    timestamps: list[float],
    processor: VJEPA2VideoProcessor,
) -> torch.Tensor:
    images = [
        _VideoImage(video=video, time=timestamp).read() for timestamp in timestamps
    ]
    data = np.array([np.array(image) for image in images])
    inputs = processor(videos=list(data), return_tensors="pt")
    return inputs["pixel_values_videos"]


def test_processed_frame_window_matches_full_vjepa2_preprocessing() -> None:
    processor = VJEPA2VideoProcessor(crop_size=32)
    video = _FakeVideo(height=40, width=32)
    num_frames = 64
    subtimes = [k / num_frames * 4.0 for k in reversed(range(num_frames))]
    cache: dict[float, torch.Tensor] = {}

    for time in (1.0, 1.5, 2.0):
        pixels, cache = _read_processed_frame_window(
            video, time, subtimes, cache, processor, None
        )
        timestamps = [float(max(0.0, time - offset)) for offset in subtimes]
        full = _full_processed(video, timestamps, processor)
        assert torch.equal(pixels, full)


def test_token_mean_accumulates_half_input_in_float32() -> None:
    hidden = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float16
    )

    actual = _token_mean(hidden)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, hidden.float().mean(dim=1))


def test_tiny_vjepa2_half_means_return_float32() -> None:
    model = _tiny_model().half()
    pixel_values = torch.randn(2, 4, 3, 32, 32, dtype=torch.float16)

    with torch.inference_mode():
        outputs = model(
            pixel_values_videos=pixel_values,
            output_hidden_states=True,
            skip_predictor=True,
        )
        actual = _vjepa2_token_means(model.encoder, pixel_values, None)

    assert outputs.hidden_states is not None
    expected = torch.stack(
        [state.float() for state in outputs.hidden_states], dim=1
    ).mean(dim=2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_rejects_float16_on_cpu() -> None:
    with pytest.raises(ValueError, match="requires a CUDA image device"):
        EfficientVJEPA2Video(
            frequency=2,
            compute_dtype="float16",
            image={**MODEL_IMAGE, "device": "cpu"},
        )


def test_compute_dtype_participates_in_cache_uid() -> None:
    image = {**MODEL_IMAGE, "cache_n_layers": 20, "device": "cuda"}
    fp32 = EfficientVJEPA2Video(frequency=2, compute_dtype="float32", image=image)
    fp16 = EfficientVJEPA2Video(frequency=2, compute_dtype="float16", image=image)

    assert fp32.infra.uid() != fp16.infra.uid()
    assert "compute_dtype=float16" in fp16.infra.uid()
