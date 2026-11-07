# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import typing as tp
import weakref

import pytest
import torch
from torch.nn import functional as F

from neuralset.base import Frequency
from neuralset.events import etypes as evts
from neuralset.extractors.audio import Wav2VecBert
from neuralset.extractors.base import BaseExtractor

from tribev2.audio_features import (
    SUPPORTED_MODEL,
    EfficientWav2VecBert,
    _stack_interpolated_hidden_states,
)

N_LAYERS = 25
N_DIMS = 4
INPUT_FREQUENCY = 16_000
OUTPUT_FREQUENCY = 2.0
DURATION = 1.3


class _FakeFeatureExtractor:
    sampling_rate = INPUT_FREQUENCY

    def __call__(self, wav, **kwargs):
        return {"input_values": wav.unsqueeze(0)}


class _FakeModel(torch.nn.Module):
    def __init__(self, states):
        super().__init__()
        self.states = states

    def forward(self, input_values, output_hidden_states=False):
        assert output_hidden_states is True
        return types.SimpleNamespace(hidden_states=self.states)


class _FakeAudio(evts.Audio):
    wav: tp.Any

    def _read(self):
        return self.wav


def make_extractor(*, cache_n_layers=20):
    return EfficientWav2VecBert(
        model_name=SUPPORTED_MODEL,
        frequency=OUTPUT_FREQUENCY,
        cache_n_layers=cache_n_layers,
        layers=[0.5, 0.75, 1.0],
        layer_aggregation="group_mean",
        device="cpu",
    )


def make_states(native_timepoints=7):
    torch.manual_seed(0)
    return tuple(
        torch.randn(1, native_timepoints, N_DIMS) for _ in range(N_LAYERS)
    )


def make_event(wav):
    return _FakeAudio(
        filepath="fake:audio",
        start=0.0,
        timeline="test",
        duration=DURATION,
        frequency=INPUT_FREQUENCY,
        wav=wav,
    )


def run_get_data(extractor, events):
    method = EfficientWav2VecBert.model_fields["infra"].default._infra_method.method
    return list(method(extractor, events))


@pytest.mark.parametrize(
    ("native_timepoints", "target_timepoints"),
    [(7, 3), (3, 7), (5, 5)],
)
def test_helper_matches_parent_cpu_interpolation(
    native_timepoints, target_timepoints
):
    states = make_states(native_timepoints)
    reference = torch.stack(states).squeeze(1).detach().cpu().transpose(-1, -2)
    if target_timepoints != native_timepoints:
        reference = F.interpolate(reference, target_timepoints)

    actual = _stack_interpolated_hidden_states(states, target_timepoints)

    assert actual.dtype == torch.float32
    assert actual.shape == (N_LAYERS, N_DIMS, target_timepoints)
    assert torch.equal(actual, reference)


def test_get_data_preserves_all_layers_with_cache_n_layers_20():
    extractor = make_extractor(cache_n_layers=20)
    extractor._feature_extractor = _FakeFeatureExtractor()
    extractor._model = _FakeModel(make_states())
    event = make_event(torch.randn(INPUT_FREQUENCY, 1))

    tarrays = run_get_data(extractor, [event])

    assert len(tarrays) == 1
    tarray = tarrays[0]
    assert tarray.data.shape == (
        N_LAYERS,
        N_DIMS,
        Frequency(OUTPUT_FREQUENCY).to_ind(DURATION),
    )
    assert tarray.frequency == OUTPUT_FREQUENCY
    assert tarray.duration == Frequency(OUTPUT_FREQUENCY).to_sec(
        Frequency(OUTPUT_FREQUENCY).to_ind(DURATION)
    )


def test_rejects_non_audio_event():
    extractor = make_extractor()
    extractor._feature_extractor = _FakeFeatureExtractor()
    extractor._model = _FakeModel(make_states())

    with pytest.raises(ValueError, match="only supports Audio"):
        run_get_data(extractor, [object()])


def test_hidden_states_released_before_next_forward():
    class EphemeralModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.previous_states = []

        def forward(self, input_values, output_hidden_states=False):
            assert all(ref() is None for ref in self.previous_states)
            states = make_states()
            self.previous_states = [weakref.ref(state) for state in states]
            return types.SimpleNamespace(hidden_states=states)

    extractor = make_extractor()
    extractor._feature_extractor = _FakeFeatureExtractor()
    extractor._model = EphemeralModel()
    event = make_event(torch.randn(INPUT_FREQUENCY, 1))
    arrays = run_get_data(extractor, [event, event])
    expected = _stack_interpolated_hidden_states(make_states(), Frequency(2).to_ind(DURATION))
    assert len(arrays) == 2
    for array in arrays:
        assert torch.equal(torch.from_numpy(array.data), expected)


def test_has_own_registration_and_cache_namespace():
    extractor = make_extractor(cache_n_layers=20)

    assert extractor.cache_n_layers == 20
    assert BaseExtractor._CLASSES["EfficientWav2VecBert"] is EfficientWav2VecBert

    parent_infra = Wav2VecBert.model_fields["infra"].default
    own_infra = EfficientWav2VecBert.model_fields["infra"].default
    assert own_infra is not parent_infra
    assert own_infra.timeout_min == parent_infra.timeout_min
    assert own_infra.gpus_per_node == parent_infra.gpus_per_node
    assert own_infra.cpus_per_task == parent_infra.cpus_per_task
    assert own_infra.min_samples_per_job == parent_infra.min_samples_per_job
    assert own_infra.version == parent_infra.version
    assert own_infra._infra_method.method.__qualname__ == (
        "EfficientWav2VecBert._get_data"
    )


def test_rejects_unsupported_model():
    with pytest.raises(ValueError, match="only supports"):
        EfficientWav2VecBert(
            model_name="facebook/wav2vec2-large-xlsr-53",
            frequency=OUTPUT_FREQUENCY,
            cache_n_layers=20,
            device="cpu",
        )


def test_rejects_convolutional_layers():
    with pytest.raises(ValueError, match="layer_type='transformer'"):
        EfficientWav2VecBert(
            frequency=OUTPUT_FREQUENCY,
            cache_n_layers=20,
            layer_type="convolution",
            device="cpu",
        )


def test_rejects_missing_layer_cache():
    with pytest.raises(ValueError, match="cache_n_layers"):
        EfficientWav2VecBert(
            frequency=OUTPUT_FREQUENCY,
            cache_n_layers=None,
            device="cpu",
        )


@pytest.mark.parametrize("frequency", ["native", 0.0, -1.0])
def test_rejects_native_or_non_positive_frequency(frequency):
    with pytest.raises(ValueError, match="positive numeric frequency"):
        EfficientWav2VecBert(
            frequency=frequency,
            cache_n_layers=20,
            device="cpu",
        )


def test_rejects_non_audio_event_types():
    with pytest.raises(ValueError, match="event_types='Audio'"):
        EfficientWav2VecBert(
            event_types="Video",
            frequency=OUTPUT_FREQUENCY,
            cache_n_layers=20,
            device="cpu",
        )
