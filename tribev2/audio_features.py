# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Wav2Vec2-BERT audio features with downsampling before the CPU transfer."""

import typing as tp

import torch
from exca import MapInfra
from torch.nn import functional as F

from neuralset import base as nsbase
from neuralset.events import etypes as evts
from neuralset.extractors.audio import Wav2VecBert

SUPPORTED_MODEL = "facebook/w2v-bert-2.0"


def _stack_interpolated_hidden_states(
    states: tp.Sequence[torch.Tensor], timepoints: int
) -> torch.Tensor:
    """Interpolate each hidden state to timepoints, then stack the layers.

    Every state has shape ``(1, tokens, dim)``. Interpolating the
    ``(dim, tokens)`` layout keeps the full native sequence on the model
    device and transfers only the requested timepoints to the CPU.
    """
    layers = [
        F.interpolate(state.transpose(1, 2), timepoints)[0] for state in states
    ]
    return torch.stack(layers).detach().cpu()


class EfficientWav2VecBert(Wav2VecBert):
    """Wav2Vec2-BERT variant that downsamples hidden states before CPU transfer.

    The model, waveform reading, resampling, normalization, feature extraction,
    hidden-state computation, and inherited layer aggregation are unchanged.
    This variant only requires a numeric output frequency and per-layer caching
    so that the all-layer cache is preserved for the standard aggregation step.
    """

    model_name: str = SUPPORTED_MODEL

    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=4096,
        version="v5",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        if self.model_name != SUPPORTED_MODEL:
            msg = (
                "EfficientWav2VecBert only supports "
                f"{SUPPORTED_MODEL!r}, got {self.model_name!r}"
            )
            raise ValueError(msg)
        if self.layer_type != "transformer":
            msg = (
                "EfficientWav2VecBert only supports layer_type='transformer', "
                f"got {self.layer_type!r}"
            )
            raise ValueError(msg)
        if self.cache_n_layers is None and not self.cache_all_layers:
            msg = (
                "EfficientWav2VecBert requires per-layer caching through "
                "cache_n_layers (cache_all_layers is deprecated in neuralset 0.0.2)"
            )
            raise ValueError(msg)
        if self.frequency == "native" or float(self.frequency) <= 0.0:
            msg = (
                "EfficientWav2VecBert requires a positive numeric frequency, "
                f"got {self.frequency!r}"
            )
            raise ValueError(msg)
        if self.event_types not in ("Audio", ("Audio",)):
            msg = (
                "EfficientWav2VecBert only supports event_types='Audio', "
                f"got {self.event_types!r}"
            )
            raise ValueError(msg)
        super().model_post_init(log__)

    @infra.apply(
        item_uid=lambda event: f"{event.study_relative_path()}_{event.offset:.2f}_{event.duration:.2f}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: list[evts.Audio]) -> tp.Iterator[nsbase.TimedArray]:
        if len(events) > 1:
            from tqdm import tqdm

            events = tqdm(events, desc="Computing audio embeddings")  # type: ignore

        for event in events:
            if not isinstance(event, evts.Audio):
                msg = (
                    "EfficientWav2VecBert only supports Audio events, "
                    f"got {type(event).__name__}"
                )
                raise ValueError(msg)
            wav = event.read()
            wav = self._resample_wav(wav, event.frequency, self._input_frequency)
            wav = self._preprocess_wav(wav)
            features = self._get_features(wav)
            with torch.no_grad():
                outputs = self.model(
                    features.to(self.device), output_hidden_states=True
                )

            timepoints = nsbase.Frequency(float(self.frequency)).to_ind(event.duration)
            data = _stack_interpolated_hidden_states(outputs.hidden_states, timepoints)
            # Release native-rate GPU states before the next event's forward.
            del outputs, features
            yield nsbase.TimedArray(
                data=data.numpy(),
                frequency=float(self.frequency),
                start=nsbase._UNSET_START,
                duration=event.duration,
            )
