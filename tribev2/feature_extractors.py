# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in feature extractors with the same inputs and cached outputs as neuralset."""

import typing as tp

import numpy as np
import torch
from exca import MapInfra
from tqdm import tqdm

from neuralset import base as nsbase
from neuralset.events import etypes as evts
from neuralset.extractors.image import _fix_pixel_values
from neuralset.extractors.video import HuggingFaceVideo, _HFVideoModel, _VideoImage

_VJEPA2_MODEL_PREFIX = "vjepa2"


def _selected_layer_indices(
    num_hidden_states: int, cache_n_layers: int | None
) -> list[int] | None:
    """Layer indices selected by the neuralset layer cache contract."""
    if cache_n_layers is None or cache_n_layers >= num_hidden_states:
        return None
    return [
        int(round(x))
        for x in np.linspace(0, num_hidden_states - 1, cache_n_layers)
    ]


def _token_mean(hidden_states: torch.Tensor) -> torch.Tensor:
    """Mean tokens, accumulating in float32 for half-precision inputs."""
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.float()
    return hidden_states.mean(dim=1)


def _vjepa2_token_means(
    encoder: torch.nn.Module,
    pixel_values: torch.Tensor,
    cache_n_layers: int | None,
) -> torch.Tensor:
    """Return token means for the requested VJEPA2 hidden states.

    VJEPA2Model records the embedding output at index 0 and each transformer
    layer output at indices 1..N. The final layer norm is not part of that
    tuple, so it is intentionally not applied here. All encoder layers still
    run, but only the selected token means are retained.
    """
    hidden_states = encoder.embeddings(pixel_values)
    num_hidden_states = len(encoder.layer) + 1
    selected = _selected_layer_indices(num_hidden_states, cache_n_layers)
    wanted = None if selected is None else set(selected)

    means: dict[int, torch.Tensor] = {}
    if wanted is None or 0 in wanted:
        means[0] = _token_mean(hidden_states)

    for layer_idx, layer in enumerate(encoder.layer, start=1):
        hidden_states = layer(hidden_states, None)[0]
        if wanted is None or layer_idx in wanted:
            means[layer_idx] = _token_mean(hidden_states)

    indices = range(num_hidden_states) if selected is None else selected
    return torch.stack([means[index] for index in indices], dim=1).float()


def _read_processed_frame_window(
    video: tp.Any,
    time: float,
    subtimes: tp.Sequence[float],
    previous: dict[float, torch.Tensor],
    processor: tp.Any,
    max_imsize: int | None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Preprocess one window's frames, reusing exact timestamps.

    Timestamps are not rounded, and clamped zero timestamps share one frame
    within the window. The processor runs only for newly read frames; the
    returned cache contains only timestamps used by this window.
    """
    timestamps = [float(max(0.0, time - offset)) for offset in subtimes]
    unique = list(dict.fromkeys(timestamps))
    current = {
        timestamp: previous[timestamp]
        for timestamp in unique
        if timestamp in previous
    }
    missing = [timestamp for timestamp in unique if timestamp not in previous]

    if missing:
        images = [
            _VideoImage(video=video, time=timestamp).read()
            for timestamp in missing
        ]
        if max_imsize is not None and images:
            factor = max(images[0].size) / max_imsize
            if factor > 1:
                size = tuple(int(s / factor) for s in images[0].size)
                images = [image.resize(size) for image in images]
        data = np.array([np.array(image) for image in images])
        inputs = processor(videos=list(data), return_tensors="pt")
        _fix_pixel_values(inputs)
        for timestamp, frame in zip(missing, inputs["pixel_values_videos"][0]):
            current[timestamp] = frame

    pixels = torch.stack([current[timestamp] for timestamp in timestamps], dim=0)
    return pixels[None], current


class EfficientVJEPA2Video(HuggingFaceVideo):
    """VJEPA2 video features without the unused predictor or full hidden states.

    This extractor keeps the parent's frame sampling, preprocessing, layer
    indices, token aggregation, and cached output layout. It runs the encoder
    directly, retains just the selected token means, and reuses processed frames
    whose exact timestamps fall in the previous 4-second window.
    compute_dtype may be set to float16 on CUDA; token means are still
    accumulated and returned in float32.
    """

    compute_dtype: tp.Literal["float32", "float16"] = "float32"
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=128,
        version="v5",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        model_name = self.image.model_name
        if _VJEPA2_MODEL_PREFIX not in model_name:
            raise ValueError(
                "EfficientVJEPA2Video requires a VJEPA2 checkpoint, "
                f"got {model_name!r}"
            )
        if self.layer_type:
            raise ValueError(
                "EfficientVJEPA2Video does not support layer_type, "
                f"got {self.layer_type!r}"
            )
        if self.image.token_aggregation != "mean":
            raise ValueError(
                "EfficientVJEPA2Video requires token_aggregation='mean', "
                f"got {self.image.token_aggregation!r}"
            )
        super().model_post_init(log__)
        if self.compute_dtype == "float16" and not str(self.image.device).startswith(
            "cuda"
        ):
            raise ValueError(
                "compute_dtype='float16' requires a CUDA image device, "
                f"got {self.image.device!r}"
            )

    @infra.apply(
        item_uid=lambda event: f"{event.study_relative_path()}_{event.offset:.2f}_{event.duration:.2f}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: list[evts.Video]) -> tp.Iterator[nsbase.TimedArray]:
        wrapper = _HFVideoModel(
            model_name=self.image.model_name,
            pretrained=self.image.pretrained,
            layer_type=self.layer_type,
            num_frames=self.num_frames,
        )
        encoder = wrapper.model.encoder
        processor = wrapper.processor
        num_frames = wrapper.num_frames
        # Drop the predictor and keep only the encoder weights.
        del wrapper
        dtype = torch.float16 if self.compute_dtype == "float16" else torch.float32
        encoder.eval()
        encoder.to(self.image.device, dtype=dtype)
        device = next(encoder.parameters()).device

        freq = events[0].frequency if self.frequency == "native" else self.frequency
        clip_duration = 1 / freq if self.clip_duration is None else self.clip_duration
        subtimes = [
            k / num_frames * clip_duration for k in reversed(range(num_frames))
        ]

        for event in events:
            frame_cache: dict[float, torch.Tensor] = {}
            video = event.read()
            try:
                freq = self.frequency if self.frequency != "native" else event.frequency
                expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
                times = np.linspace(0, video.duration, expect_frames + 1)[1:]
                output = np.array([])
                for k, t in tqdm(
                    enumerate(times), total=len(times), desc="Encoding video"
                ):
                    pixels, frame_cache = _read_processed_frame_window(
                        video, t, subtimes, frame_cache, processor, self.max_imsize
                    )
                    pixels = pixels.to(device=device, dtype=dtype)
                    with torch.inference_mode():
                        embd = _vjepa2_token_means(
                            encoder,
                            pixels,
                            self.image.cache_n_layers,
                        )[0]
                    embd = embd.cpu().numpy()
                    if (
                        not self.image.cache_all_layers
                        and self.image.cache_n_layers is None
                    ):
                        embd = self.image._aggregate_layers(embd)
                    if not output.size:
                        output = np.zeros((len(times),) + embd.shape)
                    output[k] = embd
            finally:
                video.close()

            output = output.transpose(list(range(1, output.ndim)) + [0])
            yield nsbase.TimedArray(
                data=output.astype(np.float32),
                frequency=freq,
                start=nsbase._UNSET_START,
                duration=event.duration,
            )
