# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Text feature extraction with a reduced device-to-CPU transfer."""

import typing as tp

import numpy as np
import torch
from exca import MapInfra
from exca.utils import environment_variables
from neuralset import events as _ev
from neuralset.extractors.text import HuggingFaceText, TextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

SUPPORTED_MODEL = "meta-llama/Llama-3.2-3B"


class EfficientHuggingFaceText(HuggingFaceText):
    """HuggingFaceText variant that transfers only selected layers and tokens.

    The model, tokenizer, contexts, batch size and cache format are inherited
    unchanged. Supported only for Llama-3.2-3B, whose forward accepts
    ``use_cache=False`` so the unused KV cache is not allocated.
    """

    model_name: str = SUPPORTED_MODEL

    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=10,
        min_samples_per_job=4096,
        version="v6",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        if self.model_name != SUPPORTED_MODEL:
            msg = (
                "EfficientHuggingFaceText only supports "
                f"{SUPPORTED_MODEL!r}, got {self.model_name!r}"
            )
            raise ValueError(msg)
        if self.token_aggregation != "mean":
            msg = (
                "EfficientHuggingFaceText only supports token_aggregation='mean', "
                f"got {self.token_aggregation!r}"
            )
            raise ValueError(msg)
        super().model_post_init(log__)

    def _extract_word_state(
        self,
        states: tp.Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        batch_index: int,
        target_word: str,
        context_text: str,
    ) -> np.ndarray:
        n_tokens = states[0].shape[1]
        n_pads = sum(input_ids[batch_index].cpu().numpy() == self._pad_id)

        end = n_tokens - n_pads
        if self.contextualized:
            prefix = context_text[: -len(target_word)].rstrip()
            n_prefix = (
                len(self.tokenizer.encode(prefix, add_special_tokens=False))
                if prefix
                else 0
            )
            n_target = end - n_prefix
        else:
            n_target = end
        start = max(0, end - max(1, n_target))

        layer_indices = self._layer_subselection(np.arange(len(states)))
        word_state = torch.stack(
            [states[index][batch_index, start:end].cpu() for index in layer_indices]
        )
        word_state = self._aggregate_tokens(word_state)
        out = word_state.cpu().numpy()
        if not self.cache_all_layers and self.cache_n_layers is None:
            out = self._aggregate_layers(out)
        if np.isnan(out).any():
            msg = f"NaN in output for target_word {target_word} with context {context_text}"
            raise ValueError(msg)
        return out

    @infra.apply(
        item_uid=lambda event: f"{event.text}_{getattr(event, 'context', '')}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(
        self, events: list[_ev.etypes.Word | _ev.etypes.Sentence]
    ) -> tp.Iterator[np.ndarray]:
        dataset = TextDataset(events)
        dloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if len(dloader) > 1:
            dloader = tqdm(dloader, desc="Computing word embeddings")  # type: ignore
        device = "auto" if self.device == "accelerate" else self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            for target_words, context in dloader:
                with environment_variables(TOKENIZERS_PARALLELISM="false"):
                    text = context if self.contextualized else target_words
                    if isinstance(text, tuple):
                        # temporary fix for tokenizers==0.20.2
                        # https://github.com/huggingface/tokenizers/issues/1672
                        text = list(text)
                    if not all(text):
                        msg = f"Empty text or context for target_words {target_words!r}"
                        raise ValueError(msg)
                    inputs = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                outputs = self.model(
                    **inputs, output_hidden_states=True, use_cache=False
                )
                if "hidden_states" in outputs:
                    states = outputs.hidden_states
                else:  # bart (encoder/decoder)
                    states = outputs.encoder_hidden_states + outputs.decoder_hidden_states

                for i, target_word in enumerate(target_words):
                    yield self._extract_word_state(
                        states, inputs["input_ids"], i, target_word, context[i]
                    )
                del states, outputs, inputs
                if self.device == "accelerate" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
