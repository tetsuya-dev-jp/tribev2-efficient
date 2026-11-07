import numpy as np
import pytest
import torch

from tribev2.text_features import SUPPORTED_MODEL, EfficientHuggingFaceText

PAD_ID = 99
N_LAYERS = 29
N_TOKENS = 10
N_DIMS = 4
N_BATCH = 3

PREFIX_LENGTHS = {
    "a b c d e": 5,
    "x y z": 3,
    "long prefix here": 20,
}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(PREFIX_LENGTHS[text]))


def make_extractor(*, cache_n_layers=None, contextualized=True):
    extractor = EfficientHuggingFaceText(
        model_name=SUPPORTED_MODEL,
        event_types="Word",
        contextualized=contextualized,
        token_aggregation="mean",
        layers="all",
        cache_n_layers=cache_n_layers,
        device="cpu",
        batch_size=N_BATCH,
    )
    extractor._model = object()
    extractor._tokenizer = FakeTokenizer()
    extractor._pad_id = PAD_ID
    return extractor


def make_states():
    values = torch.arange(
        N_LAYERS * N_BATCH * N_TOKENS * N_DIMS, dtype=torch.float32
    ).reshape(N_LAYERS, N_BATCH, N_TOKENS, N_DIMS)
    return tuple(values.unbind(0))


def make_input_ids(n_pads):
    input_ids = torch.full((N_BATCH, N_TOKENS), 7, dtype=torch.long)
    for batch_index, n_pad in enumerate(n_pads):
        if n_pad:
            input_ids[batch_index, -n_pad:] = PAD_ID
    return input_ids


def reference_word_state(
    extractor, states, input_ids, batch_index, target_word, context_text
):
    hidden_states = torch.stack([layer.cpu() for layer in states])
    hidden_state = hidden_states[:, batch_index]

    n_pads = sum(input_ids[batch_index].cpu().numpy() == extractor._pad_id)
    if n_pads:
        hidden_state = hidden_state[:, :-n_pads]

    n_tokens = hidden_states.shape[2]
    if extractor.contextualized:
        prefix = context_text[: -len(target_word)].rstrip()
        n_prefix = (
            len(extractor.tokenizer.encode(prefix, add_special_tokens=False))
            if prefix
            else 0
        )
        n_target = n_tokens - n_pads - n_prefix
        word_state = hidden_state[:, -max(1, n_target) :]
    else:
        word_state = hidden_state

    word_state = extractor._aggregate_tokens(word_state)
    out = word_state.cpu().numpy()
    if not extractor.cache_all_layers and extractor.cache_n_layers is None:
        out = extractor._aggregate_layers(out)
    return out


N_PADS = [0, 3, 2]
CASES = [
    (0, "a b c d e target0", "target0"),
    (1, "x y z target1", "target1"),
    (2, "long prefix here target2", "target2"),
]


@pytest.mark.parametrize("cache_n_layers", [None, 20, 5])
def test_target_token_and_layer_selection_matches_reference(cache_n_layers):
    extractor = make_extractor(cache_n_layers=cache_n_layers)
    states = make_states()
    input_ids = make_input_ids(N_PADS)

    for batch_index, context_text, target_word in CASES:
        expected = reference_word_state(
            extractor, states, input_ids, batch_index, target_word, context_text
        )
        actual = extractor._extract_word_state(
            states, input_ids, batch_index, target_word, context_text
        )
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
        if cache_n_layers is None:
            assert actual.shape == (N_DIMS,)
        else:
            assert actual.shape == (cache_n_layers, N_DIMS)


def test_non_contextualized_selection_matches_reference():
    extractor = make_extractor(cache_n_layers=20, contextualized=False)
    states = make_states()
    input_ids = make_input_ids(N_PADS)

    for batch_index, context_text, target_word in CASES:
        expected = reference_word_state(
            extractor, states, input_ids, batch_index, target_word, context_text
        )
        actual = extractor._extract_word_state(
            states, input_ids, batch_index, target_word, context_text
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (20, N_DIMS)


def test_left_truncation_negative_target_uses_last_token():
    extractor = make_extractor(cache_n_layers=20)
    states = make_states()
    input_ids = make_input_ids(N_PADS)
    batch_index = 2

    hidden = torch.stack([layer.cpu() for layer in states])[:, batch_index]
    hidden = hidden[:, : N_TOKENS - N_PADS[batch_index]]
    expected_last_token = extractor._aggregate_tokens(hidden[:, -1:])

    actual = extractor._extract_word_state(
        states, input_ids, batch_index, "target2", "long prefix here target2"
    )
    np.testing.assert_array_equal(actual, expected_last_token.numpy())


def test_rejects_unsupported_model():
    with pytest.raises(ValueError, match="only supports"):
        EfficientHuggingFaceText(model_name="openai-community/gpt2", device="cpu")


@pytest.mark.parametrize("token_aggregation", ["sum", "first", "last", "max", None])
def test_rejects_unsupported_token_aggregation(token_aggregation):
    with pytest.raises(ValueError, match="token_aggregation='mean'"):
        EfficientHuggingFaceText(
            model_name=SUPPORTED_MODEL,
            token_aggregation=token_aggregation,
            device="cpu",
        )
