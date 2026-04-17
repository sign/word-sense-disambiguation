import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from wsd.letters import NUM_LETTERS
from wsd.model import WSDModernBertForMaskedLM
from wsd.model_surgery import prune_decoder

BASE_MODEL = "answerdotai/ModernBERT-Large-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL)


@pytest.fixture(scope="module")
def pruned(tokenizer):
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    letter_set = prune_decoder(model, tokenizer)
    return model, letter_set


def test_decoder_out_features_matches_letter_count(pruned):
    model, letter_set = pruned
    assert model.decoder.out_features == NUM_LETTERS
    assert len(letter_set.letters) == NUM_LETTERS


def test_class_rebound_to_wsd_subclass(pruned):
    model, _ = pruned
    assert isinstance(model, WSDModernBertForMaskedLM)


def test_config_updated(pruned):
    model, _ = pruned
    assert model.config.tie_word_embeddings is False
    assert model.config.answer_vocab_size == NUM_LETTERS
    assert model.config.architectures == [WSDModernBertForMaskedLM.__name__]


def test_pruned_weights_come_from_original_rows(tokenizer):
    """The compact decoder row for letter_i must equal the original decoder row
    for letter_i's token id."""
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    original_weight = model.decoder.weight.data.clone()
    letter_set = prune_decoder(model, tokenizer)

    for compact_id, source_id in enumerate(letter_set.token_ids):
        torch.testing.assert_close(
            model.decoder.weight.data[compact_id],
            original_weight[source_id],
        )


def test_forward_returns_compact_logits(pruned, tokenizer):
    model, _ = pruned
    model.eval()
    text = f"Sentence here.\nAnswer: [unused0] {tokenizer.mask_token}"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.logits.shape[-1] == NUM_LETTERS


def test_embeddings_retain_full_vocab(pruned):
    """The decoder shrinks but input embeddings must still cover the full vocab."""
    model, _ = pruned
    assert model.get_input_embeddings().num_embeddings == model.config.vocab_size
    assert model.config.vocab_size > NUM_LETTERS
