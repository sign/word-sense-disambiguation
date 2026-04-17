import pytest
from transformers import AutoTokenizer

from wsd.letters import NUM_LETTERS, LetterSet, build_letters


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-Large-Instruct")


def test_build_letters_returns_expected_size(tokenizer):
    ls = build_letters(tokenizer)
    assert isinstance(ls, LetterSet)
    assert len(ls.letters) == NUM_LETTERS
    assert len(ls.token_ids) == NUM_LETTERS


def test_build_letters_all_distinct(tokenizer):
    ls = build_letters(tokenizer)
    assert len(set(ls.letters)) == NUM_LETTERS
    assert len(set(ls.token_ids)) == NUM_LETTERS


def test_build_letters_all_single_token_with_space(tokenizer):
    ls = build_letters(tokenizer)
    for letter, token_id in zip(ls.letters, ls.token_ids, strict=True):
        encoded = tokenizer.encode(" " + letter, add_special_tokens=False)
        assert len(encoded) == 1, f"letter {letter!r} is not a single token"
        assert encoded[0] == token_id


def test_build_letters_deterministic(tokenizer):
    assert build_letters(tokenizer) == build_letters(tokenizer)


def test_build_letters_starts_with_latin(tokenizer):
    ls = build_letters(tokenizer)
    # Latin A-Z, a-z should all be valid single-token on modern tokenizers.
    assert list(ls.letters[:26]) == [chr(ord("A") + i) for i in range(26)]
    assert list(ls.letters[26:52]) == [chr(ord("a") + i) for i in range(26)]
