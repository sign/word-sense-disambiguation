import pytest
import torch

from wsd.masked_language_model import (
    PromptMaskError,
    UnmaskResult,
    load_model,
    unmask_token,
    unmask_token_batch,
)


def _mc_prompt(mask_token: str, correct: str) -> str:
    """Build a trivial multiple-choice prompt where the correct letter is obvious."""
    return (
        "Pick the letter that matches.\n"
        f"Letter to pick: {correct}\n"
        f"Answer: [unused0] {mask_token}"
    )


def test_load_model():
    """Test that model loads successfully"""
    components = load_model()

    # Validate model components
    assert components.model is not None
    assert components.tokenizer is not None
    assert components.device in ['cuda', 'mps', 'cpu']
    assert len(components.letter_set.letters) == 128

    # Validate model is on correct device
    assert str(next(components.model.parameters()).device).startswith(components.device)

    # Validate tokenizer has mask token
    assert components.tokenizer.mask_token is not None


def test_unmask_token_returns_answer_letter():
    """Single unmask call returns a 128-wide probs tensor and an answer letter."""
    components = load_model()
    text = _mc_prompt(components.tokenizer.mask_token, "A")

    result = unmask_token(text)

    assert isinstance(result.token, str)
    assert result.token in components.letter_set.letters

    assert isinstance(result.probabilities, torch.Tensor)
    assert result.probabilities.shape[-1] == len(components.letter_set.letters)
    assert result.probabilities.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_no_mask_token_error():
    """Test that PromptMaskError is raised when no mask token is present"""
    text = "This text has no mask token."

    with pytest.raises(PromptMaskError):
        unmask_token(text)


def test_multiple_mask_tokens():
    """Test behavior with multiple mask tokens (should use first one)"""
    components = load_model()
    text = f"The capital of {components.tokenizer.mask_token} is {components.tokenizer.mask_token}."

    result = unmask_token(text)

    # Should still work (uses first mask token)
    assert isinstance(result.token, str)
    assert len(result.token.strip()) > 0
    assert isinstance(result.probabilities, torch.Tensor)


def test_unmask_token_batch_basic():
    """Test batch processing with multiple texts"""
    components = load_model()
    texts = [
        _mc_prompt(components.tokenizer.mask_token, "A"),
        _mc_prompt(components.tokenizer.mask_token, "B"),
    ]

    results = unmask_token_batch(texts)

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, UnmaskResult)
        assert isinstance(result.token, str)
        assert result.token in components.letter_set.letters
        assert result.probabilities.shape[-1] == len(components.letter_set.letters)
        assert result.probabilities.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_unmask_token_batch_single_item():
    """Test batch processing with single item"""
    components = load_model()
    texts = [_mc_prompt(components.tokenizer.mask_token, "A")]

    results = unmask_token_batch(texts)

    assert len(results) == 1
    assert isinstance(results[0], UnmaskResult)


def test_unmask_token_batch_empty_list():
    """Test batch processing with empty list"""
    results = unmask_token_batch([])
    assert results == []


def test_unmask_token_batch_no_mask_error():
    """Test that batch processing raises error when any text lacks mask token"""
    texts = [
        "This text has no mask token.",
        "Neither does this one.",
    ]

    with pytest.raises(PromptMaskError):
        unmask_token_batch(texts)


def test_unmask_token_batch_consistency():
    """Test that batch processing produces same results as sequential processing"""
    components = load_model()
    texts = [
        _mc_prompt(components.tokenizer.mask_token, "A"),
        _mc_prompt(components.tokenizer.mask_token, "B"),
    ]

    batch_results = unmask_token_batch(texts)
    sequential_results = [unmask_token(text) for text in texts]

    for batch_result, seq_result in zip(batch_results, sequential_results, strict=False):
        assert batch_result.token == seq_result.token


def test_unmask_token_batch_preserves_order_with_varied_lengths():
    """Varied prompt lengths exercise the internal length-bucketing sort; the
    returned results must still be in input order, not sorted order.

    Uses >_BUCKET_CHUNK_SIZE (8) interleaved prompts so the batch is split
    across multiple chunks and the tail-chunk/index-remapping path is hit.
    """
    components = load_model()
    mask = components.tokenizer.mask_token
    # Ten interleaved lengths so sorted-order != input-order *and* the batch
    # spans at least two chunks of 8.
    repeats_by_prompt = [0, 80, 5, 60, 10, 40, 15, 20, 25, 70]
    texts = [
        _mc_prompt(mask, chr(ord("A") + i))
        + (("\n" + "extra context. " * repeats) if repeats else "")
        for i, repeats in enumerate(repeats_by_prompt)
    ]

    batch_results = unmask_token_batch(texts)
    sequential_results = [unmask_token(t) for t in texts]

    assert len(batch_results) == len(texts)
    for batch_result, seq_result in zip(batch_results, sequential_results, strict=True):
        assert batch_result.token == seq_result.token


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
