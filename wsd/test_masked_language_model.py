import pytest
import torch

from wsd.masked_language_model import (
    PromptMaskError,
    UnmaskResult,
    load_model,
    unmask_token,
    unmask_token_batch,
)


def test_load_model():
    """Test that model loads successfully"""
    components = load_model()

    # Validate model components
    assert components.model is not None
    assert components.tokenizer is not None
    assert components.device in ['cuda', 'mps', 'cpu']

    # Validate model is on correct device
    assert str(next(components.model.parameters()).device).startswith(components.device)

    # Validate tokenizer has mask token
    assert components.tokenizer.mask_token is not None


@pytest.mark.parametrize(("question", "expected"), [
    ("Is Paris the capital of France?", "yes"),
    ("Is London the capital of Germany?", "no"),
])
def test_unmask_token_yes_no(question, expected):
    """Test yes/no question format with various questions"""
    components = load_model()
    text = f"Answer 'Yes' or 'No'.\nQUESTION: {question}\nANSWER: [unused0] {components.tokenizer.mask_token}"

    result = unmask_token(text)

    # Should return a valid token
    assert isinstance(result.token, str)
    assert len(result.token.strip()) > 0

    # Probabilities should be valid
    assert isinstance(result.probabilities, torch.Tensor)
    assert result.probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)

    assert result.token.lower() == expected, f"Expected '{expected}', got '{result.token}' for question: {question}"


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
        (
            f"Answer 'Yes' or 'No'.\n"
            f"QUESTION: Is Paris the capital of France?\n"
            f"ANSWER: [unused0] {components.tokenizer.mask_token}"
        ),
        (
            f"Answer 'Yes' or 'No'.\n"
            f"QUESTION: Is London the capital of Germany?\n"
            f"ANSWER: [unused0] {components.tokenizer.mask_token}"
        ),
    ]

    results = unmask_token_batch(texts)

    # Should return correct number of results
    assert len(results) == len(texts)

    # Each result should be an UnmaskResult
    for result in results:
        assert isinstance(result, UnmaskResult)
        assert isinstance(result.token, str)
        assert len(result.token.strip()) > 0
        assert isinstance(result.probabilities, torch.Tensor)
        assert result.probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)

    # Check expected answers
    assert results[0].token.lower() == "yes"
    assert results[1].token.lower() == "no"


def test_unmask_token_batch_single_item():
    """Test batch processing with single item"""
    components = load_model()
    texts = [
        (
            f"Answer 'Yes' or 'No'.\n"
            f"QUESTION: Is Paris the capital of France?\n"
            f"ANSWER: [unused0] {components.tokenizer.mask_token}"
        )
    ]

    results = unmask_token_batch(texts)

    assert len(results) == 1
    assert isinstance(results[0], UnmaskResult)
    assert results[0].token.lower() == "yes"


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
        (
            f"Answer 'Yes' or 'No'.\n"
            f"QUESTION: Is Paris the capital of France?\n"
            f"ANSWER: [unused0] {components.tokenizer.mask_token}"
        ),
        (
            f"Answer 'Yes' or 'No'.\n"
            f"QUESTION: Is London the capital of Germany?\n"
            f"ANSWER: [unused0] {components.tokenizer.mask_token}"
        ),
    ]

    # Get batch results
    batch_results = unmask_token_batch(texts)

    # Get sequential results
    sequential_results = [unmask_token(text) for text in texts]

    # Compare tokens
    for batch_result, seq_result in zip(batch_results, sequential_results, strict=False):
        assert batch_result.token == seq_result.token


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
