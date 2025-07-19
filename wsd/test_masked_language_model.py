import pytest
import torch

from wsd.masked_language_model import PromptMaskError, load_model, unmask_token


def test_load_model():
    """Test that model loads successfully"""
    model, tokenizer, device = load_model()

    # Validate model components
    assert model is not None
    assert tokenizer is not None
    assert device in ['cuda', 'mps', 'cpu']

    # Validate model is on correct device
    assert str(next(model.parameters()).device).startswith(device)

    # Validate tokenizer has mask token
    assert tokenizer.mask_token is not None


@pytest.mark.parametrize(("question", "expected"), [
    ("Is Paris the capital of France?", "yes"),
    ("Is London the capital of Germany?", "no"),
])
def test_unmask_token_yes_no(question, expected):
    """Test yes/no question format with various questions"""
    _, tokenizer, _ = load_model()
    text = f"Answer 'Yes' or 'No'.\nQUESTION: {question}\nANSWER: [unused0] {tokenizer.mask_token}"

    token, probs = unmask_token(text)

    # Should return a valid token
    assert isinstance(token, str)
    assert len(token.strip()) > 0

    # Probabilities should be valid
    assert isinstance(probs, torch.Tensor)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    assert token.lower() == expected, f"Expected '{expected}', got '{token}' for question: {question}"


def test_no_mask_token_error():
    """Test that PromptMaskError is raised when no mask token is present"""
    text = "This text has no mask token."

    with pytest.raises(PromptMaskError):
        unmask_token(text)


def test_multiple_mask_tokens():
    """Test behavior with multiple mask tokens (should use first one)"""
    _, tokenizer, _ = load_model()
    text = f"The capital of {tokenizer.mask_token} is {tokenizer.mask_token}."

    token, probs = unmask_token(text)

    # Should still work (uses first mask token)
    assert isinstance(token, str)
    assert len(token.strip()) > 0
    assert isinstance(probs, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
