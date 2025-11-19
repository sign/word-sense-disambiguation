import pytest
import torch

from wsd.masked_language_model import PromptMaskError, load_model, unmask_token


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
