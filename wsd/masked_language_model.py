from functools import cache

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class PromptMaskError(ValueError):
    def __init__(self):
        super().__init__("No mask token found for prompt")

@cache
def load_model(model_name: str = "answerdotai/ModernBERT-Large-Instruct"):
    """Load and cache the ModernBERT model and tokenizer"""
    # Device selection priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with appropriate optimizations
    attn_implementation = None
    if device == 'cuda':
        try:
            import flash_attn_interface
            attn_implementation = "flash_attention_2"
        except ImportError:
            print("FlashAttention not available, using standard attention for CUDA.")

    model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation=attn_implementation)

    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return model, tokenizer, device


def unmask_token(text: str):
    model, tokenizer, device = load_model()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()

    if len(mask_idx) == 0:
        raise PromptMaskError()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, mask_idx[0, 1]]
    probs = torch.softmax(logits, dim=-1)
    token_id = torch.argmax(probs).item()

    return tokenizer.decode(token_id).strip(), probs


def main():
    _, tokenizer, _ = load_model()
    text = f"Answer 'Yes' or 'No'.\nQUESTION: Is Paris the capital of France?\nANSWER: [unused0] {tokenizer.mask_token}"
    token, probs = unmask_token(text)
    print(text)
    print(f"Answer: {token}")


if __name__ == "__main__":
    main()
