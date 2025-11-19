from functools import cache

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class PromptMaskError(ValueError):
    def __init__(self):
        super().__init__("No mask token found for prompt")

@cache
def load_model(model_name: str = "answerdotai/ModernBERT-Large-Instruct"):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.float16 if device == "cuda" else None,
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")
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
