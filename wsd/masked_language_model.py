import time
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class PromptMaskError(ValueError):
    def __init__(self):
        super().__init__("No mask token found for prompt")


@dataclass
class ModelComponents:
    """Components returned by load_model"""
    model: Any
    tokenizer: Any
    device: str


@dataclass
class UnmaskResult:
    """Result of unmasking a single token"""
    token: str
    probabilities: torch.Tensor

@cache
def load_model(model_name: str = "answerdotai/ModernBERT-Large-Instruct") -> ModelComponents:
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
        torch_dtype=torch.float16 if device == "cuda" else None,
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")
    return ModelComponents(model=model, tokenizer=tokenizer, device=device)


def unmask_token(text: str) -> UnmaskResult:
    components = load_model()

    inputs = components.tokenizer(text, return_tensors="pt").to(components.device)
    mask_idx = (inputs.input_ids == components.tokenizer.mask_token_id).nonzero()

    if len(mask_idx) == 0:
        raise PromptMaskError()

    with torch.no_grad():
        outputs = components.model(**inputs)

    logits = outputs.logits[0, mask_idx[0, 1]]
    probs = torch.softmax(logits, dim=-1)
    token_id = torch.argmax(probs).item()

    decoded_token = components.tokenizer.decode(token_id).strip()
    return UnmaskResult(token=decoded_token, probabilities=probs)


def unmask_token_batch(texts: list[str]) -> list[UnmaskResult]:
    """
    Batch version of unmask_token that processes multiple texts in parallel.

    Args:
        texts: List of strings, each containing a mask token

    Returns:
        List of UnmaskResult objects for each input text

    Raises:
        PromptMaskError: If any text doesn't contain a mask token
    """
    components = load_model()

    # Tokenize all texts with padding
    inputs = components.tokenizer(texts, return_tensors="pt", padding=True).to(components.device)

    # Find mask token positions for each example in the batch
    mask_positions = []
    for i in range(len(texts)):
        mask_idx = (inputs.input_ids[i] == components.tokenizer.mask_token_id).nonzero()
        if len(mask_idx) == 0:
            raise PromptMaskError()
        mask_positions.append((i, mask_idx[0, 0].item()))

    # Run batched forward pass
    with torch.no_grad():
        outputs = components.model(**inputs)

    # Extract predictions for each mask token
    results = []
    for batch_idx, seq_idx in mask_positions:
        logits = outputs.logits[batch_idx, seq_idx]
        probs = torch.softmax(logits, dim=-1)
        token_id = torch.argmax(probs).item()
        decoded_token = components.tokenizer.decode(token_id).strip()
        results.append(UnmaskResult(token=decoded_token, probabilities=probs))

    return results


def main():
    components = load_model()
    text = f"Answer 'Yes' or 'No'.\nQUESTION: Is Paris the capital of France?\nANSWER: [unused0] {components.tokenizer.mask_token}"

    # Single example
    start_time = time.time()
    result = unmask_token(text)
    single_time = time.time() - start_time

    print(text)
    print(f"Answer: {result.token}")
    print(f"Time: {single_time:.4f}s")

    # Create test texts with different batch sizes
    base_texts = [
        text,
        text.replace("France", "The United States"),
        text.replace("France", "England").replace("Paris", "London"),
        text.replace("France", "Germany").replace("Paris", "Berlin"),
        text.replace("France", "Italy").replace("Paris", "Rome"),
        text.replace("France", "Spain").replace("Paris", "Madrid"),
        text.replace("France", "Portugal").replace("Paris", "Lisbon"),
        text.replace("France", "Belgium").replace("Paris", "Brussels"),
    ]

    # Test with batch size of 3
    texts = base_texts[:3]

    # Process 3 examples individually
    print("\n" + "="*60)
    print("Sequential processing (3 examples):")
    print("="*60)
    start_time = time.time()
    for text_input in texts:
        _ = unmask_token(text_input)
    sequential_time = time.time() - start_time
    print(f"Total time: {sequential_time:.4f}s")
    print(f"Average per example: {sequential_time/len(texts):.4f}s")

    # Process 3 examples in batch
    print("\n" + "="*60)
    print("Batched processing (3 examples):")
    print("="*60)
    start_time = time.time()
    results = unmask_token_batch(texts)
    batch_time = time.time() - start_time

    for i, (text_input, result) in enumerate(zip(texts, results)):
        # Extract the question from the text
        question_line = text_input.split('\n')[1]
        print(f"\n{i+1}. {question_line}")
        print(f"   Answer: {result.token}")

    print(f"\nTotal time: {batch_time:.4f}s")
    print(f"Average per example: {batch_time/len(texts):.4f}s")

    # Compare performance for batch of 3
    speedup = sequential_time / batch_time
    print("\n" + "="*60)
    print("Performance Comparison (batch size 3):")
    print("="*60)
    print(f"Sequential: {sequential_time:.4f}s")
    print(f"Batched:    {batch_time:.4f}s")
    print(f"Speedup:    {speedup:.2f}x faster")

    # Test with larger batch size (8)
    texts_8 = base_texts[:8]
    print("\n" + "="*60)
    print("Testing with batch size 8:")
    print("="*60)

    # Sequential
    start_time = time.time()
    for text_input in texts_8:
        _ = unmask_token(text_input)
    sequential_time_8 = time.time() - start_time

    # Batched
    start_time = time.time()
    _ = unmask_token_batch(texts_8)
    batch_time_8 = time.time() - start_time

    speedup_8 = sequential_time_8 / batch_time_8
    print(f"Sequential: {sequential_time_8:.4f}s ({sequential_time_8/8:.4f}s per example)")
    print(f"Batched:    {batch_time_8:.4f}s ({batch_time_8/8:.4f}s per example)")
    print(f"Speedup:    {speedup_8:.2f}x faster")


if __name__ == "__main__":
    main()
