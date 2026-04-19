import os
import time
from dataclasses import dataclass
from functools import cache
from typing import cast

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from wsd.letters import LetterSet, build_letters
from wsd.model import WSDModernBertForMaskedLM
from wsd.model_surgery import prune_decoder

# Allow overriding the model source (e.g. a local checkpoint directory) for
# benchmarking or evaluation without editing call sites.
_DEFAULT_MODEL = os.environ.get("WSD_MODEL", "sign/ModernBERT-Large-Instruct-WSD")


class PromptMaskError(ValueError):
    def __init__(self):
        super().__init__("No mask token found for prompt")


@dataclass
class ModelComponents:
    """Components returned by load_model"""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str
    letter_set: LetterSet


@dataclass
class UnmaskResult:
    """Result of unmasking a single token"""
    token: str
    probabilities: torch.Tensor


@cache
def load_model(model_name: str = _DEFAULT_MODEL) -> ModelComponents:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    letter_set = build_letters(tokenizer)

    model = WSDModernBertForMaskedLM.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.float16 if device == "cuda" else None,
    )
    # Stock checkpoints ship with a full-vocab decoder; prune it to the 128
    # answer letters so decoder outputs are indexed by compact ids. Checkpoints
    # already trained with the pruned decoder have out_features == 128 and this
    # is a no-op.
    if model.decoder.out_features != len(letter_set.letters):
        letter_set = prune_decoder(model, tokenizer)
    model.eval()
    return ModelComponents(model=model, tokenizer=tokenizer, device=device, letter_set=letter_set)


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
    compact_id = int(torch.argmax(probs).item())

    decoded_token = components.letter_set.letters[compact_id]
    return UnmaskResult(token=decoded_token, probabilities=probs)


# Sub-batch size used when length-bucketing inside ``unmask_token_batch``.
# Eight was the throughput sweet spot on GB10 — small enough that
# padding-within-chunk stays cheap, large enough to keep the GPU fed.
_BUCKET_CHUNK_SIZE = 8


def unmask_token_batch(texts: list[str]) -> list[UnmaskResult]:
    """
    Batch version of unmask_token that processes multiple texts in parallel.

    Inputs are sorted by tokenized length and processed in fixed-size chunks
    so each forward pass only pads up to the longest prompt *in its chunk*
    rather than the longest in the whole batch. Results are un-sorted before
    return, so callers still see outputs in input order. For the typical WSD
    prompt distribution (57..262 tokens) this roughly doubles throughput at
    large batch sizes.

    Args:
        texts: List of strings, each containing a mask token

    Returns:
        List of UnmaskResult objects for each input text

    Raises:
        PromptMaskError: If any text doesn't contain a mask token
    """
    if not texts:
        return []

    components = load_model()

    # Sort by pre-tokenized length so each chunk below has similar-length
    # prompts. We call the tokenizer twice (once here for lengths, once below
    # for padded tensors), but the first call is Python-side and cheap.
    lengths = [
        len(components.tokenizer(t, add_special_tokens=True)["input_ids"])
        for t in texts
    ]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    results: list[UnmaskResult | None] = [None] * len(texts)
    for start in range(0, len(order), _BUCKET_CHUNK_SIZE):
        chunk_idx = order[start : start + _BUCKET_CHUNK_SIZE]
        chunk_texts = [texts[i] for i in chunk_idx]
        for orig_idx, result in zip(
            chunk_idx, _unmask_chunk(chunk_texts, components), strict=True,
        ):
            results[orig_idx] = result

    # Every slot must be populated — callers (e.g. disambiguate_word_batch)
    # index positionally, so a short list would surface as a confusing
    # IndexError downstream rather than a clear failure here.
    assert all(r is not None for r in results), "unmask_token_batch left slots unfilled"
    return cast(list[UnmaskResult], results)


def _unmask_chunk(texts: list[str], components: ModelComponents) -> list[UnmaskResult]:
    """Single forward pass for a length-homogeneous chunk."""
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
        compact_id = int(torch.argmax(probs).item())
        decoded_token = components.letter_set.letters[compact_id]
        results.append(UnmaskResult(token=decoded_token, probabilities=probs))
    return results


def main():
    components = load_model()
    text = (
        f"Answer 'Yes' or 'No'.\n"
        f"QUESTION: Is Paris the capital of France?\n"
        f"ANSWER: [unused0] {components.tokenizer.mask_token}"
    )

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

    for i, (text_input, result) in enumerate(zip(texts, results, strict=False)):
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
