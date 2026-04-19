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

    # Prefer bf16 on GPUs that support it (Ampere+, most AMD MI200+) — it
    # matches the dtype training uses, so inference doesn't incur a numeric
    # mismatch versus the trained weights. Fall back to fp16 on older GPUs.
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = None

    model = WSDModernBertForMaskedLM.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
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
    return _unmask_chunk([text], components)[0]


# Sub-batch size used when length-bucketing inside ``unmask_token_batch``.
# Four wins on real WSD traffic on GB10: a disambiguated sentence has 6-20
# content words whose prompt lengths span 2-4x (few-sense vs many-sense
# words), so within-chunk padding waste dominates once the chunk grows
# past ~4. Larger chunks look better only on artificially length-homogeneous
# batches.
_BUCKET_CHUNK_SIZE = 4


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

    # Build the chunk list up front so we can dispatch in one loop.
    chunks: list[tuple[list[int], list[str]]] = []
    for start in range(0, len(order), _BUCKET_CHUNK_SIZE):
        chunk_idx = order[start : start + _BUCKET_CHUNK_SIZE]
        chunks.append((chunk_idx, [texts[i] for i in chunk_idx]))

    # On CUDA, launch each chunk on its own stream so the GPU can overlap
    # kernel execution across chunks instead of us serializing on the host.
    # Measured ~13% speedup on 20-content-word sentences vs sequential
    # chunks. Other devices (CPU/MPS) see no benefit from streams and fall
    # back to straight sequential dispatch.
    if components.device == "cuda" and len(chunks) > 1:
        chunk_results = _unmask_chunks_cuda_parallel(chunks, components)
    else:
        chunk_results = [_unmask_chunk(texts_, components) for _, texts_ in chunks]

    results: list[UnmaskResult | None] = [None] * len(texts)
    for (chunk_idx, _), chunk_res in zip(chunks, chunk_results, strict=True):
        for orig_idx, res in zip(chunk_idx, chunk_res, strict=True):
            results[orig_idx] = res

    # Every slot must be populated — callers (e.g. disambiguate_word_batch)
    # index positionally, so a short list would surface as a confusing
    # IndexError downstream rather than a clear failure here.
    assert all(r is not None for r in results), "unmask_token_batch left slots unfilled"
    return cast(list[UnmaskResult], results)


def _prediction_positions(input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """LongTensor ``(batch,)`` column index of the first ``[MASK]`` per row.

    Returned as integer positions (not a boolean mask) so the model can gather
    rows via indexed select. A bool ``masked_select`` would force the GPU to
    report ``mask.sum()`` back to the host before sizing its output — a sync
    that drains the stream and serializes the multi-chunk dispatch in
    ``_unmask_chunks_cuda_parallel``. Rows with multiple masks use the first
    (argmax returns the first max); rows with no mask raise.
    """
    is_mask = input_ids == mask_token_id
    if bool((is_mask.sum(dim=1) == 0).any()):
        raise PromptMaskError()
    return is_mask.int().argmax(dim=1)


def _logits_to_results(
    logits: torch.Tensor, letters: tuple[str, ...],
) -> list[UnmaskResult]:
    """Turn ``(batch, answer_vocab)`` logits into per-example UnmaskResults."""
    probs = torch.softmax(logits, dim=-1)
    compact_ids = torch.argmax(probs, dim=-1).tolist()
    return [
        UnmaskResult(token=letters[cid], probabilities=p)
        for cid, p in zip(compact_ids, probs, strict=True)
    ]


def _unmask_chunks_cuda_parallel(
    chunks: list[tuple[list[int], list[str]]],
    components: ModelComponents,
) -> list[list[UnmaskResult]]:
    """Dispatch each chunk's forward pass on its own CUDA stream.

    Tokenization and mask validation run on CPU before we enter the stream
    context so the per-chunk dispatch can overlap — on-GPU validation would
    force a ``.item()`` sync that drains the stream and erases the parallelism.
    """
    mask_id = components.tokenizer.mask_token_id
    streams = [torch.cuda.Stream() for _ in chunks]
    pending: list[tuple[torch.Tensor, torch.cuda.Stream]] = []

    for (_, chunk_texts), stream in zip(chunks, streams, strict=True):
        cpu_inputs = components.tokenizer(
            chunk_texts, return_tensors="pt", padding=True,
        )
        positions_cpu = _prediction_positions(cpu_inputs.input_ids, mask_id)

        with torch.cuda.stream(stream), torch.no_grad():
            inputs = cpu_inputs.to(components.device, non_blocking=True)
            positions = positions_cpu.to(components.device, non_blocking=True)
            outputs = components.model(**inputs, prediction_positions=positions)
        pending.append((outputs.logits, stream))

    letters = components.letter_set.letters
    results: list[list[UnmaskResult]] = []
    for logits, stream in pending:
        stream.synchronize()
        results.append(_logits_to_results(logits, letters))
    return results


def _unmask_chunk(texts: list[str], components: ModelComponents) -> list[UnmaskResult]:
    """Single forward pass for a length-homogeneous chunk."""
    # Compute mask positions on the CPU tensor before moving to device so the
    # validation sum/any doesn't force a device→host sync.
    cpu_inputs = components.tokenizer(texts, return_tensors="pt", padding=True)
    positions_cpu = _prediction_positions(
        cpu_inputs.input_ids, components.tokenizer.mask_token_id,
    )

    inputs = cpu_inputs.to(components.device)
    positions = positions_cpu.to(components.device)
    with torch.no_grad():
        outputs = components.model(**inputs, prediction_positions=positions)

    return _logits_to_results(outputs.logits, components.letter_set.letters)


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
