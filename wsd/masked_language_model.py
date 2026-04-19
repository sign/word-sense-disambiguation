import os
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


def _unmask_chunks_cuda_parallel(
    chunks: list[tuple[list[int], list[str]]],
    components: ModelComponents,
) -> list[list[UnmaskResult]]:
    """Dispatch each chunk's forward pass on its own CUDA stream.

    Tokenization and mask-position lookup run on CPU before we enter the
    stream context. Doing the lookup on-GPU would require a ``.item()``
    sync that waits for the chunk's own stream to drain, which serialized
    the per-chunk dispatch with the host and erased most of the stream
    parallelism. On CPU the lookup is ~microseconds per chunk.
    """
    mask_id = components.tokenizer.mask_token_id
    streams = [torch.cuda.Stream() for _ in chunks]
    pending: list[tuple[list[tuple[torch.Tensor, int]], torch.cuda.Stream]] = []

    for (_, chunk_texts), stream in zip(chunks, streams, strict=True):
        cpu_inputs = components.tokenizer(
            chunk_texts, return_tensors="pt", padding=True,
        )
        mask_positions: list[tuple[int, int]] = []
        for i in range(len(chunk_texts)):
            positions = (cpu_inputs.input_ids[i] == mask_id).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise PromptMaskError()
            mask_positions.append((i, int(positions[0])))

        with torch.cuda.stream(stream), torch.no_grad():
            inputs = cpu_inputs.to(components.device, non_blocking=True)
            outputs = components.model(**inputs)
            logits_per_example = [
                (outputs.logits[bi, si], bi) for bi, si in mask_positions
            ]
        pending.append((logits_per_example, stream))

    results: list[list[UnmaskResult]] = []
    for logits_per_example, stream in pending:
        stream.synchronize()
        chunk_out: list[UnmaskResult] = []
        for logits, _ in logits_per_example:
            probs = torch.softmax(logits, dim=-1)
            compact_id = int(torch.argmax(probs).item())
            decoded = components.letter_set.letters[compact_id]
            chunk_out.append(UnmaskResult(token=decoded, probabilities=probs))
        results.append(chunk_out)
    return results


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


