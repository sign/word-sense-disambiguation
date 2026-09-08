import contextlib
import os
from dataclasses import dataclass
from functools import cache
from typing import cast

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from wsd.letters import LetterSet, build_letters
from wsd.model import WSDModernBertForMaskedLM
from wsd.model_surgery import prune_decoder

DEFAULT_MODEL = "sign/Ettin-150m-WSD"  # WSD_MODEL overrides (a Hub id or a local checkpoint directory)


def attn_implementation() -> str | None:
    """Prefer flash-attention 2 when installed: ModernBERT then unpads the batch,
    so padding waste disappears and large mixed-length batches run at full speed.
    Falls back to the transformers default (sdpa) otherwise."""
    if not torch.cuda.is_available():
        return None
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return None
    return "flash_attention_2"


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
    probabilities: list[float]  # one per answer letter; plain floats so callers never index a tensor


@cache
def load_model(model_name: str | None = None) -> ModelComponents:
    model_name = model_name or os.environ.get("WSD_MODEL", DEFAULT_MODEL)
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

    model = WSDModernBertForMaskedLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attn_implementation())
    model.to(device)
    # Stock checkpoints ship with a full-vocab decoder; prune it to the 128
    # answer letters so decoder outputs are indexed by compact ids. Checkpoints
    # already trained with the pruned decoder have out_features == 128 and this
    # is a no-op.
    if model.decoder.out_features != len(letter_set.letters):
        letter_set = prune_decoder(model, tokenizer)
    model.eval()
    # torch.compile of the encoder: ~1.5x on H100 batch inference, at the cost
    # of ~50s compile per process (needs a C compiler for triton). Off by
    # default for the latency-sensitive server; wsd.batch turns it on.
    if os.environ.get("WSD_COMPILE") == "1" and device == "cuda":
        model.model = torch.compile(model.model, dynamic=True)  # static shapes and CUDA graphs measured slower
    return ModelComponents(model=model, tokenizer=tokenizer, device=device, letter_set=letter_set)


# Sub-batch size used when length-bucketing inside ``unmask_token_batch``.
# With sdpa attention every row is padded to the chunk's longest prompt, and
# on GB10 a chunk of 4 won on single-sentence traffic (6-20 prompts spanning
# 2-4x in length). With flash-attention 2 the batch is unpadded, so padding
# waste is gone and big chunks win outright (H100: 4 -> 512 is ~10x).
# Override with WSD_CHUNK_SIZE.
_BUCKET_CHUNK_SIZE = int(os.environ.get("WSD_CHUNK_SIZE", 512 if attn_implementation() else 4))


# Prompts tokenized and padded together. Kernel launches are asynchronous, so
# while the GPU works through one slice's chunks the CPU tokenizes and pads the
# next: measured on H100, host work otherwise leaves the GPU idle ~1/3 of the time.
_SLICE_SIZE = 8192
_NUM_STREAMS = 4


def unmask_token_batch(texts: list[str]) -> list[UnmaskResult]:
    """
    Batch version of unmask_token that processes multiple texts in parallel.

    Texts are handled in slices of ``_SLICE_SIZE``: each slice is tokenized once
    (the fast tokenizer parallelizes a list), sorted by length and padded into
    fixed-size chunks so each forward pass only pads up to the longest prompt in
    its chunk. On CUDA every chunk is launched asynchronously as soon as it is
    ready and results are collected at the end, so tokenizing the next slice
    overlaps the GPU work of the previous one. Results are returned in input order.

    Raises:
        PromptMaskError: If any text doesn't contain a mask token
    """
    if not texts:
        return []

    components = load_model()
    tokenizer = components.tokenizer
    cuda = components.device == "cuda"
    streams = [torch.cuda.Stream() for _ in range(_NUM_STREAMS)] if cuda else []

    results: list[UnmaskResult | None] = [None] * len(texts)
    pending: list[tuple[list[int], torch.Tensor, torch.cuda.Stream]] = []
    for slice_start in range(0, len(texts), _SLICE_SIZE):
        # the Rust batch encoder directly: the Python wrapper's per-encoding conversion
        # (`_convert_encoding`) costs ~16 us per prompt, about as much as the encoding itself
        chunk_texts = texts[slice_start:slice_start + _SLICE_SIZE]
        encodings = [e.ids for e in tokenizer.backend_tokenizer.encode_batch(chunk_texts)]
        if any(tokenizer.mask_token_id not in ids for ids in encodings):
            raise PromptMaskError()
        order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]))
        for local_idx in _chunks(order):
            chunk_idx = [slice_start + i for i in local_idx]
            input_ids, attention_mask = _pad([encodings[i] for i in local_idx], tokenizer.pad_token_id)
            positions = _prediction_positions(input_ids, tokenizer.mask_token_id)
            stream = streams[len(pending) % _NUM_STREAMS] if cuda else None
            pending.append((chunk_idx, _forward(input_ids, attention_mask, positions, components, stream), stream))

    letters = components.letter_set.letters
    for chunk_idx, logits, stream in pending:
        if stream is not None:
            stream.synchronize()
        for orig_idx, res in zip(chunk_idx, _logits_to_results(logits, letters), strict=True):
            results[orig_idx] = res

    # Every slot must be populated — callers (e.g. disambiguate_word_batch)
    # index positionally, so a short list would surface as a confusing
    # IndexError downstream rather than a clear failure here.
    assert all(r is not None for r in results), "unmask_token_batch left slots unfilled"
    return cast(list[UnmaskResult], results)


def _chunks(order: list[int]) -> list[list[int]]:
    """Split length-sorted indices into chunks of ``_BUCKET_CHUNK_SIZE``, never a chunk of one row:
    a 1-row batch fails the compiled graph's "batch >= 2" guard and costs a 15-25 s recompile."""
    starts = list(range(0, len(order), _BUCKET_CHUNK_SIZE))
    if len(order) > 1 and len(order) - starts[-1] == 1:
        starts.pop()  # fold the lone remainder into the previous chunk
    chunks = [order[a:b] for a, b in zip(starts, starts[1:] + [len(order)], strict=True)]
    return [c * 2 if len(c) == 1 else c for c in chunks]  # a single prompt runs twice instead


def _pad(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token id lists into ``(input_ids, attention_mask)`` LongTensors.

    Vectorized: ``tokenizer.pad`` loops in Python and cost about as much as the
    tokenization itself on 256-row chunks.
    """
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    width = int(lengths.max())
    mask = np.arange(width)[None, :] < lengths[:, None]
    input_ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
    input_ids[mask] = np.concatenate([np.asarray(s, dtype=np.int64) for s in sequences])
    return torch.from_numpy(input_ids), torch.from_numpy(mask.astype(np.int64))


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
    """Turn ``(batch, answer_vocab)`` logits into per-example UnmaskResults.

    Probabilities come back as Python lists in one copy per chunk; callers index
    them per option, which on a tensor costs a Python/C round trip each (and on a
    GPU tensor a device sync).
    """
    probs = torch.softmax(logits.float(), dim=-1).cpu()
    compact_ids = torch.argmax(probs, dim=-1).tolist()
    return [
        UnmaskResult(token=letters[cid], probabilities=p)
        for cid, p in zip(compact_ids, probs.tolist(), strict=True)  # one conversion per chunk
    ]


def _forward(input_ids, attention_mask, positions, components: ModelComponents, stream) -> torch.Tensor:
    """One chunk's forward pass; on CUDA queued on ``stream`` and returned before it is computed
    (the caller synchronizes when collecting). Plain copies: pinning per chunk takes a driver-wide
    lock and serializes the 8 processes sharing a node."""
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext(), torch.no_grad():
        dev = components.device
        return components.model(input_ids=input_ids.to(dev), attention_mask=attention_mask.to(dev),
                                prediction_positions=positions.to(dev)).logits
