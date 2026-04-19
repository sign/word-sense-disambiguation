"""Minimal inference benchmark for the HuggingFace transformers backend.

Measures end-to-end :func:`disambiguate_word_batch` latency and throughput
at a few batch sizes using realistic prompts sampled from WordNet (same
path as wsd.benchmark and wsd.bias_probe). Baseline numbers to decide
whether ONNX/TensorRT backends are worth pursuing.

Usage::

    WORDNET_URL=NONE python -m wsd.bench_inference \\
        --batch-sizes 1 8 32 --warmup 3 --iters 20 --n-examples 200
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import wn

from wsd.benchmark import WordNetExample, collect_wordnet_examples
from wsd.masked_language_model import load_model
from wsd.prompt import Definition
from wsd.word_sense_disambiguation import (
    DisambiguationInput,
    disambiguate_word_batch,
)


def _load_wn() -> wn.Wordnet:
    try:
        return wn.Wordnet(lexicon="omw-en:1.4")
    except wn.Error:
        wn.download("omw-en:1.4")
        return wn.Wordnet(lexicon="omw-en:1.4")


def _defs_for(en: wn.Wordnet, ex: WordNetExample) -> list[Definition] | None:
    pos_options = {"a", "s"} if ex.pos == "a" else {ex.pos}
    defs: dict[str, str] = {}
    for word in en.words(form=ex.lemma):
        for synset in word.synsets():
            if synset.pos in pos_options and synset.definition():
                defs[synset.id] = synset.definition()
    if ex.synset_id not in defs:
        return None
    return [Definition(synset_id=sid, definition=t) for sid, t in defs.items()]


def _build_inputs(n: int) -> list[DisambiguationInput]:
    en = _load_wn()
    out: list[DisambiguationInput] = []
    for ex in collect_wordnet_examples():
        defs = _defs_for(en, ex)
        if defs is None:
            continue
        out.append(DisambiguationInput(
            word=ex.word_form, marked_sentence=ex.marked_text, definitions=defs,
        ))
        if len(out) >= n:
            break
    return out


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_batch(batch: list[DisambiguationInput]) -> float:
    _sync()
    t0 = time.perf_counter()
    disambiguate_word_batch(batch)
    _sync()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--n-examples", type=int, default=200,
                   help="Pool of distinct examples to cycle through.")
    p.add_argument("--bucket", action="store_true",
                   help="Sort pool by prompt token length (reduces padding waste "
                        "when neighbouring examples are batched together).")
    args = p.parse_args()

    print(f"Loading model (device resolved inside load_model)...")
    comp = load_model()
    print(f"  device: {comp.device}, dtype: {next(comp.model.parameters()).dtype}")

    print(f"Building {args.n_examples} inputs from WordNet...")
    pool = _build_inputs(args.n_examples)
    print(f"  got {len(pool)} inputs")

    # Tokenize the whole pool so we have lengths both for reporting and for
    # optional bucketing.
    tok = comp.tokenizer
    from wsd.prompt import create_multiple_choice_prompt
    all_lengths = []
    for ex in pool:
        prompt = create_multiple_choice_prompt(
            word=ex.word, mask_token=tok.mask_token,
            marked_sentence=ex.marked_sentence, definitions=ex.definitions,
            tokenizer=tok,
        )
        all_lengths.append(len(tok(prompt)["input_ids"]))
    print(
        f"  prompt tokens: min={min(all_lengths)} "
        f"p50={statistics.median(all_lengths)} "
        f"p95={sorted(all_lengths)[int(0.95*len(all_lengths))]} "
        f"max={max(all_lengths)}"
    )

    if args.bucket:
        order = sorted(range(len(pool)), key=lambda i: all_lengths[i])
        pool = [pool[i] for i in order]
        print("  pool sorted by prompt length (bucketing enabled)")

    print(f"\n{'batch':>6} {'warmup_s':>10} {'mean_ms':>10} "
          f"{'p50_ms':>9} {'p95_ms':>9} {'ex/s':>8}")
    print("-" * 60)

    for bs in args.batch_sizes:
        # Round-robin the pool to get distinct batches each iter.
        batches = [
            [pool[(i * bs + j) % len(pool)] for j in range(bs)]
            for i in range(args.warmup + args.iters)
        ]

        wt0 = time.perf_counter()
        for b in batches[: args.warmup]:
            _time_batch(b)
        warmup_s = time.perf_counter() - wt0

        times = [_time_batch(b) for b in batches[args.warmup :]]
        times_ms = sorted(t * 1000 for t in times)
        mean_ms = statistics.mean(times_ms)
        p50 = times_ms[len(times_ms) // 2]
        p95 = times_ms[int(0.95 * len(times_ms))]
        throughput = bs / (mean_ms / 1000)

        print(f"{bs:>6d} {warmup_s:>10.2f} {mean_ms:>10.2f} "
              f"{p50:>9.2f} {p95:>9.2f} {throughput:>8.1f}")


if __name__ == "__main__":
    main()
