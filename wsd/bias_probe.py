"""Probe whether the WSD model has a bias toward letter A (the first option).

Training always lays definitions on letters[0], letters[1], ... so the model
might be exploiting "the correct answer is one of the first few letters"
rather than actually attending to each (letter, definition) pairing. This
script shifts the entire option block to a non-zero ``start_offset`` and
measures how benchmark accuracy degrades.

The NOTA slot is held fixed at :data:`wsd.letters.NOTA_LETTER_INDEX` so that
the model still has a familiar "reject" signal even when the options live at
unfamiliar letters.

Definitions are pulled directly from the ``wn`` library (same backend as
``training.build_eval_examples_from_wn``) so this script is self-contained
— no WordNet HTTP service required.

Usage::

    WORDNET_URL=NONE python -m wsd.bias_probe \\
        --offsets 0 1 5 10 26 52 100 \\
        --n-examples 1000 \\
        --seed 42

(``WORDNET_URL`` is only referenced at import time by wsd.env; a dummy value
is fine here since we never hit the HTTP endpoint.)
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from tqdm import tqdm

from wsd.benchmark import (
    WordNetExample,
    collect_wordnet_examples,
    fetch_synset_definitions,
    load_wn_english,
)
from wsd.letters import NOTA_LETTER_INDEX
from wsd.prompt import Definition
from wsd.word_sense_disambiguation import (
    DisambiguationInput,
    disambiguate_word_batch,
)


@dataclass
class OffsetResult:
    offset: int
    n_evaluated: int  # examples whose option count fit under the offset
    n_skipped: int    # examples skipped because offset + len(defs) > NOTA
    n_correct: int
    n_nota: int       # times the model said "none of the above"

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_evaluated if self.n_evaluated else 0.0


def _sample_examples_with_defs(
    n_examples: int, seed: int,
) -> tuple[list[WordNetExample], list[list[Definition]]]:
    """Sample ``n_examples`` and fetch their definitions once (reused across offsets).

    Skips examples whose own ``synset_id`` isn't in the wn lookup — that happens
    when the lexicon disagrees with the example's metadata and we'd have no
    correct answer to compare against.
    """
    en = load_wn_english()

    all_examples = list(collect_wordnet_examples())
    rng = random.Random(seed)
    rng.shuffle(all_examples)

    out_ex: list[WordNetExample] = []
    out_defs: list[list[Definition]] = []
    for ex in all_examples:
        defs = fetch_synset_definitions(en, ex.lemma, ex.pos)
        if ex.synset_id not in defs:
            continue
        out_ex.append(ex)
        out_defs.append(
            [Definition(synset_id=sid, definition=text) for sid, text in defs.items()],
        )
        if len(out_ex) >= n_examples:
            break
    return out_ex, out_defs


def _evaluate_at_offset(
    examples: list[WordNetExample],
    all_definitions: list[list[Definition]],
    offset: int,
    batch_size: int,
) -> OffsetResult:
    """Run batched disambiguation at a single offset and count correctness."""
    # Skip any example whose option block would collide with NOTA at this offset.
    pairs = [
        (ex, defs) for ex, defs in zip(examples, all_definitions, strict=True)
        if offset + len(defs) <= NOTA_LETTER_INDEX
    ]
    skipped = len(examples) - len(pairs)

    correct = 0
    nota = 0
    for start in tqdm(
        range(0, len(pairs), batch_size), desc=f"offset={offset}", leave=False,
    ):
        chunk = pairs[start : start + batch_size]
        batch = [
            DisambiguationInput(
                word=ex.word_form, marked_sentence=ex.marked_text, definitions=defs,
            )
            for ex, defs in chunk
        ]
        preds = disambiguate_word_batch(batch, start_offset=offset)
        for (ex, _), pred in zip(chunk, preds, strict=True):
            if pred.synset_id == ex.synset_id:
                correct += 1
            if pred.synset_id == "":  # NOTA
                nota += 1

    return OffsetResult(
        offset=offset,
        n_evaluated=len(pairs),
        n_skipped=skipped,
        n_correct=correct,
        n_nota=nota,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offsets", type=int, nargs="+",
        default=[0, 1, 5, 10, 26, 52, 100],
        help="Letter-offset values to evaluate (each must leave room for NOTA).",
    )
    parser.add_argument(
        "--n-examples", type=int, default=1000,
        help="Number of wn examples to sample (deterministic given --seed).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Sampling seed for example selection.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size.",
    )
    args = parser.parse_args()

    print(f"Sampling {args.n_examples} examples (seed={args.seed})...")
    examples, all_definitions = _sample_examples_with_defs(args.n_examples, args.seed)
    print(
        f"Got {len(examples)} examples with definitions "
        f"(min defs per example = {min(len(d) for d in all_definitions)}, "
        f"max = {max(len(d) for d in all_definitions)})."
    )

    results: list[OffsetResult] = []
    for offset in args.offsets:
        r = _evaluate_at_offset(examples, all_definitions, offset, args.batch_size)
        results.append(r)
        print(
            f"offset={offset:>4}  "
            f"acc={r.accuracy:.4f}  "
            f"n={r.n_evaluated}  "
            f"skipped={r.n_skipped}  "
            f"nota={r.n_nota}"
        )

    # Final table
    print("\n" + "=" * 72)
    print(f"{'offset':>8} {'accuracy':>10} {'n':>8} {'skipped':>10} {'nota':>8}")
    print("=" * 72)
    for r in results:
        print(
            f"{r.offset:>8} {r.accuracy:>10.4f} {r.n_evaluated:>8} "
            f"{r.n_skipped:>10} {r.n_nota:>8}"
        )
    base = next((r.accuracy for r in results if r.offset == 0), None)
    if base is not None:
        print("\nDelta vs offset=0:")
        for r in results:
            delta = r.accuracy - base
            print(f"  offset={r.offset:>4}: {delta:+.4f}")


if __name__ == "__main__":
    main()
