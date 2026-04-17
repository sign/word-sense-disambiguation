"""WordNet example loading for training-time eval and final benchmark.

Provides a single deterministic split between the eval subset (used during
training) and the benchmark subset (used for final evaluation). Training and
benchmark MUST use the same seed so the two disjoint sets never overlap.

The underlying iteration lives in :mod:`wsd.benchmark` so the loader and the
evaluation script can't drift apart.
"""
import random

from wsd.benchmark import WordNetExample, collect_wordnet_examples


def split(
    n_eval: int = 1000, seed: int = 42
) -> tuple[list[WordNetExample], list[WordNetExample]]:
    """Return (eval_examples, benchmark_examples) disjoint at the synset level.

    Examples sharing a ``synset_id`` are never split across the two sets, so the
    benchmark measures generalization to synsets unseen during training-time eval.
    """
    by_synset: dict[str, list[WordNetExample]] = {}
    for ex in collect_wordnet_examples():
        by_synset.setdefault(ex.synset_id, []).append(ex)

    synset_ids = list(by_synset.keys())
    rng = random.Random(seed)
    rng.shuffle(synset_ids)

    eval_examples: list[WordNetExample] = []
    benchmark_examples: list[WordNetExample] = []
    for sid in synset_ids:
        group = by_synset[sid]
        if len(eval_examples) < n_eval:
            eval_examples.extend(group)
        else:
            benchmark_examples.extend(group)
    return eval_examples, benchmark_examples
