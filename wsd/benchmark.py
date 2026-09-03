"""Accuracy + speed benchmark on WordNet's own example sentences.

Every polysemous (word form, example sentence) pair in OMW English is a test
case; the correct answer is the synset the example came from. Needs the WordNet
API (``WORDNET_URL``, see README) and the ``wn`` library (``pip install ".[benchmark]"``).

    python -m wsd.benchmark                      # whole set (~27.8k examples)
    python -m wsd.benchmark --split eval         # the held-out slice training evaluates on
    python -m wsd.benchmark --failures fails.jsonl

Under torchrun the examples are sharded across ranks and the counts gathered,
so an 8-GPU node benchmarks 8x faster.
"""
import argparse
import fcntl
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import wn
from tqdm import tqdm

from wsd.prompt import SentenceAlreadyMarkedError, WordNotFoundError, mark_word_in_sentence
from wsd.word_sense_disambiguation import (
    DisambiguationInput,
    WordQuery,
    disambiguate_word_batch,
    get_definitions,
)


@cache
def load_wn_english() -> wn.Wordnet:
    """Plain ``omw-en:1.4`` via the ``wn`` library, used only to iterate WordNet's
    own example sentences (definitions for prompts always come from the API).

    Kept in its own node-local data directory, separate from any other lexicon
    the installed ``wn`` may hold (the API image merges Wikidata lexemes into
    omw-en), so the example set and the seeded held-out split stay identical
    across environments. Local disk also matters: iterating all synsets is
    ~500k sqlite queries, ~6 ms each over a network filesystem vs ~0.1 ms locally.
    """
    data_dir = Path(tempfile.gettempdir()) / "wsd-benchmark-wn"
    data_dir.mkdir(exist_ok=True)
    wn.config.data_directory = data_dir
    with open(data_dir / ".lock", "w") as lock:  # one download per node, even with one process per GPU
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not wn.lexicons(lexicon="omw-en:1.4"):
            wn.download("omw-en:1.4")
    return wn.Wordnet(lexicon="omw-en:1.4")


@dataclass
class WordNetExample:
    synset_id: str
    word_form: str
    lemma: str
    pos: str
    marked_text: str  # sentence with *word* markers
    sentence: str  # original, unmarked example


def collect_wordnet_examples():
    """Yield every usable (synset, word form, example) tuple from OMW English.

    Skips monosemous target words (nothing to disambiguate) and sentences where
    the form can't be marked with clean word boundaries.
    """

    en = load_wn_english()

    # Iterate through all synsets
    for synset in en.synsets():
        examples = synset.examples()
        if len(examples) == 0:
            continue

        for word in synset.words():
            # Skip monosemous target words — nothing to disambiguate.
            if len(word.synsets()) <= 1:
                continue
            for form in word.forms():
                for example in examples:
                    try:
                        marked_text = mark_word_in_sentence(example, form)
                    except (WordNotFoundError, SentenceAlreadyMarkedError):
                        continue
                    yield WordNetExample(
                        synset.id, form, word.lemma(), word.pos, marked_text, example
                    )


def evaluate(examples: list[WordNetExample], batch_size: int = 64, failures_path: str | None = None,
             progress: bool = True) -> tuple[int, int, float]:
    """Return ``(correct, total, seconds)`` over ``examples``; optionally dump misses as JSONL."""
    correct = 0
    fails = open(failures_path, "w") if failures_path else None
    start = time.time()
    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating", disable=not progress):
        batch = examples[i:i + batch_size]
        all_definitions = get_definitions([WordQuery(form=ex.lemma, pos=ex.pos) for ex in batch])
        batch_data = [
            DisambiguationInput(word=ex.word_form, marked_sentence=ex.marked_text, definitions=defs)
            for ex, defs in zip(batch, all_definitions, strict=True)
        ]
        predictions = disambiguate_word_batch(batch_data)
        for ex, defs, result in zip(batch, all_definitions, predictions, strict=True):
            if result.synset_id == ex.synset_id:
                correct += 1
            elif fails:
                gold = next((d.definition for d in defs if d.synset_id == ex.synset_id), None)
                fails.write(json.dumps({
                    **asdict(ex), "n_options": len(defs), "gold_definition": gold,
                    "predicted_id": result.synset_id, "predicted_definition": result.definition,
                    "confidence": result.confidence,
                }) + "\n")
    if fails:
        fails.close()
    return correct, len(examples), time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["all", "eval", "benchmark"], default="all",
                        help="'eval' = the held-out slice training uses; 'benchmark' = the rest")
    parser.add_argument("--n-eval", type=int, default=5000, help="size of the held-out slice")
    parser.add_argument("--seed", type=int, default=42, help="seed of the held-out split")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--failures", type=str, help="write mispredictions to this JSONL file")
    parser.add_argument("--limit", type=int, help="only evaluate this many examples")
    args, _ = parser.parse_known_args()  # tolerate the launcher's --nodes

    if args.split == "all":
        examples = list(collect_wordnet_examples())
    else:
        from training.wn_data import split

        eval_examples, benchmark_examples = split(n_eval=args.n_eval, seed=args.seed)
        examples = eval_examples if args.split == "eval" else benchmark_examples
    random.Random(args.seed).shuffle(examples)  # varied batches; deterministic
    if args.limit:
        examples = examples[:args.limit]

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("gloo")
        shard = examples[rank::world]
        failures = f"{args.failures}.{rank}" if args.failures else None
        correct, total, seconds = evaluate(shard, args.batch_size, failures, progress=rank == 0)
        gathered = [None] * world
        dist.all_gather_object(gathered, (correct, total, seconds))
        dist.destroy_process_group()
        if rank != 0:
            return
        correct = sum(g[0] for g in gathered)
        total = sum(g[1] for g in gathered)
        seconds = max(g[2] for g in gathered)
        if args.failures:
            with open(args.failures, "w") as out:
                for r in range(world):
                    with open(f"{args.failures}.{r}") as part:
                        out.write(part.read())
                    os.remove(f"{args.failures}.{r}")
    else:
        correct, total, seconds = evaluate(examples, args.batch_size, args.failures)

    print(f"split={args.split} n={total} accuracy={correct / max(total, 1):.4f} "
          f"time={seconds:.0f}s ({total / max(seconds, 1e-9):.0f} examples/s over {world} GPU(s))")


if __name__ == "__main__":
    main()
