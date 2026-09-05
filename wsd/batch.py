"""Offline disambiguation of a sentence corpus: one input line -> one JSON line.

    python -m wsd.batch --input 'corpus/*.txt' --output-dir out/ [--batch-size 512] [--no-entities]

Input files hold one sentence per line. Output ``<output-dir>/<name>.jsonl``
lines have the same schema as the server's JSON response. Under torchrun each
rank takes every WORLD_SIZE-th input file, so pre-split a large corpus into many
files (``split -n l/256 corpus.txt shard-``) and launch on as many GPUs as you
like. Finished outputs are skipped, so a killed job resumes where it stopped.

Per batch of sentences: spaCy (GPU) -> one WordNet lookup for all content words
(WordNet API batch endpoint) -> one model batch for all prompts -> write.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path

from wsd.env import detach_from_torchrun

# Each rank is an independent single-GPU worker (spaCy's transformer and the
# WSD model must share a device); must run before torch/cupy are imported.
RANK, WORLD = detach_from_torchrun()
# Thousands of prompts per batch: big length-sorted chunks win even without flash-attn (H100: 256 ~ 512).
os.environ.setdefault("WSD_CHUNK_SIZE", "256")
os.environ.setdefault("WSD_COMPILE", "1")  # ~1.5x model throughput after a one-time compile
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
# The BPE tokenizer's rayon pool contends on a shared cache: 224 threads burn 185 CPU-s for a 0.9 s call,
# 16 threads take 0.22 s. Must be set before `tokenizers` initializes (i.e. before transformers is imported).
os.environ.setdefault("RAYON_NUM_THREADS", "16")

from wsd.word_sense_disambiguation import disambiguate_docs, light_doc  # noqa: E402


def _read_batches(src, batch_size: int):
    while True:
        # range first: with the file first, zip would consume (and drop) a line when the range runs out
        texts = [line.rstrip("\n") for _, line in zip(range(batch_size), src, strict=False)]
        texts = [t for t in texts if t.strip()]
        if not texts:
            return
        yield texts


def _spacy_worker(path: str, batch_size: int, entities: bool, queue: mp.Queue) -> None:
    """Parse one file in batches and hand picklable docs to the model process.

    Runs in its own process (spawned, so it gets its own CUDA/cupy context on the
    same GPU): in one process spaCy and the model fight over the GIL and nothing
    overlaps; as a separate process spaCy's ~20% is hidden behind the model.
    """
    from wsd.spacy_utils import run_spacy_pipe

    with open(path) as src:
        for texts in _read_batches(src, batch_size):
            t0 = time.time()
            docs = [light_doc(d) for d in run_spacy_pipe(texts, batch_size=batch_size, entities=entities)]
            queue.put((texts, docs, time.time() - t0))
    queue.put(None)


def process_file(path: Path, out_path: Path, batch_size: int, entities: bool, skip_single_sense: bool,
                 log) -> tuple[int, int]:
    """Disambiguate one file; returns ``(sentences, prompts)``."""
    tmp = out_path.with_suffix(".jsonl.tmp")
    n_sentences = n_prompts = 0
    t_spacy = t_wsd = 0.0
    start = time.time()
    queue: mp.Queue = mp.get_context("spawn").Queue(maxsize=2)
    worker = mp.get_context("spawn").Process(
        target=_spacy_worker, args=(str(path), batch_size, entities, queue), daemon=True,
    )
    worker.start()
    with open(tmp, "w") as out:
        while (item := queue.get()) is not None:
            texts, docs, dt_spacy = item
            t1 = time.time()
            results = disambiguate_docs(docs, skip_single_sense=skip_single_sense)
            t2 = time.time()
            for text, result in zip(texts, results, strict=True):
                n_prompts += sum(tok.confidence is not None for tok in result.tokens)
                out.write(json.dumps({"text": text, **asdict(result)}) + "\n")
            n_sentences += len(texts)
            t_spacy += dt_spacy
            t_wsd += t2 - t1
    worker.join()
    os.replace(tmp, out_path)
    elapsed = time.time() - start
    log(f"{path.name}: {n_sentences} sentences, {n_prompts} prompts in {elapsed:.0f}s "
        f"({n_sentences / max(elapsed, 1e-9):.0f} sent/s; spacy {t_spacy:.0f}s (overlapped), wsd {t_wsd:.0f}s)")
    return n_sentences, n_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="glob of input text files (one sentence per line)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512, help="sentences per spaCy/model batch")
    parser.add_argument("--no-entities", action="store_true", help="skip the (CPU-bound) entity linker")
    parser.add_argument("--skip-single-sense", action="store_true",
                        help="assign words with one candidate sense directly instead of asking the model "
                             "(about 20%% fewer prompts)")
    parser.add_argument("--nodes", type=int, help=argparse.SUPPRESS)  # appended by run_distributed.py
    args = parser.parse_args()

    rank, world = RANK, WORLD
    files = sorted(Path(p) for p in glob.glob(args.input))[rank::world]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    total_sentences = total_prompts = 0
    start = time.time()
    for path in files:
        out_path = args.output_dir / f"{path.stem}.jsonl"
        if out_path.exists():
            continue
        s, p = process_file(path, out_path, args.batch_size, not args.no_entities, args.skip_single_sense, log)
        total_sentences += s
        total_prompts += p
    elapsed = time.time() - start
    log(f"done: {total_sentences} sentences, {total_prompts} prompts in {elapsed:.0f}s "
        f"({total_sentences / max(elapsed, 1e-9):.0f} sent/s, {total_prompts / max(elapsed, 1e-9):.0f} prompts/s)")


if __name__ == "__main__":
    main()
