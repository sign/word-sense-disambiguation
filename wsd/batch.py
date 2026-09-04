"""Offline disambiguation of a sentence corpus: one input line -> one JSON line.

    python -m wsd.batch --input 'corpus/*.txt' --output-dir out/ [--batch-size 2048] [--no-entities]

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
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from wsd.env import detach_from_torchrun


def _start_mps() -> None:
    """Start CUDA MPS for this node so the spaCy process and the model process on
    each GPU share it concurrently instead of time-slicing contexts (measured:
    model +9%, spaCy +10% when both run on one H100). Rank 0 starts the daemon,
    the others wait for it; a missing binary or WSD_MPS=0 leaves things as is."""
    if os.environ.get("WSD_MPS", "1") != "1" or not shutil.which("nvidia-cuda-mps-control"):
        return
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/wsd-mps-pipe")
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", "/tmp/wsd-mps-log")
    for d in (os.environ["CUDA_MPS_PIPE_DIRECTORY"], os.environ["CUDA_MPS_LOG_DIRECTORY"]):
        os.makedirs(d, exist_ok=True)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=False, capture_output=True)  # no-op if running
    time.sleep(3)  # let the daemon come up before any process touches CUDA


_start_mps()
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

from wsd import word_sense_disambiguation  # noqa: E402
from wsd.word_sense_disambiguation import disambiguate_docs, light_doc  # noqa: E402

# Time spent inside the model call, to split the WSD stage into model vs host work in the per-file log.
# Patched where it is looked up (word_sense_disambiguation imported the name).
_model_seconds = 0.0
_unmask = word_sense_disambiguation.unmask_token_batch


def _timed_unmask(texts):
    global _model_seconds
    t0 = time.time()
    try:
        return _unmask(texts)
    finally:
        _model_seconds += time.time() - t0


word_sense_disambiguation.unmask_token_batch = _timed_unmask


def _read_batches(src, batch_size: int):
    while True:
        # range first: with the file first, zip would consume (and drop) a line when the range runs out
        texts = [line.rstrip("\n") for _, line in zip(range(batch_size), src, strict=False)]
        texts = [t for t in texts if t.strip()]
        if not texts:
            return
        yield texts


def _spacy_worker(paths: list[str], batch_size: int, entities: bool, queue: mp.Queue,
                  worker: int, workers: int) -> None:
    """Parse every ``workers``-th batch of each file and hand picklable docs to the model process.

    Runs in its own process (spawned, so it gets its own CUDA/cupy context on the
    same GPU): in one process spaCy and the model fight over the GIL and nothing
    overlaps. spaCy is mostly Python/CPU-bound with a small GPU footprint, so
    several workers per GPU scale it; the consumer round-robins their queues,
    which keeps the output in input order. Workers live for all of a rank's
    files: loading the pipeline costs ~15 s, too much to pay per file.
    """
    # Under MPS, confine this client to a share of the SMs: spaCy's ~50k tiny kernels per 3k sentences otherwise
    # take the whole GPU in turns with the model. Must be set before the CUDA context is created.
    os.environ.setdefault("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", os.environ.get("WSD_SPACY_SM_PERCENT", "25"))
    from wsd.spacy_utils import run_spacy_pipe

    for path in paths:
        with open(path) as src:
            for i, texts in enumerate(_read_batches(src, batch_size)):
                if i % workers != worker:
                    continue
                t0 = time.time()
                docs = [light_doc(d) for d in run_spacy_pipe(texts, batch_size=batch_size, entities=entities)]
                queue.put((texts, docs, time.time() - t0))
        queue.put(None)  # end of this file for this worker


class SpacyPool:
    """``spacy_workers`` persistent spaCy processes feeding batches of the given files, in file order."""

    def __init__(self, paths: list[Path], batch_size: int, entities: bool, spacy_workers: int):
        ctx = mp.get_context("spawn")
        self.queues = [ctx.Queue(maxsize=2) for _ in range(spacy_workers)]
        self.workers = [
            ctx.Process(target=_spacy_worker, daemon=True,
                        args=([str(p) for p in paths], batch_size, entities, q, k, spacy_workers))
            for k, q in enumerate(self.queues)
        ]
        for w in self.workers:
            w.start()

    def batches(self):
        """Batches of the next file in input order: worker k produced batches k, k+n, k+2n, ..."""
        finished: set[int] = set()
        i = 0
        while len(finished) < len(self.queues):
            k = i % len(self.queues)
            i += 1
            if k in finished:
                continue
            item = self.queues[k].get()
            if item is None:
                finished.add(k)
                continue
            yield item

    def close(self):
        for w in self.workers:
            w.join()


def process_file(path: Path, out_path: Path, pool: SpacyPool, skip_single_sense: bool, log) -> tuple[int, int]:
    """Disambiguate one file (whose batches the pool is producing next); returns ``(sentences, prompts)``."""
    global _model_seconds
    tmp = out_path.with_suffix(".jsonl.tmp")
    n_sentences = 0
    t_spacy = t_wsd = 0.0
    _model_seconds = 0.0
    start = time.time()

    def write(texts, results, out):
        count = 0
        for text, result in zip(texts, results, strict=True):
            count += sum(tok.confidence is not None for tok in result.tokens)
            out.write(json.dumps({"text": text, **asdict(result)}) + "\n")
        return count

    # JSON serialization of batch i runs on a writer thread while batch i+1 is
    # tokenized (GIL released in Rust) and launched; one worker keeps file order.
    with open(tmp, "w") as out, ThreadPoolExecutor(max_workers=1) as writer:
        writes = []
        for texts, docs, dt_spacy in pool.batches():
            t1 = time.time()
            results = disambiguate_docs(docs, skip_single_sense=skip_single_sense)
            t2 = time.time()
            writes.append(writer.submit(write, texts, results, out))
            n_sentences += len(texts)
            t_spacy += dt_spacy
            t_wsd += t2 - t1
        n_prompts = sum(f.result() for f in writes)
    os.replace(tmp, out_path)
    elapsed = time.time() - start
    log(f"{path.name}: {n_sentences} sentences, {n_prompts} prompts in {elapsed:.0f}s "
        f"({n_sentences / max(elapsed, 1e-9):.0f} sent/s; spacy {t_spacy:.0f}s summed over workers, "
        f"wsd {t_wsd:.0f}s of which model {_model_seconds:.0f}s)")
    return n_sentences, n_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="glob of input text files (one sentence per line)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="sentences per spaCy/model batch (>= 2048 keeps each model call above one slice)")
    parser.add_argument("--no-entities", action="store_true", help="skip the (CPU-bound) entity linker")
    parser.add_argument("--spacy-workers", type=int, default=2,
                        help="spaCy processes per GPU (spaCy is mostly CPU-bound; 2 keeps up with the model)")
    parser.add_argument("--skip-single-sense", action="store_true",
                        help="assign words with one candidate sense directly instead of asking the model "
                             "(about 20%% fewer prompts)")
    parser.add_argument("--nodes", type=int, help=argparse.SUPPRESS)  # appended by run_distributed.py
    args = parser.parse_args()

    rank, world = RANK, WORLD
    import torch

    if torch.cuda.is_available():
        # The caching allocator otherwise grows to most of the GPU (72 GB seen) and
        # starves the spaCy worker processes sharing the device (~3 GB each).
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("WSD_GPU_MEM_FRACTION", "0.7")))
    files = sorted(Path(p) for p in glob.glob(args.input))[rank::world]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    total_sentences = total_prompts = 0
    start = time.time()
    todo = [p for p in files if not (args.output_dir / f"{p.stem}.jsonl").exists()]
    pool = SpacyPool(todo, args.batch_size, not args.no_entities, args.spacy_workers)
    for path in todo:
        s, p = process_file(path, args.output_dir / f"{path.stem}.jsonl", pool, args.skip_single_sense, log)
        total_sentences += s
        total_prompts += p
    pool.close()
    elapsed = time.time() - start
    log(f"done: {total_sentences} sentences, {total_prompts} prompts in {elapsed:.0f}s "
        f"({total_sentences / max(elapsed, 1e-9):.0f} sent/s, {total_prompts / max(elapsed, 1e-9):.0f} prompts/s)")


if __name__ == "__main__":
    main()
