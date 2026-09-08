"""Run the pipeline's CPU side (spaCy + lookups + prompt building) over a corpus and dump the model prompts,
one JSON line each, for unlabeled distillation (train with --unlabeled-prompts OUT --teacher DIR).
Usage: python scripts/dump_prompts.py IN_GLOB OUT_JSONL [MAX_SENTENCES]"""
import glob
import json
import sys

from thinc.api import use_ops

import wsd.spacy_utils as su
import wsd.word_sense_disambiguation as w
from wsd.masked_language_model import UnmaskResult

su._PIPELINE_MODELS["en"] = "en_core_web_lg"  # CPU pipeline; entity linking is skipped below
files = sorted(glob.glob(sys.argv[1]))
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
captured: list[str] = []


def fake_unmask(prompts):
    """Record the prompts instead of running the model."""
    captured.extend(prompts)
    return [UnmaskResult(token="A", probabilities=[1.0] + [0.0] * 127) for _ in prompts]


w.unmask_token_batch = fake_unmask
n = 0
with open(sys.argv[2], "w") as out:
    for f in files:
        with open(f) as src:
            sents = [line.rstrip("\n") for line in src][:max(0, limit - n)]
        if not sents:
            break
        for i in range(0, len(sents), 2000):
            chunk = sents[i:i + 2000]
            with use_ops("numpy"):
                docs = [w.light_doc(d) for d in su.run_spacy_pipe(chunk, "en", 64, False)]
            captured.clear()
            w.disambiguate_docs(docs)
            for p in captured:
                out.write(json.dumps({"prompt": p}) + "\n")
            n += len(chunk)
            print(f"{n} sentences, {len(captured)} prompts in last chunk", flush=True)
