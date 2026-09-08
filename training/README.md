# Training

`training/train.py` fine-tunes a ModernBERT-architecture encoder to pick the right WordNet definition for a
marked word: a multiple-choice prompt (the repository's compact template) whose answer is one masked letter.
`python -m training.train --help` lists every option; the two recipes that produced the published models are
in `training/sweeps/2026-09-08-published.json`.

## Data

- Generated sentences for 97k synsets (`training/data/generate.py` → `training/data/generated.tar.xz`, read
  directly by `--data-dir`, the default). Each `<word>.json` lists the word's synsets with a source and an
  alternative definition and a few example sentences. One "none of the above" example per word shows the
  definitions of the most frequent POS with a sentence from another POS (`--no-nota-examples` drops them).
- `--wn-train`: WordNet's own example sentences, minus the held-out eval slice (`--eval-wn-count`, default
  5,000, the same split as `python -m wsd.benchmark --split eval`).
- `--semcor PREFIX --sense-index dict/index.sense`: a corpus in Raganato et al. (2017) XML format, e.g. SemCor
  from http://lcl.uniroma1.it/wsdeval/ (sense keys are mapped to `omw-en` ids through WordNet 3.0's `index.sense`).
- `--wngt glosstag/ [--wngt-parts def,ex] [--wngt-tags man,auto]`: the Princeton WordNet Gloss Corpus
  (https://wordnetcode.princeton.edu/glosstag-files/WordNet-3.0-glosstag.tar.bz2); held-out synsets are skipped.
- `--unlabeled-prompts FILE --teacher DIR`: prompts without gold, dumped by `scripts/dump_prompts.py` from any
  corpus with the pipeline's own lookups; they train on the teacher's answer distribution alone.

Adjective (`a`) and satellite (`s`) senses form one option set, as at inference. Weights stay fp32 with bf16
autocast (pure-bf16 weights round away most updates). Flash attention 2 is used when installed.

## Distillation

`--teacher DIR` adds `alpha · T²·KL(teacher ‖ student)` over the 128 answer letters to `(1 - alpha)` times the
label cross-entropy (`--distill-alpha`, `--distill-temperature`); the frozen bf16 teacher runs in the same
process. Small encoders gain the most, and unlabeled in-domain prompts are what lift them further: the
published 150m (`sign/Ettin-150m-WSD`) is `jhu-clsp/ettin-encoder-150m` distilled from `sign/Ettin-1B-WSD`
on the labeled data plus 1.06M Wikipedia prompts (alpha 0.7, 3 epochs, lr 5e-5), 80.8% on SemEval ALL;
the same data without the Wikipedia prompts saturates at 80.0 ± 0.4.

## Sweeps on a Slurm node

`training/sweep.py` trains one config per GPU (no DDP) and benchmarks each result on the held-out slice and,
with `--eval-raganato`, on SemEval:

```shell
uv run --no-project --with nemo_run run_distributed.py --nodes 1 --nodelist cnodeXXX \
  --script training.sweep --image /mnt/nfs-1/amit/wsd/wsd-train.sqsh --image_name wsd-train --detach \
  --configs training/sweeps/2026-09-08-published.json \
  --eval-raganato .../WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL --sense-index .../dict/index.sense
```

Results land in `<output-root>/<config>/{train.log,result.json,final/}`. The image is built with
`training/Enrootfile.sh` from the NeMo base image and bundles the WordNet API, which each process starts for
itself when `WORDNET_URL` is unset. Measured results of every round are in `wsd/README.md`.

## Output

`<output-dir>/final/` holds the model, tokenizer and `answer_letters.json` (the compact decoder's letter order);
load it with `WSD_MODEL=<output-dir>/final` in any `wsd` entry point. Do not train longer on the generated
data alone: WordNet-example accuracy rises while real-text accuracy collapses.
