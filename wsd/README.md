# Word Sense Disambiguation (WSD)

A Python implementation of word sense disambiguation using ModernBERT and spaCy for natural language processing.
This tool identifies the correct meaning of ambiguous words in context by leveraging dictionary definitions and
transformer-based language models.

This is designed for synthetic data generation, and can run at about 1 sentence per second.
If this needs to be done in production, a word-vector search approach would be much more efficient.

![Example of word sense disambiguation](assets/wsd-example.png)

## Possible Improvements

- [ ] For every definition, also include hypernym and other forms for the same synset
- [ ] Automatically search for a better prompt over a smaller benchmark dataset
- [ ] Use the benchmark dataset as a training dataset, and fine tune the model (with shuffled definitions).
- [ ] Use https://huggingface.co/swap-uniba/LLM-wsd-FT-ALL to generate training data
- [ ] Also search for noun phrases like "bus driver" or "bass player" and disambiguate the whole phrase
- [ ] Batch processing of words within a sentence to improve performance

## Example

For the sentence:
> The bass player adjusted the bass on his amplifier while fishing for bass.

For each content word, we generate a prompt: the sentence with the word marked, one WordNet definition per
option letter, a fixed "none of the above" letter, and the answer slot:

```txt
The *bass* player adjusted the bass on his amplifier while fishing for bass.
A. the lowest part of the musical range
B. the lowest part in polyphonic music
C. an adult male singer with the lowest voice
D. the lean flesh of a saltwater fish of the family Serranidae
E. any of various North American freshwater fish with lean flesh (especially of the genus Micropterus)
F. the lowest adult male singing voice
G. the member with the lowest range of a family of musical instruments
H. nontechnical name for any of numerous edible marine and freshwater spiny-finned fishes
а. none of the above
[unused0] [MASK]
```

The results show reasonable word sense disambiguation:

- "bass" (position 1): "the lowest part of the musical range" (22% confidence) ✓ Correct - musical context
- "player": "someone who plays a musical instrument (as a profession)" (81% confidence) ✓ Correct - high confidence
- "adjusted": "alter or regulate so as to achieve accuracy or conform to a standard" (48% confidence) ✓ Correct -
  moderate confidence
- "bass" (position 5): "the lowest part of the musical range" (44% confidence) ✓ Correct - audio equipment context
- "amplifier": "electronic equipment that increases strength of signals passing through it" (70% confidence) ✓ Correct -
  strong confidence
- "fishing": "catch or try to catch fish or shellfish" (87% confidence) ✓ Correct - very high confidence
- "bass" (position 12): "the lowest part of the musical range" (30% confidence) ✗ Wrong - should be fish in fishing
  context

## Benchmark

Benchmarking helps us understand whether a new model, prompt, or any change improves or
hurts the performance of the method.

In `benchmark.py`, we collect all non-trivial examples from the English WordNet -
cases where a word form has multiple possible meanings and appears in the example text.
We automatically mark target words with asterisks (*word*) in the example sentences.

For each example, we perform WSD given the marked sentence, the lemma, and the part-of-speech tag,
then compare the predicted synset ID against the ground truth.

Install the `wn` library (`pip install ".[benchmark]"`) and have the WordNet API running (`WORDNET_URL`, see the
top-level README), then:

```shell
python -m wsd.benchmark                       # all 27.8k examples
python -m wsd.benchmark --split eval          # the 5k-example slice held out during training
python -m wsd.benchmark --failures fails.jsonl  # dump mispredictions for analysis
python -m wsd.benchmark --raganato .../Evaluation_Datasets/ALL/ALL --sense-index .../dict/index.sense  # SemEval "ALL"
```

Under `torchrun` the examples are sharded across GPUs. With flash attention on NVIDIA DGX Spark, set
`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

| Device           | Model      | Time      | Accuracy | Notes                                    |
|------------------|------------|-----------|----------|------------------------------------------|
| Macbook Pro M4   | ModernBERT | 00:36:16  | 54.0%    | Initial test                             |
| NVIDIA DGX Spark | ModernBERT | 00:19:06  | 53.8%    | On GPU (float16)                         |
| NVIDIA DGX Spark | ModernBERT | 00:15:05  | 54.6%    | Batched (32)                             |
| NVIDIA DGX Spark | ModernBERT | 00:11:37  | 54.6%    | With Flash Attention                     |
| NVIDIA DGX Spark | none       | 00:08:36  | 0%       | Just Definitions                         |
| NVIDIA DGX Spark | none       | 00:01:02  | 0%       | Definitions Batch Endpoint               |
| NVIDIA DGX Spark | ModernBERT | 00:03:27  | 38.5%    | Batch size 64                            |
| NVIDIA DGX Spark | ModernBERT | 00:03:37  | 58.1%    | Prompt Optimizations                     |
| NVIDIA DGX Spark | ModernBERT | 00:03:37  | 66.0%    | After 1 Epoch (56320 sentences, all pos) |
| NVIDIA DGX Spark | ModernBERT | 00:04:59* | 67.1%    | After 1 Epoch (300k+ sentences)          |

*machine under other load, time is not reliable

### Accuracy on real text (2026-09)

WordNet's example sentences are short phrases; the corpus we actually process is running text. The two
disagree, so we also report the standard SemEval "ALL" set (Senseval-2/3, SemEval-07/13/15; 7,247 instances,
any gold key accepted, natural punctuation). Held-out slice = `--split eval` (5,000 WordNet examples, seed 42,
never trained on). Trained on 8xH100 via `training/sweep.py`; configs in `training/sweeps/`.

| Model                                                                  | WN held-out | SemEval ALL |
|------------------------------------------------------------------------|------------:|------------:|
| `sign/ModernBERT-Large-Instruct-WSD` (published, 2026-04)              | 70.0%       | 59.2%       |
| generated data only, fixed recipe, 1 epoch                             | 75.4%       | 53.4%       |
| generated data only, 3 epochs                                          | 75-76%      | 40-45%      |
| + WordNet sentences + SemCor (space-tokenized), 2 epochs (S5)          | 78.1%       | 68.0%       |
| + WordNet sentences + SemCor (detokenized), 2 epochs (R5)              | 77.8%       | 80.4%       |
| + WordNet Gloss Corpus, manual tags, 2 epochs (W4)                     | 78.6%       | 80.7%       |
| **same with the compact prompt template, lr 2e-5 (C3; +20% throughput; published)** | **78.3%** | **80.6%** |
| + WordNet Gloss Corpus, definitions only (W3) / all tags (W1)          | 78.1%       | 80.9% / 80.4% |
| same, 3 epochs (R1)                                                    | 77.8%       | 79.7%       |
| same, 4 epochs, lr 2e-5 (R4)                                           | 77.8%       | 78.9%       |
| same, without cross-POS "none of the above" examples (R2)              | 77.3%       | 79.3%       |
| SemCor+OMSTI (1.1M instances) instead of SemCor, 1 epoch (R3)          | 77.6%       | 79.1%       |
| ModernBERT-base, same data, 3 epochs (R7; ~2.5x cheaper per prompt)    | 73.5%       | 77.9%       |

Recipe for W4: `--wn-train --semcor SemCor/semcor --wngt glosstag --wngt-tags man --sense-index dict/index.sense
--lr-scheduler cosine --label-smoothing 0.1 --weight-decay 0.01 --learning-rate 3e-5 --num-epochs 2 --batch-size 64`.
The gloss corpus adds ~0.5 points on both benchmarks; its variants (definitions only, all tags, lr 2e-5/4e-5)
are within noise of each other; 3 epochs (77.6% / 80.3%) and 1 epoch (77.5% / 79.9%) are both worse than 2.

What we learned: more epochs on the synthetic data alone overfit its style and destroy real-text accuracy;
SemCor (222k gold-annotated sentences) fixes that, and it must be detokenized to match natural text (S5 vs R5).
The published model's largest failure class on real text was over-predicting "none of the above" (34% of its
misses); the SemCor-trained models almost never do (0.2%), so `WSD_NOTA_THRESHOLD` is no longer needed for them. Remaining errors are
mostly fine-grained sense splits ("shake, as from cold" vs "tremble, as from fear").

## Throughput

Offline batch pipeline (`python -m wsd.batch`), 100k Wikipedia sentences (~10.7 model prompts per sentence),
one process per GPU, steady state per H100 80GB. spaCy `en_core_web_trf` runs on the GPU too.

| Configuration                                                   | Sentences/s per GPU | Notes                                            |
|-----------------------------------------------------------------|--------------------:|--------------------------------------------------|
| Original code (`WSD_CHUNK_SIZE=512`)                            | 27                  | WordNet sqlite over NFS dominated (6 ms/lookup)  |
| + local `wn.db` copy, tokenize once, CPU probabilities          | 55                  |                                                  |
| + `WSD_COMPILE=1` (torch.compile, default in `wsd.batch`)       | 75                  | one-time ~60s compile per process                |
| definitions via the WordNet API instead of the local file       | 60                  | ~0.35 ms of server work per query                |
| + memoized API lookups, 16 tokenizer threads, spaCy in its own process | 125          | output identical; entity linking on    |
| + tokenize/pad the next slice while the GPU runs (vectorized pad), CUDA MPS, 2 persistent spaCy workers per GPU | 160-195 | output identical |
| + 2,048 sentences per batch (model calls span several tokenization slices) | **175-230** | output identical |
| + compact prompt template (retrained model, now `main` on the Hub)     | 205-265     | 78.3% / 80.6% vs 78.6% / 80.7% |
| `--skip-single-sense`                                           | ~+20% (est.)        | 1-sense words assigned directly (20% of prompts) |

A billion sentences at ~1,600 sentences/s per 8-GPU node is roughly 7 node-days.

Where the forward pass stands (one H100, 32.6k real prompts, `torch.profiler` + `nvidia-smi dmon`): the model with
pre-padded inputs runs at 4,100 prompts/s at 99% SM utilization, so the kernels (GEMM ~55%, compile-fused
elementwise ~25%, cuDNN SDPA attention ~10%; padding waste 1.6%) are near the practical ceiling for these shapes.
End to end the same call ran at 2,700 prompts/s at 63% utilization; the gap was host work: HF `tokenizer.pad`
in Python per chunk, and tokenization done before any GPU work. `unmask_token_batch` now tokenizes and pads one
8,192-prompt slice while the previous slice's kernels run and pads with numpy: 3,700 prompts/s at 96%. A spaCy
process on the same GPU costs the model 16%; CUDA MPS (`wsd.batch` starts it) cuts that to 8%, and two
persistent spaCy workers per GPU keep spaCy from pacing the pipeline.

Also measured: spaCy's transformer in mixed precision (`WSD_SPACY_MIXED_PRECISION=1`, opt-in) makes spaCy ~20% faster but
leaves the model's throughput unchanged next to it (3,342 vs 3,311 prompts/s), so it stays off.

What did not work:
fp8 dynamic quantization (torchao) is +16% speed for a collapse to 67% on SemEval; a spaCy prefetch *thread*
gains nothing (GIL), a separate process does; the BPE tokenizer's default 224 rayon threads contend on a lock
(185 CPU-s for a 0.9 s call), 16 threads are 4x faster. Remaining levers: flash-attention in the serve image
(needs a CUDA toolkit to build; at 1.6% padding and 10% attention time it has little to gain), static-shape
compilation with widths bucketed to 64 (3.1k vs 3.6k prompts/s), CUDA graphs (compile did not finish in 45 min),
chunk 512 (2.96k). The prompt template was then shortened (no question line, no labels; 16% fewer tokens) and the model retrained
with it (C3 = W4 recipe, lr 2e-5): 78.3% held-out / 80.6% ALL, i.e. within noise of W4, at 4,326 vs 3,614
prompts/s (+20%). This is the only template in the code and the published `main` model is trained with it; the
earlier template and models are not compatible with each other. Remaining lever: ModernBERT-base (~2.5x cheaper,
-3 points).

## More Examples

#### the big brown fox jumps over the lazy dog

- "big": "above average in size or number or quantity or magnitude or extent" (46% confidence) ✓ Correct - size descriptor
- "brown": "(of skin) deeply suntanned" (55% confidence) ✗ Wrong - should be color, not skin tone
- "fox": "alert carnivorous mammal with pointed muzzle and ears and a bushy tail; most are predators that do not hunt in packs" (84% confidence) ✓ Correct - high confidence animal identification
- "jumps": "move or jump suddenly, as if in surprise or alarm" (26% confidence) ✗ Wrong - should be physical leaping action, not startled movement
- "lazy": "disinclined to work or exertion" (98% confidence) ✓ Correct - very high confidence
- "dog": "a member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds" (69% confidence) ✓ Correct - strong confidence animal identification

#### Nagish captions your calls and empowers you to communicate using text or voice. It's fast, private, and accurate.

- "captions": "translation of foreign dialogue of a movie or TV program; usually displayed at the bottom of the screen" (61% confidence) ✓ Correct - subtitle/text display context
- "calls": "none of the above" (18% confidence) ✗ Wrong - should be phone calls/communication, low confidence suggests difficulty with proper nouns in context
- "empowers": "give or delegate power or authority to" (83% confidence) ✓ Correct - high confidence enablement meaning
- "communicate": "be in verbal contact; interchange information or ideas" (50% confidence) ✓ Correct - moderate confidence information exchange
- "using": "put into service; make work or employ for a particular purpose or for its inherent or natural purpose" (43% confidence) ✓ Correct - moderate confidence utilization meaning
- "text": "the words of something written" (50% confidence) ✓ Correct - moderate confidence written communication
- "voice": "the sound made by the vibration of vocal folds modified by the resonance of the vocal tract" (34% confidence) ✓ Correct - low confidence but correct vocal sound meaning
- "fast": "(used of timepieces) indicating a time ahead of or later than the correct time" (59% confidence) ✗ Wrong - should be speed/quick, not time accuracy
- "private": "confined to particular persons or groups or providing privacy" (41% confidence) ✓ Correct - moderate confidence privacy meaning
- "accurate": "conforming exactly or almost exactly to fact or to a standard or performing with total accuracy" (61% confidence) ✓ Correct - good confidence precision meaning
