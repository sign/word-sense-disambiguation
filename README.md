# Word Sense Disambiguation

We use spaCy for classical analysis (tokenization, POS tagging, etc.) + 
[entity linking](https://pypi.org/project/spacy-entity-linker/), 
and have our own implementation for [word sense disambiguation](./wsd/word_sense_disambiguation.py).

We expose a [web server](./wsd/server.py) that can be used to disambiguate words in sentences.

## Environment Variables

- `WORDNET_URL`: URL of the WordNet API server (see below). Required; the server image sets it. When it is
  unset and the API package is installed (the cluster images built by `training/Enrootfile.sh` and
  `wsd/Enrootfile.sh`), the process starts a private API instance and uses that, so batch, training and
  benchmark jobs need no separate service.
- `WSD_MODEL`: model name or local checkpoint directory (default `sign/Ettin-150m-WSD`; `sign/ModernBERT-Large-Instruct-WSD` is the larger, 0.5-point more accurate model at 2.3x the cost, `sign/Ettin-1B-WSD` the most accurate at 7x).

### WordNet API server

Definitions come from the WordNet API in https://github.com/sign/wn (the `wn` library plus a REST layer, with
Wikidata lexemes merged into the English lexicon). Run it with Docker:

```shell
docker run -d --name wn -p 8000:8080 ghcr.io/sign/wn:latest
export WORDNET_URL=http://127.0.0.1:8000
curl "$WORDNET_URL/health"
```

The server listens on `$PORT` (default 8080). It is stateless, so one instance can serve many clients; the
batch endpoint used here (`POST /lexicons/omw-en:1.4/definitions`) answers ~1,000 queries per request.
Without Docker (e.g. a Slurm node with enroot): `enroot import -o wn.sqsh docker://ghcr.io#sign/wn:latest`
and run `uvicorn wn.web:app --host 0.0.0.0 --port 8080` inside it, or rely on the built-in local start above.

WSD also looks up function words supported by this lexicon: pronouns, determiners, adpositions,
conjunctions, auxiliaries, particles, and interjections. Their senses use the same disambiguation and
token-span response as content words. Punctuation, spaces, symbols, and unknown POS tags are not queried;
words with no candidates or a model-rejected sense still have no synset.

### Running locally

```shell
export WORDNET_URL=http://127.0.0.1:8000
uvicorn --reload wsd.server:app --port 8080
```

### Running with Docker

```shell
docker build --platform="linux/amd64" -t wsd .                                                        # GPU image
docker build --platform="linux/amd64" --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t wsd .  # CPU-only, ~3 GB smaller
docker run -p 8005:8080 -e PORT=8080 -e WORDNET_URL=http://host.docker.internal:8000 wsd
```

`cloudbuild.yaml` builds both as `:gpu` and `:cpu` tags (`gcloud builds submit --config cloudbuild.yaml`).
On the CPU image the default model answers a typical sentence in about 0.25 s on one core, 0.1 s on four.

## Batch processing

To disambiguate a corpus offline (one sentence per line, one JSON line out per sentence), split it into
many files and run `wsd.batch`; under `torchrun` each rank takes every N-th file and finished files are
skipped on restart:

```shell
split -n l/256 corpus.txt shards/part-
python -m wsd.batch --input 'shards/part-*' --output-dir out/ [--no-entities] [--skip-single-sense]
```

See [wsd/README.md](./wsd/README.md#throughput) for measured throughput.

### Entities

Each result carries Wikidata links for spaCy's person, organisation, place and facility spans (`entities`:
id, token span, label, description, URL), chosen from the spacy-entity-linker alias table by popularity.
On AIDA-CoNLL test with Wikidata gold this scores F1 0.48 with the CPU pipeline `en_core_web_lg` (0.51 with
`en_core_web_trf`, which finds more spans but needs a GPU to be fast); nationalities and other groups are
left to word sense disambiguation because the popularity prior links them badly.

Multiword expressions that WordNet lists (`test tube`, `New York`, `give up`) are disambiguated as one unit
first; each accepted expression produces one synset span. Only when the
model answers "none of the above" for the expression are its words disambiguated individually.

### Response format

Both `/disambiguate` and batch JSON lines return three arrays:

- `tokens`: `word`, `lemma`, `pos`, `position`, `start_char`, `end_char`, `morph`.
- `entities`: `id`, `start_token`, `end_token`, `text`, `description`, `url`.
- `synsets`: `id`, `start_token`, `end_token`, `definition`, `confidence`, `expression`.

Entity and synset token indices are zero-based and **inclusive** at both ends. Character offsets are
zero-based with an exclusive `end_char`. Synsets are sorted by token position. A single-word sense has
equal start/end indices and `expression: null`; a multiword sense includes its canonical WordNet form.
Words with no definitions or a "none of the above" answer have no synset entry. Entity and synset spans
are independent and may overlap. Repeated occurrences of the same sense have separate spans.

`morph` contains spaCy's morphological features as a string-to-string dictionary (`token.morph.to_dict()`),
for example `{"Number": "Plur"}` for **books** and `{"Tense": "Past", "VerbForm": "Fin"}` for **visited**.
It is `{}` when no features are available; multi-valued features retain spaCy's comma-separated strings.
These describe each original token, including tokens inside entity or synset spans, not the whole span.
The API and batch output preserve the same features; they do not interpret them as target-language inflections.

For example, the accepted expression in `She held a test tube.` appears once in `synsets`:

```json
{
  "id": "omw-en-04415921-n",
  "start_token": 3,
  "end_token": 4,
  "definition": "glass tube closed at one end",
  "confidence": 0.8,
  "expression": "test tube"
}
```

## Usage

To view an output, visit this [example link](http://localhost:8005/disambiguate?text=Obama%20told%20the%20bus%20driver,%20to%20drive%20to%20D.C.&lang=en&output=html) (adjust port if running locally):
![Example of our system's output](assets/output-example.png)
