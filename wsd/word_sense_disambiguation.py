import logging
from dataclasses import dataclass, field

import requests

from wsd.env import WORDNET_URL
from wsd.letters import NOTA_LETTER_INDEX
from wsd.masked_language_model import load_model, unmask_token_batch
from wsd.multiword import find_spans
from wsd.prompt import (
    NONE_OF_THE_ABOVE,
    Definition,
    create_multiple_choice_prompt,
)

logger = logging.getLogger(__name__)

# Constants
NO_DEFINITIONS_FOUND = "No definitions found"
_MAX_QUERIES_PER_REQUEST = 1000
# (form, pos) -> definitions, filled from API responses. A corpus has a bounded vocabulary, so
# after warm-up almost every lookup is a hit; each API round trip otherwise costs ~0.35 ms per
# query of server-side work. ponytail: unbounded (a few hundred k entries at most).
_definitions_cache: dict[tuple[str, str], list[Definition]] = {}


@dataclass
class WordQuery:
    """Query for word definitions"""
    form: str
    pos: str


@dataclass
class DisambiguatedToken:
    """Token-level linguistic information; meanings live in separate spans."""
    word: str
    lemma: str
    pos: str
    position: int
    start_char: int
    end_char: int


@dataclass
class Synset:
    """An accepted word sense over an inclusive, zero-based token span."""
    id: str
    start_token: int
    end_token: int
    definition: str
    confidence: float
    expression: str | None = None  # canonical WordNet form for a multiword expression


@dataclass
class Entity:
    id: str
    start_token: int
    end_token: int
    text: str
    description: str | None = None
    url: str | None = None


@dataclass
class DisambiguationResult:
    """Result of word sense disambiguation"""
    synset_id: str
    definition: str
    confidence: float


@dataclass
class DisambiguationInput:
    """Input for batch disambiguation"""
    marked_sentence: str
    definitions: list[Definition]


@dataclass
class WordSenseDisambiguation:
    tokens: list[DisambiguatedToken]
    entities: list[Entity]
    synsets: list[Synset] = field(default_factory=list)


@dataclass(frozen=True)
class LightToken:
    """The spaCy ``Token`` attributes this module reads, as a plain picklable record."""
    text: str
    lemma_: str
    pos_: str
    i: int
    idx: int
    is_punct: bool
    is_space: bool
    whitespace_: str
    dep_: str = ""
    head_i: int = -1  # index of the syntactic head (spaCy ``token.head.i``)


@dataclass
class LightDoc:
    """Picklable stand-in for a parsed spaCy ``Doc`` (tokens + linked entities), so
    spaCy can run in another process and hand results to :func:`disambiguate_docs`."""
    tokens: list[LightToken]
    entities: list[Entity]

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i]


def light_doc(doc) -> LightDoc:
    return LightDoc(
        tokens=[
            LightToken(t.text, t.lemma_, t.pos_, t.i, t.idx, t.is_punct, t.is_space, t.whitespace_, t.dep_, t.head.i)
            for t in doc
        ],
        entities=_extract_entities(doc),
    )


def _fetch_definitions(queries: list[WordQuery]) -> list[list[Definition]] | None:
    """One batch request to the WordNet API; ``None`` on failure."""
    url = f"{WORDNET_URL}/lexicons/omw-en:1.4/definitions"
    payload = {
        "queries": [{"form": q.form, "pos": q.pos} for q in queries]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as e:
        logger.warning("WordNet batch request to %s failed: %s", url, e)
        return None

    if response.status_code != 200:
        logger.warning("WordNet API returned status code %s", response.status_code)
        return None

    try:
        data = response.json()
    except requests.RequestException as e:
        logger.warning("WordNet API returned non-JSON body: %s", e)
        return None

    # Parse response and maintain order. Definitions are kept in the order
    # returned by the API — WordNet's sense (frequency) order — so the
    # more common senses land on earlier letter slots.
    results = [
        [
            Definition(synset_id=synset_id, definition=definition_text)
            for synset_id, definition_text in item.get("definitions", {}).items()
        ]
        for item in data.get("data", [])
    ]

    if len(results) < len(queries):
        logger.warning(
            "WordNet API returned %d items, expected %d", len(results), len(queries),
        )
        return None
    return results[:len(queries)]


def get_definitions(queries: list[WordQuery]) -> list[list[Definition]]:
    """Definitions per query, in input order, from the WordNet API's batch endpoint. Adjectives
    (``pos="a"``) also get their satellite (``"s"``) senses. Memoized per (form, pos); only distinct
    misses go to the server, in requests of ``_MAX_QUERIES_PER_REQUEST``."""
    expanded = [(i, WordQuery(q.form, p)) for i, q in enumerate(queries)
                for p in (("a", "s") if q.pos == "a" else (q.pos,))]
    misses = list(dict.fromkeys((q.form, q.pos) for _, q in expanded if (q.form, q.pos) not in _definitions_cache))
    for start in range(0, len(misses), _MAX_QUERIES_PER_REQUEST):  # training sends hundreds of thousands at once
        chunk = misses[start:start + _MAX_QUERIES_PER_REQUEST]
        fetched = _fetch_definitions([WordQuery(f, p) for f, p in chunk])
        if fetched is not None:  # a failed request is not cached, so it is retried next time
            _definitions_cache.update(zip(chunk, fetched, strict=True))
    results: list[list[Definition]] = [[] for _ in queries]
    for i, q in expanded:
        results[i].extend(_definitions_cache.get((q.form, q.pos), []))
    return results


def _result_from_probs(probs: list[float], definitions: list[Definition]) -> DisambiguationResult:
    """Pick the best option. With the pruned decoder ``probs`` are in answer-letter order: option
    ``i`` is letter ``i`` and "none of the above" the fixed :data:`NOTA_LETTER_INDEX`. Confidence is
    renormalized over the shown options."""
    choice_probs = [probs[i] for i in range(len(definitions))] + [probs[NOTA_LETTER_INDEX]]
    best = choice_probs.index(max(choice_probs))
    total = sum(choice_probs)
    confidence = choice_probs[best] / total if total > 0 else 0.0
    if best == len(definitions):
        return DisambiguationResult(synset_id="", definition=NONE_OF_THE_ABOVE, confidence=confidence)
    return DisambiguationResult(definitions[best].synset_id, definitions[best].definition, confidence)


def disambiguate_word_batch(
    batch_data: list[DisambiguationInput],
) -> list[DisambiguationResult]:
    """
    Batch version of disambiguate_word that processes multiple words in parallel.

    Args:
        batch_data: List of DisambiguationInput objects

    Returns:
        List of DisambiguationResult objects for each input
    """
    if not batch_data:
        return []

    components = load_model()

    # Build prompts only for inputs with definitions. Inputs without definitions
    # get a fixed NO_DEFINITIONS_FOUND result without touching the model.
    results: list[DisambiguationResult] = [
        DisambiguationResult(synset_id=NO_DEFINITIONS_FOUND, definition="", confidence=0.0)
        for _ in batch_data
    ]
    valid = [(i, inp) for i, inp in enumerate(batch_data) if inp.definitions]
    if not valid:
        return results

    prompts = [
        create_multiple_choice_prompt(components.tokenizer.mask_token, inp.marked_sentence, inp.definitions,
                                      components.tokenizer)
        for _, inp in valid
    ]
    batch_results = unmask_token_batch(prompts)

    for (i, inp), unmask_result in zip(valid, batch_results, strict=True):
        results[i] = _result_from_probs(unmask_result.probabilities, inp.definitions)
    return results


# spaCy POS tag → WordNet POS tag. Fixed mapping shared by token creation and
# the lemma/pos query builder; both sides must agree or definitions end up
# attached to the wrong tokens.
_SPACY_TO_WORDNET_POS: dict[str, str] = {
    # n
    'NOUN': 'n',
    'PROPN': 'n',
    'PRON': 'h',  # pronouns come from the Wikidata-lexeme extension of sign/wn (WN-LMF code h)
    'NUM': 'n',
    'INTJ': 'n',  # hello→n, alas/ouch/wow→r (but only noun available)
    # v
    'VERB': 'v',
    # a / s
    'ADJ': 'a',
    # r
    'ADV': 'r',
}


def _is_content(token) -> bool:
    return token.pos_ in _SPACY_TO_WORDNET_POS and not token.is_punct and not token.is_space


def _create_base_tokens(doc) -> tuple[list[DisambiguatedToken], list[int]]:
    """Output tokens for a doc, and the indices of the content words to disambiguate."""
    tokens = [DisambiguatedToken(word=t.text, lemma=t.lemma_.lower(), pos=t.pos_, position=t.i, start_char=t.idx,
                                 end_char=t.idx + len(t.text)) for t in doc]
    return tokens, [t.i for t in doc if _is_content(t)]


def _word_definitions(tokens) -> list[list[Definition]]:
    """Candidate senses per token: the union over its lookup forms (see :func:`_queries`),
    in one WordNet request, keeping the first form's order and deduplicating synsets."""
    per_token = [_queries(t) for t in tokens]
    flat = get_definitions([q for qs in per_token for q in qs])
    out: list[list[Definition]] = []
    pos = 0
    for qs in per_token:
        merged: dict[str, Definition] = {}
        for defs in flat[pos:pos + len(qs)]:
            for definition in defs:
                merged.setdefault(definition.synset_id, definition)
        pos += len(qs)
        out.append(list(merged.values()))
    return out


def _queries(token) -> list[WordQuery]:
    """Lookup forms for one token, whose senses are pooled: the spaCy lemma and POS, the
    surface form (spaCy keeps proper nouns unlemmatized, lemmatizes "best" to "good" and
    "species" to "specie"; "works"/"years" have senses of their own) and a naive singular
    for proper nouns. A verb used as a modifier ("a damaged gene", "is concerned") is a
    participial adjective, so its adjective senses are listed first."""
    pos = _SPACY_TO_WORDNET_POS[token.pos_]
    lemma, surface = token.lemma_.lower(), token.text.lower()
    candidates = [(lemma, pos)]
    if token.pos_ == "VERB" and token.dep_ in ("amod", "acomp"):
        candidates.insert(0, (surface, "a"))
    if surface != lemma:
        candidates.append((surface, pos))
    if token.pos_ == "PROPN" and len(lemma) > 3 and lemma.endswith("s"):  # ponytail: naive plural strip
        candidates.append((lemma[:-1], pos))
    return [WordQuery(form=f, pos=p) for f, p in dict.fromkeys(candidates)]


def _mark_span(doc, start: int, end: int) -> str:
    """Sentence text with tokens ``start:end`` wrapped in one ``*...*`` pair."""
    text = ""
    for token in doc:
        if token.i == start:
            text += "*"
        text += token.text
        if token.i == end - 1:
            text += "*"
        text += token.whitespace_
    return text


def _extract_entities(doc) -> list[Entity]:
    """Wikidata links for the doc's named-entity spans (empty when the NER pipe is disabled)."""
    if isinstance(doc, LightDoc):
        return doc.entities
    from wsd.entities import (
        link_named_entities,  # needs the spacy-entity-linker knowledge base (not in the training image)
    )
    return [
        Entity(id=item_id, start_token=span.start, end_token=span.end - 1, text=label, description=description,
               url=f"https://www.wikidata.org/wiki/Q{item_id}")
        for span, item_id, label, description in link_named_entities(doc)
    ]


@dataclass
class _Unit:
    """One thing to disambiguate: a WordNet multiword span or a single word (token range ``[start, end)``
    of doc ``doc``). A span carries its ``words`` as fallback units, run when the compound reading is rejected."""
    doc: int
    start: int
    end: int
    definitions: list[Definition]
    expression: str | None = None  # the WordNet multiword form, None for a single word
    words: list["_Unit"] = field(default_factory=list)


def _units(docs, per_doc) -> list[_Unit]:
    """Phase-1 units for all docs, with every definition fetched in one lookup: each WordNet
    multiword span (its words attached as fallbacks) and each content word outside a span.
    A span whose form has no definitions for its POS is replaced by its words right away."""
    span_queries: list[WordQuery] = []
    span_units: list[_Unit] = []
    word_tokens: list = []
    word_slots: list[tuple[int, int, list[_Unit]]] = []  # (doc, token, list the word unit joins)
    units: list[_Unit] = []
    for d, (doc, (_, content_idx)) in enumerate(zip(docs, per_doc, strict=True)):
        owner: dict[int, _Unit] = {}
        for span in find_spans(doc):
            unit = _Unit(d, span.start, span.end, [], span.form)
            span_queries.append(WordQuery(form=span.form, pos=_SPACY_TO_WORDNET_POS.get(doc[span.head].pos_, "n")))
            span_units.append(unit)
            units.append(unit)
            owner.update(dict.fromkeys(range(span.start, span.end), unit))
        for i in content_idx:
            word_tokens.append(doc[i])
            word_slots.append((d, i, owner[i].words if i in owner else units))
    for unit, defs in zip(span_units, get_definitions(span_queries), strict=True):
        unit.definitions = defs
    for (d, i, target), defs in zip(word_slots, _word_definitions(word_tokens), strict=True):
        if defs:
            target.append(_Unit(d, i, i + 1, defs))
    return [w for u in units for w in ([u] if u.definitions else u.words)]


def _build_batch(
    docs, results, units: list[_Unit], skip_single_sense: bool,
) -> tuple[list[DisambiguationInput], list[_Unit]]:
    """Model inputs for the units; under ``skip_single_sense`` a unit with one candidate is assigned directly."""
    batch: list[DisambiguationInput] = []
    kept: list[_Unit] = []
    for u in units:
        if skip_single_sense and len(u.definitions) == 1:
            results[u.doc].synsets.append(Synset(
                id=u.definitions[0].synset_id, start_token=u.start, end_token=u.end - 1,
                definition=u.definitions[0].definition, confidence=1.0, expression=u.expression,
            ))
            continue
        batch.append(DisambiguationInput(_mark_span(docs[u.doc], u.start, u.end), u.definitions))
        kept.append(u)
    return batch, kept


def _apply_results(results, kept: list[_Unit], model_results) -> list[_Unit]:
    """Record accepted synset spans and return rejected expressions for word-level fallback."""
    rejected = []
    for u, result in zip(kept, model_results, strict=True):
        if result.definition == NONE_OF_THE_ABOVE:
            if u.expression is not None:
                rejected.append(u)
            continue
        results[u.doc].synsets.append(Synset(
            id=result.synset_id, start_token=u.start, end_token=u.end - 1,
            definition=result.definition, confidence=result.confidence, expression=u.expression,
        ))
    return rejected


@dataclass
class PreparedBatch:
    """Everything :func:`disambiguate_docs` computes before the model runs (CPU only, picklable), so a
    producer process can do it while the model process only runs the forward pass."""
    docs: list
    results: list[WordSenseDisambiguation]
    batch: list[DisambiguationInput]
    kept: list[_Unit]


def prepare_docs(docs: list, skip_single_sense: bool = False) -> PreparedBatch:
    """Phase 1 without the model: entities, WordNet lookups for expressions and words, model inputs."""
    per_doc = [_create_base_tokens(doc) for doc in docs]
    results = [WordSenseDisambiguation(tokens=tokens, entities=_extract_entities(doc))
               for doc, (tokens, _) in zip(docs, per_doc, strict=True)]
    batch, kept = _build_batch(docs, results, _units(docs, per_doc), skip_single_sense)
    return PreparedBatch(docs, results, batch, kept)


def complete_docs(
    prepared: PreparedBatch, model_results, skip_single_sense: bool = False,
) -> list[WordSenseDisambiguation]:
    """Apply the phase-1 answers, then disambiguate the words of rejected expressions (phase 2)."""
    rejected = _apply_results(prepared.results, prepared.kept, model_results)
    words = [w for u in rejected for w in u.words]
    batch, kept = _build_batch(prepared.docs, prepared.results, words, skip_single_sense)
    _apply_results(prepared.results, kept, disambiguate_word_batch(batch))
    for result in prepared.results:
        result.synsets.sort(key=lambda synset: synset.start_token)
    return prepared.results


def disambiguate_docs(docs: list, skip_single_sense: bool = False) -> list[WordSenseDisambiguation]:
    """Disambiguate already-parsed spaCy docs together: one WordNet lookup and
    one model batch for all content words of all docs (the batch path).

    Multiword expressions that WordNet lists ("test tube", "give up") are
    disambiguated as one unit first; their words are only disambiguated
    separately (a second, small model batch) when the expression's answer is
    "none of the above", i.e. when the compound reading does not apply.

    ``skip_single_sense`` assigns a word's (or expression's) only candidate sense
    directly (confidence 1.0) instead of asking the model whether it is "none of
    the above"; about a fifth of prompts in running text, so a real saving at scale.
    """
    prepared = prepare_docs(docs, skip_single_sense)
    return complete_docs(prepared, disambiguate_word_batch(prepared.batch), skip_single_sense)


def disambiguate(text: str, language: str = "en") -> WordSenseDisambiguation:
    # spaCy is only needed for full-text disambiguation; keep it out of the
    # import graph so training and benchmarks run in environments without it.
    from wsd.spacy_utils import run_spacy_pipeline

    return disambiguate_docs([run_spacy_pipeline(text, language)])[0]
