import logging
import os
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
# (form, pos, language) -> definitions, filled from API responses. A corpus has a
# bounded vocabulary, so after warm-up almost every lookup is a hit; each API
# round trip otherwise costs ~0.35 ms per query of server-side work.
# ponytail: unbounded (a few hundred k entries at most); add an LRU if memory matters.
_definitions_cache: dict[tuple[str, str, str], list[Definition]] = {}
# Minimum (renormalized) probability for "none of the above" to win. 0 keeps
# plain argmax; >1 disables NOTA. The model over-predicts NOTA on long natural
# sentences (trained on short WordNet-style examples), so batch users may want
# to raise this. Read at call time so the sweep can vary it.
_nota_threshold = lambda: float(os.environ.get("WSD_NOTA_THRESHOLD", "0"))  # noqa: E731


@dataclass
class WordQuery:
    """Query for word definitions"""
    form: str
    pos: str


@dataclass
class DisambiguatedToken:
    """Token with disambiguation results"""
    word: str
    lemma: str
    pos: str
    position: int
    start_char: int
    end_char: int
    synset_id: str | None = None
    synset_definition: str | None = None
    confidence: float | None = None
    # WordNet multiword form this token is part of ("test tube"); the sense fields then
    # describe the whole expression and are repeated on each of its tokens.
    expression: str | None = None


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
    word: str
    marked_sentence: str
    definitions: list[Definition]


@dataclass
class WordSenseDisambiguation:
    tokens: list[DisambiguatedToken]
    entities: list[Entity]


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


def _get_definitions_raw(queries: list[WordQuery], language: str = "en") -> list[list[Definition]]:
    """Fetch definitions for exact (form, pos) queries from the WordNet API batch
    endpoint, one list per query, in input order. Results are memoized per
    (form, pos), and only distinct misses are sent to the server."""
    if not queries:
        return []

    keys = [(q.form, q.pos, language) for q in queries]
    misses = list(dict.fromkeys(k for k in keys if k not in _definitions_cache))
    for start in range(0, len(misses), _MAX_QUERIES_PER_REQUEST):  # training sends hundreds of thousands at once
        chunk = misses[start:start + _MAX_QUERIES_PER_REQUEST]
        fetched = _fetch_definitions([WordQuery(form=f, pos=p) for f, p, _ in chunk], language)
        if fetched is not None:  # a failed request is not cached, so it is retried next time
            _definitions_cache.update(zip(chunk, fetched, strict=True))
    return [list(_definitions_cache.get(k, [])) for k in keys]


def _fetch_definitions(queries: list[WordQuery], language: str) -> list[list[Definition]] | None:
    """One batch request to the WordNet API; ``None`` on failure."""

    url = f"{WORDNET_URL}/lexicons/omw-{language}:1.4/definitions"
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


def get_definitions(queries: list[WordQuery], language: str = "en") -> list[list[Definition]]:
    """Fetch definitions for multiple words using the batch endpoint.

    For adjectives (``pos="a"``), fetches both ``"a"`` (adjective) and ``"s"``
    (satellite adjective) and concatenates them; other POS tags pass through
    unchanged. Output is in input order.
    """
    if not queries:
        return []

    # Expand "a" queries to (a, s); track which output slot each expanded query
    # feeds. Non-adjective queries map to exactly one slot, so the merge below
    # uses a uniform extend() and there's no need to tag pos_type separately.
    expanded_queries: list[WordQuery] = []
    origin: list[int] = []
    for i, q in enumerate(queries):
        if q.pos == "a":
            expanded_queries.append(WordQuery(form=q.form, pos="a"))
            expanded_queries.append(WordQuery(form=q.form, pos="s"))
            origin.extend([i, i])
        else:
            expanded_queries.append(q)
            origin.append(i)

    expanded_results = _get_definitions_raw(expanded_queries, language)

    results: list[list[Definition]] = [[] for _ in queries]
    for orig_idx, defs in zip(origin, expanded_results, strict=True):
        results[orig_idx].extend(defs)
    return results


def get_choice_probabilities(probs, definitions: list[Definition]) -> list[float]:
    """Get probabilities for all choice letters including 'none of the above'.

    With a pruned decoder, logits (and therefore ``probs``) are already laid out
    in answer-letter order. ``definitions[i]`` occupies letter ``i``; NOTA
    always lives at the fixed index :data:`wsd.letters.NOTA_LETTER_INDEX`. The
    returned list has one entry per definition followed by the NOTA probability.
    """
    choice_probs = [float(probs[i]) for i in range(len(definitions))]
    choice_probs.append(float(probs[NOTA_LETTER_INDEX]))  # "none of the above"
    return choice_probs


def _result_from_probs(
    probs, definitions: list[Definition],
) -> DisambiguationResult:
    """Pick the best choice from ``probs`` and package it as a result.

    Confidence is renormalized over the valid choices only.
    """
    choice_probs = get_choice_probabilities(probs, definitions)
    total_prob = sum(choice_probs)
    best_choice_idx = choice_probs.index(max(choice_probs))
    if best_choice_idx == len(definitions) and choice_probs[-1] < _nota_threshold() * total_prob:
        best_choice_idx = choice_probs.index(max(choice_probs[:-1]))  # best real sense instead of NOTA
    normalized_score = choice_probs[best_choice_idx] / total_prob if total_prob > 0 else 0.0

    if best_choice_idx == len(definitions):  # NOTA slot
        return DisambiguationResult(
            synset_id="",
            definition=NONE_OF_THE_ABOVE,
            confidence=normalized_score,
        )
    best_definition = definitions[best_choice_idx]
    return DisambiguationResult(
        synset_id=best_definition.synset_id,
        definition=best_definition.definition,
        confidence=normalized_score,
    )


def disambiguate_word(
    word: str,
    marked_sentence: str,
    definitions: list[Definition],
) -> DisambiguationResult:
    """Use ModernBERT to disambiguate word sense given context and definitions"""
    results = disambiguate_word_batch(
        [DisambiguationInput(word=word, marked_sentence=marked_sentence, definitions=definitions)],
    )
    return results[0]


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
        create_multiple_choice_prompt(
            inp.word,
            components.tokenizer.mask_token,
            inp.marked_sentence,
            inp.definitions,
            components.tokenizer,
        )
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
    """Create base tokens and identify content word indices"""
    tokens = []
    content_word_indices = []

    for token in doc:
        # Create base token info
        disambiguated_token = DisambiguatedToken(
            word=token.text,
            lemma=token.lemma_.lower(),
            pos=token.pos_,
            position=token.i,
            start_char=token.idx,
            end_char=token.idx + len(token.text)
        )
        tokens.append(disambiguated_token)

        # Track content words that need disambiguation
        if _is_content(token):
            content_word_indices.append(token.i)

    return tokens, content_word_indices


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


def _run_units(docs, results, units: list[_Unit], skip_single_sense: bool) -> list[_Unit]:
    """Disambiguate the units in one model batch and write the answers on their tokens.
    Returns the multiword units the model rejected ("none of the above"); their tokens stay unset."""
    batch: list[DisambiguationInput] = []
    kept: list[_Unit] = []
    for u in units:
        if skip_single_sense and len(u.definitions) == 1:
            for tok in results[u.doc].tokens[u.start:u.end]:
                tok.synset_id, tok.synset_definition = u.definitions[0].synset_id, u.definitions[0].definition
                tok.confidence, tok.expression = 1.0, u.expression
            continue
        word = " ".join(t.text for t in docs[u.doc][u.start:u.end])
        batch.append(DisambiguationInput(word=word, marked_sentence=_mark_span(docs[u.doc], u.start, u.end),
                                         definitions=u.definitions))
        kept.append(u)
    rejected = []
    for u, result in zip(kept, disambiguate_word_batch(batch), strict=True):
        if result.definition == NONE_OF_THE_ABOVE and u.expression is not None:
            rejected.append(u)  # not the compound reading: the words get their own turn
            continue
        for tok in results[u.doc].tokens[u.start:u.end]:
            tok.confidence, tok.expression = result.confidence, u.expression
            if result.definition != NONE_OF_THE_ABOVE:  # NOTA leaves the synset fields None
                tok.synset_id, tok.synset_definition = result.synset_id, result.definition
    return rejected


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
    per_doc = [_create_base_tokens(doc) for doc in docs]
    results = [WordSenseDisambiguation(tokens=tokens, entities=_extract_entities(doc))
               for doc, (tokens, _) in zip(docs, per_doc, strict=True)]
    rejected = _run_units(docs, results, _units(docs, per_doc), skip_single_sense)
    _run_units(docs, results, [w for u in rejected for w in u.words], skip_single_sense)
    return results


def disambiguate(text: str, language: str = "en") -> WordSenseDisambiguation:
    # spaCy is only needed for full-text disambiguation; keep it out of the
    # import graph so training and benchmarks run in environments without it.
    from wsd.spacy_utils import run_spacy_pipeline

    return disambiguate_docs([run_spacy_pipeline(text, language)])[0]
