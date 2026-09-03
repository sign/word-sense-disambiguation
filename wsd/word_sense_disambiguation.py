import logging
import os
from dataclasses import dataclass

import requests

from wsd.env import WORDNET_URL
from wsd.letters import NOTA_LETTER_INDEX
from wsd.masked_language_model import load_model, unmask_token_batch
from wsd.prompt import (
    NONE_OF_THE_ABOVE,
    Definition,
    create_marked_sentence,
    create_multiple_choice_prompt,
)

logger = logging.getLogger(__name__)

# Constants
NO_DEFINITIONS_FOUND = "No definitions found"
_MAX_QUERIES_PER_REQUEST = 1000
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


def _get_definitions_raw(queries: list[WordQuery], language: str = "en") -> list[list[Definition]]:
    """Fetch definitions for exact (form, pos) queries from the WordNet API batch
    endpoint, one list per query, in input order."""
    if not queries:
        return []

    # Training builds hundreds of thousands of queries at once; keep requests bounded.
    if len(queries) > _MAX_QUERIES_PER_REQUEST:
        return [
            d
            for start in range(0, len(queries), _MAX_QUERIES_PER_REQUEST)
            for d in _get_definitions_raw(queries[start:start + _MAX_QUERIES_PER_REQUEST], language)
        ]

    url = f"{WORDNET_URL}/lexicons/omw-{language}:1.4/definitions"
    payload = {
        "queries": [{"form": q.form, "pos": q.pos} for q in queries]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as e:
        logger.warning("WordNet batch request to %s failed: %s", url, e)
        return [[] for _ in queries]

    if response.status_code != 200:
        logger.warning("WordNet API returned status code %s", response.status_code)
        return [[] for _ in queries]

    try:
        data = response.json()
    except requests.RequestException as e:
        logger.warning("WordNet API returned non-JSON body: %s", e)
        return [[] for _ in queries]

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
        return [[] for _ in queries]
    return results


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
    'NUM': 'n',
    'INTJ': 'n',  # hello→n, alas/ouch/wow→r (but only noun available)
    # v
    'VERB': 'v',
    # a / s
    'ADJ': 'a',
    # r
    'ADV': 'r',
}


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
        if token.pos_ in _SPACY_TO_WORDNET_POS and not token.is_punct and not token.is_space:
            content_word_indices.append(token.i)

    return tokens, content_word_indices


def _prepare_disambiguation_batch(
    doc,
    tokens: list[DisambiguatedToken],
    content_word_indices: list[int],
    all_definitions: list[list[Definition]]
) -> tuple[list[DisambiguationInput], list[int]]:
    """Prepare batch data for disambiguation"""
    batch_data = []
    valid_indices = []

    for idx, definitions in zip(content_word_indices, all_definitions, strict=True):
        if definitions:
            marked_sentence = create_marked_sentence(doc, idx)
            batch_data.append(
                DisambiguationInput(
                    word=tokens[idx].word,
                    marked_sentence=marked_sentence,
                    definitions=definitions
                )
            )
            valid_indices.append(idx)

    return batch_data, valid_indices


def _update_tokens_with_results(
    tokens: list[DisambiguatedToken],
    valid_indices: list[int],
    predictions: list[DisambiguationResult]
) -> None:
    """Update tokens with disambiguation results"""
    for token_idx, result in zip(valid_indices, predictions, strict=True):
        tokens[token_idx].confidence = result.confidence
        # NOTA → leave synset_id/synset_definition at their dataclass defaults (None).
        if result.definition != NONE_OF_THE_ABOVE:
            tokens[token_idx].synset_id = result.synset_id
            tokens[token_idx].synset_definition = result.definition


def _extract_entities(doc) -> list[Entity]:
    """Extract linked entities from spaCy doc (empty when the entityLinker pipe is disabled)."""
    entities = []
    for ent in getattr(doc._, "linkedEntities", None) or []:
        span = ent.get_span()
        entities.append(Entity(
            id=ent.identifier,
            start_token=span.start,
            end_token=span.end - 1,
            text=ent.label,
            description=ent.description,
            url=ent.url,
        ))
    return entities


def disambiguate_docs(docs: list, skip_single_sense: bool = False) -> list[WordSenseDisambiguation]:
    """Disambiguate already-parsed spaCy docs together: one WordNet lookup and
    one model batch for all content words of all docs (the batch path).

    ``skip_single_sense`` assigns a word's only candidate sense directly
    (confidence 1.0) instead of asking the model whether it is "none of the
    above"; about a fifth of prompts in running text, so a real saving at scale.
    """
    per_doc = [_create_base_tokens(doc) for doc in docs]

    # One definitions lookup for every content word of every doc.
    queries = [
        WordQuery(form=tokens[i].lemma, pos=_SPACY_TO_WORDNET_POS[tokens[i].pos])
        for tokens, content_word_indices in per_doc
        for i in content_word_indices
    ]
    all_definitions = iter(get_definitions(queries))

    batch_data: list[DisambiguationInput] = []
    doc_slices: list[tuple[list[int], int, int]] = []  # (valid token indices, start, end) into batch_data
    for doc, (tokens, content_word_indices) in zip(docs, per_doc, strict=True):
        doc_definitions = [next(all_definitions) for _ in content_word_indices]
        if skip_single_sense:
            for i, definitions in zip(content_word_indices, doc_definitions, strict=True):
                if len(definitions) == 1:
                    tokens[i].synset_id = definitions[0].synset_id
                    tokens[i].synset_definition = definitions[0].definition
                    tokens[i].confidence = 1.0
            doc_definitions = [[] if len(d) == 1 else d for d in doc_definitions]
        doc_batch, valid_indices = _prepare_disambiguation_batch(doc, tokens, content_word_indices, doc_definitions)
        doc_slices.append((valid_indices, len(batch_data), len(batch_data) + len(doc_batch)))
        batch_data.extend(doc_batch)

    predictions = disambiguate_word_batch(batch_data)

    results = []
    for doc, (tokens, _), (valid_indices, start, end) in zip(docs, per_doc, doc_slices, strict=True):
        _update_tokens_with_results(tokens, valid_indices, predictions[start:end])
        results.append(WordSenseDisambiguation(tokens=tokens, entities=_extract_entities(doc)))
    return results


def disambiguate(text: str, language: str = "en") -> WordSenseDisambiguation:
    # spaCy is only needed for full-text disambiguation; keep it out of the
    # import graph so training and benchmarks run in environments without it.
    from wsd.spacy_utils import run_spacy_pipeline

    return disambiguate_docs([run_spacy_pipeline(text, language)])[0]
