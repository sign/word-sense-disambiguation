import logging
from dataclasses import dataclass
from functools import cache

import requests
import spacy
from colorama import Fore, Style, init

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


@cache
def get_spacy_pipeline(language: str = "en"):
    """Get or load and cache spaCy model for given language"""
    model_map = {
        'en': 'en_core_web_trf',
        # Add more language models as needed
    }
    if language in model_map:
        nlp = spacy.load(model_map[language])
        nlp.add_pipe("entityLinker", last=True)
        return nlp

    msg = f"Language '{language}' not supported"
    raise ValueError(msg)


def _get_definitions_raw(queries: list[WordQuery], language: str = "en") -> list[list[Definition]]:
    """
    Internal function to fetch definitions for multiple words using the batch endpoint.

    Args:
        queries: List of WordQuery objects with form and pos
        language: Language code (default: "en")

    Returns:
        List of definition lists, one per query (in same order as input).
        Each definition list contains Definition objects.
    """
    if not queries:
        return []

    # Prepare the request payload
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
    # returned by the API — typically WordNet's frequency order — so the
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


def get_choice_probabilities(
    probs, definitions: list[Definition], start_offset: int = 0,
) -> list[float]:
    """Get probabilities for all choice letters including 'none of the above'.

    With a pruned decoder, logits (and therefore ``probs``) are already laid out
    in answer-letter order. ``definitions[i]`` occupies letter
    ``start_offset + i``; NOTA always lives at the fixed index
    :data:`wsd.letters.NOTA_LETTER_INDEX`. The returned list has one entry per
    definition followed by the NOTA probability.
    """
    choice_probs = [float(probs[start_offset + i]) for i in range(len(definitions))]
    choice_probs.append(float(probs[NOTA_LETTER_INDEX]))  # "none of the above"
    return choice_probs


def _result_from_probs(
    probs, definitions: list[Definition], start_offset: int = 0,
) -> DisambiguationResult:
    """Pick the best choice from ``probs`` and package it as a result.

    NOTA is indexed at ``len(definitions)`` in the ``choice_probs`` list (it's
    the appended last entry), not at :data:`NOTA_LETTER_INDEX`; confidence is
    renormalized over the valid choices only.
    """
    choice_probs = get_choice_probabilities(probs, definitions, start_offset)
    best_choice_idx = choice_probs.index(max(choice_probs))
    total_prob = sum(choice_probs)
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
    start_offset: int = 0,
) -> DisambiguationResult:
    """Use ModernBERT to disambiguate word sense given context and definitions"""
    results = disambiguate_word_batch(
        [DisambiguationInput(word=word, marked_sentence=marked_sentence, definitions=definitions)],
        start_offset=start_offset,
    )
    return results[0]


def disambiguate_word_batch(
        batch_data: list[DisambiguationInput],
        start_offset: int = 0,
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
            start_offset=start_offset,
        )
        for _, inp in valid
    ]
    batch_results = unmask_token_batch(prompts)

    for (i, inp), unmask_result in zip(valid, batch_results, strict=True):
        results[i] = _result_from_probs(
            unmask_result.probabilities, inp.definitions, start_offset=start_offset,
        )
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
        # Handle "none of the above" case
        if result.definition == NONE_OF_THE_ABOVE:
            tokens[token_idx].synset_id = None
            tokens[token_idx].synset_definition = None
            tokens[token_idx].confidence = result.confidence
        else:
            tokens[token_idx].synset_id = result.synset_id
            tokens[token_idx].synset_definition = result.definition
            tokens[token_idx].confidence = result.confidence


def _extract_entities(doc) -> list[Entity]:
    """Extract linked entities from spaCy doc"""
    entities = []
    if hasattr(doc._, 'linkedEntities'):
        for ent in doc._.linkedEntities:
            span = ent.get_span()
            entity = Entity(
                id=ent.identifier,
                start_token=span.start,
                end_token=span.end - 1,
                text=ent.label,
                description=ent.description,
                url=ent.url
            )
            entities.append(entity)
    return entities


def disambiguate(text: str, language: str = "en") -> WordSenseDisambiguation:
    nlp = get_spacy_pipeline(language)
    doc = nlp(text)

    # First pass: Create all base tokens and identify content words to disambiguate
    tokens, content_word_indices = _create_base_tokens(doc)

    # Batch fetch definitions for all content words
    if content_word_indices:
        queries = [
            WordQuery(form=tokens[i].lemma, pos=_SPACY_TO_WORDNET_POS[tokens[i].pos])
            for i in content_word_indices
        ]
        all_definitions = get_definitions(queries, language)

        # Prepare batch data for disambiguation
        batch_data, valid_indices = _prepare_disambiguation_batch(
            doc, tokens, content_word_indices, all_definitions
        )

        # Batch disambiguate all content words with definitions
        if batch_data:
            predictions = disambiguate_word_batch(batch_data)
            _update_tokens_with_results(tokens, valid_indices, predictions)

    # Extract linked entities using entityLinker
    entities = _extract_entities(doc)

    return WordSenseDisambiguation(tokens=tokens, entities=entities)


def _get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level"""
    if confidence >= 0.7:
        return Fore.GREEN
    elif confidence >= 0.4:
        return Fore.YELLOW
    else:
        return Fore.RED


def _create_char_to_token_mapping(results: list[DisambiguatedToken]) -> dict:
    """Create a mapping from character positions to tokens"""
    char_to_token = {}
    for token in results:
        for i in range(token.start_char, token.end_char):
            char_to_token[i] = token
    return char_to_token


def _build_colored_sentence(text: str, char_to_token: dict) -> str:
    """Build the colored sentence string"""
    colored_sentence = ""
    i = 0
    while i < len(text):
        if i in char_to_token:
            token = char_to_token[i]
            word = text[token.start_char:token.end_char]

            if token.confidence is not None:
                if token.synset_id:
                    color = _get_confidence_color(token.confidence)
                    colored_sentence += f"{color}{Style.BRIGHT}{word}{Style.RESET_ALL}"
                else:
                    colored_sentence += f"{Fore.MAGENTA}{Style.BRIGHT}{word}{Style.RESET_ALL}"
            else:
                colored_sentence += f"{Fore.CYAN}{word}{Style.RESET_ALL}"

            i = token.end_char
        else:
            colored_sentence += text[i]
            i += 1
    return colored_sentence


def _print_legend() -> None:
    """Print the color legend"""
    print(f"{Style.BRIGHT}Legend:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{Style.BRIGHT}Green{Style.RESET_ALL}: High confidence (≥0.7)")
    print(f"  {Fore.YELLOW}{Style.BRIGHT}Yellow{Style.RESET_ALL}: Medium confidence (0.4-0.7)")
    print(f"  {Fore.RED}{Style.BRIGHT}Red{Style.RESET_ALL}: Low confidence (<0.4)")
    print(f"  {Fore.MAGENTA}{Style.BRIGHT}Magenta{Style.RESET_ALL}: None of the above")
    print(f"  {Fore.CYAN}{Style.BRIGHT}Cyan{Style.RESET_ALL}: Function words/punctuation")


def _print_detailed_breakdown(results: list[DisambiguatedToken]) -> None:
    """Print detailed breakdown of disambiguation results"""
    print(f"{Style.BRIGHT}Detailed Breakdown:{Style.RESET_ALL}")
    print("-" * 60)

    for token in results:
        if token.confidence is not None:
            color = _get_confidence_color(token.confidence) if token.synset_id else Fore.MAGENTA
            print(f"{color}{Style.BRIGHT}{token.word}{Style.RESET_ALL} ({token.pos.lower()})")

            if token.synset_id:
                print(f"  📖 Definition: {token.synset_definition}")
                print(f"  🎯 Confidence: {token.confidence:.3f}")
                print(f"  🔗 Synset ID: {token.synset_id}")
            else:
                print("  ❓ No matching definition found")
                print(f"  🎯 Confidence: {token.confidence:.3f}")
            print()


def visualize_sentence(text: str, results: list[DisambiguatedToken]) -> None:
    """Visualize the sentence with colorful disambiguation results"""
    init()  # Initialize colorama

    char_to_token = _create_char_to_token_mapping(results)

    print(f"\n{Style.BRIGHT}Sentence Visualization:{Style.RESET_ALL}")
    print("=" * 60)

    colored_sentence = _build_colored_sentence(text, char_to_token)
    print(colored_sentence)
    print()

    _print_legend()
    print()

    _print_detailed_breakdown(results)


# Example usage
if __name__ == "__main__":
    test_sentence = "Apple is going to be a technology company."

    results = disambiguate(test_sentence)

    # Add the new colorful visualization
    visualize_sentence(test_sentence, results.tokens)
