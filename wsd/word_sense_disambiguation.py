from dataclasses import dataclass
from functools import cache
from typing import Optional

import requests
import spacy
from colorama import Fore, Style, init

from wsd.env import WORDNET_URL
from wsd.masked_language_model import load_model, unmask_token, unmask_token_batch
from wsd.prompt import Definition, get_option_letter, create_multiple_choice_prompt, NONE_OF_THE_ABOVE, \
    create_marked_sentence

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
    synset_id: Optional[str] = None
    synset_definition: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Entity:
    id: str
    start_token: int
    end_token: int
    text: str
    description: Optional[str] = None
    url: Optional[str] = None


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
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return [[] for _ in queries]

        data = response.json()

        # Parse response and maintain order
        results = []
        for item in data.get("data", []):
            definitions = []
            # Sort definitions by synset_id for consistent ordering
            definitions_dict = item.get("definitions", {})
            for synset_id in sorted(definitions_dict.keys()):
                definition_text = definitions_dict[synset_id]
                definitions.append(Definition(synset_id=synset_id, definition=definition_text))
            results.append(definitions)

        # If response doesn't have enough items, pad with empty lists
        if len(results) < len(queries):
            raise ValueError(f"API response has {len(results)} items but expected {len(queries)} queries")

        return results

    except (requests.RequestException, ValueError) as e:
        print(f"Error making batch request to {url}: {e}")
        return [[] for _ in queries]


def get_definitions(queries: list[WordQuery], language: str = "en") -> list[list[Definition]]:
    """
    Fetch definitions for multiple words using the batch endpoint.

    For adjectives (pos="a"), automatically fetches and concatenates definitions
    from both "a" (adjective) and "s" (adjective satellite) categories.

    Args:
        queries: List of WordQuery objects with form and pos
        language: Language code (default: "en")

    Returns:
        List of definition lists, one per query (in same order as input).
        Each definition list contains Definition objects.
    """
    if not queries:
        return []

    # Expand queries: when pos is "a", we need to query both "a" and "s"
    expanded_queries = []
    query_mapping = []  # Maps expanded query index to (original query index, pos_type)

    for i, q in enumerate(queries):
        if q.pos == "a":
            # Add both "a" and "s" queries
            expanded_queries.append(WordQuery(form=q.form, pos="a"))
            query_mapping.append((i, "a"))
            expanded_queries.append(WordQuery(form=q.form, pos="s"))
            query_mapping.append((i, "s"))
        else:
            expanded_queries.append(q)
            query_mapping.append((i, None))

    # Get definitions for all expanded queries
    expanded_results = _get_definitions_raw(expanded_queries, language)

    # Collapse results back to match original queries
    results = [[] for _ in queries]
    for idx, (orig_idx, pos_type) in enumerate(query_mapping):
        if pos_type is None:
            # Not an "a" query, just copy the result
            results[orig_idx] = expanded_results[idx]
        else:
            # This is an "a" or "s" part of an adjective query
            # Concatenate to the existing result
            results[orig_idx].extend(expanded_results[idx])

    return results


def get_definitions_single(word: str, pos: str, language: str = "en") -> list[Definition]:
    """
    Convenience function to fetch definitions for a single word.

    Args:
        word: Word form to look up
        pos: Part of speech
        language: Language code (default: "en")

    Returns:
        List of Definition objects for the word
    """
    query = WordQuery(form=word, pos=pos)
    results = get_definitions([query], language)
    return results[0] if results else []


def get_choice_probabilities(tokenizer, probs, definitions: list[Definition]) -> list[float]:
    """Get probabilities for all choice letters including 'none of the above'"""
    choice_probs = []
    vocab_size = len(probs)

    # Get probabilities for definition choices A, B, C, etc.
    for i in range(len(definitions)):
        letter = get_option_letter(i)

        # Get probabilities for both "A" and " A" tokens and sum them
        letter_token_id = tokenizer.convert_tokens_to_ids(letter)
        space_letter_token_id = tokenizer.convert_tokens_to_ids(f" {letter}")

        total_prob = 0.0
        if letter_token_id is not None and letter_token_id < vocab_size:
            total_prob += probs[letter_token_id].item()
        if space_letter_token_id is not None and space_letter_token_id < vocab_size:
            total_prob += probs[space_letter_token_id].item()

        choice_probs.append(total_prob)

    # Add probability for next letter (none of the above)
    none_letter = get_option_letter(len(definitions))
    none_token_id = tokenizer.convert_tokens_to_ids(none_letter)
    space_none_token_id = tokenizer.convert_tokens_to_ids(f" {none_letter}")

    none_prob = 0.0
    if none_token_id is not None and none_token_id < vocab_size:
        none_prob += probs[none_token_id].item()
    if space_none_token_id is not None and space_none_token_id < vocab_size:
        none_prob += probs[space_none_token_id].item()

    choice_probs.append(none_prob)
    return choice_probs


def disambiguate_word(word: str, marked_sentence: str, definitions: list[Definition]) -> DisambiguationResult:
    """Use ModernBERT to disambiguate word sense given context and definitions"""
    if not definitions:
        return DisambiguationResult(
            synset_id=NO_DEFINITIONS_FOUND,
            definition="",
            confidence=0.0
        )

    components = load_model()

    # Create multiple choice prompt
    text = create_multiple_choice_prompt(word, components.tokenizer.mask_token, marked_sentence, definitions)

    # Get prediction
    result = unmask_token(text)

    # Get probabilities for all choices
    choice_probs = get_choice_probabilities(components.tokenizer, result.probabilities, definitions)

    # Find best choice and normalize
    best_choice_idx = choice_probs.index(max(choice_probs))
    total_prob = sum(choice_probs)
    normalized_score = choice_probs[best_choice_idx] / total_prob if total_prob > 0 else 0.0

    # Handle "none of the above" case
    if best_choice_idx == len(definitions):  # Next letter option selected
        return DisambiguationResult(
            synset_id="",
            definition=NONE_OF_THE_ABOVE,
            confidence=normalized_score
        )
    else:
        best_definition = definitions[best_choice_idx]
        return DisambiguationResult(
            synset_id=best_definition.synset_id,
            definition=best_definition.definition,
            confidence=normalized_score
        )


def disambiguate_word_batch(
        batch_data: list[DisambiguationInput]
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

    # Prepare prompts and track which ones are valid
    prompts = []
    valid_indices = []  # Track which inputs have definitions
    for i, input_obj in enumerate(batch_data):
        if input_obj.definitions:
            text = create_multiple_choice_prompt(
                input_obj.word,
                components.tokenizer.mask_token,
                input_obj.marked_sentence,
                input_obj.definitions
            )
            prompts.append(text)
            valid_indices.append(i)

    # If no valid prompts, return empty results for all
    if not prompts:
        return [
            DisambiguationResult(
                synset_id=NO_DEFINITIONS_FOUND,
                definition="",
                confidence=0.0
            )
            for _ in batch_data
        ]

    # Get predictions for all prompts in batch
    batch_results = unmask_token_batch(prompts)

    # Process results
    results = []
    result_idx = 0
    for i, input_obj in enumerate(batch_data):
        if i not in valid_indices:
            # No definitions for this word
            results.append(
                DisambiguationResult(
                    synset_id=NO_DEFINITIONS_FOUND,
                    definition="",
                    confidence=0.0
                )
            )
        else:
            # Process the prediction for this word
            unmask_result = batch_results[result_idx]
            result_idx += 1

            # Get probabilities for all choices
            choice_probs = get_choice_probabilities(
                components.tokenizer,
                unmask_result.probabilities,
                input_obj.definitions
            )
            # Find best choice and normalize
            best_choice_idx = choice_probs.index(max(choice_probs))
            total_prob = sum(choice_probs)
            normalized_score = choice_probs[best_choice_idx] / total_prob if total_prob > 0 else 0.0

            # Handle "none of the above" case
            if best_choice_idx == len(input_obj.definitions):  # Next letter option selected
                results.append(
                    DisambiguationResult(
                        synset_id="",
                        definition=NONE_OF_THE_ABOVE,
                        confidence=normalized_score
                    )
                )
            else:
                best_definition = input_obj.definitions[best_choice_idx]
                results.append(
                    DisambiguationResult(
                        synset_id=best_definition.synset_id,
                        definition=best_definition.definition,
                        confidence=normalized_score
                    )
                )

    return results


def disambiguate(text: str, language: str = "en") -> WordSenseDisambiguation:
    nlp = get_spacy_pipeline(language)
    doc = nlp(text)
    tokens = []
    entities = []

    pos_map = {
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

    # First pass: Create all base tokens and identify content words to disambiguate
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
        if token.pos_ in pos_map and not token.is_punct and not token.is_space:
            content_word_indices.append(token.i)

    # Batch fetch definitions for all content words
    if content_word_indices:
        queries = [
            WordQuery(form=tokens[i].lemma, pos=pos_map[tokens[i].pos])
            for i in content_word_indices
        ]
        all_definitions = get_definitions(queries, language)

        # Prepare batch data for disambiguation
        batch_data = []
        valid_indices = []  # Track which content words have definitions
        for idx, definitions in zip(content_word_indices, all_definitions):
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

        # Batch disambiguate all content words with definitions
        if batch_data:
            predictions = disambiguate_word_batch(batch_data)

            # Update tokens with disambiguation results
            for token_idx, result in zip(valid_indices, predictions):
                # Handle "none of the above" case
                if result.definition == NONE_OF_THE_ABOVE:
                    tokens[token_idx].synset_id = None
                    tokens[token_idx].synset_definition = None
                    tokens[token_idx].confidence = result.confidence
                else:
                    tokens[token_idx].synset_id = result.synset_id
                    tokens[token_idx].synset_definition = result.definition
                    tokens[token_idx].confidence = result.confidence

    # Extract linked entities using entityLinker
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
