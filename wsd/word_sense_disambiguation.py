from dataclasses import dataclass
from functools import cache
from typing import Optional

import requests
import spacy
from colorama import Fore, Style, init

from wsd.env import WORDNET_URL
from wsd.masked_language_model import load_model, unmask_token, unmask_token_batch


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


def get_definitions(queries: list[WordQuery], language: str = "en") -> list[list[tuple[str, str]]]:
    """
    Fetch definitions for multiple words using the batch endpoint.

    Args:
        queries: List of WordQuery objects with form and pos
        language: Language code (default: "en")

    Returns:
        List of definition lists, one per query (in same order as input).
        Each definition list contains (synset_id, definition) tuples.
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
        for i, item in enumerate(data.get("data", [])):
            definitions = []
            # Sort definitions by synset_id for consistent ordering
            definitions_dict = item.get("definitions", {})
            for synset_id in sorted(definitions_dict.keys()):
                definition = definitions_dict[synset_id]
                definitions.append((synset_id, definition))
            results.append(definitions)

            # Debug: print first few examples
            if i < 3:
                query = queries[i]
                print(f"DEBUG: Query {i}: form={query.form}, pos={query.pos}")
                print(f"DEBUG: Got {len(definitions)} definitions")
                if definitions:
                    print(f"DEBUG: First definition: {definitions[0]}")
                print()

        # If response doesn't have enough items, pad with empty lists
        if len(results) < len(queries):
            raise Exception("API response has fewer items than queries")

    except (requests.RequestException, ValueError) as e:
        print(f"Error making batch request to {url}: {e}")
        return [[] for _ in queries]
    else:
        return results


def get_definitions_single(word: str, pos: str, language: str = "en") -> list[tuple[str, str]]:
    """
    Convenience function to fetch definitions for a single word.

    Args:
        word: Word form to look up
        pos: Part of speech
        language: Language code (default: "en")

    Returns:
        List of (synset_id, definition) tuples for the word
    """
    query = WordQuery(form=word, pos=pos)
    results = get_definitions([query], language)
    return results[0] if results else []


def create_marked_sentence(doc, target_position: int) -> str:
    """Create sentence with target word marked with asterisks"""
    text = ""
    for token in doc:
        if token.i == target_position:
            text += f"*{token.text}*"
        else:
            text += token.text
        text += token.whitespace_
    return text


def create_multiple_choice_prompt(word: str, mask_token: str, marked_sentence: str,
                                  definitions: list[tuple[str, str]]) -> str:
    """Create multiple choice prompt for word sense disambiguation"""
    choices = []
    for i, (_, definition) in enumerate(definitions):
        letter = chr(ord('A') + i)
        choices.append(f"- {letter}: {definition}")

    # Add "none of the above" option using next sequential letter
    none_letter = chr(ord('A') + len(definitions))
    choices.append(f"- {none_letter}: none of the above")
    choices_lines = "\n".join(choices)
    return f"""Disambiguate the meaning of the highlighted word based on its usage in the sentence.
Choose the most appropriate sense from the list.

Sentence:
{marked_sentence}

Question:
Which sense best matches the meaning of the highlighted word (*{word}*) as used in this sentence?

Choices:
{choices_lines}

Answer: [unused0] {mask_token}"""


def get_choice_probabilities(tokenizer, probs, definitions: list[tuple[str, str]]) -> list[float]:
    """Get probabilities for all choice letters including 'none of the above'"""
    choice_probs = []

    # Get probabilities for definition choices A, B, C, etc.
    for i in range(len(definitions)):
        letter = chr(ord('A') + i)

        # Get probabilities for both "A" and " A" tokens and sum them
        letter_token_id = tokenizer.convert_tokens_to_ids(letter)
        space_letter_token_id = tokenizer.convert_tokens_to_ids(f" {letter}")

        total_prob = 0.0
        if letter_token_id is not None:
            total_prob += probs[letter_token_id].item()
        if space_letter_token_id is not None:
            total_prob += probs[space_letter_token_id].item()

        choice_probs.append(total_prob)

    # Add probability for next letter (none of the above)
    none_letter = chr(ord('A') + len(definitions))
    none_token_id = tokenizer.convert_tokens_to_ids(none_letter)
    space_none_token_id = tokenizer.convert_tokens_to_ids(f" {none_letter}")

    none_prob = 0.0
    if none_token_id is not None:
        none_prob += probs[none_token_id].item()
    if space_none_token_id is not None:
        none_prob += probs[space_none_token_id].item()

    choice_probs.append(none_prob)
    return choice_probs


def disambiguate_word(word: str, marked_sentence: str, definitions: list[tuple[str, str]]) -> tuple[str, str, float]:
    """Use ModernBERT to disambiguate word sense given context and definitions"""
    if not definitions:
        return "No definitions found", "", 0.0

    model, tokenizer, device = load_model()

    # Create multiple choice prompt
    text = create_multiple_choice_prompt(word, tokenizer.mask_token, marked_sentence, definitions)

    # Get prediction
    predicted_token, probs = unmask_token(text)

    # Get probabilities for all choices
    choice_probs = get_choice_probabilities(tokenizer, probs, definitions)

    # Find best choice and normalize
    best_choice_idx = choice_probs.index(max(choice_probs))
    total_prob = sum(choice_probs)
    normalized_score = choice_probs[best_choice_idx] / total_prob if total_prob > 0 else 0.0

    # Handle "none of the above" case
    if best_choice_idx == len(definitions):  # Next letter option selected
        return "", "none of the above", normalized_score
    else:
        best_synset, best_definition = definitions[best_choice_idx]
        return best_synset, best_definition, normalized_score


def disambiguate_word_batch(
    batch_data: list[tuple[str, str, list[tuple[str, str]]]]
) -> list[tuple[str, str, float]]:
    """
    Batch version of disambiguate_word that processes multiple words in parallel.

    Args:
        batch_data: List of tuples, each containing:
            - word: The word to disambiguate
            - marked_sentence: Sentence with the word marked
            - definitions: List of (synset_id, definition) tuples

    Returns:
        List of tuples (synset_id, definition, confidence) for each input
    """
    if not batch_data:
        return []

    model, tokenizer, device = load_model()

    # Prepare prompts and track which ones are valid
    prompts = []
    valid_indices = []  # Track which inputs have definitions
    for i, (word, marked_sentence, definitions) in enumerate(batch_data):
        if definitions:
            text = create_multiple_choice_prompt(word, tokenizer.mask_token, marked_sentence, definitions)
            prompts.append(text)
            valid_indices.append(i)

    # If no valid prompts, return empty results for all
    if not prompts:
        return [("No definitions found", "", 0.0) for _ in batch_data]

    # Get predictions for all prompts in batch
    batch_results = unmask_token_batch(prompts)

    # Process results
    results = []
    result_idx = 0
    for i, (_word, _marked_sentence, definitions) in enumerate(batch_data):
        if i not in valid_indices:
            # No definitions for this word
            results.append(("No definitions found", "", 0.0))
        else:
            # Process the prediction for this word
            _, probs = batch_results[result_idx]
            result_idx += 1

            # Get probabilities for all choices
            choice_probs = get_choice_probabilities(tokenizer, probs, definitions)

            # Find best choice and normalize
            best_choice_idx = choice_probs.index(max(choice_probs))
            total_prob = sum(choice_probs)
            normalized_score = choice_probs[best_choice_idx] / total_prob if total_prob > 0 else 0.0

            # Handle "none of the above" case
            if best_choice_idx == len(definitions):  # Next letter option selected
                results.append(("", "none of the above", normalized_score))
            else:
                best_synset, best_definition = definitions[best_choice_idx]
                results.append((best_synset, best_definition, normalized_score))

    return results


def disambiguate(text: str, language: str = "en") -> WordSenseDisambiguation:
    nlp = get_spacy_pipeline(language)
    doc = nlp(text)
    tokens = []
    entities = []

    pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a'}

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
                batch_data.append((tokens[idx].word, marked_sentence, definitions))
                valid_indices.append(idx)

        # Batch disambiguate all content words with definitions
        if batch_data:
            predictions = disambiguate_word_batch(batch_data)

            # Update tokens with disambiguation results
            for token_idx, (synset_id, best_def, confidence) in zip(valid_indices, predictions):
                # Handle "none of the above" case
                if best_def == "none of the above":
                    tokens[token_idx].synset_id = None
                    tokens[token_idx].synset_definition = None
                    tokens[token_idx].confidence = confidence
                else:
                    tokens[token_idx].synset_id = synset_id
                    tokens[token_idx].synset_definition = best_def
                    tokens[token_idx].confidence = confidence

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
    test_sentence = "Apple is a technology company."

    results = disambiguate(test_sentence)

    # Add the new colorful visualization
    visualize_sentence(test_sentence, results.tokens)
