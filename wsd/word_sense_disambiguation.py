from dataclasses import dataclass
from functools import cache
from typing import Optional

import requests
import spacy
from colorama import Fore, Style, init

from wsd.env import WORDNET_URL
from wsd.masked_language_model import load_model, unmask_token


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


def get_definitions(word: str, pos: str, language: str = "en") -> list[tuple[str, str]]:
    """Fetch definitions for a word from the API"""
    url = f"{WORDNET_URL}/lexicons/omw-{language}:1.4/words?form={word}&pos={pos}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            definitions = []
            for item in data.get('data', []):
                for included in item.get('included', []):
                    definition = included.get('attributes', {}).get('definition', '')
                    if definition:
                        definitions.append((included['id'], definition))
            return definitions
        else:
            return []
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching definitions for {word}: {e}")
        return []


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
    return f"""Disambiguate the meaning of the highlighted word based on its usage in the sentence.
Choose the most appropriate sense from the list.

Sentence:
{marked_sentence}

Question:
Which sense best matches the meaning of the highlighted word (*{word}*) as used in this sentence?

Choices:
{"\n".join(choices)}

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
        return "No definitions found", 0.0

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


def disambiguate(text: str, language: str = "en") -> WordSenseDisambiguation:
    nlp = get_spacy_pipeline(language)
    doc = nlp(text)
    tokens = []
    entities = []

    pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a'}

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

        # Only disambiguate content words
        if token.pos_ in pos_map and not token.is_punct and not token.is_space:
            pos = pos_map[token.pos_]
            definitions = get_definitions(token.lemma_.lower(), pos, language)

            if definitions:
                marked_sentence = create_marked_sentence(doc, token.i)
                synset_id, best_def, confidence = disambiguate_word(token.text, marked_sentence, definitions)

                # Handle "none of the above" case
                if best_def == "none of the above":
                    disambiguated_token.synset_id = None
                    disambiguated_token.synset_definition = None
                    disambiguated_token.confidence = confidence
                else:
                    # For now, use definition as synset_id placeholder
                    # In a real implementation, you'd map definitions to synset IDs
                    disambiguated_token.synset_id = synset_id
                    disambiguated_token.synset_definition = best_def
                    disambiguated_token.confidence = confidence

        tokens.append(disambiguated_token)

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
