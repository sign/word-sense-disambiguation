import re
from dataclasses import dataclass

NONE_OF_THE_ABOVE = "none of the above"

# Index of the letter reserved for the "none of the above" (NOTA) slot.
# Fixed across all prompts so the model sees a single, consistent reject token
# instead of the NOTA meaning rotating across every letter based on option
# count. The letter lives at the top of the available range so it does not
# collide with normal option slots; callers must pass at most this many
# definitions.
NOTA_LETTER_INDEX = 108


class OptionLetterIndexError(ValueError):
    """Raised when index is too large for option letter"""
    def __init__(self, index: int):
        super().__init__(f"Index too large for option letter {index}")


class WordNotFoundError(ValueError):
    """Raised when *word* cannot be found in *sentence* with word boundaries."""
    def __init__(self, word: str, sentence: str):
        super().__init__(f"Word {word!r} not found with word boundaries in sentence: {sentence!r}")
        self.word = word
        self.sentence = sentence


def mark_word_in_sentence(sentence: str, word: str) -> str:
    """Mark the first word-boundary occurrence of *word* in *sentence* with asterisks.

    Case-insensitive. Uses regex word boundaries so the match does not fire
    inside longer words ("bank" does not match inside "bankrupt"). Exactly
    one span is marked (the first match), so the output always contains
    exactly one ``*...*`` pair.

    Raises ``WordNotFoundError`` if no word-boundary match is found, or if
    *sentence* already contains an asterisk (which would be ambiguous with
    our marker character).
    """
    if "*" in sentence:
        raise WordNotFoundError(word, sentence)
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if match is None:
        raise WordNotFoundError(word, sentence)
    start, end = match.span()
    marked = sentence[:start] + "*" + sentence[start:end] + "*" + sentence[end:]
    # Invariant: exactly one marked span (two asterisks) in the output.
    assert marked.count("*") == 2, marked
    return marked


@dataclass
class Definition:
    """A single word definition from WordNet"""
    synset_id: str
    definition: str


def get_option_letter(index: int) -> str:
    """
    Get option letter for given index.
    Uses A-Z (26), then a-z (26) for total of 52 possible options.
    Each option is a single character token for masked language model prediction.
    """
    if index < 26:
        return chr(ord('A') + index)
    elif index < 52:
        return chr(ord('a') + (index - 26))
    elif index < 76:
        return chr(ord('α') + (index - 52))
    elif index < 109:
        return chr(ord('А') + (index - 76))
    else:
        raise OptionLetterIndexError(index)


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


def create_multiple_choice_prompt(word: str,
                                  mask_token: str,
                                  marked_sentence: str,
                                  definitions: list[Definition]) -> str:
    """Create multiple choice prompt for word sense disambiguation.

    The letter at :data:`NOTA_LETTER_INDEX` is always reserved for the
    "none of the above" option, so it is a stable reject signal across all
    prompts. Definitions occupy letters ``0..NOTA_LETTER_INDEX-1``.
    """
    if len(definitions) > NOTA_LETTER_INDEX:
        raise OptionLetterIndexError(len(definitions))

    choices = []
    for i, definition_obj in enumerate(definitions):
        letter = get_option_letter(i)
        choices.append(f"{letter}. {definition_obj.definition}")

    # NOTA always uses the reserved letter, regardless of how many
    # definitions preceded it. Training and inference agree on this index.
    none_letter = get_option_letter(NOTA_LETTER_INDEX)
    choices.append(f"{none_letter}. {NONE_OF_THE_ABOVE}")
    choices_lines = "\n".join(choices)
    return f"""What is the meaning of *{word}* in this sentence?

Sentence: {marked_sentence}

{choices_lines}

Answer: [unused0] {mask_token}"""
