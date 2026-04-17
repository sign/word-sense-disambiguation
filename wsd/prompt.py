import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from wsd.letters import NOTA_LETTER_INDEX, build_letters

NONE_OF_THE_ABOVE = "none of the above"


class WordNotFoundError(ValueError):
    """Raised when *word* cannot be found in *sentence* with word boundaries."""
    def __init__(self, word: str, sentence: str):
        super().__init__(f"Word {word!r} not found with word boundaries in sentence: {sentence!r}")
        self.word = word
        self.sentence = sentence


def mark_word_in_sentence(sentence: str, word: str) -> str:
    """Mark the first word-boundary occurrence of *word* in *sentence* with asterisks.

    Case-insensitive. Uses regex word boundaries so the match does not fire
    inside longer words ("100" does not match "100th"). Exactly one span is
    marked (the first match), so the output always contains exactly one
    ``*...*`` pair. This is the single source of truth for marking shared by
    training data generation and benchmark/inference paths; identical output
    here means identical prompts.

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


class OptionLetterIndexError(ValueError):
    """Raised when index is too large for option letter"""
    def __init__(self, index: int):
        super().__init__(f"Index too large for option letter {index}")


@dataclass
class Definition:
    """A single word definition from WordNet"""
    synset_id: str
    definition: str


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
                                  definitions: list[Definition],
                                  tokenizer: PreTrainedTokenizerBase,
                                  start_offset: int = 0) -> str:
    """Create multiple choice prompt for word sense disambiguation.

    ``definitions[i]`` is rendered with letter ``start_offset + i``; the
    default ``start_offset=0`` keeps the historical "A, B, C, ..." layout.
    Non-zero offsets are used by the bias probe to test whether the model
    depends on the specific A-first mapping it saw during training.

    The last letter (index :data:`wsd.letters.NOTA_LETTER_INDEX`) is always
    reserved for the "none of the above" option. The offset window must not
    collide with NOTA: ``start_offset + len(definitions) <= NOTA_LETTER_INDEX``.
    """
    letters = build_letters(tokenizer).letters
    if start_offset < 0 or start_offset + len(definitions) > NOTA_LETTER_INDEX:
        raise OptionLetterIndexError(start_offset + len(definitions))

    choices = []
    for i, definition_obj in enumerate(definitions):
        letter = letters[start_offset + i]
        choices.append(f"{letter}. {definition_obj.definition}")

    # NOTA always uses the reserved last letter, regardless of how many
    # definitions preceded it. Training and inference agree on this index.
    none_letter = letters[NOTA_LETTER_INDEX]
    choices.append(f"{none_letter}. {NONE_OF_THE_ABOVE}")
    choices_lines = "\n".join(choices)
    return f"""What is the meaning of *{word}* in this sentence?

Sentence: {marked_sentence}

{choices_lines}

Answer: [unused0] {mask_token}"""
