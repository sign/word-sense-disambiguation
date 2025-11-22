from dataclasses import dataclass

NONE_OF_THE_ABOVE = "none of the above"


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
        raise ValueError("Index too large for option letter")


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
    """Create multiple choice prompt for word sense disambiguation"""
    choices = []
    for i, definition_obj in enumerate(definitions):
        letter = get_option_letter(i)
        choices.append(f"{letter}. {definition_obj.definition}")

    # Add "none of the above" option using next sequential letter
    none_letter = get_option_letter(len(definitions))
    choices.append(f"{none_letter}. {NONE_OF_THE_ABOVE}")
    choices_lines = "\n".join(choices)
    return f"""What is the meaning of *{word}* in this sentence?

Sentence: {marked_sentence}

{choices_lines}

Answer: {mask_token}"""
