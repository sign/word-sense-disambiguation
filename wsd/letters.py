"""
Build the fixed set of single-token option letters for WSD multiple-choice prompts.

The list and order are deterministic for a given tokenizer. Training and inference
must call `build_letters(tokenizer)` with the same tokenizer to agree on the mapping.
"""
import string
from dataclasses import dataclass
from functools import cache

from transformers import PreTrainedTokenizerBase

NUM_LETTERS = 128

# Index of the letter reserved for the "none of the above" slot. Fixed across
# all prompts so the model sees a single, consistent reject token instead of
# the NOTA meaning rotating across every letter based on option count.
NOTA_LETTER_INDEX = NUM_LETTERS - 1


class NotEnoughSingleTokenLettersError(RuntimeError):
    def __init__(self, found: int, needed: int):
        super().__init__(f"Tokenizer yielded only {found} single-token letters, need {needed}")


@dataclass(frozen=True)
class LetterSet:
    """Fixed, deterministic mapping between compact answer indices and letters/token ids."""
    letters: tuple[str, ...]     # length == NUM_LETTERS
    token_ids: tuple[int, ...]   # length == NUM_LETTERS; tokenizer.encode(' ' + letter)[0]


def _candidate_pools() -> list[list[str]]:
    """Priority-ordered pools of candidate answer-letter characters.

    The first pool has the most familiar / readable letters; we fill remaining
    slots from later pools only if earlier pools don't yield enough single-token
    characters on the given tokenizer.
    """
    latin = list(string.ascii_uppercase + string.ascii_lowercase)
    digits = list(string.digits)
    # '.' and other format-significant punctuation excluded (clashes with "A. " template)
    safe_symbols = list("!@#$%^&*+=<>?/|~`'()[]{}_-")
    greek_upper = [chr(c) for c in range(0x0391, 0x03A9 + 1) if c != 0x03A2]
    greek_lower = [chr(c) for c in range(0x03B1, 0x03C9 + 1)]
    cyrillic_upper = [chr(c) for c in range(0x0410, 0x042F + 1)]
    cyrillic_lower = [chr(c) for c in range(0x0430, 0x044F + 1)]
    return [latin, digits, safe_symbols, greek_upper, greek_lower, cyrillic_upper, cyrillic_lower]


@cache
def build_letters(tokenizer: PreTrainedTokenizerBase) -> LetterSet:
    """Select exactly NUM_LETTERS characters that are single-token when space-prefixed.

    Deterministic: always yields the same list for the same tokenizer. Safe to call
    independently in training and inference, they will agree.
    """
    letters: list[str] = []
    ids: list[int] = []
    seen: set[int] = set()
    unk_id = tokenizer.unk_token_id

    for pool in _candidate_pools():
        for c in pool:
            encoded = tokenizer.encode(" " + c, add_special_tokens=False)
            if len(encoded) != 1:
                continue
            tid = encoded[0]
            if tid == unk_id or tid in seen:
                continue
            letters.append(c)
            ids.append(tid)
            seen.add(tid)
            if len(letters) >= NUM_LETTERS:
                break
        if len(letters) >= NUM_LETTERS:
            break

    if len(letters) < NUM_LETTERS:
        raise NotEnoughSingleTokenLettersError(len(letters), NUM_LETTERS)

    return LetterSet(letters=tuple(letters), token_ids=tuple(ids))
