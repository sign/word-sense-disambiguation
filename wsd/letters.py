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


@dataclass(frozen=True)
class LetterSet:
    """Fixed, deterministic mapping between compact answer indices and letters/token ids."""
    letters: tuple[str, ...]     # length == NUM_LETTERS
    token_ids: tuple[int, ...]   # length == NUM_LETTERS; tokenizer.encode(' ' + letter)[0]


# Priority-ordered pools of candidate answer letters: the most readable first, later pools only fill
# slots when earlier ones do not yield enough single-token characters on the tokenizer. Excluded:
# '.' (clashes with the "A. " option template), '*' (the *word* marker), "'" (looks like a contraction).
_CANDIDATE_POOLS = [
    list(string.ascii_uppercase + string.ascii_lowercase),
    list(string.digits),
    list("!@#$%^&+=<>?/|~`()[]{}_-"),
    [chr(c) for c in range(0x0391, 0x03A9 + 1) if c != 0x03A2],  # Greek upper
    [chr(c) for c in range(0x03B1, 0x03C9 + 1)],  # Greek lower
    [chr(c) for c in range(0x0410, 0x042F + 1)],  # Cyrillic upper
    [chr(c) for c in range(0x0430, 0x044F + 1)],  # Cyrillic lower
]


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

    for pool in _CANDIDATE_POOLS:
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
        raise RuntimeError(f"Tokenizer yielded only {len(letters)} single-token letters, need {NUM_LETTERS}")

    return LetterSet(letters=tuple(letters), token_ids=tuple(ids))
