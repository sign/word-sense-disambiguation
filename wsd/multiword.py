"""Multiword expressions ("test tube", "give up", "New York") from the WordNet lexicon.

WordNet has ~64k multiword forms, 42% of its vocabulary and 94% nouns. Disambiguating
them as one unit gives the compound's own sense (issue #1) and saves prompts.
"""
import logging
import re
from dataclasses import dataclass
from functools import cache

import requests

from wsd.env import WORDNET_URL

logger = logging.getLogger(__name__)
MAX_TOKENS = 5


@dataclass(frozen=True)
class MultiwordSpan:
    start: int  # token index, inclusive
    end: int  # token index, exclusive
    form: str  # WordNet form as listed in the lexicon (casing preserved), used for lookups
    head: int  # token index whose POS stands for the expression


@cache
def _index() -> dict[str, dict[tuple[str, ...], str]]:
    """``first lowercase token -> {lowercase token tuple: canonical form}`` for every multiword form."""
    url = f"{WORDNET_URL}/lexicons/omw-en:1.4/forms"
    try:
        forms = requests.get(url, timeout=120).json()["data"]
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("could not load WordNet forms from %s (%s); multiword expressions disabled", url, e)
        return {}
    index: dict[str, dict[tuple[str, ...], str]] = {}
    for form in forms:
        # WordNet keeps hyphens inside forms; prose and spaCy may use spaces
        # or separate dash tokens. All spellings still query the canonical form.
        variants = {
            tuple(form.lower().split()),
            tuple(re.split(r"[\s-]+", form.lower())),
            tuple(re.findall(r"[^\s-]+|-", form.lower())),
        }
        for tokens in variants:
            if 1 < len(tokens) <= MAX_TOKENS * 2 - 1 and sum(t != "-" for t in tokens) <= MAX_TOKENS:
                index.setdefault(tokens[0], {}).setdefault(tokens, form)
    return index


def find_spans(doc) -> list[MultiwordSpan]:
    """Greedy, longest-first, non-overlapping matches of WordNet multiword forms in a
    spaCy (or light) doc, on lemmas or on lowercased surface tokens."""
    index = _index()
    tokens = list(doc)
    lemmas = [t.lemma_.lower() for t in tokens]
    lowers = [t.text.lower() for t in tokens]
    matches = []
    for i in range(len(tokens)):
        candidates = {**index.get(lemmas[i], {}), **index.get(lowers[i], {})}
        for n in range(min(MAX_TOKENS * 2 - 1, len(tokens) - i), 1, -1):
            form = candidates.get(tuple(lemmas[i:i + n])) or candidates.get(tuple(lowers[i:i + n]))
            if form:
                head = _head(tokens[i:i + n], i, tokens)
                if head is not None:
                    matches.append(MultiwordSpan(start=i, end=i + n, form=form, head=head))
    # Compare lengths across starts: "pull off" must not consume the beginning
    # of the longer "off the shelf" candidate before the model can judge it.
    spans, covered = [], set()
    for span in sorted(matches, key=lambda s: (s.start - s.end, s.start)):
        positions = set(range(span.start, span.end))
        if not covered.intersection(positions):
            spans.append(span)
            covered.update(positions)
    return sorted(spans, key=lambda s: s.start)


def _head_i(token) -> int:  # spaCy Token.head.i or LightToken.head_i
    return getattr(token, "head_i", None) if hasattr(token, "head_i") else token.head.i


def _head(span, start: int, tokens) -> int | None:
    """Token index whose POS stands for the expression, or None when the match is spurious.

    Compounds are head-final ("test tube" -> "tube"). A span containing a verb is a
    verb-particle expression ("gave up", "take part") only when its other tokens depend
    directly on that verb and head nothing outside the span; a preposition with its own
    object ("go to the store", "live in hopes") is a phrase boundary, not a compound."""
    verbs = [j for j, t in enumerate(span) if t.pos_ == "VERB"]
    if not verbs:
        return start + len(span) - 1
    verb = start + verbs[0]
    inside = range(start, start + len(span))
    for j, t in enumerate(span):
        if start + j == verb:
            continue
        if _head_i(t) != verb or any(_head_i(o) == start + j for k, o in enumerate(tokens) if k not in inside):
            return None
    return verb
