"""Wikidata entity linking for spaCy named-entity spans.

The linker used to run spacy-entity-linker's own candidate extractor over every noun chunk and
pick the most-viewed Wikidata item, which linked "lamb", "consumers" and "scientists" alongside
real names: on AIDA-CoNLL test it emitted 2.5 links per gold mention (precision 20%, F1 0.29).
Linking only spaCy's person, organisation, place and facility spans through the same alias table
and prior gives precision 58% at nearly the same recall (F1 0.50), and skips the linker's expensive
noun-chunk pass. Common nouns
are covered by word sense disambiguation instead.
"""
from functools import lru_cache

from spacy_entity_linker.DatabaseConnection import get_wikidata_instance

# Labels whose spans link well through an alias table and a popularity prior. Nationalities and groups (NORP)
# do not: "American" -> a 2010 film, "Republican" -> a newspaper (8% right on AIDA); word sense disambiguation
# covers those adjectives. Events, works, products, laws and languages were too few and too noisy to keep.
NAMED_ENTITY_LABELS = frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC"})


def _variants(text: str) -> list[str]:
    stripped = text.removeprefix("the ").removeprefix("The ").removesuffix("'s")
    return list(dict.fromkeys([text, stripped, text.replace(".", ""), stripped.replace(".", "")]))


@lru_cache(maxsize=200_000)
def best_candidate(text: str) -> tuple[int, str, str | None] | None:
    """(Wikidata item id, label, description) of the most-viewed item whose alias matches the mention
    (case-insensitive), trying the mention without a leading article, possessive or dots; None if no alias."""
    db = get_wikidata_instance()
    for variant in _variants(text):
        rows = db.get_entities_from_alias(variant)
        if rows:
            item_id, label, description, views, _inlinks, _alias = max(rows, key=lambda r: r[3] or 0)
            return int(item_id), label, description
    return None


def _merge_adjacent(ents):
    """spaCy often splits "Washington, D.C." or "New York, NY" into two spans; when the text of two adjacent
    named spans (joined by a comma and/or a space) has an alias of its own, link that instead of the parts."""
    ents = [e for e in ents if e.label_ in NAMED_ENTITY_LABELS]
    merged = []
    i = 0
    while i < len(ents):
        span = ents[i]
        if i + 1 < len(ents):
            gap = span.doc.text[span.end_char:ents[i + 1].start_char]
            if gap in (", ", " ", ",") and best_candidate(span.doc.text[span.start_char:ents[i + 1].end_char]):
                span = span.doc[span.start:ents[i + 1].end]
                i += 1
        merged.append(span)
        i += 1
    return merged


def link_named_entities(doc):
    """Yield ``(span, item_id, label, description)`` for each named-entity span of a spaCy doc with a Wikidata match."""
    for span in _merge_adjacent(doc.ents):
        hit = best_candidate(span.text)
        if hit:
            yield span, *hit
