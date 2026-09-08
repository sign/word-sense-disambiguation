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


def link_named_entities(doc):
    """Yield ``(span, item_id, label, description)`` for each named-entity span of a spaCy doc with a Wikidata match."""
    for ent in doc.ents:
        if ent.label_ not in NAMED_ENTITY_LABELS:
            continue
        hit = best_candidate(ent.text)
        if hit:
            yield ent, *hit
