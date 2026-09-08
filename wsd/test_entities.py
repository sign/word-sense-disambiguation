"""Named-entity linking over the alias table, offline (the knowledge base is monkeypatched)."""
from types import SimpleNamespace

import wsd.entities as entities

# alias -> rows as returned by spacy-entity-linker: (item_id, label, description, views, inlinks, alias)
ROWS = {
    "apple": [(312, "Apple Inc.", "American technology company", 5000, 900, "Apple"),
              (89, "apple", "fruit", 3000, 800, "apple")],
    "us": [(30, "United States", "country in North America", 9000, 9999, "US")],
    "washington": [(23, "George Washington", "first president", 8000, 900, "Washington")],
    "d.c.": [(61, "Washington, D.C.", "capital of the United States", 7000, 800, "D.C.")],
    "washington, d.c.": [(61, "Washington, D.C.", "capital of the United States", 7000, 800, "Washington, D.C.")],
}


def fake_db():
    return SimpleNamespace(get_entities_from_alias=lambda a: ROWS.get(a.lower(), []))


def test_best_candidate_prefers_views_and_tries_variants(monkeypatch):
    monkeypatch.setattr(entities, "get_wikidata_instance", fake_db)
    entities.best_candidate.cache_clear()
    assert entities.best_candidate("Apple") == (312, "Apple Inc.", "American technology company")
    assert entities.best_candidate("the Apple's") == (312, "Apple Inc.", "American technology company")
    assert entities.best_candidate("U.S.") == (30, "United States", "country in North America")  # dots stripped
    assert entities.best_candidate("Zzyzx") is None


def test_only_named_entity_labels_are_linked(monkeypatch):
    monkeypatch.setattr(entities, "get_wikidata_instance", fake_db)
    entities.best_candidate.cache_clear()
    doc = SimpleNamespace(ents=[SimpleNamespace(text="Apple", label_="ORG"),
                                SimpleNamespace(text="Apple", label_="CARDINAL")])
    assert [(item_id, label) for _, item_id, label, _ in entities.link_named_entities(doc)] == [(312, "Apple Inc.")]


def test_adjacent_spans_with_a_joint_alias_are_linked_as_one(monkeypatch):
    import spacy
    from spacy.tokens import Span

    monkeypatch.setattr(entities, "get_wikidata_instance", fake_db)
    entities.best_candidate.cache_clear()
    doc = spacy.blank("en")("Obama lived in Washington, D.C. for years.")
    doc.set_ents([Span(doc, 3, 4, label="GPE"), Span(doc, 5, 6, label="GPE")])  # "Washington", "D.C."
    linked = [(span.text, item_id) for span, item_id, _, _ in entities.link_named_entities(doc)]
    assert linked == [("Washington, D.C.", 61)]
