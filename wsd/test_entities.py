"""Named-entity linking over the alias table, offline (the knowledge base is monkeypatched)."""
from types import SimpleNamespace

import wsd.entities as entities

# alias -> rows as returned by spacy-entity-linker: (item_id, label, description, views, inlinks, alias)
ROWS = {
    "apple": [(312, "Apple Inc.", "American technology company", 5000, 900, "Apple"),
              (89, "apple", "fruit", 3000, 800, "apple")],
    "us": [(30, "United States", "country in North America", 9000, 9999, "US")],
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
