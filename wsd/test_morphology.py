"""Morphology survives the API and pickled batch-worker paths without WSD inference."""
import pickle
from dataclasses import asdict

import pytest
import spacy
from starlette.testclient import TestClient

import wsd.word_sense_disambiguation as wsd


@pytest.fixture
def doc(monkeypatch):
    doc = spacy.blank("en")("She visited books.")
    for token, lemma, pos, morph in zip(
        doc,
        ["she", "visit", "book", "."],
        ["PRON", "VERB", "NOUN", "PUNCT"],
        ["Case=Acc,Nom|Number=Sing", "Tense=Past|VerbForm=Fin", "Number=Plur", ""],
        strict=True,
    ):
        token.lemma_ = lemma
        token.pos_ = pos
        token.set_morph(morph)
    monkeypatch.setattr("wsd.entities.link_named_entities", lambda doc: [])
    monkeypatch.setattr(wsd, "get_definitions", lambda queries: [[] for _ in queries])
    monkeypatch.setattr(wsd, "find_spans", lambda doc: [])
    return doc


def test_morphology_survives_batch_worker(doc):
    light = pickle.loads(pickle.dumps(wsd.light_doc(doc)))
    direct, batch = wsd.disambiguate_docs([doc, light])
    assert direct == batch
    assert [token.morph for token in batch.tokens] == [
        {"Case": "Acc,Nom", "Number": "Sing"},
        {"Tense": "Past", "VerbForm": "Fin"},
        {"Number": "Plur"},
        {},
    ]


def test_api_serializes_morphology(doc, monkeypatch):
    import wsd.server as server

    result, = wsd.disambiguate_docs([doc])
    monkeypatch.setenv("WSD_WARMUP", "0")
    monkeypatch.setattr(server, "disambiguate", lambda **kwargs: result)
    with TestClient(server.app) as client:
        response = client.get("/disambiguate", params={"text": doc.text, "lang": "en"})
    assert response.status_code == 200
    assert response.json() == asdict(result)
    assert response.json()["tokens"][2]["morph"] == {"Number": "Plur"}


def test_morphology_defaults_are_independent():
    first = wsd.DisambiguatedToken("book", "book", "NOUN", 0, 0, 4)
    second = wsd.DisambiguatedToken("book", "book", "NOUN", 0, 0, 4)
    first.morph["Number"] = "Sing"
    assert second.morph == {}
    light = wsd.LightToken("book", "book", "NOUN", 0, 0, False, False, "")
    assert wsd._create_base_tokens(wsd.LightDoc([light], []))[0][0].morph == {}


def test_spacy_detects_plural_and_past():
    nlp = spacy.load("en_core_web_lg", exclude=["ner"])
    tokens = {token.text: token for token in nlp("Ada visited and bought 42 books.")}
    assert tokens["books"].lemma_ == "book"
    assert tokens["books"].morph.to_dict() == {"Number": "Plur"}
    assert tokens["visited"].morph.to_dict() == {"Tense": "Past", "VerbForm": "Fin"}
