"""Span-first disambiguation of WordNet multiword expressions (issue #1), offline."""
import pytest

import wsd.word_sense_disambiguation as wsd
from wsd.multiword import MultiwordSpan, find_spans
from wsd.prompt import NONE_OF_THE_ABOVE, Definition
from wsd.word_sense_disambiguation import LightDoc, LightToken, disambiguate_docs

DEFS = {
    ("test tube", "n"): [Definition("test_tube", "glass tube closed at one end")],
    ("hot dog", "n"): [Definition("hot_dog", "a frankfurter served hot on a bun")],
    ("hold", "v"): [Definition("hold-1", "keep in a certain state"), Definition("hold-2", "have in one's hands")],
    ("hot", "a"): [Definition("hot", "used of physical heat")],
    ("dog", "n"): [Definition("dog", "a member of the genus Canis")],
    ("bark", "v"): [Definition("bark", "speak in an unfriendly tone")],
}
INDEX = {"test": {("test", "tube"): "test tube"}, "hot": {("hot", "dog"): "hot dog"},
         "give": {("give", "up"): "give up"}}


def tok(i, text, lemma, pos, idx, ws=" ", dep="", head=-1):
    return LightToken(text, lemma, pos, i, idx, pos == "PUNCT", False, ws, dep, head)


TEST_TUBE = LightDoc([
    tok(0, "She", "she", "PRON", 0), tok(1, "held", "hold", "VERB", 4), tok(2, "a", "a", "DET", 9),
    tok(3, "test", "test", "NOUN", 11), tok(4, "tube", "tube", "NOUN", 16, ""), tok(5, ".", ".", "PUNCT", 20, ""),
], [])
HOT_DOGS = LightDoc([
    tok(0, "The", "the", "DET", 0), tok(1, "hot", "hot", "ADJ", 4), tok(2, "dogs", "dog", "NOUN", 8),
    tok(3, "barked", "bark", "VERB", 13, ""), tok(4, ".", ".", "PUNCT", 19, ""),
], [])


@pytest.fixture
def fake_model(monkeypatch):
    calls = []

    def batch(inputs):
        words = [i.marked_sentence.split("*")[1] for i in inputs]
        calls.append(list(zip(words, [i.marked_sentence for i in inputs], strict=True)))
        return [
            wsd.DisambiguationResult("", NONE_OF_THE_ABOVE, 0.9) if word == "hot dogs"  # compound reading rejected
            else wsd.DisambiguationResult(i.definitions[0].synset_id, i.definitions[0].definition, 0.8)
            for word, i in zip(words, inputs, strict=True)
        ]

    monkeypatch.setattr(wsd, "disambiguate_word_batch", batch)
    monkeypatch.setattr(wsd, "get_definitions", lambda queries: [DEFS.get((q.form, q.pos), []) for q in queries])
    monkeypatch.setattr("wsd.multiword._index", lambda: INDEX)
    return calls


def test_find_spans_on_lemmas(fake_model):
    assert find_spans(HOT_DOGS) == [MultiwordSpan(start=1, end=3, form="hot dog", head=2)]


def test_verb_spans_need_a_particle(fake_model):
    particle = LightDoc([tok(0, "gave", "give", "VERB", 0), tok(1, "up", "up", "ADP", 5, "", "prt", 0)], [])
    # "gave up the hill": "up" heads its own object, so "give up" is not the reading
    preposition = LightDoc([
        tok(0, "gave", "give", "VERB", 0), tok(1, "up", "up", "ADP", 5, " ", "prep", 0),
        tok(2, "the", "the", "DET", 8, " ", "det", 3), tok(3, "hill", "hill", "NOUN", 12, "", "pobj", 1),
    ], [])
    assert find_spans(particle) == [MultiwordSpan(start=0, end=2, form="give up", head=0)]
    assert find_spans(preposition) == []


def test_span_first_then_word_fallback(fake_model):
    tubes, dogs = disambiguate_docs([TEST_TUBE, HOT_DOGS])
    # phase 1: expressions and the words outside them, one batch; phase 2: only the rejected expression's words
    assert sorted(w for w, _ in fake_model[0]) == ["barked", "held", "hot dogs", "test tube"]
    assert dict(fake_model[0])["test tube"] == "She held a *test tube*."
    assert [w for w, _ in fake_model[1]] == ["hot", "dogs"]
    assert tubes.synsets == [
        wsd.Synset("hold-1", 1, 1, DEFS["hold", "v"][0].definition, 0.8),
        wsd.Synset("test_tube", 3, 4, DEFS["test tube", "n"][0].definition, 0.8, "test tube"),
    ]
    assert [(s.start_token, s.end_token, s.expression, s.id) for s in dogs.synsets] == [
        (1, 1, None, "hot"), (2, 2, None, "dog"), (3, 3, None, "bark"),
    ]


def test_skip_single_sense_covers_expressions(fake_model):
    result, = disambiguate_docs([TEST_TUBE], skip_single_sense=True)
    assert [w for w, _ in fake_model[0]] == ["held"]  # the only-candidate expression is assigned without a prompt
    assert [(s.start_token, s.end_token, s.confidence, s.expression) for s in result.synsets] == [
        (1, 1, 0.8, None), (3, 4, 1.0, "test tube"),
    ]


def test_rejected_words_and_missing_definitions_have_no_synsets(fake_model, monkeypatch):
    monkeypatch.setattr(wsd, "disambiguate_word_batch", lambda inputs: [
        wsd.DisambiguationResult("", NONE_OF_THE_ABOVE, 0.9) for _ in inputs
    ])
    result, = disambiguate_docs([TEST_TUBE])
    assert result.synsets == []
    assert [t.word for t in result.tokens] == ["She", "held", "a", "test", "tube", "."]


def test_repeated_synsets_keep_distinct_spans(fake_model):
    doc = LightDoc([tok(0, "dog", "dog", "NOUN", 0), tok(1, "dog", "dog", "NOUN", 4, "")], [])
    result, = disambiguate_docs([doc])
    assert [(s.id, s.start_token, s.end_token) for s in result.synsets] == [("dog", 0, 0), ("dog", 1, 1)]


def test_empty_doc(fake_model):
    result, = disambiguate_docs([LightDoc([], [])])
    assert result == wsd.WordSenseDisambiguation(tokens=[], entities=[], synsets=[])


@pytest.fixture
def president_doc(fake_model, monkeypatch):
    text = "Obama lived in Washington, D.C. while he was President of the United States."
    words = ["Obama", "lived", "in", "Washington", ",", "D.C.", "while", "he", "was",
             "President", "of", "the", "United", "States", "."]
    poses = ["PROPN", "VERB", "ADP", "PROPN", "PUNCT", "PROPN", "SCONJ", "PRON", "AUX",
             "PROPN", "ADP", "DET", "PROPN", "PROPN", "PUNCT"]
    tokens = []
    offset = 0
    for i, (word, pos) in enumerate(zip(words, poses, strict=True)):
        start = text.index(word, offset)
        offset = start + len(word)
        tokens.append(tok(i, word, word.lower(), pos, start, " " if text[offset:offset + 1] == " " else ""))
    form = "President of the United States"
    monkeypatch.setitem(INDEX, "president", {tuple(form.lower().split()): form})
    monkeypatch.setitem(DEFS, (form, "n"), [
        Definition("president", "the person who holds the office of head of state"),
    ])
    return text, LightDoc(tokens, [
        wsd.Entity("30", 12, 13, "United States", "country", "https://www.wikidata.org/wiki/Q30"),
    ])


def test_api_expression_span(president_doc, monkeypatch):
    from starlette.testclient import TestClient

    import wsd.server as server

    text, doc = president_doc
    monkeypatch.setattr(server, "disambiguate", lambda text, language: disambiguate_docs([doc])[0])
    response = TestClient(server.app).get("/disambiguate", params={"text": text, "lang": "en"})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"tokens", "entities", "synsets"}
    for token in result["tokens"]:
        assert set(token) == {"word", "lemma", "pos", "position", "start_char", "end_char"}
        assert text[token["start_char"]:token["end_char"]] == token["word"]
    assert result["synsets"] == [{
        "id": "president", "start_token": 9, "end_token": 13,
        "definition": "the person who holds the office of head of state", "confidence": 0.8,
        "expression": "President of the United States",
    }]
    # The entity inside the expression keeps its own independent span.
    assert (result["entities"][0]["start_token"], result["entities"][0]["end_token"]) == (12, 13)


def test_html_expression_span(president_doc, monkeypatch):
    from html.parser import HTMLParser

    from starlette.testclient import TestClient

    import wsd.server as server

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.rows.append([])
            elif tag == "td":
                self.rows[-1].append(int(dict(attrs).get("colspan", 1)))

    text, doc = president_doc
    monkeypatch.setattr(server, "disambiguate", lambda text, language: disambiguate_docs([doc])[0])
    response = TestClient(server.app).get("/disambiguate", params={"text": text, "lang": "en", "output": "html"})
    assert response.status_code == 200
    assert response.text.count("the person who holds the office of head of state") == 1
    parser = TableParser()
    parser.feed(response.text)
    assert [sum(row) for row in parser.rows] == [len(doc.tokens)] * 3
    assert parser.rows[1] == [1] * 9 + [5, 1]
    assert parser.rows[2] == [1] * 12 + [2, 1]
