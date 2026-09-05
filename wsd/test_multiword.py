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
INDEX = {"test": {("test", "tube"): "test tube"}, "hot": {("hot", "dog"): "hot dog"}}


def tok(i, text, lemma, pos, idx, ws=" "):
    return LightToken(text, lemma, pos, i, idx, pos == "PUNCT", False, ws)


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
        calls.append([(i.word, i.marked_sentence) for i in inputs])
        return [
            wsd.DisambiguationResult("", NONE_OF_THE_ABOVE, 0.9) if i.word == "hot dogs"  # compound reading rejected
            else wsd.DisambiguationResult(i.definitions[0].synset_id, i.definitions[0].definition, 0.8)
            for i in inputs
        ]

    monkeypatch.setattr(wsd, "disambiguate_word_batch", batch)
    monkeypatch.setattr(wsd, "get_definitions", lambda queries: [DEFS.get((q.form, q.pos), []) for q in queries])
    monkeypatch.setattr("wsd.multiword._index", lambda language="en": INDEX)
    return calls


def test_find_spans_on_lemmas(fake_model):
    assert find_spans(HOT_DOGS) == [MultiwordSpan(start=1, end=3, form="hot dog", head=2)]


def test_span_first_then_word_fallback(fake_model):
    tubes, dogs = disambiguate_docs([TEST_TUBE, HOT_DOGS])
    # phase 1: expressions and the words outside them, one batch; phase 2: only the rejected expression's words
    assert [w for w, _ in fake_model[0]] == ["test tube", "held", "hot dogs", "barked"]
    assert fake_model[0][0][1] == "She held a *test tube*."
    assert [w for w, _ in fake_model[1]] == ["hot", "dogs"]
    assert [(t.expression, t.synset_id) for t in tubes.tokens[3:5]] == [("test tube", "test_tube")] * 2
    assert (tubes.tokens[1].synset_id, tubes.tokens[1].expression) == ("hold-1", None)
    assert [(t.expression, t.synset_id) for t in dogs.tokens[1:3]] == [(None, "hot"), (None, "dog")]


def test_skip_single_sense_covers_expressions(fake_model):
    result, = disambiguate_docs([TEST_TUBE], skip_single_sense=True)
    assert [w for w, _ in fake_model[0]] == ["held"]  # the only-candidate expression is assigned without a prompt
    assert (result.tokens[3].confidence, result.tokens[3].expression) == (1.0, "test tube")
