"""Function-word candidates use the extended lexicon, not a content-POS allowlist."""

import pytest

import wsd.word_sense_disambiguation as wsd


def token(word, pos):
    return wsd.LightToken(word, word, pos, 0, 0, pos == "PUNCT", pos == "SPACE", "", "", 0)


@pytest.mark.parametrize(("word", "pos", "codes"), [
    ("we", "PRON", ["h"]),
    ("into", "ADP", ["p"]),
    ("and", "CCONJ", ["c"]),
    ("where", "SCONJ", ["c"]),
    ("the", "DET", ["d"]),
    ("can", "AUX", ["v"]),
    ("to", "PART", ["y", "r"]),
    ("wow", "INTJ", ["i", "n", "r"]),
])
def test_supported_pos_are_queried(word, pos, codes):
    t = token(word, pos)
    assert wsd._is_content(t)
    assert [(q.form, q.pos) for q in wsd._queries(t)] == [(word, code) for code in codes]


@pytest.mark.parametrize("pos", ["PUNCT", "SPACE", "SYM", "X", ""])
def test_nonlexical_pos_are_skipped(pos):
    assert not wsd._is_content(token("?", pos))


def test_no_general_cross_pos_fallback():
    assert wsd._queries(token("boreal", "NOUN")) == [wsd.WordQuery("boreal", "n")]


@pytest.mark.parametrize(("word", "pos", "code", "sense_id", "definition"), [
    ("into", "ADP", "p", "wikidata-en-L3042-S1", "to move inside of (something)"),
    ("and", "CCONJ", "c", "wikidata-en-L1385-S1", "joins words and sentence parts together"),
    ("to", "PART", "y", "wikidata-en-L2985-S1", "infinitive marker"),
    ("not", "PART", "r", "omw-en-00024073-r", "negation of a word or group of words"),
    ("hello", "INTJ", "n", "omw-en-06632511-n", "an expression of greeting"),
])
def test_function_word_senses_reach_spans(monkeypatch, word, pos, code, sense_id, definition):
    monkeypatch.setattr("wsd.multiword._index", lambda: {})
    monkeypatch.setattr(wsd, "get_definitions", lambda queries: [
        [wsd.Definition(sense_id, definition)] if (q.form, q.pos) == (word, code) else [] for q in queries
    ])
    calls = []

    def model(inputs):
        calls.extend(inputs)
        return [wsd.DisambiguationResult(sense_id, definition, 0.9) for _ in inputs]

    monkeypatch.setattr(wsd, "disambiguate_word_batch", model)
    result, = wsd.disambiguate_docs([wsd.LightDoc([token(word, pos)], [])])
    assert [i.marked_sentence for i in calls] == [f"*{word}*"]
    assert result.synsets == [wsd.Synset(sense_id, 0, 0, definition, 0.9)]
