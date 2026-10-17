"""Hyphenated lexical forms must be candidates, not forced idiom selections."""
from types import SimpleNamespace

import pytest

from wsd import multiword
from wsd import word_sense_disambiguation as wsd
from wsd.prompt import NONE_OF_THE_ABOVE, Definition
from wsd.word_sense_disambiguation import LightDoc, LightToken


@pytest.fixture
def expression_index(monkeypatch):
    monkeypatch.setattr(multiword.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {"data": ["off-the-shelf", "pull off", "test tube"]}))
    multiword._index.cache_clear()
    yield multiword._index()
    multiword._index.cache_clear()


def document(words):
    tokens, offset = [], 0
    for i, (word, pos, dep, head) in enumerate(words):
        tokens.append(LightToken(word, word.lower(), pos, i, offset, pos == "PUNCT", False, " ", dep, head))
        offset += len(word) + 1
    return LightDoc(tokens, [])


def pull_off_shelf():
    # The production parse that previously committed to the shorter PULL OFF.
    return document([("pull", "VERB", "ROOT", 0), ("off", "ADP", "prt", 0),
                     ("the", "DET", "det", 3), ("shelf", "NOUN", "dobj", 0)])


def test_index_includes_spaced_and_spacy_dash_token_spellings(expression_index):
    assert expression_index["off"][("off", "the", "shelf")] == "off-the-shelf"
    assert expression_index["off"][("off", "-", "the", "-", "shelf")] == "off-the-shelf"
    assert expression_index["test"][("test", "tube")] == "test tube"


def test_longer_expression_wins_over_an_earlier_shorter_overlap(expression_index):
    spans = multiword.find_spans(pull_off_shelf())
    assert spans == [multiword.MultiwordSpan(1, 4, "off-the-shelf", 3)]


def test_hyphen_tokens_preserve_original_coordinates(expression_index):
    doc = document([("off", "ADP", "prep", 4), ("-", "PUNCT", "punct", 4),
                    ("the", "DET", "det", 4), ("-", "PUNCT", "punct", 4),
                    ("shelf", "NOUN", "ROOT", 4)])
    assert multiword.find_spans(doc) == [multiword.MultiwordSpan(0, 5, "off-the-shelf", 4)]


def test_no_phrase_match_across_unrelated_punctuation(expression_index):
    doc = document([("off", "ADP", "ROOT", 0), (",", "PUNCT", "punct", 0),
                    ("the", "DET", "det", 3), ("shelf", "NOUN", "conj", 0)])
    assert multiword.find_spans(doc) == []


@pytest.mark.parametrize("accept", [True, False])
def test_adjective_candidates_reach_model_and_can_still_be_rejected(expression_index, monkeypatch, accept):
    compound = Definition("omw-en-00672932-s", "made in standard sizes and available from merchandise in stock")
    shelf = Definition("shelf", "a horizontal surface for holding objects")
    lookup = {("off-the-shelf", "a"): [compound], ("shelf", "n"): [shelf]}
    monkeypatch.setattr(wsd, "get_definitions", lambda queries: [lookup.get((q.form, q.pos), []) for q in queries])
    prompts = []

    def model(inputs):
        prompts.extend(inputs)
        return [wsd.DisambiguationResult(
            "" if i.definitions == [compound] and not accept else i.definitions[0].synset_id,
            NONE_OF_THE_ABOVE if i.definitions == [compound] and not accept else i.definitions[0].definition,
            0.9,
        ) for i in inputs]

    monkeypatch.setattr(wsd, "disambiguate_word_batch", model)
    result, = wsd.disambiguate_docs([pull_off_shelf()])
    assert prompts[0].marked_sentence == "pull *off the shelf* "
    assert prompts[0].definitions == [compound]
    if accept:
        assert [(s.id, s.start_token, s.end_token, s.expression) for s in result.synsets] == [
            (compound.synset_id, 1, 3, "off-the-shelf"), ("shelf", 3, 3, None)]
    else:
        assert [(s.id, s.start_token, s.end_token, s.expression) for s in result.synsets] == [
            ("shelf", 3, 3, None)]


def test_existing_pos_definitions_are_not_broadened(monkeypatch):
    noun = Definition("test-tube", "laboratory container")
    queries_seen = []

    def definitions(queries):
        queries_seen.extend(queries)
        return [[noun] for _ in queries]

    monkeypatch.setattr(wsd, "get_definitions", definitions)
    assert wsd._span_definitions([wsd.WordQuery("test tube", "n")]) == [[noun]]
    assert queries_seen == [wsd.WordQuery("test tube", "n")]
