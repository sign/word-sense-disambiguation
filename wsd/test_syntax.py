"""The HTTP and multiprocessing paths expose the same language-neutral syntax."""
import pickle
from dataclasses import asdict

import spacy

from wsd import word_sense_disambiguation as wsd
from wsd.batch import _to_dict


def test_syntax_and_sentences_survive_batch(monkeypatch):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp("Dr. Smith is here. We meet next Tuesday.")
    monkeypatch.setattr("wsd.entities.link_named_entities", lambda doc: [])
    monkeypatch.setattr(wsd, "get_definitions", lambda queries: [[] for _ in queries])
    monkeypatch.setattr(wsd, "find_spans", lambda doc: [])
    light = pickle.loads(pickle.dumps(wsd.light_doc(doc)))
    direct, batch = wsd.disambiguate_docs([doc, light])
    assert direct == batch
    assert _to_dict(batch) == asdict(direct)
    assert direct.sentences == [wsd.Sentence(s.start, s.end - 1) for s in doc.sents]
    assert len(direct.sentences) == 2  # Dr. is not a sentence boundary.
    for output, original in zip(direct.tokens, doc, strict=True):
        assert output.dep == original.dep_
        assert output.head == original.head.i
        assert output.ent_type == original.ent_type_
    assert any(t.ent_type == "DATE" for t in direct.tokens)


def test_empty_and_unparsed_documents_have_no_boundaries():
    nlp = spacy.blank("en")
    assert wsd._sentences(nlp("")) == []
    assert wsd._sentences(nlp("No parser.")) == []
