"""Offline coverage of the batch JSON writer and API schema parity."""
import os
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import wsd.word_sense_disambiguation as wsd


@pytest.fixture
def batch_module(monkeypatch):
    # Importing the CLI configures its worker environment and model timer.
    monkeypatch.setattr(wsd, "unmask_token_batch", wsd.unmask_token_batch)
    with patch.dict(os.environ):
        import wsd.batch as batch

        yield batch


@pytest.fixture
def result():
    return wsd.WordSenseDisambiguation(
        tokens=[
            wsd.DisambiguatedToken("test", "test", "NOUN", 0, 0, 4),
            wsd.DisambiguatedToken("tube", "tube", "NOUN", 1, 5, 9),
        ],
        entities=[wsd.Entity("1", 0, 1, "test tube", "example", "https://example.com")],
        synsets=[wsd.Synset("test_tube", 0, 1, "a laboratory tube", 0.8, "test tube")],
    )


def test_serialization_matches_api(batch_module, result):
    assert batch_module._to_dict(result) == asdict(result)


def test_process_file_writes_spans_and_counts_accepted_synsets(batch_module, result, monkeypatch, tmp_path):
    import json

    rejected = wsd.WordSenseDisambiguation(tokens=result.tokens, entities=[])
    prepared = SimpleNamespace(batch=[])
    pool = SimpleNamespace(batches=lambda: iter([(["test tube", "test tube"], prepared, 0.0)]))
    monkeypatch.setattr(batch_module, "disambiguate_word_batch", lambda batch: [])
    monkeypatch.setattr(batch_module, "complete_docs", lambda *args: [result, rejected])
    output = tmp_path / "corpus.jsonl"
    logs = []

    counts = batch_module.process_file(tmp_path / "corpus.txt", output, pool, False, logs.append)

    assert counts == (2, 1)  # One expression span; words with no accepted sense contribute zero.
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"text": "test tube", **asdict(result)},
        {"text": "test tube", **asdict(rejected)},
    ]
    assert not output.with_suffix(".jsonl.tmp").exists()
    assert "2 sentences, 1 synsets" in logs[0]
    assert "prompts" not in logs[0]
