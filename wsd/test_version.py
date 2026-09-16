"""The gateway probes health and learns the same version from WSD responses."""

from dataclasses import asdict
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

import wsd.server as server
from wsd.word_sense_disambiguation import WordSenseDisambiguation


@pytest.mark.parametrize("version", ["build-123-cpu", ""])
def test_version_contract(monkeypatch, version):
    monkeypatch.setenv("WSD_WARMUP", "0")
    monkeypatch.setattr(server, "MODEL_VERSION", version)
    result = WordSenseDisambiguation(tokens=[], entities=[], synsets=[])
    disambiguate = Mock(return_value=result)
    monkeypatch.setattr(server, "disambiguate", disambiguate)
    with TestClient(server.app) as client:
        health = client.get("/health", headers={"Origin": "https://example.com"})
        assert health.status_code == 200
        assert health.json()["version"] == health.headers["X-Model-Tag"] == version
        assert "X-Model-Tag" in health.headers["Access-Control-Expose-Headers"]
        disambiguate.assert_not_called()
        for output in ("json", "html"):
            response = client.get("/disambiguate", params={"text": "", "lang": "en", "output": output})
            assert response.status_code == 200
            assert response.headers["X-Model-Tag"] == version
            if output == "json":
                assert response.json() == asdict(result)
