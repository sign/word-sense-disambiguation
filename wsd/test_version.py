"""Version attestation does not change WSD output or run inference on health probes."""

from dataclasses import asdict

import pytest
from starlette.testclient import TestClient

import wsd.server as server
from wsd.word_sense_disambiguation import WordSenseDisambiguation


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WSD_WARMUP", "0")
    monkeypatch.setattr(server, "MODEL_VERSION", "wsd-build-123-cpu")
    with TestClient(server.app) as client:
        yield client


def test_health_attests_version_without_inference(client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Health must not run disambiguation")

    monkeypatch.setattr(server, "disambiguate", forbidden)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == response.headers["X-Model-Tag"] == "wsd-build-123-cpu"


@pytest.mark.parametrize("output", ["json", "html"])
def test_disambiguation_attests_same_version_as_health(client, monkeypatch, output):
    result = WordSenseDisambiguation(tokens=[], entities=[], synsets=[])
    monkeypatch.setattr(server, "disambiguate", lambda **kwargs: result)
    response = client.get("/disambiguate", params={"text": "", "lang": "en", "output": output})
    assert response.status_code == 200
    assert response.headers["X-Model-Tag"] == client.get("/health").json()["version"]
    if output == "json":
        assert response.json() == asdict(result)
    else:
        assert "text/html" in response.headers["content-type"]


def test_version_is_exposed_to_browser_clients(client):
    response = client.get("/health", headers={"Origin": "https://example.com"})
    assert "X-Model-Tag" in response.headers["Access-Control-Expose-Headers"]


def test_request_errors_keep_version_header(client):
    response = client.get("/disambiguate")
    assert response.status_code == 400
    assert response.headers["X-Model-Tag"] == "wsd-build-123-cpu"


@pytest.mark.parametrize("version", [None, ""])
def test_unversioned_development_does_not_attest_cache_identity(client, monkeypatch, version):
    monkeypatch.setattr(server, "MODEL_VERSION", version)
    response = client.get("/health")
    assert not response.json()["version"]
    assert "X-Model-Tag" not in response.headers
