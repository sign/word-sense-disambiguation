import socket
import subprocess
import time
from contextlib import closing
from urllib.parse import urlencode

import pytest


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_port(host, port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


class ServerProcess:
    def __init__(self):
        self.host = "localhost"
        self.port = find_free_port()
        self.process = None

    def start(self):
        self.process = subprocess.Popen([
            "uvicorn", "wsd.server:app",
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "warning"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not wait_for_port(self.host, self.port, timeout=10):
            stdout, stderr = self.process.communicate(timeout=2)
            print(f"[server stdout]\n{stdout.decode()}")
            print(f"[server stderr]\n{stderr.decode()}")
            self.process.kill()
            raise RuntimeError(f"Server failed to start on {self.host}:{self.port}")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()


@pytest.fixture(scope="session")
def server():
    server_proc = ServerProcess()
    server_proc.start()
    yield server_proc
    server_proc.stop()


def test_e2e_index_endpoint(server):
    """Test index endpoint"""
    import requests

    base_url = f"http://{server.host}:{server.port}"

    response = requests.get(f"{base_url}/")
    assert response.status_code == 200

    result = response.json()
    assert "endpoints" in result
    assert isinstance(result["endpoints"], dict)


def test_e2e_health_endpoint(server):
    """Test health check endpoint"""
    from datetime import UTC, datetime

    import requests

    base_url = f"http://{server.host}:{server.port}"

    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200

    result = response.json()
    assert result["status"] == "healthy"
    assert "timestamp" in result
    assert "service" in result

    # Validate timestamp is recent (within last 10 seconds)
    response_timestamp = datetime.fromisoformat(result["timestamp"])
    now = datetime.now(tz=UTC)
    time_diff = abs((now - response_timestamp).total_seconds())
    assert time_diff < 10, f"Timestamp {result['timestamp']} is not recent (diff: {time_diff}s)"


def test_e2e_disambiguate_endpoint(server):
    """End-to-end test that spins up server and calls disambiguate endpoint"""
    import requests

    base_url = f"http://{server.host}:{server.port}"

    # Test parameters
    text = "Apple is a technology company"
    lang = "en"

    # Build URL with query parameters
    params = {"text": text, "lang": lang}
    url = f"{base_url}/disambiguate?" + urlencode(params)

    # Make request
    response = requests.get(url)

    # Validate response
    assert response.status_code == 200

    result = response.json()

    # Validate response structure
    assert "tokens" in result
    assert "entities" in result
    assert isinstance(result["tokens"], list)
    assert isinstance(result["entities"], list)

    # Validate that we have tokens and entities
    assert len(result["tokens"]) > 0
    assert len(result["entities"]) > 0

    # Validate last token "company" which should have synset information
    company_token = result["tokens"][3].copy()
    company_token.pop("confidence", None)  # Remove confidence for comparison

    expected_company_token = {
        'end_char': 21,
        'lemma': 'technology',
        'pos': 'NOUN',
        'position': 3,
        'start_char': 11,
        'synset_definition': 'the discipline dealing with the art or science of '
                             'applying scientific knowledge to practical problems',
        'synset_id': 'omw-en-06125041-n',
        'word': 'technology'
    }
    assert company_token == expected_company_token

    # Validate first entity exactly
    expected_first_entity = {
        "id": 312,
        "start_token": 0,
        "end_token": 0,
        "text": "Apple Inc.",
        "description": "American producer of hardware, software, and services, based in Cupertino, California",
        "url": "https://www.wikidata.org/wiki/Q312"
    }
    assert result["entities"][0] == expected_first_entity


def test_e2e_disambiguate_missing_params(server):
    """Test error handling for missing parameters"""
    import requests

    base_url = f"http://{server.host}:{server.port}"

    # Test missing text parameter
    response = requests.get(f"{base_url}/disambiguate?lang=en")
    assert response.status_code == 400
    error_data = response.json()
    assert "Missing 'text' query parameter" in error_data["error"]["message"]

    # Test missing lang parameter
    response = requests.get(f"{base_url}/disambiguate?text=test")
    assert response.status_code == 400
    error_data = response.json()
    assert "Missing 'lang' query parameter" in error_data["error"]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
