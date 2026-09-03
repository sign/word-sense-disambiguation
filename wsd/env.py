import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


class MissingEnvironmentVariableError(Exception):
    """Raised when a required environment variable is missing."""

    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        super().__init__(f"Missing required environment variable: {variable_name}")


# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def start_local_wordnet_api(timeout: float = 120.0) -> str:
    """Start the WordNet API (``wn.web`` from https://github.com/sign/wn, the same
    server as ``ghcr.io/sign/wn``) in a child process on a free port and return its URL.

    Meant for batch/training/benchmark jobs on cluster images that bundle the
    API; the process is stopped at exit. Raises ImportError if ``wn.web`` is not
    installed.
    """
    import wn.web  # noqa: F401 - probe availability before spawning

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "wn.web:app", "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "warning",
    ])
    atexit.register(proc.terminate)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2):
                return url
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(0.5)
    raise RuntimeError("local WordNet API did not become healthy")  # noqa: TRY003


# URL of the WordNet API server (ghcr.io/sign/wn). When unset and the API package
# is installed (cluster images), a local instance is started for this process.
WORDNET_URL = os.environ.get("WORDNET_URL")
if not WORDNET_URL:
    try:
        WORDNET_URL = start_local_wordnet_api()
    except ImportError:
        raise MissingEnvironmentVariableError("WORDNET_URL") from None
    os.environ["WORDNET_URL"] = WORDNET_URL


def detach_from_torchrun() -> tuple[int, int]:
    """Turn a torchrun worker into a plain single-GPU process; returns ``(rank, world_size)``.

    Call before importing torch/transformers/cupy. Pins ``CUDA_VISIBLE_DEVICES``
    to this rank's GPU and drops the rendezvous variables so libraries that read
    ``LOCAL_RANK`` (accelerate, transformers' loader) don't reach for other GPUs.
    Used by scripts that shard work across GPUs without any collective ops.
    """
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if "LOCAL_RANK" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(var, None)
    return rank, world
