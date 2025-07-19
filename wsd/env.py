import os
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


# Environment variables
WORDNET_URL = os.environ.get("WORDNET_URL")
if not WORDNET_URL:
    raise MissingEnvironmentVariableError("WORDNET_URL")
