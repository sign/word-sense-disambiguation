import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from wsd.env import WORDNET_URL

OLLAMA_URL = "http://localhost:11434/api/chat"

# Fetch all forms from WordNet
print("Fetching forms from WordNet...")
forms_response = requests.get(f"{WORDNET_URL}/lexicons/omw-en:1.4/forms")
forms_response.raise_for_status()
forms_data = forms_response.json()
forms = forms_data["data"]

# Create generated directory if it doesn't exist
generated_dir = Path(__file__).parent / "generated"
generated_dir.mkdir(exist_ok=True)

skip_file = Path(__file__).parent / "skip_forms.txt"
if skip_file.exists():
    with open(skip_file) as f:
        skip_forms = {line.strip() for line in f if line.strip()}
    forms = [form for form in forms if form not in skip_forms]
    print(f"Skipping {len(skip_forms)} forms from skip_forms.txt")


def _fetch_synsets(form: str) -> list[dict]:
    """Fetch synsets for a form from WordNet API"""
    url = f"{WORDNET_URL}/lexicons/omw-en:1.4/words?form={form}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    all_synsets = []
    for pos in data["data"]:
        synsets = [row for row in pos["included"] if row["type"] == "synset"]
        for synset in synsets:
            del synset["type"]
            del synset["links"]
        all_synsets.extend(synsets)

    return all_synsets


def _create_ollama_payload(form: str, all_synsets: list[dict]) -> dict:
    """Create payload for Ollama API request"""
    return {
        "model": "gpt-oss:120b",
        "stream": True,
        "options": {
            "temperature": 0
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a lexicographer working with WordNet-style synsets.\n"
                    "You receive a JSON array of synset objects. Each object has:\n"
                    "- id: synset ID\n"
                    "- attributes.pos: part of speech\n"
                    "- attributes.definition: original definition\n"
                    "- attributes.examples: example sentences\n\n"
                    "For each input synset, you MUST output exactly one object with:\n"
                    "- id: same as input\n"
                    "- pos: same as attributes.pos\n"
                    "- source_definition: same as attributes.definition\n"
                    "- alternative_definition: a clear, modern paraphrase of the same sense\n"
                    "- examples: EXACTLY 3 natural example sentences illustrating that sense. "
                    f"each example sentence must contain the word form '{form}' (case sensitive).\n\n"
                    "You may use a <think>...</think> block for your internal reasoning, "
                    "but AFTER that you MUST output ONLY valid JSON that matches the schema. "
                    "Do NOT wrap the JSON in markdown or any extra text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(all_synsets),
            },
        ],
        "format": {
            "type": "array",
            "minItems": len(all_synsets),
            "maxItems": len(all_synsets),
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "pos": {"type": "string"},
                    "source_definition": {"type": "string"},
                    "alternative_definition": {"type": "string"},
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                "required": [
                    "id",
                    "pos",
                    "source_definition",
                    "alternative_definition",
                    "examples",
                ],
            },
        },
    }


def _ollama_response(payload: dict) -> str:
    """Complete message content from the Ollama API (no streaming: nothing consumes increments)."""
    try:
        resp = requests.post(OLLAMA_URL, json={**payload, "stream": False}, timeout=3600)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"\nError: Failed to connect to Ollama endpoint at {OLLAMA_URL}\nDetails: {e}")
        print("\nPlease ensure Ollama is running and accessible.")
        exit(1)
    return resp.json()["message"]["content"]


def process_form(form: str) -> tuple[str, bool, str | None]:
    """Process a single form and return (form, success, error_message)"""
    output_file = generated_dir / f"{form}.json"

    # Skip if file already exists
    if output_file.exists():
        return (form, True, "skipped - file exists")

    try:
        # Fetch synsets from WordNet API
        all_synsets = _fetch_synsets(form)

        # Skip if only one synset
        if len(all_synsets) == 1:
            with open(skip_file, "a") as f:
                f.write(f"{form}\n")
            return (form, True, "skipped - only one synset")

        # Skip if no synsets found
        if not all_synsets:
            return (form, True, "skipped - no synsets found")

        # Create payload and stream response from Ollama
        payload = _create_ollama_payload(form, all_synsets)
        parsed = json.loads(_ollama_response(payload))

        # Save to file
        with open(output_file, "w") as f:
            json.dump(parsed, f, indent=2)

    except (requests.RequestException, json.JSONDecodeError, OSError) as e:
        return (form, False, str(e))
    else:
        return (form, True, None)

# Process forms in parallel with 4 workers
print(f"Processing {len(forms)} forms with 4 parallel requests...")
with ThreadPoolExecutor(max_workers=4) as executor:
    for form, success, message in tqdm(executor.map(process_form, forms), total=len(forms), desc="Processing forms"):
        if not success:
            tqdm.write(f"Error processing {form}: {message}")
        elif message and message != "skipped - file exists":
            tqdm.write(f"{form}: {message}")

print(f"\nCompleted! Results saved to {generated_dir}/")
