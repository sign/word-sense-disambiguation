from pathlib import Path
import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    with open(skip_file, "r") as f:
        skip_forms = set(line.strip() for line in f if line.strip())
    forms = [form for form in forms if form not in skip_forms]
    print(f"Skipping {len(skip_forms)} forms from skip_forms.txt")

def process_form(form: str) -> tuple[str, bool, str | None]:
    """Process a single form and return (form, success, error_message)"""
    output_file = generated_dir / f"{form}.json"

    # Skip if file already exists
    if output_file.exists():
        return (form, True, "skipped - file exists")

    try:
        # Prepare the request payload
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
        if len(all_synsets) == 1:
            with open(skip_file, "a") as f:
                f.write(f"{form}\n")
            return (form, True, "skipped - only one synset")

        if not all_synsets:
            return (form, True, "skipped - no synsets found")

        payload = {
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

        try:
            resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=3600)
            resp.raise_for_status()
        except Exception as e:
            print(f"\nError: Failed to connect to Ollama endpoint at {OLLAMA_URL}")
            print(f"Details: {e}")
            print("\nPlease ensure Ollama is running and accessible.")
            exit(1)

        content_chunks: list[str] = []

        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))

            msg = chunk.get("message") or {}
            content = msg.get("content")

            if content:
                content_chunks.append(content)

            if chunk.get("done"):
                break

        full_content = "".join(content_chunks)
        parsed = json.loads(full_content)

        # Save to file
        with open(output_file, "w") as f:
            json.dump(parsed, f, indent=2)

        return (form, True, None)

    except Exception as e:
        return (form, False, str(e))

# Process forms in parallel with 4 workers
print(f"Processing {len(forms)} forms with 4 parallel requests...")
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all tasks 
    future_to_form = {executor.submit(process_form, form): form for form in forms}

    # Process results as they complete with progress bar
    with tqdm(total=len(forms), desc="Processing forms") as pbar:
        for future in as_completed(future_to_form):
            form = future_to_form[future]
            try:
                form_result, success, message = future.result()
                if not success:
                    tqdm.write(f"Error processing {form}: {message}")
                elif message and message != "skipped - file exists":
                    tqdm.write(f"{form}: {message}")
            except Exception as e:
                tqdm.write(f"Exception processing {form}: {e}")
            pbar.update(1)

print(f"\nCompleted! Results saved to {generated_dir}/")