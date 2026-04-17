#!/usr/bin/env python3

import requests

OLLAMA_BASE = "http://localhost:11434"

print("Checking Ollama setup...\n")

# Check if Ollama is running
try:
    response = requests.get(f"{OLLAMA_BASE}/api/tags")
    response.raise_for_status()
    models = response.json()

    print("✓ Ollama is running")
    print("\nAvailable models:")
    for model in models.get("models", []):
        print(f"  - {model['name']}")

    # Check if the specific model exists
    model_name = "gpt-oss:120b"
    model_exists = any(m["name"] == model_name for m in models.get("models", []))

    if model_exists:
        print(f"\n✓ Model '{model_name}' is available")

        # Test the chat endpoint (only meaningful if the model is actually present)
        print("\nTesting chat endpoint...")
        test_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "test"}],
            "stream": False
        }

        chat_response = requests.post(f"{OLLAMA_BASE}/api/chat", json=test_payload)
        if chat_response.status_code == 200:
            print("✓ Chat endpoint is working")
        else:
            print(f"✗ Chat endpoint returned status {chat_response.status_code}")
            print(f"  Response: {chat_response.text}")
    else:
        print(f"\n✗ Model '{model_name}' NOT found")
        print("  You need to pull this model first:")
        print(f"  ollama pull {model_name}")

except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to Ollama")
    print("  Make sure Ollama is running: ollama serve")
except requests.RequestException as e:
    print(f"✗ Error: {e}")
