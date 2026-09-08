"""File to be executed in order to make sure the docker image is primed with all necessary resources."""
import os

from wsd.masked_language_model import load_model

# Make sure it does not interact with the wordnet
os.environ["WORDNET_URL"] = "NONE"

from wsd.spacy_utils import run_spacy_pipeline
from wsd.word_sense_disambiguation import DisambiguationInput, _extract_entities, disambiguate_word_batch

# Download spaCy entities knowledge base (600MB~)
print("Priming spaCy model...")
doc = run_spacy_pipeline("Apple is a technology company.")
assert [e.id for e in _extract_entities(doc)] == [312], "entity linking is broken"  # Apple Inc.

# Download HuggingFace Language model
print("Priming WSD model...")
load_model()
disambiguate_word_batch([DisambiguationInput("Apple is a technology company.", [])])
