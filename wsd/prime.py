"""File to be executed in order to make sure the docker image is primed with all necessary resources."""
import os

from wsd.masked_language_model import load_model

# Make sure it does not interact with the wordnet
os.environ["WORDNET_URL"] = "NONE"

from wsd.word_sense_disambiguation import disambiguate_word, get_spacy_pipeline

# Download spaCy entities knowledge base (600MB~)
print("Priming spaCy model...")
nlp = get_spacy_pipeline()
nlp("Apple is a technology company.")

# Download HuggingFace Language model
print("Priming WSD model...")
load_model()
disambiguate_word("Apple", "Apple is a technology company.", [])
