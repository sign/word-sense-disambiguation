import itertools
import re
from dataclasses import dataclass

import wn
from tqdm import tqdm

from wsd.word_sense_disambiguation import (
    WordQuery,
    DisambiguationInput,
    get_definitions,
    disambiguate_word_batch,
)


@dataclass
class WordNetExample:
    synset_id: str
    word_form: str
    lemma: str
    pos: str
    marked_text: str

def collect_wordnet_examples():
    """List all English words and example sentences from WordNet"""

    # Ensure omw-en:1.4 is downloaded
    try:
        en = wn.Wordnet(lexicon="omw-en:1.4")
    except wn.Error:
        print("Downloading omw-en:1.4...")
        wn.download("omw-en:1.4")
        en = wn.Wordnet(lexicon="omw-en:1.4")

    # Iterate through all synsets
    for synset in en.synsets():
        synset_id = synset.id
        word_synsets = sum(len(word.synsets()) for word in synset.words())
        if word_synsets == 1:
            # Trivial case, only one possible meaning
            continue

        examples = synset.examples()
        if len(examples) == 0:
            continue

        for word in synset.words():
            for form in word.forms():
                for example in examples:
                    regex_form = r'\b' + re.escape(form) + r'\b'
                    if re.search(regex_form, example, re.IGNORECASE):
                        marked_text = re.sub(regex_form, r'*\g<0>*', example, flags=re.IGNORECASE)
                        yield WordNetExample(synset_id, form, word.lemma(), word.pos, marked_text)

if __name__ == "__main__":
    examples = collect_wordnet_examples()
    examples = list(tqdm(examples, desc="Collecting examples"))
    correct = 0
    batch_size = 64

    # Process examples in batches
    num_batches = (len(examples) + batch_size - 1) // batch_size

    with tqdm(total=len(examples), desc="Evaluating examples") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(examples))
            batch = examples[start_idx:end_idx]

            # Fetch definitions for all examples in batch using the batch endpoint
            queries = [WordQuery(form=example.lemma, pos=example.pos) for example in batch]
            all_definitions = get_definitions(queries)

            # Prepare batch data with fetched definitions
            batch_data = [
                DisambiguationInput(
                    word=example.word_form,
                    marked_sentence=example.marked_text,
                    definitions=definitions
                )
                for example, definitions in zip(batch, all_definitions)
            ]

            # Process entire batch at once
            predictions = disambiguate_word_batch(batch_data)

            # Check predictions and update accuracy
            for i, (example, result) in enumerate(zip(batch, predictions)):
                is_correct = result.synset_id == example.synset_id
                if is_correct:
                    correct += 1

                # Debug: print first few mismatches
                if not is_correct and (start_idx + i) < 10:
                    print(f"\nDEBUG Mismatch #{start_idx + i}:")
                    print(f"  Word: {example.word_form} (lemma: {example.lemma}, pos: {example.pos})")
                    print(f"  Expected: {example.synset_id}")
                    print(f"  Predicted: {result.synset_id} ({result.definition}, {result.confidence})")
                    print(f"  Sentence: {example.marked_text}")

            # Update progress bar
            accuracy = correct / end_idx
            pbar.set_description(f"Accuracy: {accuracy:.3f}")
            pbar.update(len(batch))

    print(f"Accuracy: {accuracy:.3f}")