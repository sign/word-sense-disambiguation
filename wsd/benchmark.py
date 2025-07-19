
import re
from dataclasses import dataclass

import wn
from tqdm import tqdm

from wsd.word_sense_disambiguation import disambiguate_word, get_definitions


@dataclass
class WordNetExample:
    synset_id: str
    word_form: str
    lemma: str
    pos: str
    marked_text: str

def collect_wordnet_examples():
    """List all English words and example sentences from WordNet"""

    # Get English WordNet
    en = wn.Wordnet(lang="en")

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

        for example in examples:
            for word in synset.words():
                for form in word.forms():
                    regex_form = r'\b' + re.escape(form) + r'\b'
                    if re.search(regex_form, example, re.IGNORECASE):
                        marked_text = re.sub(regex_form, r'*\g<0>*', example, flags=re.IGNORECASE)
                        yield WordNetExample(synset_id, form, word.lemma(), word.pos, marked_text)

if __name__ == "__main__":
    examples = list(tqdm(collect_wordnet_examples(), desc="Collecting WordNet Examples"))
    correct = 0

    with tqdm(examples, desc="Evaluating examples") as pbar:
        for example in pbar:
            definitions = get_definitions(example.lemma, example.pos)
            predicted_synset_id, _, _ = disambiguate_word(example.word_form, example.marked_text, definitions)
            if predicted_synset_id == example.synset_id:
                correct += 1
            accuracy = correct / (pbar.n + 1)
            pbar.set_description(f"Accuracy: {accuracy:.3f}")
            pbar.update()

