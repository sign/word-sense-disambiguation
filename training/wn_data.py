"""WordNet example loading for training-time eval and final benchmark.

Provides a single deterministic split between the eval subset (used during
training) and the benchmark subset (used for final evaluation). Training and
benchmark MUST use the same seed so the two disjoint sets never overlap.
"""
import random
from dataclasses import dataclass

import wn

from wsd.prompt import WordNotFoundError, mark_word_in_sentence


@dataclass
class WnExample:
    """A single WordNet-derived disambiguation example."""
    synset_id: str
    word_form: str  # surface form shown in the sentence
    lemma: str
    pos: str
    marked_text: str  # sentence with *word* markers
    sentence: str  # original, unmarked


def _ensure_lexicon() -> wn.Wordnet:
    try:
        return wn.Wordnet(lexicon="omw-en:1.4")
    except wn.Error:
        wn.download("omw-en:1.4")
        return wn.Wordnet(lexicon="omw-en:1.4")


def collect_all(seed: int = 42) -> list[WnExample]:
    """Enumerate every usable (synset, word form, example) combination.

    Filters out monosemous synsets (nothing to disambiguate) and sentences
    where the form can't be marked with clean word boundaries. Returns a
    deterministically shuffled list.
    """
    en = _ensure_lexicon()
    out: list[WnExample] = []
    for synset in en.synsets():
        word_synsets = sum(len(word.synsets()) for word in synset.words())
        if word_synsets == 1:
            continue
        examples = synset.examples()
        if not examples:
            continue
        for word in synset.words():
            for form in word.forms():
                for example in examples:
                    try:
                        marked = mark_word_in_sentence(example, form)
                    except WordNotFoundError:
                        continue
                    out.append(
                        WnExample(
                            synset_id=synset.id,
                            word_form=form,
                            lemma=word.lemma(),
                            pos=word.pos,
                            marked_text=marked,
                            sentence=example,
                        )
                    )
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def split(n_eval: int = 1000, seed: int = 42) -> tuple[list[WnExample], list[WnExample]]:
    """Return (eval_examples, benchmark_examples) with no overlap."""
    all_examples = collect_all(seed=seed)
    return all_examples[:n_eval], all_examples[n_eval:]
