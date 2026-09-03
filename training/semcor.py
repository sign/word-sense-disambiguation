"""Sense-annotated corpora in Raganato et al. (2017) XML format, as WordNetExamples.

Covers SemCor (the standard WSD training corpus, ~226k instances) and the
unified ``ALL`` evaluation set (Senseval-2/3, SemEval-07/13/15), from
http://lcl.uniroma1.it/wsdeval/ . Sense keys are mapped to omw-en:1.4 synset ids
through WordNet 3.0's ``index.sense`` (omw-en:1.4 *is* WordNet 3.0, so
``omw-en-<offset>-<pos>`` ids line up).

    python -m training.semcor /path/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor \
        /path/dict/index.sense
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from wsd.benchmark import WordNetExample

_SS_TYPE_TO_POS = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "s"}
# Raganato XML uses Penn-style coarse tags; WordNet lookups take word-level pos.
_XML_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


_DETOK = [
    (re.compile(r" ([,.;:!?%)\]}])"), r"\1"),      # no space before closing punctuation
    (re.compile(r"([(\[{$]) "), r"\1"),            # no space after opening punctuation
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),  # clitics
    (re.compile(r"`` ?"), '"'), (re.compile(r" ?''"), '"'),  # PTB quotes
    (re.compile(r" - "), "-"),
]


def detokenize(tokens: list[str]) -> str:
    """Join Penn-Treebank-style tokens back into natural text (SemCor / SemEval
    are pre-tokenized: ``the dog 's bowl , and`` -> ``the dog's bowl, and``)."""
    text = " ".join(tokens)
    for pattern, repl in _DETOK:
        text = pattern.sub(repl, text)
    return text


def load_sense_index(index_sense: Path) -> dict[str, str]:
    """``sense_key -> omw-en synset id`` from WordNet 3.0's index.sense."""
    mapping = {}
    with open(index_sense) as f:
        for line in f:
            key, offset, *_ = line.split()
            ss_type = key.split("%")[1].split(":")[0]
            mapping[key] = f"omw-en-{offset}-{_SS_TYPE_TO_POS[ss_type]}"
    return mapping


def _iter_sentences(path: Path):
    """Stream ``<sentence>`` elements. Handles SemCor+OMSTI (1.3 GB, two
    ``<corpus>`` roots in one file) by wrapping the stream in a synthetic root
    and freeing each sentence after it is yielded."""
    parser = ET.XMLPullParser(events=("end",))
    parser.feed("<root>")
    with open(path, "rb") as f:
        first = True
        while chunk := f.read(1 << 20):
            if first:
                chunk = re.sub(rb"<\?xml[^>]*\?>", b"", chunk)
                first = False
            parser.feed(chunk)
            for _, elem in parser.read_events():
                if elem.tag == "sentence":
                    yield elem
                    elem.clear()
    parser.feed("</root>")
    for _, elem in parser.read_events():
        if elem.tag == "sentence":
            yield elem
            elem.clear()


def load_raganato(prefix: Path, sense_index: dict[str, str]) -> list[WordNetExample]:
    """Load ``<prefix>.data.xml`` + ``<prefix>.gold.key.txt``.

    One example per annotated instance; the instance's own token is marked, so
    repeated words in a sentence are marked at the right position. Instances
    whose first gold key is unknown, or whose text already contains ``*``, are
    skipped.
    """
    gold: dict[str, list[str]] = {}
    with open(f"{prefix}.gold.key.txt") as f:
        for line in f:
            inst, *keys = line.split()
            gold[inst] = keys  # the first key is the canonical gold sense; all are accepted in scoring

    examples = []
    for sentence in _iter_sentences(Path(f"{prefix}.data.xml")):
        tokens = [(el.text or "", el) for el in sentence]
        words = [t for t, _ in tokens]
        text = detokenize(words)
        if "*" in text:
            continue
        for i, (tok, el) in enumerate(tokens):
            if el.tag != "instance":
                continue
            gold_ids = [sid for key in gold.get(el.get("id"), []) if (sid := sense_index.get(key))]
            pos = _XML_POS.get(el.get("pos", ""))
            if not gold_ids or pos is None:
                continue
            synset_id = gold_ids[0]
            marked = detokenize([f"*{t}*" if j == i else t for j, t in enumerate(words)])
            examples.append(WordNetExample(
                synset_id=synset_id,
                word_form=tok,
                lemma=el.get("lemma", tok).replace("_", " "),
                pos=pos,
                marked_text=marked,
                sentence=text,
                gold_ids=tuple(gold_ids[1:]),
            ))
    return examples


if __name__ == "__main__":
    prefix, index_sense = Path(sys.argv[1]), Path(sys.argv[2])
    exs = load_raganato(prefix, load_sense_index(index_sense))
    from collections import Counter

    print(len(exs), "instances;", Counter(e.pos for e in exs))
    print(exs[0])
    assert detokenize("the dog 's bowl , and ( maybe ) `` more '' .".split()) == 'the dog\'s bowl, and (maybe) "more".'
    print("detokenize ok")
