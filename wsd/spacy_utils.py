"""spaCy pipeline management: one lazily loaded English pipeline, on the GPU when cupy is installed
(the cluster batch image), on the CPU otherwise (the serving image)."""
import logging
import threading

import spacy
from thinc.api import use_ops

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_trf"
# (pipeline, thinc backend it was built on: "numpy" or "cupy"). Every call must scope
# `use_ops(backend)`: input tensors are created on the backend active in the *calling*
# context, which has to match the device the pipeline's weights live on.
_pipeline: tuple[spacy.language.Language, str] | None = None
_lock = threading.Lock()


def _get_pipeline(language: str) -> tuple[spacy.language.Language, str]:
    global _pipeline
    if language != "en":
        raise ValueError(f"Language '{language}' not supported")
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                gpu = spacy.prefer_gpu()
                logger.info("spaCy GPU activated: %s", gpu)
                _pipeline = (spacy.load(SPACY_MODEL), "cupy" if gpu else "numpy")
    return _pipeline


def run_spacy_pipeline(text: str, language: str = "en"):
    """Run the pipeline on a text, on the backend the pipeline was built for."""
    nlp, backend = _get_pipeline(language)
    with use_ops(backend):
        return nlp(text)


def run_spacy_pipe(texts: list[str], language: str = "en", batch_size: int = 256, entities: bool = True) -> list:
    """Run the pipeline over many texts at once (``nlp.pipe``); ``entities=False``
    skips NER (and therefore entity linking, which works on the NER spans)."""
    nlp, backend = _get_pipeline(language)
    with use_ops(backend), nlp.select_pipes(disable=[] if entities else ["ner"]):
        return list(nlp.pipe(texts, batch_size=batch_size))
