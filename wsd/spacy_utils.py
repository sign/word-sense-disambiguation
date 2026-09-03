"""spaCy pipeline management: lazy loading, CPU/GPU backends, and the
serve-on-CPU-then-swap-to-GPU warmup used by the web server."""
import asyncio
import logging
import threading

import spacy
from thinc.api import use_ops

logger = logging.getLogger(__name__)

_PIPELINE_MODELS = {
    'en': 'en_core_web_trf',
    # Add more language models as needed
}
# language -> (pipeline, thinc backend it was built on: "numpy" or "cupy").
# Every call must scope `use_ops(backend)`: input tensors are created on the
# backend active in the *calling* context, which has to match the device the
# pipeline's weights live on, and context state does not survive across
# threads or asyncio tasks.
_pipelines: dict[str, tuple[spacy.language.Language, str]] = {}
_pipeline_lock = threading.Lock()


def _load_pipeline(language: str) -> spacy.language.Language:
    if language not in _PIPELINE_MODELS:
        msg = f"Language '{language}' not supported"
        raise ValueError(msg)
    nlp = spacy.load(_PIPELINE_MODELS[language])
    nlp.add_pipe("entityLinker", last=True)
    return nlp


def _get_pipeline_entry(language: str) -> tuple[spacy.language.Language, str]:
    entry = _pipelines.get(language)
    if entry is None:
        with _pipeline_lock:
            entry = _pipelines.get(language)
            if entry is None:
                gpu_activated = spacy.prefer_gpu()
                logger.info("spaCy GPU activated: %s", gpu_activated)
                entry = (_load_pipeline(language), "cupy" if gpu_activated else "numpy")
                _pipelines[language] = entry
    return entry


def get_spacy_pipeline(language: str = "en") -> spacy.language.Language:
    """Get the current spaCy pipeline for a language, loading it on first use.

    Runs on GPU when one is available (requires cupy); falls back to CPU
    otherwise.
    """
    return _get_pipeline_entry(language)[0]


def run_spacy_pipeline(text: str, language: str = "en"):
    """Run the pipeline on a text, on the backend the pipeline was built for."""
    nlp, backend = _get_pipeline_entry(language)
    with use_ops(backend):
        return nlp(text)


def run_spacy_pipe(texts: list[str], language: str = "en", batch_size: int = 256, entities: bool = True) -> list:
    """Run the pipeline over many texts at once (``nlp.pipe``), optionally
    skipping the CPU-bound entity linker."""
    nlp, backend = _get_pipeline_entry(language)
    disable = [] if entities else ["entityLinker"]
    with use_ops(backend), nlp.select_pipes(disable=disable):
        return list(nlp.pipe(texts, batch_size=batch_size))


def warm_cpu_spacy_pipeline(language: str = "en") -> None:
    """Load and warm a CPU pipeline (~2.5s) so requests can be served
    immediately, without paying the GPU pipeline's one-time ~6s kernel
    compilation first."""
    with _pipeline_lock:
        if language not in _pipelines:
            with use_ops("numpy"):
                cpu_nlp = _load_pipeline(language)
                cpu_nlp("bank")
            _pipelines[language] = (cpu_nlp, "numpy")


def _build_gpu_spacy_pipeline(language: str = "en"):
    """Build and warm a GPU pipeline; returns None when no GPU is available.

    Heavy (one-time ~6s CUDA kernel compilation) — run it in a worker thread.
    """
    if not spacy.prefer_gpu():
        logger.info("No GPU available for spaCy; staying on CPU")
        return None
    gpu_nlp = _load_pipeline(language)
    # entityLinker stays on CPU and its sqlite connection is a process-wide
    # singleton bound to the serving thread, so it must not run here; the
    # transformer stack is what needs the one-time kernel compilation.
    with gpu_nlp.select_pipes(disable=["entityLinker"]):
        gpu_nlp("bank")
    return gpu_nlp


async def swap_spacy_to_gpu(language: str = "en") -> None:
    """Build and warm the GPU pipeline in a worker thread, then publish it.

    The publish runs back on the event loop, between requests, so the swap is
    atomic with respect to request handling.
    """
    gpu_nlp = await asyncio.to_thread(_build_gpu_spacy_pipeline, language)
    if gpu_nlp is not None:
        _pipelines[language] = (gpu_nlp, "cupy")
        logger.info("spaCy pipeline swapped to GPU")
