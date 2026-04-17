"""
Runtime utility to take a stock ModernBertForMaskedLM and reduce it to a
128-way WSD answer classifier.

Usage (training):

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from wsd.model_surgery import prune_decoder

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    letter_set = prune_decoder(model, tokenizer)

    # model is now a WSDModernBertForMaskedLM with a Linear(hidden, 128) decoder,
    # and labels passed to model.forward must be compact ids in [0, 128).
"""
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

from wsd.letters import LetterSet, build_letters
from wsd.model import WSDModernBertForMaskedLM


class UnexpectedDecoderTypeError(TypeError):
    def __init__(self, actual: type):
        super().__init__(f"expected nn.Linear decoder, got {actual.__name__}")


def prune_decoder(model: ModernBertForMaskedLM, tokenizer: PreTrainedTokenizerBase) -> LetterSet:
    """Prune ``model.decoder`` to the rows corresponding to answer-letter tokens.

    Mutates the model in place:
      * ``model.decoder`` is replaced with ``Linear(hidden, N)`` where N == len(letter_set).
      * The output head is untied from input embeddings (config.tie_word_embeddings=False).
      * ``config.answer_vocab_size`` is set to N.
      * ``model.__class__`` is rebound to ``WSDModernBertForMaskedLM`` so that
        the forward pass uses the compact loss dimension.

    Returns the LetterSet used to select the rows — save this alongside the
    checkpoint so inference can decode compact ids back to their letters.
    """
    letter_set = build_letters(tokenizer)
    allowed_ids = list(letter_set.token_ids)

    decoder = model.decoder
    if not isinstance(decoder, nn.Linear):
        raise UnexpectedDecoderTypeError(type(decoder))

    hidden = decoder.in_features
    n_out = len(allowed_ids)
    has_bias = decoder.bias is not None

    new_decoder = nn.Linear(hidden, n_out, bias=has_bias)
    with torch.no_grad():
        new_decoder.weight.copy_(decoder.weight.data[allowed_ids].clone())
        if has_bias:
            new_decoder.bias.copy_(decoder.bias.data[allowed_ids].clone())
    new_decoder = new_decoder.to(device=decoder.weight.device, dtype=decoder.weight.dtype)

    model.decoder = new_decoder
    model.config.tie_word_embeddings = False
    model.config.answer_vocab_size = n_out
    model.config.architectures = [WSDModernBertForMaskedLM.__name__]

    # Rebind method resolution so model.forward uses the compact loss vocab.
    model.__class__ = WSDModernBertForMaskedLM

    return letter_set
