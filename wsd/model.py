"""
ModernBert-for-MaskedLM subclass whose decoder emits logits only over the
128 answer letters used by WSD multiple-choice prompts.

The input embeddings remain at full tokenizer vocab size (config.vocab_size).
The output decoder is sized to config.answer_vocab_size.

Config fields consumed
----------------------
- ``config.answer_vocab_size`` (int): output dim of the decoder. If missing,
  falls back to ``config.vocab_size`` (so this class behaves like the stock
  ModernBertForMaskedLM).
- ``config.tie_word_embeddings`` must be False once the decoder is pruned
  (otherwise HF will try to re-tie the compact decoder to the full embeddings
  and fail).
"""
from torch import nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertConfig,
    ModernBertForMaskedLM,
)


def _answer_vocab_size(config: ModernBertConfig) -> int:
    return int(getattr(config, "answer_vocab_size", config.vocab_size))


class WSDModernBertForMaskedLM(ModernBertForMaskedLM):
    """ModernBertForMaskedLM with a compact answer-only decoder."""

    # Never tie: decoder shape (answer_vocab_size, hidden) does not match the
    # input embedding table (vocab_size, hidden).
    _tied_weights_keys = None

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        n_out = _answer_vocab_size(config)
        if n_out != config.vocab_size:
            # Replace the default Linear(hidden, vocab_size) with a compact one.
            self.decoder = nn.Linear(config.hidden_size, n_out, bias=config.decoder_bias)
            # Redo initialization for the replaced module.
            self._init_weights(self.decoder)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = self.decoder(self.head(last_hidden_state))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits, labels, vocab_size=self.decoder.out_features, **kwargs
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
