"""
Training script for word sense disambiguation using masked language modeling.

This script trains a model to predict the correct definition of a word in context
by treating it as a multiple-choice classification task using masked language modeling.
"""

import argparse
import json
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from training.wn_data import WordNetExample
from training.wn_data import split as split_wn_examples
from wsd.benchmark import fetch_synset_definitions, load_wn_english
from wsd.letters import NOTA_LETTER_INDEX, LetterSet, build_letters
from wsd.model import WSDModernBertForMaskedLM
from wsd.model_surgery import prune_decoder
from wsd.prompt import (
    Definition,
    SentenceAlreadyMarkedError,
    WordNotFoundError,
    create_multiple_choice_prompt,
    mark_word_in_sentence,
)

# Constants
DEFAULT_MODEL = "answerdotai/ModernBERT-Large-Instruct"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_LR_SCHEDULER = "linear"
NONE_SUFFIX = "_none"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str = DEFAULT_MODEL
    data_dir: Path = Path(__file__).parent / "data" / "generated"
    output_dir: Path = Path(__file__).parent / "output"
    max_length: int = DEFAULT_MAX_LENGTH
    num_epochs: int = 1
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    random_seed: int = DEFAULT_RANDOM_SEED
    report_to: str = "wandb"
    max_steps: int = -1  # -1 means no limit (train full epochs)
    eval_steps: int = 500  # run eval every N steps
    eval_wn_count: int = 1000  # held-out wn examples used as eval set
    eval_wn_seed: int = 42  # seed controlling wn eval/benchmark split
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING
    lr_scheduler: str = DEFAULT_LR_SCHEDULER


@dataclass
class TrainingExample:
    """A single training example with prompt and answer."""
    word: str
    sentence: str
    marked_sentence: str
    correct_synset_id: str
    correct_answer_letter: str
    prompt: str


def _random_start_offset(n_definitions: int) -> int:
    """Random letter offset that keeps the options block clear of the NOTA slot.

    Training spreads the correct answer across the whole letter range so the
    model doesn't learn "correct answer clusters near A". The offset window
    must leave room for all definitions before NOTA's fixed slot at
    :data:`wsd.letters.NOTA_LETTER_INDEX`.
    """
    max_offset = NOTA_LETTER_INDEX - n_definitions
    return random.randint(0, max_offset) if max_offset > 0 else 0


def create_examples_for_synset(
    synset: dict,
    word: str,
    all_synsets: list[dict],
    tokenizer: PreTrainedTokenizer,
) -> list[TrainingExample]:
    """
    Create training examples for a single synset.

    Args:
        synset: The synset to create examples for
        word: The word form
        all_synsets: All synsets for this word
        tokenizer: The tokenizer to use

    Returns:
        List of training examples for this synset
    """
    examples = []
    synset_id = synset["id"]
    synset_pos = synset["pos"]

    # Build one Definition per same-POS synset, picking a source or alternative
    # definition uniformly; then shuffle in place and locate the correct slot
    # by synset_id. Locating after the shuffle drops the parallel
    # synset_to_definition / index_mapping dicts the old version carried.
    shuffled_definitions = [
        Definition(
            synset_id=s["id"],
            definition=random.choice([s["source_definition"], s["alternative_definition"]]),
        )
        for s in all_synsets if s["pos"] == synset_pos
    ]
    random.shuffle(shuffled_definitions)
    correct_shuffled_idx = next(
        i for i, d in enumerate(shuffled_definitions) if d.synset_id == synset_id
    )

    # Create one example for each sentence
    letters = build_letters(tokenizer).letters
    for sentence in synset["examples"]:
        try:
            marked_sentence = mark_word_in_sentence(sentence, word)
        except (WordNotFoundError, SentenceAlreadyMarkedError):
            # Sentence doesn't contain the word with clean word boundaries
            # (e.g. "100" inside "100th"), or the sentence already uses '*';
            # skip so training matches inference.
            continue

        start_offset = _random_start_offset(len(shuffled_definitions))
        correct_letter = letters[start_offset + correct_shuffled_idx]

        prompt = create_multiple_choice_prompt(
            word=word,
            mask_token=tokenizer.mask_token,
            marked_sentence=marked_sentence,
            definitions=shuffled_definitions,
            tokenizer=tokenizer,
            start_offset=start_offset,
        )

        examples.append(TrainingExample(
            word=word,
            sentence=sentence,
            marked_sentence=marked_sentence,
            correct_synset_id=synset_id,
            correct_answer_letter=correct_letter,
            prompt=prompt
        ))

    return examples


def create_none_of_above_example(
    word: str,
    all_synsets: list[dict],
    most_frequent_pos: str,
    tokenizer: PreTrainedTokenizer,
) -> TrainingExample | None:
    """
    Create a "none of the above" training example.

    This creates an example where the sentence uses a word in one POS,
    but the definitions shown are from a different POS.

    Args:
        word: The word form
        all_synsets: All synsets for this word
        most_frequent_pos: The most frequent POS tag
        tokenizer: The tokenizer to use

    Returns:
        A "none of the above" training example, or None if not possible
    """
    # Find synsets with different POS tags
    other_pos_synsets = [s for s in all_synsets if s["pos"] != most_frequent_pos]

    if not other_pos_synsets:
        return None

    # Pick a random synset+sentence from a different POS where the word appears
    # with clean word boundaries. If no valid sentence exists across any of the
    # other-POS synsets, we cannot build a faithful "none of the above" example.
    candidate_sentences = [
        (s, ex) for s in other_pos_synsets for ex in s["examples"]
    ]
    random.shuffle(candidate_sentences)
    chosen_synset = None
    chosen_sentence = None
    marked_sentence = None
    for s, ex in candidate_sentences:
        try:
            marked_sentence = mark_word_in_sentence(ex, word)
        except (WordNotFoundError, SentenceAlreadyMarkedError):
            continue
        chosen_synset = s
        chosen_sentence = ex
        break
    if marked_sentence is None:
        return None

    # Collect definitions only from the most frequent POS tag
    frequent_pos_definitions = []
    for synset in all_synsets:
        if synset["pos"] == most_frequent_pos:
            chosen_def = random.choice([
                synset["source_definition"],
                synset["alternative_definition"]
            ])
            frequent_pos_definitions.append(
                Definition(synset_id=synset["id"], definition=chosen_def)
            )

    random.shuffle(frequent_pos_definitions)

    # The correct answer is "none of the above" — always the fixed NOTA letter.
    none_letter = build_letters(tokenizer).letters[NOTA_LETTER_INDEX]

    start_offset = _random_start_offset(len(frequent_pos_definitions))

    prompt = create_multiple_choice_prompt(
        word=word,
        mask_token=tokenizer.mask_token,
        marked_sentence=marked_sentence,
        definitions=frequent_pos_definitions,
        tokenizer=tokenizer,
        start_offset=start_offset,
    )

    return TrainingExample(
        word=word,
        sentence=chosen_sentence,
        marked_sentence=marked_sentence,
        correct_synset_id=f"{chosen_synset['id']}{NONE_SUFFIX}",
        correct_answer_letter=none_letter,
        prompt=prompt
    )


def build_eval_examples_from_wn(
    wn_examples: list[WordNetExample],
    tokenizer: PreTrainedTokenizer,
) -> list[TrainingExample]:
    """Convert :class:`WordNetExample` objects into :class:`TrainingExample` objects.

    Mirrors the benchmark's definition-lookup path: for each example, fetch all
    synset definitions for ``(lemma, pos)`` via the wn library and position the
    correct answer at the letter matching its index. Definitions are kept in
    wn iteration order (roughly frequency order) so the more common sense
    lands on earlier letter slots, matching the inference-time API order.
    Skips examples where the correct synset isn't among the fetched definitions
    (can happen when the wn lexicon disagrees with the example's own synset
    metadata), and any where we'd exceed the letter budget.
    """
    en = load_wn_english()
    letters = build_letters(tokenizer).letters
    max_definitions = len(letters) - 1  # last letter reserved for "none of the above"

    out: list[TrainingExample] = []
    for ex in wn_examples:
        defs = fetch_synset_definitions(en, ex.lemma, ex.pos)
        if ex.synset_id not in defs:
            continue
        if len(defs) > max_definitions:
            continue
        definitions = [
            Definition(synset_id=sid, definition=text) for sid, text in defs.items()
        ]
        correct_idx = next(
            i for i, d in enumerate(definitions) if d.synset_id == ex.synset_id
        )
        correct_letter = letters[correct_idx]
        prompt = create_multiple_choice_prompt(
            word=ex.word_form,
            mask_token=tokenizer.mask_token,
            marked_sentence=ex.marked_text,
            definitions=definitions,
            tokenizer=tokenizer,
        )
        out.append(TrainingExample(
            word=ex.word_form,
            sentence=ex.sentence,
            marked_sentence=ex.marked_text,
            correct_synset_id=ex.synset_id,
            correct_answer_letter=correct_letter,
            prompt=prompt,
        ))
    return out


def load_training_data(data_dir: Path, tokenizer: PreTrainedTokenizer) -> list[TrainingExample]:
    """
    Load all training examples from generated JSON files.

    For each word file:
    1. Creates examples for each synset using only same-POS definitions
    2. Creates one "none of the above" example using cross-POS confusion

    Args:
        data_dir: Directory containing JSON files with synset data
        tokenizer: The tokenizer to use for creating prompts

    Returns:
        List of all training examples
    """
    examples = []
    json_files = list(data_dir.glob("*.json"))

    print(f"Loading data from {len(json_files)} files...")

    for json_file in json_files:
        word = json_file.stem

        try:
            with open(json_file) as f:
                synsets = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"Failed to load {json_file}: {e}", stacklevel=2)
            continue

        if not synsets:
            continue

        # Find most frequent POS tag
        pos_counter = Counter(synset["pos"] for synset in synsets)
        most_frequent_pos, _ = pos_counter.most_common(1)[0]

        # Create examples for each synset
        for synset in synsets:
            synset_examples = create_examples_for_synset(synset, word, synsets, tokenizer)
            examples.extend(synset_examples)

        # Create one "none of the above" example per word
        none_example = create_none_of_above_example(word, synsets, most_frequent_pos, tokenizer)
        if none_example:
            examples.append(none_example)

    print(f"Loaded {len(examples)} training examples")
    return examples


class WSDDataset(Dataset):
    """Dataset for word sense disambiguation training."""

    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: PreTrainedTokenizer,
        letter_set: LetterSet,
        max_length: int = DEFAULT_MAX_LENGTH
    ):
        self.tokenizer = tokenizer
        self.letter_to_compact = {letter: i for i, letter in enumerate(letter_set.letters)}
        self.max_length = max_length

        # Filter out examples whose prompt has no mask token after truncation.
        # A mask-less example produces all-(-100) labels, which contributes
        # nothing to the loss but still costs a full forward pass — and the
        # old __getitem__ path emitted one warning per access (N epochs ×
        # num_workers), drowning real warnings. Catch them once, here.
        mask_id = tokenizer.mask_token_id
        kept: list[TrainingExample] = []
        dropped = 0
        for ex in examples:
            input_ids = tokenizer(
                ex.prompt, truncation=True, max_length=max_length,
            )["input_ids"]
            if mask_id in input_ids:
                kept.append(ex)
            else:
                dropped += 1
        if dropped:
            warnings.warn(
                f"Dropped {dropped}/{len(examples)} WSD example(s) whose prompt "
                f"has no mask token after truncation to max_length={max_length}",
                stacklevel=2,
            )
        self.examples = kept

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        encoding = self.tokenizer(
            example.prompt, truncation=True, max_length=self.max_length,
        )
        input_ids = encoding["input_ids"]
        # __init__ guarantees a mask survives truncation, so .index is safe.
        mask_pos = input_ids.index(self.tokenizer.mask_token_id)
        answer_compact_id = self.letter_to_compact[example.correct_answer_letter]

        labels = [-100] * len(input_ids)
        labels[mask_pos] = answer_compact_id

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


class WSDDataCollator:
    """Custom data collator that pads to longest sequence in batch."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the collator.

        Args:
            tokenizer: The tokenizer to use for padding
        """
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of feature dictionaries

        Returns:
            Dictionary of padded tensors
        """
        # Extract and convert to tensors
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        # Pad to longest sequence in batch
        input_ids_padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_padded = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0
        )
        labels_padded = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }


def print_device_info():
    """Print information about available compute devices."""
    print("=" * 80)
    print("DEVICE INFORMATION")
    print("=" * 80)
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  Memory allocated: {allocated:.2f} GB")
            print(f"  Memory reserved: {reserved:.2f} GB")
    else:
        print("WARNING: CUDA not available, training will use CPU")

    print("=" * 80 + "\n")


def print_gpu_memory():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return

    print("\nGPU Memory:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  Device {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def print_sample_example(example: TrainingExample):
    """Print a sample training example."""
    print("\n" + "=" * 80)
    print("Sample training example:")
    print("=" * 80)
    print(f"Word: {example.word}")
    print(f"Synset ID: {example.correct_synset_id}")
    print(f"Sentence: {example.sentence}")
    print(f"Correct answer: {example.correct_answer_letter}")
    print(f"\nPrompt:\n{example.prompt}")
    print("=" * 80)


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a word sense disambiguation model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--data-dir", type=Path, help="Directory containing training data")
    parser.add_argument("--output-dir", type=Path, help="Directory to save model outputs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (-1 for no limit, useful for debugging)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--report-to", type=str, default=TrainingConfig.report_to,
                        help="Where Trainer should log (e.g. 'wandb', 'none')")
    parser.add_argument("--freeze-embeddings", action="store_true",
                        help="Freeze the input embedding layer (~51M params)")
    parser.add_argument("--eval-steps", type=int, default=TrainingConfig.eval_steps,
                        help="Run eval every N steps")
    parser.add_argument("--eval-wn-count", type=int, default=TrainingConfig.eval_wn_count,
                        help="Hold out this many wn benchmark examples as the eval set (0 disables eval)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
        help="Label smoothing factor passed to TrainingArguments.label_smoothing_factor",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=DEFAULT_LR_SCHEDULER,
        help="HuggingFace LR scheduler type (e.g. linear, cosine, cosine_with_restarts)",
    )
    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        model_name=args.model,
        data_dir=args.data_dir or TrainingConfig.data_dir,
        output_dir=args.output_dir or TrainingConfig.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        random_seed=args.seed,
        report_to=args.report_to,
        eval_steps=args.eval_steps,
        eval_wn_count=args.eval_wn_count,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lr_scheduler=args.lr_scheduler,
    )

    # Set random seeds for reproducibility
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Print device information
    print_device_info()

    # Load model and tokenizer
    print(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = WSDModernBertForMaskedLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Run the LM head only on mask positions — every training example has
    # exactly one unmasked label (the answer slot), so the head skips ~250x
    # non-mask positions (avg prompt length ~150, one mask per prompt).
    # Inference uses a parallel path via ``prediction_positions`` in model.py.
    model.sparse_prediction = True

    # If we loaded a pristine checkpoint the decoder is still full-vocab; prune
    # it down to the 128 answer-letter rows. When resuming from a previously
    # pruned checkpoint the decoder already has 128 outputs and we skip prune.
    letter_set = build_letters(tokenizer)
    if model.decoder.out_features != len(letter_set.letters):
        letter_set = prune_decoder(model, tokenizer)
        print(
            f"Pruned decoder to {len(letter_set.letters)} output tokens: "
            f"{''.join(letter_set.letters[:32])}..."
        )
    else:
        print(f"Loaded pre-pruned checkpoint with {len(letter_set.letters)} output tokens")

    # Optionally freeze the input embedding layer (~51M params on ModernBERT).
    if args.freeze_embeddings:
        frozen = 0
        for p in model.model.embeddings.parameters():
            p.requires_grad = False
            frozen += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"Froze embeddings: {frozen/1e6:.1f}M params frozen; "
            f"trainable {trainable/1e6:.1f}M / total {total/1e6:.1f}M"
        )

    if hasattr(model, 'hf_device_map'):
        print(f"\nModel device map: {model.hf_device_map}")
    print(f"Model dtype: {model.dtype}")

    # Load training data
    print(f"\nLoading training data from: {config.data_dir}")
    training_examples = load_training_data(config.data_dir, tokenizer)

    # Shuffle the training examples
    random.shuffle(training_examples)
    print(f"Shuffled {len(training_examples)} training examples")

    # Use a deterministic slice of the wn benchmark set as the eval split.
    # Because benchmark_local.py uses the same split/seed and skips the held-out
    # slice, eval metrics track final benchmark accuracy without leaking.
    if config.eval_wn_count > 0:
        wn_eval, _ = split_wn_examples(
            n_eval=config.eval_wn_count,
            seed=config.eval_wn_seed,
        )
        eval_examples = build_eval_examples_from_wn(wn_eval, tokenizer)
        print(
            f"Held out {len(eval_examples)} wn examples as eval "
            f"(requested {config.eval_wn_count}, seed {config.eval_wn_seed})"
        )
    else:
        eval_examples = []

    # Create datasets and data collator
    train_dataset = WSDDataset(training_examples, tokenizer, letter_set, config.max_length)
    eval_dataset = (
        WSDDataset(eval_examples, tokenizer, letter_set, config.max_length)
        if eval_examples else None
    )
    data_collator = WSDDataCollator(tokenizer)

    # Print a sample example
    if training_examples:
        print_sample_example(training_examples[0])

    # Accuracy on the held-out eval set. With ``sparse_prediction``, the model
    # returns logits of shape (num_masks, answer_vocab) — one row per label
    # that survived the ``!= -100`` filter. ``preprocess_logits_for_metrics``
    # collapses those to predicted compact-ids so Trainer doesn't accumulate
    # per-vocab logits across the eval set. ``compute_metrics`` flattens
    # labels the same way (row-major over (batch, seq), selecting non-ignored
    # positions) so predictions and labels line up 1:1.
    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred  # predictions: (N_masks,), labels: (B, L)
        labels_flat = labels[labels != -100]
        # Alignment between preds and labels depends on both sides flattening
        # row-major; if that invariant ever drifts (e.g. a preprocess hook
        # reshapes labels), accuracy would silently go wrong rather than error.
        assert predictions.shape == labels_flat.shape, (
            f"sparse prediction/label shape mismatch: "
            f"{predictions.shape} vs {labels_flat.shape}"
        )
        correct = (predictions == labels_flat).sum()
        total = labels_flat.size
        return {"accuracy": float(correct) / max(int(total), 1)}

    # When eval is enabled we save at the same cadence so
    # load_best_model_at_end can compare eval metrics to saved checkpoints
    # and restore the best-accuracy one at the end of training.
    eval_enabled = eval_dataset is not None
    save_strategy = "steps" if eval_enabled else (
        "epoch" if config.max_steps == -1 else "steps"
    )
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        label_smoothing_factor=config.label_smoothing,
        lr_scheduler_type=config.lr_scheduler,
        logging_steps=10,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=config.eval_steps if eval_enabled else None,
        save_strategy=save_strategy,
        save_steps=config.eval_steps if save_strategy == "steps" else None,
        save_total_limit=2,
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="accuracy" if eval_enabled else None,
        greater_is_better=True if eval_enabled else None,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=config.report_to,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics if eval_enabled else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_enabled else None,
    )

    # Print training info
    if config.max_steps > 0:
        print(f"\nStarting training for max {config.max_steps} step(s) (debugging mode)...")
    else:
        print(f"\nStarting training for {config.num_epochs} epoch(s)...")
    print(f"Using device: {training_args.device}")
    print(f"Number of GPUs: {training_args.n_gpu}")
    print(f"FP16/BF16: {training_args.fp16}/{training_args.bf16}")
    print_gpu_memory()

    # Train
    trainer.train()

    print_gpu_memory()

    # Save the final model
    print("\nTraining complete!")
    print(f"Model saved to: {config.output_dir}")

    final_model_path = config.output_dir / "final"
    print(f"Saving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save the answer-letter sidecar so consumers can decode compact ids without
    # re-running the tokenizer heuristic.
    sidecar = final_model_path / "answer_letters.json"
    with open(sidecar, "w") as f:
        json.dump({
            "letters": list(letter_set.letters),
            "token_ids_in_source_tokenizer": list(letter_set.token_ids),
            "num_letters": len(letter_set.letters),
        }, f, indent=2)
    print(f"Wrote answer-letter sidecar to: {sidecar}")
    print("Done!")


if __name__ == "__main__":
    main()
