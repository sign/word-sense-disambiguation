"""
Training script for word sense disambiguation using masked language modeling.

This script trains a model to predict the correct definition of a word in context
by treating it as a multiple-choice classification task using masked language modeling.
"""

import argparse
import io
import json
import os
import random
import tarfile
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
from wsd.letters import NOTA_LETTER_INDEX, LetterSet, build_letters
from wsd.masked_language_model import attn_implementation
from wsd.model import WSDModernBertForMaskedLM
from wsd.model_surgery import prune_decoder
from wsd.prompt import (
    Definition,
    SentenceAlreadyMarkedError,
    WordNotFoundError,
    create_multiple_choice_prompt,
    mark_word_in_sentence,
)
from wsd.word_sense_disambiguation import WordQuery, get_definitions

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
    data_dir: Path = Path(__file__).parent / "data" / "generated.tar.xz"
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
    eval_wn_count: int = 5000  # held-out wn examples used as eval set
    eval_wn_seed: int = 42  # seed controlling wn eval/benchmark split
    wn_train: bool = False  # also train on the non-held-out WordNet examples
    nota_examples: bool = True  # include the one cross-POS "none of the above" example per word
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


def _pos_group(pos: str) -> str:
    """Adjectives ("a") and satellite adjectives ("s") form one option set at
    inference (``get_definitions`` fetches both for any adjective), so training
    must present them together too."""
    return "a" if pos == "s" else pos


def _random_start_offset(n_definitions: int) -> int:
    """Random letter offset that keeps the options block clear of the NOTA slot.

    Training spreads the correct answer across the whole letter range so the
    model doesn't learn "correct answer clusters near A". The offset window
    must leave room for all definitions before NOTA's fixed slot at
    :data:`wsd.letters.NOTA_LETTER_INDEX`.
    """
    max_offset = NOTA_LETTER_INDEX - n_definitions
    return random.randint(0, max_offset) if max_offset > 0 else 0


def _augmented_example(
    word: str,
    sentence: str,
    marked_sentence: str,
    definitions: list[Definition],
    correct_synset_id: str | None,
    tokenizer: PreTrainedTokenizer,
) -> TrainingExample:
    """Build one training example with shuffled options and a random letter offset.

    ``correct_synset_id=None`` means the answer is "none of the above".
    """
    definitions = list(definitions)
    random.shuffle(definitions)
    letters = build_letters(tokenizer).letters
    start_offset = _random_start_offset(len(definitions))
    if correct_synset_id is None:
        correct_letter = letters[NOTA_LETTER_INDEX]
    else:
        correct_idx = next(i for i, d in enumerate(definitions) if d.synset_id == correct_synset_id)
        correct_letter = letters[start_offset + correct_idx]
    prompt = create_multiple_choice_prompt(
        word=word,
        mask_token=tokenizer.mask_token,
        marked_sentence=marked_sentence,
        definitions=definitions,
        tokenizer=tokenizer,
        start_offset=start_offset,
    )
    return TrainingExample(
        word=word,
        sentence=sentence,
        marked_sentence=marked_sentence,
        correct_synset_id=correct_synset_id if correct_synset_id is not None else "",
        correct_answer_letter=correct_letter,
        prompt=prompt,
    )


def create_examples_for_synset(
    synset: dict,
    word: str,
    all_synsets: list[dict],
    tokenizer: PreTrainedTokenizer,
) -> list[TrainingExample]:
    """Create training examples for a single synset: one per example sentence,
    with the same-POS-group synsets as options (one definition each, picking
    source or alternative uniformly)."""
    examples = []
    synset_id = synset["id"]
    group = _pos_group(synset["pos"])

    definitions = [
        Definition(
            synset_id=s["id"],
            definition=random.choice([s["source_definition"], s["alternative_definition"]]),
        )
        for s in all_synsets if _pos_group(s["pos"]) == group
    ]

    for sentence in synset["examples"]:
        try:
            marked_sentence = mark_word_in_sentence(sentence, word)
        except (WordNotFoundError, SentenceAlreadyMarkedError):
            # Sentence doesn't contain the word with clean word boundaries
            # (e.g. "100" inside "100th"), or the sentence already uses '*';
            # skip so training matches inference.
            continue
        examples.append(_augmented_example(word, sentence, marked_sentence, definitions, synset_id, tokenizer))

    return examples


def create_none_of_above_example(
    word: str,
    all_synsets: list[dict],
    most_frequent_group: str,
    tokenizer: PreTrainedTokenizer,
) -> TrainingExample | None:
    """Create a "none of the above" example: a sentence using the word in one
    POS group, with the definitions shown from a different (the most frequent)
    POS group. Returns None if no other-POS sentence can be marked."""
    other_pos_synsets = [s for s in all_synsets if _pos_group(s["pos"]) != most_frequent_group]
    candidate_sentences = [(s, ex) for s in other_pos_synsets for ex in s["examples"]]
    random.shuffle(candidate_sentences)
    for s, sentence in candidate_sentences:
        try:
            marked_sentence = mark_word_in_sentence(sentence, word)
        except (WordNotFoundError, SentenceAlreadyMarkedError):
            continue
        definitions = [
            Definition(
                synset_id=syn["id"],
                definition=random.choice([syn["source_definition"], syn["alternative_definition"]]),
            )
            for syn in all_synsets if _pos_group(syn["pos"]) == most_frequent_group
        ]
        example = _augmented_example(word, sentence, marked_sentence, definitions, None, tokenizer)
        example.correct_synset_id = f"{s['id']}{NONE_SUFFIX}"
        return example
    return None


def build_examples_from_wn(
    wn_examples: list[WordNetExample],
    tokenizer: PreTrainedTokenizer,
    augment: bool = False,
) -> list[TrainingExample]:
    """Convert WordNet examples into prompts using the inference-time option set
    (``get_definitions``: WordNet sense order, adjectives merged with satellites).

    ``augment=False`` (eval) keeps that order and letter offset 0, exactly what
    inference builds. ``augment=True`` (training) shuffles options and
    randomizes the offset like the generated data. Skips examples whose gold
    synset isn't among the fetched options or that exceed the letter budget.
    """
    letters = build_letters(tokenizer).letters
    max_definitions = len(letters) - 1  # last letter reserved for "none of the above"

    all_definitions = get_definitions([WordQuery(form=ex.lemma, pos=ex.pos) for ex in wn_examples])
    out: list[TrainingExample] = []
    for ex, definitions in zip(wn_examples, all_definitions, strict=True):
        if not 0 < len(definitions) <= max_definitions:
            continue
        if not any(d.synset_id == ex.synset_id for d in definitions):
            continue
        if augment:
            out.append(_augmented_example(ex.word_form, ex.sentence, ex.marked_text, definitions,
                                          ex.synset_id, tokenizer))
            continue
        correct_idx = next(i for i, d in enumerate(definitions) if d.synset_id == ex.synset_id)
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
            correct_answer_letter=letters[correct_idx],
            prompt=prompt,
        ))
    return out


def _iter_word_files(data_path: Path):
    """Yield ``(word, synsets)`` from a directory of ``<word>.json`` files or a
    ``.tar.xz`` of them (one bulk read: friendlier to network filesystems)."""
    if data_path.is_dir():
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    yield json_file.stem, json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                warnings.warn(f"Failed to load {json_file}: {e}", stacklevel=2)
        return
    with tarfile.open(data_path, "r:xz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            try:
                yield Path(member.name).stem, json.load(io.TextIOWrapper(tar.extractfile(member), encoding="utf-8"))
            except json.JSONDecodeError as e:
                warnings.warn(f"Failed to load {member.name}: {e}", stacklevel=2)


def load_training_data(data_path: Path, tokenizer: PreTrainedTokenizer,
                       nota_examples: bool = True) -> list[TrainingExample]:
    """Load all training examples from the generated word files.

    For each word:
    1. Creates examples for each synset using only same-POS-group definitions
    2. Creates one "none of the above" example using cross-POS confusion
    """
    examples = []
    n_words = 0
    for word, synsets in _iter_word_files(data_path):
        if not synsets:
            continue
        n_words += 1
        most_frequent_group, _ = Counter(_pos_group(s["pos"]) for s in synsets).most_common(1)[0]
        for synset in synsets:
            examples.extend(create_examples_for_synset(synset, word, synsets, tokenizer))
        none_example = create_none_of_above_example(word, synsets, most_frequent_group, tokenizer)
        if none_example and nota_examples:
            examples.append(none_example)

    print(f"Loaded {len(examples)} training examples from {n_words} words ({data_path})")
    return examples


def build_examples(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Return ``(training_examples, eval_examples)``: generated data (+ optionally
    the non-held-out WordNet examples) and the held-out WordNet eval slice.

    ``wsd.benchmark --split eval`` uses the same split/seed, so eval metrics
    track the final benchmark accuracy without leaking.
    """
    print(f"\nLoading training data from: {config.data_dir}")
    training_examples = load_training_data(config.data_dir, tokenizer, config.nota_examples)

    eval_examples: list[TrainingExample] = []
    wn_eval: list[WordNetExample] = []
    if config.eval_wn_count > 0 or config.wn_train:
        wn_eval, wn_rest = split_wn_examples(n_eval=config.eval_wn_count, seed=config.eval_wn_seed)
        if config.eval_wn_count > 0:
            eval_examples = build_examples_from_wn(wn_eval, tokenizer)
            print(f"Held out {len(eval_examples)} wn examples as eval "
                  f"(requested {config.eval_wn_count}, seed {config.eval_wn_seed})")
        if config.wn_train:
            wn_train_examples = build_examples_from_wn(wn_rest, tokenizer, augment=True)
            print(f"Adding {len(wn_train_examples)} non-held-out wn examples to training")
            training_examples.extend(wn_train_examples)


    random.shuffle(training_examples)
    print(f"Shuffled {len(training_examples)} training examples")
    return training_examples, eval_examples


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
        # nothing to the loss but still costs a full forward pass.
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
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a word sense disambiguation model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--data-dir", type=Path, help="Generated data: directory of <word>.json or a .tar.xz of them")
    parser.add_argument("--output-dir", type=Path, help="Directory to save model outputs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-epochs", type=float, default=1, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Maximum number of training steps (-1 for no limit, useful for debugging)")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--report-to", type=str, default=TrainingConfig.report_to,
                        help="Where Trainer should log (e.g. 'wandb', 'none')")
    parser.add_argument("--run-name", type=str, help="Run name for the tracker (defaults to output dir name)")
    parser.add_argument("--freeze-embeddings", action="store_true",
                        help="Freeze the input embedding layer (~51M params)")
    parser.add_argument("--eval-steps", type=int, default=TrainingConfig.eval_steps, help="Run eval every N steps")
    parser.add_argument("--eval-wn-count", type=int, default=TrainingConfig.eval_wn_count,
                        help="Hold out this many wn benchmark examples as the eval set (0 disables eval)")
    parser.add_argument("--wn-train", action="store_true",
                        help="Also train on the WordNet example sentences that are not held out for eval")
    parser.add_argument("--no-nota-examples", action="store_true",
                        help="Drop the cross-POS 'none of the above' training examples")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING,
                        help="Label smoothing applied in the model loss")
    parser.add_argument("--lr-scheduler", type=str, default=DEFAULT_LR_SCHEDULER,
                        help="HuggingFace LR scheduler type (e.g. linear, cosine, cosine_with_restarts)")
    parser.add_argument("--nodes", type=int, help=argparse.SUPPRESS)  # appended by run_distributed.py
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    """Main training function."""
    args = parse_args(argv)
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
        wn_train=args.wn_train,
        nota_examples=not args.no_nota_examples,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lr_scheduler=args.lr_scheduler,
    )
    os.environ.setdefault("WANDB_PROJECT", "modernbert-wsd-training")

    # Set random seeds for reproducibility
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Load model and tokenizer. Weights stay fp32 (bf16 autocast happens in the
    # Trainer): with pure-bf16 weights, lr ~3e-5 updates are below bf16's
    # resolution on many weights and get rounded away.
    print(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = WSDModernBertForMaskedLM.from_pretrained(
        config.model_name,
        dtype=torch.float32,
        attn_implementation=attn_implementation(),
    )

    # Run the LM head only on mask positions — every training example has
    # exactly one unmasked label (the answer slot), so the head skips ~250x
    # non-mask positions (avg prompt length ~150, one mask per prompt).
    # Inference uses a parallel path via ``prediction_positions`` in model.py.
    model.sparse_prediction = True
    model.config.label_smoothing = config.label_smoothing  # applied in WSDModernBertForMaskedLM.forward

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
    print(f"Model dtype: {model.dtype}, attention: {model.config._attn_implementation}")

    training_examples, eval_examples = build_examples(config, tokenizer)
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
        run_name=args.run_name or config.output_dir.name,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_ratio,  # float < 1 is a ratio in transformers 5
        weight_decay=config.weight_decay,
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
        seed=config.random_seed,
    )

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

    if config.max_steps > 0:
        print(f"\nStarting training for max {config.max_steps} step(s) (debugging mode)...")
    else:
        print(f"\nStarting training for {config.num_epochs} epoch(s)...")
    print(f"Using device: {training_args.device}, GPUs: {training_args.n_gpu}, bf16: {training_args.bf16}")
    print_gpu_memory()

    trainer.train()
    print_gpu_memory()

    final_model_path = config.output_dir / "final"
    print(f"\nTraining complete! Saving final model to: {final_model_path}")
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
    if eval_enabled:
        print(f"Best eval accuracy: {trainer.state.best_metric}")
    print("Done!")
    return final_model_path


if __name__ == "__main__":
    main()
