"""
Training script for word sense disambiguation using masked language modeling.

This script trains a model to predict the correct definition of a word in context
by treating it as a multiple-choice classification task using masked language modeling.
"""

import argparse
import functools
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
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from wsd.benchmark import WordNetExample
from wsd.benchmark import split as split_wn_examples
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

MAX_LENGTH = 2048
EVAL_WN_SEED = 42  # the held-out WordNet split; `python -m wsd.benchmark --split eval` uses the same seed
UNLABELED = -1  # label at the mask position of a prompt that only has a teacher distribution


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
    for _, sentence in candidate_sentences:
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
    args: argparse.Namespace, tokenizer: PreTrainedTokenizer,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Return ``(training_examples, eval_examples)``: generated data (+ optionally
    the non-held-out WordNet examples) and the held-out WordNet eval slice.

    ``wsd.benchmark --split eval`` uses the same split/seed, so eval metrics
    track the final benchmark accuracy without leaking.
    """
    print(f"\nLoading training data from: {args.data_dir}")
    training_examples = load_training_data(args.data_dir, tokenizer, (not args.no_nota_examples))

    eval_examples: list[TrainingExample] = []
    wn_eval: list[WordNetExample] = []
    if args.eval_wn_count > 0 or args.wn_train:
        wn_eval, wn_rest = split_wn_examples(n_eval=args.eval_wn_count, seed=EVAL_WN_SEED)
        if args.eval_wn_count > 0:
            eval_examples = build_examples_from_wn(wn_eval, tokenizer)
            print(f"Held out {len(eval_examples)} wn examples as eval "
                  f"(requested {args.eval_wn_count}, seed {EVAL_WN_SEED})")
        if args.wn_train:
            wn_train_examples = build_examples_from_wn(wn_rest, tokenizer, augment=True)
            print(f"Adding {len(wn_train_examples)} non-held-out wn examples to training")
            training_examples.extend(wn_train_examples)

    if args.semcor:
        from training.semcor import load_raganato, load_sense_index

        semcor_examples = build_examples_from_wn(
            load_raganato(args.semcor, load_sense_index(args.sense_index)), tokenizer, augment=True,
        )
        print(f"Adding {len(semcor_examples)} examples from {args.semcor}")
        training_examples.extend(semcor_examples)
    if args.unlabeled_prompts:
        with open(args.unlabeled_prompts) as f:
            unlabeled = [TrainingExample("", "", "", "", "", json.loads(line)["prompt"]) for line in f]
        print(f"Adding {len(unlabeled)} unlabeled prompts from {args.unlabeled_prompts} (teacher-only loss)")
        training_examples.extend(unlabeled)
    if args.wngt:
        from training.semcor import load_sense_index
        from training.wngt import load_wngt

        held_out = frozenset(ex.synset_id for ex in wn_eval) if args.eval_wn_count > 0 else frozenset()
        wngt_examples = build_examples_from_wn(
            load_wngt(args.wngt, load_sense_index(args.sense_index), parts=frozenset(args.wngt_parts.split(",")),
                      tags=frozenset(args.wngt_tags.split(",")), exclude_synsets=held_out),
            tokenizer, augment=True,
        )
        print(f"Adding {len(wngt_examples)} gloss-corpus examples from {args.wngt}")
        training_examples.extend(wngt_examples)

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
        max_length: int = MAX_LENGTH
    ):
        self.tokenizer = tokenizer
        self.letter_to_compact = {letter: i for i, letter in enumerate(letter_set.letters)}
        self.max_length = max_length

        # Drop prompts whose mask token does not survive truncation (they would cost a forward pass for nothing).
        mask_id = tokenizer.mask_token_id
        self.examples = [ex for ex in examples
                         if mask_id in tokenizer(ex.prompt, truncation=True, max_length=max_length)["input_ids"]]
        if len(self.examples) < len(examples):
            warnings.warn(f"Dropped {len(examples) - len(self.examples)} example(s) whose prompt has no mask token "
                          f"after truncation to max_length={max_length}", stacklevel=2)

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
        # UNLABELED marks a prompt without a gold answer: DistillTrainer trains it on the teacher only
        letter = example.correct_answer_letter
        answer_compact_id = self.letter_to_compact[letter] if letter else UNLABELED

        labels = [-100] * len(input_ids)
        labels[mask_pos] = answer_compact_id

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def collate(features: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch to its longest sequence (labels with -100)."""
    pad = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
    return {k: pad_sequence([torch.tensor(f[k]) for f in features], batch_first=True, padding_value=v)
            for k, v in pad.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a word sense disambiguation model")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-Large-Instruct", help="Model name or path")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data" / "generated.tar.xz",
                        help="Generated data: directory of <word>.json or a .tar.xz of them")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output",
                        help="Directory to save model outputs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=float, default=1, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Maximum number of training steps (-1 for no limit, useful for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-to", type=str, default="wandb",
                        help="Where Trainer should log (e.g. 'wandb', 'none')")
    parser.add_argument("--run-name", type=str, help="Run name for the tracker (defaults to output dir name)")
    parser.add_argument("--eval-steps", type=int, default=500, help="Run eval every N steps")
    parser.add_argument("--eval-wn-count", type=int, default=5000,
                        help="Hold out this many wn benchmark examples as the eval set (0 disables eval)")
    parser.add_argument("--wn-train", action="store_true",
                        help="Also train on the WordNet example sentences that are not held out for eval")
    parser.add_argument("--no-nota-examples", action="store_true",
                        help="Drop the cross-POS 'none of the above' training examples")
    parser.add_argument("--semcor", type=Path,
                        help="Also train on this Raganato-format corpus (prefix of .data.xml/.gold.key.txt)")
    parser.add_argument("--sense-index", type=Path, help="WordNet 3.0 index.sense (required with --semcor/--wngt)")
    parser.add_argument("--wngt", type=Path, help="Also train on the WordNet gloss corpus (glosstag directory)")
    parser.add_argument("--wngt-parts", type=str, default="def,ex", help="def,ex subset to use")
    parser.add_argument("--wngt-tags", type=str, default="man,auto", help="man,auto subset to use")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing applied in the model loss")
    parser.add_argument("--lr-scheduler", type=str, default="linear",
                        help="HuggingFace LR scheduler type (e.g. linear, cosine, cosine_with_restarts)")
    parser.add_argument("--teacher", type=Path, help="distill from this trained WSD model (same letters/prompts)")
    parser.add_argument("--unlabeled-prompts", type=Path,
                        help="jsonl of {\"prompt\": ...} rows (scripts/dump_prompts.py); needs --teacher")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="weight of the KL term vs the label loss")
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--nodes", type=int, help=argparse.SUPPRESS)  # appended by run_distributed.py
    return parser.parse_args(argv)


class DistillTrainer(Trainer):
    """Trainer whose loss mixes the label cross-entropy with KL to a frozen teacher's answer
    distribution at the same mask positions (classic soft-target distillation)."""

    def __init__(self, *args, teacher, alpha: float, temperature: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher, self.alpha, self.temperature = teacher, alpha, temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        positions = (labels != -100).int().argmax(dim=-1)  # the one answer slot per row
        target = labels.gather(1, positions[:, None]).squeeze(1)  # UNLABELED (-1) for teacher-only rows
        batch = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                 "prediction_positions": positions}
        outputs = model(**batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            teacher_logits = self.teacher(**batch).logits.float()
        logits = outputs.logits.float()
        t = self.temperature
        kl = nn.functional.kl_div(
            nn.functional.log_softmax(logits / t, dim=-1), nn.functional.log_softmax(teacher_logits / t, dim=-1),
            log_target=True, reduction="batchmean",
        ) * t * t
        labeled = target >= 0
        if labeled.any():
            ce = nn.functional.cross_entropy(
                logits[labeled], target[labeled], label_smoothing=float(getattr(model.config, "label_smoothing", 0.0)),
            )
            loss = (1 - self.alpha) * ce + self.alpha * kl
        else:
            loss = kl
        return (loss, outputs) if return_outputs else loss


def _trainer(args, device) -> tuple[type, dict]:
    """Plain Trainer, or DistillTrainer with the frozen teacher loaded on ``device``."""
    if not args.teacher:
        if args.unlabeled_prompts:
            raise ValueError("--unlabeled-prompts needs --teacher (their loss is the teacher's distribution)")
        return Trainer, {}
    teacher = WSDModernBertForMaskedLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, attn_implementation=attn_implementation(),
    ).to(device).eval().requires_grad_(False)
    teacher.sparse_prediction = True
    print(f"distilling from {args.teacher} (alpha {args.distill_alpha}, T {args.distill_temperature})")
    return DistillTrainer, {"teacher": teacher, "alpha": args.distill_alpha, "temperature": args.distill_temperature}


def main(argv: list[str] | None = None):
    """Main training function."""
    args = parse_args(argv)
    os.environ.setdefault("WANDB_PROJECT", "modernbert-wsd-training")

    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model and tokenizer. Weights stay fp32 (bf16 autocast happens in the
    # Trainer): with pure-bf16 weights, lr ~3e-5 updates are below bf16's
    # resolution on many weights and get rounded away.
    print(f"Loading model and tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = WSDModernBertForMaskedLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation=attn_implementation(),
    )

    # Run the LM head only on mask positions — every training example has
    # exactly one unmasked label (the answer slot), so the head skips ~250x
    # non-mask positions (avg prompt length ~150, one mask per prompt).
    # Inference uses a parallel path via ``prediction_positions`` in model.py.
    model.sparse_prediction = True
    model.args.label_smoothing = args.label_smoothing  # applied in WSDModernBertForMaskedLM.forward

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

    print(f"Model dtype: {model.dtype}, attention: {model.args._attn_implementation}")

    training_examples, eval_examples = build_examples(args, tokenizer)
    train_dataset = WSDDataset(training_examples, tokenizer, letter_set, MAX_LENGTH)
    eval_dataset = (
        WSDDataset(eval_examples, tokenizer, letter_set, MAX_LENGTH)
        if eval_examples else None
    )
    data_collator = functools.partial(collate, pad_id=tokenizer.pad_token_id)

    if training_examples:
        print(f"Sample prompt:\n{training_examples[0].prompt}\nanswer: {training_examples[0].correct_answer_letter}")

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
        "epoch" if args.max_steps == -1 else "steps"
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name or args.output_dir.name,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,  # float < 1 is a ratio in transformers 5
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=10,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=args.eval_steps if eval_enabled else None,
        save_strategy=save_strategy,
        save_steps=args.eval_steps if save_strategy == "steps" else None,
        save_total_limit=2,
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="accuracy" if eval_enabled else None,
        greater_is_better=True if eval_enabled else None,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=args.report_to,
        seed=args.seed,
    )

    trainer_cls, trainer_kwargs = _trainer(args, training_args.device)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics if eval_enabled else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_enabled else None,
        **trainer_kwargs,
    )

    if args.max_steps > 0:
        print(f"\nStarting training for max {args.max_steps} step(s) (debugging mode)...")
    else:
        print(f"\nStarting training for {args.num_epochs} epoch(s)...")
    print(f"Using device: {training_args.device}, GPUs: {training_args.n_gpu}, bf16: {training_args.bf16}")
    trainer.train()

    final_model_path = args.output_dir / "final"
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
