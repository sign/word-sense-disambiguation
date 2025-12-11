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
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from wsd.prompt import Definition, create_multiple_choice_prompt, get_option_letter

# Constants
DEFAULT_MODEL = "answerdotai/ModernBERT-Large-Instruct"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42
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


@dataclass
class TrainingExample:
    """A single training example with prompt and answer."""
    word: str
    sentence: str
    marked_sentence: str
    correct_synset_id: str
    correct_answer_letter: str
    prompt: str


def mark_word_in_sentence(sentence: str, word: str) -> str:
    """
    Mark the first occurrence of word in sentence with asterisks.

    Args:
        sentence: The sentence containing the word
        word: The word to mark

    Returns:
        The sentence with the first occurrence of word marked with asterisks
    """
    lower_sentence = sentence.lower()
    lower_word = word.lower()

    start_idx = lower_sentence.find(lower_word)
    if start_idx == -1:
        return sentence

    end_idx = start_idx + len(word)
    return sentence[:start_idx] + "*" + sentence[start_idx:end_idx] + "*" + sentence[end_idx:]


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

    # Collect definitions only from synsets with the same POS tag
    pos_definitions = []
    synset_to_definition = {}

    for other_synset in all_synsets:
        if other_synset["pos"] != synset_pos:
            continue

        other_synset_id = other_synset["id"]

        # Randomly pick either source or alternative definition
        chosen_def = random.choice([
            other_synset["source_definition"],
            other_synset["alternative_definition"]
        ])

        def_idx = len(pos_definitions)
        pos_definitions.append(Definition(synset_id=other_synset_id, definition=chosen_def))
        synset_to_definition[other_synset_id] = def_idx

    # Shuffle definitions for variety
    indices = list(range(len(pos_definitions)))
    random.shuffle(indices)
    shuffled_definitions = [pos_definitions[i] for i in indices]
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

    # Create one example for each sentence
    for sentence in synset["examples"]:
        marked_sentence = mark_word_in_sentence(sentence, word)

        # Find correct answer after shuffling
        correct_original_idx = synset_to_definition[synset_id]
        correct_shuffled_idx = index_mapping[correct_original_idx]
        correct_letter = get_option_letter(correct_shuffled_idx)

        prompt = create_multiple_choice_prompt(
            word=word,
            mask_token=tokenizer.mask_token,
            marked_sentence=marked_sentence,
            definitions=shuffled_definitions
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

    # Pick a random synset and sentence from a different POS
    chosen_synset = random.choice(other_pos_synsets)
    chosen_sentence = random.choice(chosen_synset["examples"])
    marked_sentence = mark_word_in_sentence(chosen_sentence, word)

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

    # The correct answer is "none of the above"
    none_letter = get_option_letter(len(frequent_pos_definitions))

    prompt = create_multiple_choice_prompt(
        word=word,
        mask_token=tokenizer.mask_token,
        marked_sentence=marked_sentence,
        definitions=frequent_pos_definitions
    )

    return TrainingExample(
        word=word,
        sentence=chosen_sentence,
        marked_sentence=marked_sentence,
        correct_synset_id=f"{chosen_synset['id']}{NONE_SUFFIX}",
        correct_answer_letter=none_letter,
        prompt=prompt
    )


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
        max_length: int = DEFAULT_MAX_LENGTH
    ):
        """
        Initialize the dataset.

        Args:
            examples: List of training examples
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]

        # Tokenize the prompt
        encoding = self.tokenizer(
            example.prompt,
            truncation=True,
            max_length=self.max_length,
        )

        # Get the token ID for the correct answer letter
        answer_token_id = self.tokenizer.encode(
            example.correct_answer_letter,
            add_special_tokens=False
        )[0]

        # Find the mask token position
        input_ids = encoding["input_ids"]
        mask_token_positions = [
            i for i, token_id in enumerate(input_ids)
            if token_id == self.tokenizer.mask_token_id
        ]

        # Create labels (all -100 except at mask position)
        labels = [-100] * len(input_ids)

        if not mask_token_positions:
            warnings.warn(
                f"No mask token found in prompt for example {idx} "
                f"(word: {example.word}). This example will be skipped during training.",
                stacklevel=2
            )
        else:
            labels[mask_token_positions[0]] = answer_token_id

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
    )

    # Set random seeds for reproducibility
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Print device information
    print_device_info()

    # Load model and tokenizer
    print(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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

    # Create dataset and data collator
    dataset = WSDDataset(training_examples, tokenizer, config.max_length)
    data_collator = WSDDataCollator(tokenizer)

    # Print a sample example
    if training_examples:
        print_sample_example(training_examples[0])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch" if config.max_steps == -1 else "steps",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=config.report_to,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
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
    print("Done!")


if __name__ == "__main__":
    main()
