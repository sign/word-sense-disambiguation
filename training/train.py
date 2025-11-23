import json
import random
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from wsd.prompt import Definition, create_multiple_choice_prompt, get_option_letter


@dataclass
class TrainingExample:
    """A single training example with prompt and answer"""
    word: str
    sentence: str
    marked_sentence: str
    correct_synset_id: str
    correct_answer_letter: str
    prompt: str


def mark_word_in_sentence(sentence: str, word: str) -> str:
    """Mark the first occurrence of word in sentence with asterisks (case-insensitive)"""
    # Find the word in the sentence (case-insensitive)
    lower_sentence = sentence.lower()
    lower_word = word.lower()

    # Find position
    start_idx = lower_sentence.find(lower_word)
    if start_idx == -1:
        # Word not found as exact match, return original
        return sentence

    # Extract the actual word with original case
    end_idx = start_idx + len(word)
    marked = sentence[:start_idx] + "*" + sentence[start_idx:end_idx] + "*" + sentence[end_idx:]
    return marked


def load_training_data(data_dir: Path, tokenizer) -> list[TrainingExample]:
    """Load all training examples from generated JSON files"""
    examples = []

    json_files = list(data_dir.glob("*.json"))
    print(f"Loading data from {len(json_files)} files...")

    for json_file in json_files:
        # The word form is the filename without extension
        word = json_file.stem

        with open(json_file) as f:
            synsets = json.load(f)

        # Collect all definitions for this word (both source and alternative)
        all_definitions = []
        synset_to_definitions = {}  # Maps synset_id to list of (definition_text, def_index)

        for synset in synsets:
            synset_id = synset["id"]
            source_def = synset["source_definition"]
            alt_def = synset["alternative_definition"]

            # Add both definitions
            source_idx = len(all_definitions)
            all_definitions.append(Definition(synset_id=synset_id, definition=source_def))

            alt_idx = len(all_definitions)
            all_definitions.append(Definition(synset_id=synset_id, definition=alt_def))

            # Track which definitions belong to this synset
            if synset_id not in synset_to_definitions:
                synset_to_definitions[synset_id] = []
            synset_to_definitions[synset_id].extend([source_idx, alt_idx])

        # Shuffle the order of definitions for variety
        indices = list(range(len(all_definitions)))
        random.shuffle(indices)
        shuffled_definitions = [all_definitions[i] for i in indices]

        # Create reverse mapping to find correct answer after shuffling
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        # For each synset, create training examples from its example sentences
        for synset in synsets:
            synset_id = synset["id"]

            # For each example sentence
            for sentence in synset["examples"]:
                # Mark the word in the sentence
                marked_sentence = mark_word_in_sentence(sentence, word)

                # Find which definitions are correct for this synset
                correct_indices = synset_to_definitions[synset_id]

                # Pick one of the correct definitions randomly
                correct_original_idx = random.choice(correct_indices)
                correct_shuffled_idx = index_mapping[correct_original_idx]
                correct_letter = get_option_letter(correct_shuffled_idx)

                # Create the prompt using wsd.prompt format
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

    print(f"Loaded {len(examples)} training examples")
    return examples


class WSDDataset(Dataset):
    """Dataset for word sense disambiguation training"""

    def __init__(self, examples: list[TrainingExample], tokenizer, max_length: int = 2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize the prompt without padding (collator will handle padding)
        encoding = self.tokenizer(
            example.prompt,
            truncation=True,
            max_length=self.max_length,
        )

        # Get the token ID for the correct answer letter
        answer_token_id = self.tokenizer.encode(example.correct_answer_letter, add_special_tokens=False)[0]

        # Find the mask token position
        input_ids = encoding["input_ids"]
        mask_token_positions = [i for i, token_id in enumerate(input_ids) if token_id == self.tokenizer.mask_token_id]

        # Create labels (all -100 except at mask position)
        labels = [-100] * len(input_ids)

        if len(mask_token_positions) == 0:
            warnings.warn(f"No mask token found in prompt for example {idx} (word: {self.examples[idx].word}). Skipping this example.")
        else:
            labels[mask_token_positions[0]] = answer_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


class WSDDataCollator:
    """Custom data collator that pads to longest sequence in batch"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Configuration
    MODEL_NAME = "answerdotai/ModernBERT-Large-Instruct"
    DATA_DIR = Path(__file__).parent / "data" / "generated"
    OUTPUT_DIR = Path(__file__).parent / "output"

    # Check CUDA availability
    print("="*80)
    print("DEVICE INFORMATION")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, training will use CPU")
    print("="*80 + "\n")

    print(f"Loading model and tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Print device map
    if hasattr(model, 'hf_device_map'):
        print(f"\nModel device map: {model.hf_device_map}")
    print(f"Model dtype: {model.dtype}")

    print(f"\nLoading training data from: {DATA_DIR}")
    training_examples = load_training_data(DATA_DIR, tokenizer)

    # Shuffle the training examples
    random.shuffle(training_examples)
    print(f"Shuffled {len(training_examples)} training examples")

    # Create dataset and data collator
    dataset = WSDDataset(training_examples, tokenizer)
    data_collator = WSDDataCollator(tokenizer)

    # Print a sample
    print("\n" + "="*80)
    print("Sample training example:")
    print("="*80)
    sample = training_examples[0]
    print(f"Word: {sample.word}")
    print(f"Synset ID: {sample.correct_synset_id}")
    print(f"Sentence: {sample.sentence}")
    print(f"Correct answer: {sample.correct_answer_letter}")
    print(f"\nPrompt:\n{sample.prompt}")
    print("="*80)

    # Training arguments - configured for 1 epoch
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=64,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),  # Use bf16 if CUDA available
        dataloader_num_workers=0,
        report_to="wandb",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting training for 1 epoch...")
    print(f"Using device: {training_args.device}")
    print(f"Number of GPUs: {training_args.n_gpu}")
    print(f"FP16/BF16: {training_args.fp16}/{training_args.bf16}")

    if torch.cuda.is_available():
        print("\nGPU Memory before training:")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(i) / 1e9:.2f} GB reserved")

    trainer.train()

    if torch.cuda.is_available():
        print("\nGPU Memory after training:")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(i) / 1e9:.2f} GB reserved")

    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT_DIR}")

    # Save the final model
    print("Saving final model...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Final model saved to: {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
