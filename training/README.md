# Word Sense Disambiguation Training

This directory contains the training pipeline for fine-tuning ModernBERT-Large-Instruct on word sense disambiguation tasks.

## Overview

The training script fine-tunes the model to predict which definition best fits a word in context, using a multiple-choice format with masked language modeling.

## Data Format

Training data is generated using `training/data/generate.py` and stored in `training/data/generated/`. Each JSON file represents one word form with multiple senses:

```json
[
  {
    "id": "omw-en-03610098-n",
    "pos": "n",
    "source_definition": "the main tower within the walls of a medieval castle or fortress",
    "alternative_definition": "the central fortified tower of a castle, often the strongest and most secure part",
    "examples": [
      "The royal guard posted sentries on the battlements while the prisoners were locked in the dungeon beneath the castle keep.",
      "Legends say the king's crown was hidden in a secret vault inside the dungeon of the keep, guarded by enchanted stones.",
      "During the siege, the defenders retreated to the keep, and the attackers tried to breach the dungeon that formed its lower level."
    ]
  }
]
```

## Training Process

The training script (`train.py`):

1. Loads all JSON files from `training/data/generated/`
2. For each word:
   - Collects ALL definitions (both source and alternative) from all synsets
   - Shuffles the order of definitions randomly
   - Creates multiple-choice prompts with ALL definitions as options
3. For each example sentence:
   - Marks the target word with asterisks (e.g., `*blare*`)
   - Creates a prompt with all shuffled definitions
   - Includes "none of the above" as the final option
   - Randomly selects which correct definition to use (source or alternative from the synset)
4. Shuffles all training examples before training
5. Tokenizes the prompts with a [MASK] token at the answer position
6. Trains the model to predict the correct letter (A, B, C, etc.)

## Usage

### Training (1 epoch)

```bash
export WANDB_PROJECT=modernbert-wsd-training
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
python -m training.train
```

This trains the model for 1 complete epoch over all training examples.

### Configuration

To adjust training parameters, modify `training/train.py`:

```python
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,  # Number of epochs
    per_device_train_batch_size=128,  # Batch size (adjust based on GPU memory)
    learning_rate=3e-5,  # Learning rate
    warmup_ratio=0.1,  # Warmup 10% of training
    logging_steps=10,  # Log every 10 steps
    save_strategy="epoch",  # Save at end of each epoch
    bf16=torch.cuda.is_available(),  # Use bf16 if CUDA available
    report_to="wandb",  # Report metrics to Weights & Biases
)
```

## Output

The trained model is saved to:
- `training/output/checkpoint-*` - Checkpoint at each epoch
- `training/output/final/` - Final model after training completes

Load the trained model using:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("training/output/final")
tokenizer = AutoTokenizer.from_pretrained("training/output/final")
```

## Example Training Output

```
Sample training example:
================================================================================
Word: instill
Synset ID: omw-en-00728393-v
Sentence: The vivid mural was designed to instill awe in anyone who passed by.
Correct answer: G

Prompt:
What is the meaning of *instill* in this sentence?

Sentence: The vivid mural was designed to *instill* awe in anyone who passed by.

A. enter drop by drop
B. to create a strong, lasting impression in someone's mind
C. to repeatedly teach or remind someone so that they internalize a lesson
D. fill, as with a certain quality
E. teach and impress by frequent repetitions or admonitions
F. to slowly teach or introduce something over time
G. produce or try to produce a vivid impression of
H. impart gradually
I. to saturate or permeate something with a particular characteristic
J. to seep into something in tiny droplets
K. none of the above

Answer: [unused0] [MASK]
================================================================================

Training metrics:
{'loss': 4.5358, 'epoch': 0.17}
{'loss': 2.5132, 'epoch': 0.33}
{'loss': 2.3022, 'epoch':  0.5}
{'loss': 2.1031, 'epoch': 0.67}
{'loss': 2.7594, 'epoch': 0.83}
{'loss': 2.1934, 'epoch': 1.0}
{'train_runtime': 28.982, 'train_loss': 2.7344929377237954, 'epoch': 1.0}
```

Note: The number of options varies by word. Words with more synsets will have more definition choices (e.g., "instill" has 10 definitions plus "none of the above").

## Configuration

Key parameters in `train.py`:

- `MODEL_NAME`: Base model to fine-tune (default: "answerdotai/ModernBERT-Large-Instruct")
- `num_train_epochs`: Number of training epochs (default: 1)
- `per_device_train_batch_size`: Batch size per device (default: 128)
- `learning_rate`: Learning rate (default: 3e-5)
- `warmup_ratio`: Percentage of training for warmup (default: 0.1)
- `max_length`: Maximum sequence length (default: 512)
- `report_to`: Metrics reporting destination (default: "wandb")

### Using Weights & Biases

The script logs training metrics to Weights & Biases. Set the project name before running:

```bash
export WANDB_PROJECT=modernbert-wsd-training
```

## Requirements

- torch
- transformers
- wandb (for experiment tracking)
- All dependencies from the main project

## Training Details

### Data Preparation
- Each word's definitions are shuffled randomly for variety
- Training examples are shuffled before training starts
- Each example randomly selects either the source or alternative definition as correct

### Model Architecture
- Uses ModernBERT's masked language modeling objective
- Predicts single-token answers (A, B, C, etc.)
- All definitions from a word are presented as options
- Includes "none of the above" as final option

### Prompt Format
Prompts are created using `wsd.prompt.create_multiple_choice_prompt()` which formats questions as:
```
What is the meaning of *word* in this sentence?

Sentence: The *word* appears in context.

A. definition 1
B. definition 2
...
Z. none of the above

Answer: [unused0] [MASK]
```
