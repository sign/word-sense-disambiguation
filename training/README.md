# Word Sense Disambiguation Training

This directory contains the training pipeline for fine-tuning ModernBERT-Large-Instruct on word sense disambiguation tasks.

## Overview

The training script fine-tunes the model to predict which definition best fits a word in context, using a multiple-choice format with masked language modeling.

### Key Features

- **POS-aware training**: Only shows definitions from the same part of speech as options
- **"None of the above" examples**: Trains the model to recognize when no definition matches
- **Flexible configuration**: Command-line arguments for easy customization
- **Modular code structure**: Clean, well-documented functions for maintainability
- **Reproducible training**: Configurable random seeds and comprehensive logging

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

The data was compressed using
```shell
tar -I "xz -9e" -cvf generated.tar.xz generated/
```

## Training Process

The training script (`train.py`):

1. Loads all JSON files from `training/data/generated/`
2. For each word and synset:
   - Collects definitions ONLY from synsets with the **same part of speech (POS) tag**
   - For each synset, randomly chooses **one** definition (either source OR alternative)
   - Shuffles the order of definitions randomly
   - Creates multiple-choice prompts with same-POS definitions as options
3. For each example sentence:
   - Marks the target word with asterisks (e.g., `*blare*`)
   - Creates a prompt with shuffled definitions from the same POS
   - Includes "none of the above" as the final option
4. Creates one "none of the above" example per word:
   - Uses a sentence from a different POS than the most frequent one
   - Shows only definitions from the most frequent POS
   - Trains the model to recognize when none of the options match
5. Shuffles all training examples before training
6. Tokenizes the prompts with a [MASK] token at the answer position
7. Trains the model to predict the correct letter (A, B, C, etc.)

## Usage

### Basic Training (Default Settings)

```bash
export WANDB_PROJECT=modernbert-wsd-training
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
python -m training.train --max-steps 1
```

This trains the model for 1 epoch with default settings.

### Advanced Training with Custom Configuration

The training script now supports command-line arguments for easy configuration:

```bash
python -m training.train \
  --model "answerdotai/ModernBERT-Large-Instruct" \
  --data-dir training/data/generated \
  --output-dir training/output2 \
  --batch-size 64 \
  --learning-rate 3e-5 \
  --num-epochs 1 \
  --seed 42
```

#### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `answerdotai/ModernBERT-Large-Instruct` | Model name or path |
| `--data-dir` | Path | `training/data/generated` | Directory containing training data |
| `--output-dir` | Path | `training/output` | Directory to save model outputs |
| `--batch-size` | int | `64` | Training batch size per device |
| `--learning-rate` | float | `3e-5` | Learning rate |
| `--num-epochs` | int | `1` | Number of training epochs |
| `--seed` | int | `42` | Random seed for reproducibility |

### Examples

Train for 3 epochs with larger batch size:
```bash
python -m training.train --num-epochs 3 --batch-size 128
```

Use a different model:
```bash
python -m training.train --model "bert-base-uncased"
```

Custom output directory:
```bash
python -m training.train --output-dir ./my-custom-model
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

**Note on option counts**:
- The number of options varies by word and POS tag
- Only synsets with the same POS are included as options
- Each synset contributes exactly one definition (randomly chosen from source/alternative)
- For example, if "instill" has 5 verb synsets, there will be 5 verb definitions plus "none of the above"

## Configuration

The training script uses the `TrainingConfig` dataclass for configuration management. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `answerdotai/ModernBERT-Large-Instruct` | Base model to fine-tune |
| `data_dir` | `training/data/generated` | Directory with training data |
| `output_dir` | `training/output` | Output directory for checkpoints |
| `max_length` | `2048` | Maximum sequence length |
| `num_epochs` | `1` | Number of training epochs |
| `batch_size` | `64` | Batch size per device |
| `learning_rate` | `3e-5` | Learning rate |
| `warmup_ratio` | `0.1` | Percentage of training for warmup |
| `random_seed` | `42` | Random seed for reproducibility |
| `report_to` | `wandb` | Metrics reporting destination |

These can be overridden via command-line arguments (see Usage section above).

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
- **POS-based filtering**: Only definitions with the same part of speech are shown as options
- **Single definition per synset**: Randomly chooses either source OR alternative definition
- **Definition shuffling**: Each word's definitions are shuffled randomly for variety
- **Example shuffling**: All training examples are shuffled before training starts
- **"None of the above" examples**: One per word, using cross-POS confusion

### Model Architecture
- Uses ModernBERT's masked language modeling objective
- Predicts single-token answers (A, B, C, etc.)
- Only same-POS definitions from a word are presented as options
- Includes "none of the above" as final option

### Code Structure
The training script is organized into modular functions:
- `create_examples_for_synset()`: Creates training examples for a single synset
- `create_none_of_above_example()`: Creates "none of the above" examples
- `load_training_data()`: Orchestrates data loading
- `WSDDataset`: PyTorch dataset for tokenization and labeling
- `WSDDataCollator`: Custom collator for dynamic padding

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
