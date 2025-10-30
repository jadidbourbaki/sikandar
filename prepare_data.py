"""
Download and prepare everyday conversations dataset for training
"""

import argparse
import logging
from pathlib import Path
from datasets import load_dataset

USER_PREFIX = "user: "
ASSISTANT_PREFIX = "assistant: "
DATASET_NAME = "HuggingFaceTB/everyday-conversations-llama3.1-2k"


def prepare_dataset(output_dir: Path, max_train_samples: int = 5000,
                    max_val_samples: int = 500) -> None:
    """
    Download a HuggingFace dataset and prepare it for training

    Args:
        output_dir: Directory to save train.txt and val.txt
        max_train_samples: Maximum number of training dialogues to use
        max_val_samples: Maximum number of validation dialogues to use
    """

    logging.info(
        "downloading %s dataset from HuggingFace...", DATASET_NAME)

    dataset = load_dataset(DATASET_NAME)

    train_dataset = dataset["train_sft"]
    val_dataset = dataset["test_sft"]

    logging.info("dataset loaded. train: %d samples, val: %d samples",
                 len(train_dataset), len(val_dataset))

    # Extract messages from dialogues
    train_lines = []
    val_lines = []

    logging.info("processing training data...")
    train_samples = train_dataset.take(max_train_samples)
    for i, example in enumerate(train_samples):
        messages = example['messages']
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', 'user')
            if content and content.strip():
                # add role prefix to help model distinguish speakers
                if role == 'user':
                    train_lines.append(f"{USER_PREFIX}{content.strip()}")
                else:
                    train_lines.append(f"{ASSISTANT_PREFIX}{content.strip()}")

        if (i + 1) % 1000 == 0:
            logging.info("processed %d/%d training dialogues",
                         i + 1, max_train_samples)

    logging.info("processing validation data...")

    val_samples = val_dataset.take(max_val_samples)
    for i, example in enumerate(val_samples):
        messages = example['messages']
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', 'user')
            if content and content.strip():
                # Add role prefix to help model distinguish speakers
                if role == 'user':
                    val_lines.append(f"{USER_PREFIX}{content.strip()}")
                else:
                    val_lines.append(f"{ASSISTANT_PREFIX}{content.strip()}")

        if (i + 1) % 100 == 0:
            logging.info("Processed %d/%d validation dialogues",
                         i + 1, max_val_samples)

    # save to files
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"

    logging.info("writing %d training lines to %s",
                 len(train_lines), train_file)
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))

    logging.info("writing %d validation lines to %s", len(val_lines), val_file)
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

    logging.info("data preparation complete!")
    logging.info("training samples: %d", len(train_lines))
    logging.info("validation samples: %d", len(val_lines))


def prepare_data() -> None:
    """Prepare a dataset for training"""
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for training")
    parser.add_argument('--output-dir', type=str, default='data',
                        help='directory to save train.txt and val.txt')
    parser.add_argument('--max-train', type=int, default=5000,
                        help='maximum number of training samples to use')
    parser.add_argument('--max-val', type=int, default=500,
                        help='maximum number of validation samples to use')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    prepare_dataset(
        output_dir=Path(args.output_dir),
        max_train_samples=args.max_train,
        max_val_samples=args.max_val
    )


if __name__ == '__main__':
    prepare_data()
