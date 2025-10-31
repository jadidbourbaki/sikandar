"""
Download and prepare TinyStories dataset for story generation training
"""

import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"
    max_train_samples: int = 10000
    max_val_samples: int = 1000


def prepare_dataset(output_dir: Path, dataset_config: DatasetConfig) -> None:
    """
    Download dataset from HuggingFace and prepare it for training

    Args:
        output_dir: Directory to save train.txt and val.txt
        max_train_samples: Maximum number of training stories to use
        max_val_samples: Maximum number of validation stories to use
    """

    logging.info(
        "downloading %s dataset from HuggingFace...", dataset_config.name)

    # TinyStories dataset has 'train' and 'validation' splits
    dataset = load_dataset(dataset_config.name)

    train_dataset = dataset.get(dataset_config.train_split, dataset.get(
        dataset_config.train_split.title(), None))
    val_dataset = dataset.get(dataset_config.val_split, dataset.get(
        dataset_config.val_split.title(), None))

    if train_dataset is None:
        raise ValueError("Could not find training split in dataset")
    if val_dataset is None:
        raise ValueError("Could not find validation split in dataset")

    logging.info("dataset loaded. train: %d samples, val: %d samples",
                 len(train_dataset), len(val_dataset))

    # Extract story texts
    train_lines = []
    val_lines = []

    logging.info("processing training data...")
    train_samples = train_dataset.take(dataset_config.max_train_samples)
    for i, example in enumerate(train_samples):
        # TinyStories has a 'text' field with the story
        story_text = example.get('text', example.get('story', ''))
        if story_text and story_text.strip():
            train_lines.append(story_text.strip())

        if (i + 1) % 1000 == 0:
            logging.info("processed %d/%d training stories",
                         i + 1, dataset_config.max_train_samples)

    logging.info("processing validation data...")

    val_samples = val_dataset.take(dataset_config.max_val_samples)
    for i, example in enumerate(val_samples):
        story_text = example.get('text', example.get('story', ''))
        if story_text and story_text.strip():
            val_lines.append(story_text.strip())

        if (i + 1) % 100 == 0:
            logging.info("Processed %d/%d validation stories",
                         i + 1, dataset_config.max_val_samples)

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
    parser.add_argument('--max-train', type=int, default=10000,
                        help='maximum number of training stories to use')
    parser.add_argument('--max-val', type=int, default=1000,
                        help='maximum number of validation stories to use')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    prepare_dataset(
        output_dir=Path(args.output_dir),
        dataset_config=DatasetConfig(
            max_train_samples=args.max_train,
            max_val_samples=args.max_val
        )
    )


if __name__ == '__main__':
    prepare_data()
