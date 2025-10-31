"""
Download and prepare TinyShakespeare dataset for training
Similar to Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
"""

import argparse
import logging
from pathlib import Path
from datasets import load_dataset


def prepare_dataset(output_dir: Path) -> None:
    """
    Download TinyShakespeare dataset from HuggingFace and prepare it for training
    Uses the same 90/10 train/val split as Karpathy's nanoGPT

    Args:
        output_dir: Directory to save train.txt and val.txt
    """

    logging.info("downloading TinyShakespeare dataset from HuggingFace...")
    dataset = load_dataset("Trelis/tiny-shakespeare")

    # Trelis/tiny-shakespeare already has train/test splits (90/10)
    train_dataset = dataset.get("train", dataset.get("Train", None))
    val_dataset = dataset.get("test", dataset.get("Test", None))

    if train_dataset is None:
        raise ValueError(
            "Could not find 'train' split in TinyShakespeare dataset")
    if val_dataset is None:
        raise ValueError(
            "Could not find 'test' split in TinyShakespeare dataset")

    logging.info("dataset loaded. train: %d rows, test: %d rows",
                 len(train_dataset), len(val_dataset))

    # Extract text from each row (dataset has 'Text' field)
    train_lines = []
    val_lines = []

    logging.info("processing training data...")
    for example in train_dataset:
        text = example.get('Text', example.get('text', ''))
        if text and text.strip():
            train_lines.append(text.strip())

    logging.info("processing validation data...")
    for example in val_dataset:
        text = example.get('Text', example.get('text', ''))
        if text and text.strip():
            val_lines.append(text.strip())

    logging.info("split TinyShakespeare: %d train lines, %d val lines",
                 len(train_lines), len(val_lines))

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
    """Prepare TinyShakespeare dataset for training"""
    parser = argparse.ArgumentParser(
        description="Prepare TinyShakespeare dataset for training")
    parser.add_argument('--output-dir', type=str, default='data',
                        help='directory to save train.txt and val.txt')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    prepare_dataset(output_dir=Path(args.output_dir))


if __name__ == '__main__':
    prepare_data()
