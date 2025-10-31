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
    dataset = load_dataset("karpathy/tiny_shakespeare")

    # TinyShakespeare is one continuous text, typically in 'train' split
    train_split = dataset.get("train", dataset.get("Train", None))
    if train_split is None:
        raise ValueError(
            "Could not find 'train' split in TinyShakespeare dataset")

    # Extract the text - TinyShakespeare usually has 'text' field with one entry
    full_text = ""
    if len(train_split) > 0:
        # Try different ways to get the text
        first_item = train_split[0] if isinstance(
            train_split, list) else train_split
        if isinstance(first_item, dict):
            full_text = first_item.get('text', '')
        elif isinstance(first_item, str):
            full_text = first_item
        else:
            # Try accessing directly from dataset
            full_text = train_split['text'][0] if 'text' in train_split.features else str(
                train_split[0])

    if not full_text:
        raise ValueError("Could not extract text from TinyShakespeare dataset")

    logging.info("TinyShakespeare: %d characters total", len(full_text))

    # Split 90/10 for train/val (same as Karpathy's nanoGPT)
    split_idx = int(len(full_text) * 0.9)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]

    # Split into lines
    train_lines = train_text.split('\n')
    val_lines = val_text.split('\n')

    # Filter empty lines
    train_lines = [line.strip() for line in train_lines if line.strip()]
    val_lines = [line.strip() for line in val_lines if line.strip()]

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
