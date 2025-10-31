"""
Story generation using trained Sikandar model
Inspired by smolGPT: https://github.com/Om-Alve/smolGPT
"""

import argparse
import logging
import pathlib
import torch
import torch.nn.functional as F

from model import SikandarModel
from tokenizer import Tokenizer


def load_model(model_path: pathlib.Path, tokenizer: Tokenizer, device: torch.device,
               d_model: int | None = None, num_heads: int | None = None,
               num_layers: int | None = None, max_len: int | None = None) -> SikandarModel:
    """Load trained model

    Args:
        model_path: Path to trained model checkpoint
        tokenizer: Tokenizer instance
        device: Device to run on
        d_model: Model dimension
        num_heads: Number of heads
        num_layers: Number of layers
        max_len: Max sequence length

    Returns:
        Loaded SikandarModel instance
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Use values from checkpoint if available, otherwise use provided args
    d_model = checkpoint.get('d_model', d_model)
    num_heads = checkpoint.get('num_heads', num_heads)
    num_layers = checkpoint.get('num_layers', num_layers)
    max_len = checkpoint.get('max_len', max_len)

    if d_model is None or num_heads is None or num_layers is None or max_len is None:
        raise ValueError(
            "model hyperparameters not found in checkpoint and not provided as arguments")

    vocab_size = tokenizer.get_vocab_size()
    model = SikandarModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def generate_story(model: SikandarModel, tokenizer: Tokenizer, prompt: str,
                   max_new_tokens: int = 200, temperature: float = 0.7,
                   top_k: int = 50, device: torch.device = torch.device('cpu')) -> str:
    """
    Generate a story continuation given a prompt

    Args:
        model: Trained SikandarModel
        tokenizer: Tokenizer instance
        prompt: Story beginning/prompt (e.g., "Once upon a time")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random, lower = more focused)
        top_k: Top-k sampling: only sample from top k most likely tokens
        device: Device to run on

    Returns:
        Generated story text (full text including prompt)
    """
    # Encode prompt (stories don't need role prefixes)
    prompt_tokens = tokenizer.encode(prompt)
    input_tokens = [tokenizer.get_special_token_ids()['<BOS>']] + prompt_tokens

    # Convert to tensor
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    prompt_length = len(input_tokens)

    # Generate tokens
    generated_token_ids = []
    special_token_ids = tokenizer.get_special_token_ids()

    for _ in range(max_new_tokens):
        # Truncate to max_len if needed
        if input_ids.size(1) >= model.pos_embedding.num_embeddings:
            input_ids = input_ids[:, -model.pos_embedding.num_embeddings + 1:]

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)  # (batch=1, seq_len, vocab_size)

        # Get logits for last token
        next_token_logits = logits[0, -1, :].clone()

        # Block undesirable tokens from being sampled
        pad_id = special_token_ids.get('<PAD>')
        unk_id = special_token_ids.get('<UNK>')
        bos_id = special_token_ids.get('<BOS>')

        if pad_id is not None:
            next_token_logits[pad_id] = float('-inf')
        if unk_id is not None:
            next_token_logits[unk_id] = float('-inf')
        if bos_id is not None:
            next_token_logits[bos_id] = float('-inf')

        # Apply temperature
        next_token_logits = next_token_logits / max(temperature, 1e-6)

        # Top-k sampling: keep only top k tokens
        if top_k > 0:
            topk_values, topk_indices = torch.topk(
                next_token_logits, min(top_k, next_token_logits.size(0)))
            filtered_logits = torch.full_like(next_token_logits, float('-inf'))
            filtered_logits.scatter_(0, topk_indices, topk_values)
            next_token_logits = filtered_logits

        # Apply softmax to get probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()

        # Check for EOS (but allow at least a few tokens before stopping)
        eos_id = special_token_ids.get('<EOS>')
        if eos_id is not None and len(generated_token_ids) >= 10:
            if next_token_id == eos_id:
                break

        # Store generated token (only tokens after the prompt)
        if len(input_ids[0]) >= prompt_length:
            generated_token_ids.append(next_token_id)

        # Append to sequence for next iteration
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Decode the full story (prompt + generated tokens)
    full_story_tokens = prompt_tokens + generated_token_ids
    return tokenizer.decode(full_story_tokens)


def main():
    """Main function for story generation"""
    parser = argparse.ArgumentParser(
        description="Generate stories using trained Sikandar model")
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True,
                        help='path to vocabulary JSON file')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                        help='story beginning/prompt')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='number of story samples to generate')
    parser.add_argument('--max-tokens', type=int, default=200,
                        help='maximum tokens to generate per story')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='sampling temperature (0.1-2.0, lower = more focused)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='top-k sampling: only sample from top k tokens (0 = disabled)')
    parser.add_argument('--d-model', type=int, default=None,
                        help='model dimension (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='number of heads (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='number of layers (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--max-len', type=int, default=None,
                        help='max sequence length (auto-loaded from checkpoint if not provided)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("using device: %s", device)

    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_vocab(args.vocab_path)
    logging.info("loaded vocabulary with %d tokens",
                 tokenizer.get_vocab_size())

    # Load model
    logging.info("loading model...")
    model = load_model(
        pathlib.Path(args.model_path),
        tokenizer,
        device,
        args.d_model,
        args.num_heads,
        args.num_layers,
        args.max_len
    )
    logging.info("model loaded!")

    # Generate stories
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n=== Story {i+1}/{args.num_samples} ===")

        story = generate_story(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        print(story)
        print()


if __name__ == "__main__":
    main()
