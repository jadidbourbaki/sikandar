"""
Simple chatbot using trained Sikandar model
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


def generate(model: SikandarModel, tokenizer: Tokenizer, prompt: str,
             max_new_tokens: int = 50, temperature: float = 0.8,
             device: torch.device = torch.device('cpu')) -> str:
    """
    Generate text given a prompt

    Args:
        model: Trained SikandarModel
        tokenizer: Tokenizer instance
        prompt: Input text string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to run on

    Returns:
        Generated text (only new tokens, not including the prompt)
    """
    # Encode prompt with role prefix (without BOS, we'll add it)
    formatted_prompt = f"user: {prompt}\nassistant:"
    prompt_tokens = tokenizer.encode(formatted_prompt)
    input_tokens = [tokenizer.get_special_token_ids()['<BOS>']] + prompt_tokens

    # Convert to tensor
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    prompt_length = len(input_tokens)

    # Generate tokens
    generated_token_ids = []
    for _ in range(max_new_tokens):
        # Truncate to max_len if needed
        if input_ids.size(1) >= model.pos_embedding.num_embeddings:
            input_ids = input_ids[:, -model.pos_embedding.num_embeddings + 1:]

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)  # (batch=1, seq_len, vocab_size)

        # Get logits for last token
        next_token_logits = logits[0, -1, :] / temperature

        # Apply softmax to get probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()

        # Check for EOS
        if next_token_id == tokenizer.get_special_token_ids()['<EOS>']:
            break

        # Store generated token (only tokens after the prompt)
        if len(input_ids[0]) >= prompt_length:
            generated_token_ids.append(next_token_id)

        # Append to sequence for next iteration
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Decode only the generated tokens (not the prompt)
    return tokenizer.decode(generated_token_ids)


def chat(model: SikandarModel, tokenizer: Tokenizer, device: torch.device,
         max_new_tokens: int = 50, temperature: float = 0.8):
    """Interactive chat interface

    Args:
        model: Trained SikandarModel
        tokenizer: Tokenizer instance
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
    """
    print("hi, i am sikandar! feel free to chat with me (type 'quit' to exit)")

    while True:
        # Get user input
        user_input = input("you: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("bye!")
            break

        if not user_input:
            continue

        # Generate response
        try:
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device
            )

            # Response already contains only new tokens, so just print it
            if response.strip():
                print(f"sikandar: {response}")
            else:
                print("sikandar: (no response generated)")

        except (ValueError, RuntimeError, IndexError) as e:
            print(f"error: {e}")
            logging.error("generation error: %s", e, exc_info=True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="chat with sikandar")
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True,
                        help='path to vocabulary JSON file')
    parser.add_argument('--d-model', type=int, default=None,
                        help='model dimension (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='number of heads (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='number of layers (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--max-len', type=int, default=None,
                        help='max sequence length (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='sampling temperature (0.1-2.0)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_vocab(args.vocab_path)
    logging.info("loaded vocabulary with %d tokens",
                 tokenizer.get_vocab_size())

    # Load model
    logging.info("loading model...")
    model = load_model(
        args.model_path,
        tokenizer,
        device,
        args.d_model,
        args.num_heads,
        args.num_layers,
        args.max_len
    )
    logging.info("model loaded!")

    # Start chat
    chat(model=model,
         tokenizer=tokenizer,
         device=device,
         max_new_tokens=args.max_tokens,
         temperature=args.temperature)


if __name__ == "__main__":
    main()
