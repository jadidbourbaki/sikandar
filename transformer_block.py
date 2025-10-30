"""
This file implements the transformer block for sikandar.
The transformer block is the main building block of the transformer architecture.
"""

import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """One transformer block: attention + feed-forward with residual connections"""

    # A helpful resource here is the original paper on transformers:
    # https://arxiv.org/abs/1706.03762

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        """Initialize the transformer block.
        Args: 
            d_model (int): the dimension of the model
            num_heads (int): the number of attention heads in the transformer blocks
            d_ff (int): the dimension of the feed-forward network
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """Forward pass for the transformer block
        Args:
            x (torch.Tensor): the input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor): the causal mask tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            torch.Tensor: the output tensor of shape (batch_size, seq_len, d_model)
        """

        # Note that a residual connection (or skip connection) adds the
        # layer's input to its output: $output = input + layer(input)$.
        # this helps the model to learn the identity function easier
        # without the residual connection, the code would be:
        # x = self.attention(self.norm1(x), mask)
        # x = self.feed_forward(self.norm2(x))
        # return x

        # self-attention with residual connection
        x = x + self.attention(self.norm1(x), mask)

        # feed-forward with residual connection
        x = x + self.feed_forward(self.norm2(x))
        return x
