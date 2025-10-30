"""
This file implements the feed-forward network for sikandar.
The feed-forward network is a simple 2-layer multi-layer perceptron (MLP) 
with a rectified linear unit (ReLU) activation function between the layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Simple 2-layer MLP with ReLU"""

    def __init__(self, d_model: int, d_ff: int):
        """Initialize the feed-forward network.
        Args:
            d_model (int): the dimension of the model
            d_ff (int): the dimension of the feed-forward network
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the feed-forward network
        Args:
            x (torch.Tensor): the input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: the output tensor of shape (batch_size, seq_len, d_model)
        """
        # the feed-forward network is a simple 2-layer MLP with ReLU activation function
        # the first layer is a linear layer that projects the
        # input tensor to the feed-forward network. the second layer is a another linear layer
        # that projects the output of the feed-forward network back to the model
        return self.linear2(F.relu(self.linear1(x)))
